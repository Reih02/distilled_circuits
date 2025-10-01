import argparse
from typing import List, Tuple, Dict, Optional
import torch
import numpy as np
import re
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoConfig
import math
import random
from tqdm import tqdm
import itertools
from sklearn.decomposition import PCA

# ----------------------------
# Command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Compute alignment score between two Transformer models (CLM or MLM).")
parser.add_argument("--teacher_model", type=str, default="meta-llama/Llama-3.1-8B", help="Hugging Face model ID for the teacher model.")
parser.add_argument("--student_model", type=str, default="nvidia/Llama-3.1-Minitron-4B-Depth-Base", help="Hugging Face model ID for the student model.")
parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use (informational, as device_map='auto' is used).")
parser.add_argument("--num_samples", type=int, default=None, help="Number of text samples to run on. If None, all texts are used.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling reproducibility.")
args = parser.parse_args()

# ----------------------------
# Model Type Helper
# ----------------------------
def is_mlm(config):
    """Check if a model config is for a Masked Language Model."""
    model_type_str = getattr(config, "model_type", "").lower()
    return "distilbert" in model_type_str or "bert" in model_type_str

# ----------------------------
# Helpers
# ----------------------------
def extract_last_numeral(text: str) -> Optional[int]:
    nums = re.findall(r'\b\d+\b', text)
    return int(nums[-1]) if nums else None

def infer_tokens(texts):
    next_tokens, target_tokens, distractor_tokens = [], [], []
    for t in texts:
        last = extract_last_numeral(t)
        if last is None: last = 0
        next_tokens.append(" " + str(last + 1))
        target_tokens.append(str(last + 1))
        distractor_tokens.append(str(last))
    return next_tokens, target_tokens, distractor_tokens

# ----------------------------
# Forward & Logit Diff Functions
# ----------------------------
def get_logits_causal_lm(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
    return outputs

def compute_logit_diff_causal_lm_smart(model, tokenizer, texts):
    """
    Computes logit difference, automatically adapting to the model's tokenizer.
    """
    test_target_ids = tokenizer.encode(" 5", add_special_tokens=False)
    test_distractor_ids = tokenizer.encode(" 4", add_special_tokens=False)
    use_llama_style_logic = False
    if not test_target_ids or not test_distractor_ids or test_target_ids[0] == test_distractor_ids[0]:
        use_llama_style_logic = True
        if len(tokenizer.encode("5", add_special_tokens=False)) != 1:
            print(f"WARNING: Tokenizer for {model.config.model_type} cannot handle single-digit numerals. Logit diff may be inaccurate.")
            use_llama_style_logic = False

    diffs = []
    for text in texts:
        last = extract_last_numeral(text)
        if last is None: continue
        try:
            if use_llama_style_logic:
                target_ids = tokenizer.encode(str(last + 1), add_special_tokens=False)
                distractor_ids = tokenizer.encode(str(last), add_special_tokens=False)
                if len(target_ids) != 1 or len(distractor_ids) != 1: continue
                target_id, distractor_id = target_ids[0], distractor_ids[0]
                prompt = text + " "
            else:
                target_id = tokenizer.encode(" " + str(last + 1), add_special_tokens=False)[0]
                distractor_id = tokenizer.encode(" " + str(last), add_special_tokens=False)[0]
                prompt = text
        except (IndexError, TypeError):
            continue
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs, return_dict=True).logits
        pred_pos = inputs.input_ids.shape[1] - 1
        logits_slice = logits[0, pred_pos]
        diffs.append((logits_slice[target_id] - logits_slice[distractor_id]).item())
    return float(np.mean(diffs)) if diffs else 0.0

def get_logits_mlm(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(model.device)
    mask_token_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
    return outputs, mask_token_indices

def compute_logit_diff_mlm_dynamic(logits, mask_token_indices, tokenizer, target_tokens, distractor_tokens):
    diffs = []
    for i, (b, pos) in enumerate(zip(*mask_token_indices)):
        tgt_id = tokenizer.convert_tokens_to_ids(target_tokens[i])
        dis_id = tokenizer.convert_tokens_to_ids(distractor_tokens[i])
        diffs.append((logits[b, pos, tgt_id] - logits[b, pos, dis_id]).item())
    return float(np.mean(diffs)) if diffs else 0.0

# ----------------------------
# Vector Extraction
# ----------------------------
def get_head_vectors_causal_lm(model, tokenizer, texts, top_heads):
    if not top_heads: return {}
    outputs = get_logits_causal_lm(model, tokenizer, texts)
    if outputs.attentions is None: raise RuntimeError("Causal LM returned no attentions.")
    vectors = {}
    for layer, head in top_heads:
        if layer < len(outputs.attentions):
            vectors[(layer, head)] = outputs.attentions[layer][:, head].mean(dim=0)
    return vectors

def get_mlp_vectors_causal_lm(model, tokenizer, texts, top_layers):
    if not top_layers: return {}
    outputs = get_logits_causal_lm(model, tokenizer, texts)
    vectors = {}
    for layer in top_layers:
        if layer + 1 < len(outputs.hidden_states):
            vectors[layer] = outputs.hidden_states[layer + 1].mean(dim=1)
    return vectors

def get_head_vectors_distilbert(model, tokenizer, texts, top_heads):
    if not top_heads: return {}
    outputs, _ = get_logits_mlm(model, tokenizer, texts)
    if outputs.attentions is None: raise RuntimeError("MLM returned no attentions.")
    vectors = {}
    for layer, head in top_heads:
        vectors[(layer, head)] = outputs.attentions[layer][:, head].mean(dim=0)
    return vectors

def get_mlp_vectors_distilbert(model, tokenizer, texts, top_layers):
    if not top_layers: return {}
    outputs, _ = get_logits_mlm(model, tokenizer, texts)
    vectors = {}
    for layer in top_layers:
        if layer + 1 < len(outputs.hidden_states):
            vectors[layer] = outputs.hidden_states[layer + 1].mean(dim=1)
    return vectors

# ----------------------------
# Similarity & Matching
# ----------------------------
def activation_eigenvector_similarity(acts1, acts2, k=3):
    n_samples = acts1.shape[0]
    n_components = min(k, n_samples)
    if n_components == 0: return 0.0
    pca1 = PCA(n_components=n_components)
    pca2 = PCA(n_components=n_components)
    pca1.fit(acts1)
    pca2.fit(acts2)
    U1_top_k, U2_top_k = pca1.components_, pca2.components_
    similarity_matrix = np.dot(U1_top_k, U2_top_k.T)
    return float(np.mean(np.abs(np.diag(similarity_matrix))))

def compute_best_matches_attn(teacher_dict, student_dict):
    matches = {}
    for t_key, t_mat in tqdm(teacher_dict.items(), desc="Matching Attention Heads"):
        best_sim, best_key = -1.0, None
        for s_key, s_mat in student_dict.items():
            min_shape = min(t_mat.shape[0], s_mat.shape[0])
            t_vec = t_mat[:min_shape, :min_shape].reshape(-1).to(torch.float32)
            s_vec = s_mat[:min_shape, :min_shape].reshape(-1).to(torch.float32)
            sim = F.cosine_similarity(t_vec, s_vec, dim=0).item()
            if sim > best_sim: best_sim, best_key = sim, s_key
        matches[t_key] = (best_key, best_sim)
    return matches

def compute_best_matches_mlp(teacher_dict, student_dict):
    matches = {}
    for t_key, t_vec in tqdm(teacher_dict.items(), desc="Matching MLP Layers"):
        best_sim, best_key = -1.0, None
        for s_key, s_vec in student_dict.items():
            sim = activation_eigenvector_similarity(t_vec.detach().float().cpu().numpy(), s_vec.detach().float().cpu().numpy())
            if sim > best_sim: best_sim, best_key = sim, s_key
        matches[t_key] = (best_key, best_sim)
    return matches

def compute_alignment_score(pairs: List[Tuple[float, float, float]]) -> float:
    return sum(S * (1 - abs(I_S - I_T)) for I_S, I_T, S in pairs) / len(pairs) if pairs else 0.0

# ----------------------------
# Ablation Hooks
# ----------------------------
def get_ablation_hook(head_idx, model_config):
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    start, end = head_idx * head_dim, (head_idx + 1) * head_dim
    def ablate_head(module, input, output):
        output_to_modify = output[0] if isinstance(output, tuple) else output
        output_to_modify[:, :, start:end] = 0
        return output
    return ablate_head

def get_mlp_ablation_hook():
    return lambda module, input, output: torch.zeros_like(output)

# ----------------------------
# Impact Calculation
# ----------------------------
def calculate_head_impacts_causal_lm(model, tokenizer, texts, baseline_diff):
    num_layers, num_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    impacts = []
    iterator = itertools.product(range(num_layers), range(num_heads))
    for layer, head in tqdm(iterator, total=num_layers*num_heads, desc="Calculating Head Impacts (CLM)"):
        module = model.transformer.h[layer].attn.c_proj if "gpt2" in model.config.model_type.lower() else model.model.layers[layer].self_attn.o_proj
        hook = module.register_forward_hook(get_ablation_hook(head, model.config))
        try:
            ablated_diff = compute_logit_diff_causal_lm_smart(model, tokenizer, texts)
            impacts.append(((layer, head), abs(100 * (1 - ablated_diff / baseline_diff) if baseline_diff != 0 else 0)))
        finally: hook.remove()
    return impacts

def calculate_mlp_impacts_causal_lm(model, tokenizer, texts, baseline_diff):
    num_layers = model.config.num_hidden_layers
    impacts = []
    for layer in tqdm(range(num_layers), desc="Calculating MLP Impacts (CLM)"):
        module = model.transformer.h[layer].mlp if "gpt2" in model.config.model_type.lower() else model.model.layers[layer].mlp
        hook = module.register_forward_hook(get_mlp_ablation_hook())
        try:
            ablated_diff = compute_logit_diff_causal_lm_smart(model, tokenizer, texts)
            impacts.append((layer, abs(100 * (1 - ablated_diff / baseline_diff) if baseline_diff != 0 else 0)))
        finally: hook.remove()
    return impacts

def calculate_head_impacts_distilbert(model, tokenizer, texts, target_tokens, distractor_tokens, baseline_diff):
    num_layers, num_heads = model.config.n_layers, model.config.n_heads
    impacts = []
    iterator = itertools.product(range(num_layers), range(num_heads))
    for layer, head in tqdm(iterator, total=num_layers*num_heads, desc="Calculating Head Impacts (MLM)"):
        module = model.distilbert.transformer.layer[layer].attention.out_lin
        hook = module.register_forward_hook(get_ablation_hook(head, model.config))
        try:
            outputs_ablated, mask_indices_ablated = get_logits_mlm(model, tokenizer, texts)
            ablated_diff = compute_logit_diff_mlm_dynamic(outputs_ablated.logits, mask_indices_ablated, tokenizer, target_tokens, distractor_tokens)
            impacts.append(((layer, head), abs(100 * (1 - ablated_diff / baseline_diff) if baseline_diff != 0 else 0)))
        finally: hook.remove()
    return impacts

def calculate_mlp_impacts_distilbert(model, tokenizer, texts, target_tokens, distractor_tokens, baseline_diff):
    num_layers = model.config.n_layers
    impacts = []
    for layer in tqdm(range(num_layers), desc="Calculating MLP Impacts (MLM)"):
        module = model.distilbert.transformer.layer[layer].ffn
        hook = module.register_forward_hook(get_mlp_ablation_hook())
        try:
            outputs_ablated, mask_indices_ablated = get_logits_mlm(model, tokenizer, texts)
            ablated_diff = compute_logit_diff_mlm_dynamic(outputs_ablated.logits, mask_indices_ablated, tokenizer, target_tokens, distractor_tokens)
            impacts.append((layer, abs(100 * (1 - ablated_diff / baseline_diff) if baseline_diff != 0 else 0)))
        finally: hook.remove()
    return impacts

# ----------------------------
# Run
# ----------------------------
def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    # --- Smart Model Loading ---
    print(f"Loading teacher model: {args.teacher_model}...")
    teacher_cfg = AutoConfig.from_pretrained(args.teacher_model)
    teacher_is_mlm = is_mlm(teacher_cfg)
    teacher_model_class = AutoModelForMaskedLM if teacher_is_mlm else AutoModelForCausalLM
    teacher = teacher_model_class.from_pretrained(args.teacher_model, config=teacher_cfg, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager")
    tokenizer_t = AutoTokenizer.from_pretrained(args.teacher_model)

    print(f"Loading student model: {args.student_model}...")
    student_cfg = AutoConfig.from_pretrained(args.student_model)
    student_is_mlm = is_mlm(student_cfg)
    student_model_class = AutoModelForMaskedLM if student_is_mlm else AutoModelForCausalLM
    student = student_model_class.from_pretrained(args.student_model, config=student_cfg, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager")
    tokenizer_s = AutoTokenizer.from_pretrained(args.student_model)
    
    # --- Tokenizer Padding Setup ---
    for tok, is_mlm_flag in [(tokenizer_t, teacher_is_mlm), (tokenizer_s, student_is_mlm)]:
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        if not is_mlm_flag: tok.padding_side = "left"

    teacher.eval(); student.eval()

    # --- Data Preparation ---
    texts_full = ['Van done in 1. Hat done in 2. Ring done in 3. Desk done in 4. Sun done in', 'Ice done in 2. Snow done in 3. Watch done in 4. Sun done in 5. Table done in', 'Ring done in 3. Moon done in 4. Queen done in 5. Book done in 6. Rose done in', 'Queen done in 4. Oil done in 5. Rose done in 6. Desk done in 7. Car done in', 'Light done in 5. Arm done in 6. Road done in 7. Book done in 8. Ice done in', 'Ball done in 6. Cow done in 7. Book done in 8. Rose done in 9. Key done in', 'Road done in 7. Key done in 8. Ocean done in 9. Key done in 10. Queen done in', 'House done in 8. Rose done in 9. Key done in 10. Hat done in 11. Van done in', 'Ring done in 1. Car done in 2. Apple done in 3. Pear done in 4. Moon done in', 'Star done in 2. Sun done in 3. Road done in 4. Queen done in 5. Box done in', 'Hill done in 3. Ant done in 4. Apple done in 5. House done in 6. Hat done in', 'Chair done in 4. Van done in 5. Orange done in 6. Queen done in 7. Zip done in', 'Gate done in 5. Desk done in 6. Wolf done in 7. Rain done in 8. Flag done in', 'Queen done in 6. House done in 7. Light done in 8. Cat done in 9. Moon done in', 'Zip done in 7. Car done in 8. Ring done in 9. Hand done in 10. Fish done in', 'Snake done in 8. Queen done in 9. Window done in 10. Ear done in 11. Orange done in', 'Ice done in 1. Sand done in 2. Desk done in 3. Van done in 4. Hat done in', 'Camera done in 2. Watch done in 3. Dog done in 4. Book done in 5. Desk done in', 'Rain done in 3. Dog done in 4. Rat done in 5. Nut done in 6. Ocean done in', 'Night done in 4. Hat done in 5. Wall done in 6. Book done in 7. Tree done in', 'Gate done in 5. Chair done in 6. Ring done in 7. Hill done in 8. Car done in', 'Ax done in 6. Fan done in 7. Cat done in 8. Ring done in 9. Ring done in', 'Rose done in 7. Night done in 8. Van done in 9. Orange done in 10. Apple done in', 'Orange done in 8. Ear done in 9. Book done in 10. Ring done in 11. Bird done in', 'Ball done in 1. Wind done in 2. Wallet done in 3. Tree done in 4. Sand done in', 'Table done in 2. Jam done in 3. Apple done in 4. Pear done in 5. Ice done in', 'Gate done in 3. Train done in 4. Hat done in 5. Fan done in 6. Key done in', 'Window done in 4. Desk done in 5. Year done in 6. Van done in 7. Camera done in', 'Car done in 5. Watch done in 6. Oil done in 7. Queen done in 8. Fan done in', 'Mouse done in 6. Rose done in 7. Table done in 8. Clock done in 9. Queen done in', 'Year done in 7. Wheel done in 8. Wolf done in 9. Arm done in 10. Ax done in', 'Key done in 8. Ring done in 9. Night done in 10. Fan done in 11. Clock done in', 'Glass done in 1. Key done in 2. Nut done in 3. Rat done in 4. Van done in', 'Star done in 2. Pear done in 3. Bug done in 4. Bird done in 5. Ring done in', 'Road done in 3. Wind done in 4. Rat done in 5. Wolf done in 6. Map done in', 'Car done in 4. Ant done in 5. Rose done in 6. Van done in 7. Wolf done in', 'Zip done in 5. Hat done in 6. Wind done in 7. Rain done in 8. Car done in', 'Ring done in 6. Light done in 7. Jar done in 8. Dog done in 9. Fish done in', 'Bug done in 7. Key done in 8. Gate done in 9. Orange done in 10. Ring done in', 'Flag done in 8. Fire done in 9. Pear done in 10. Ax done in 11. Ear done in', 'Ring done in 1. Flag done in 2. Queen done in 3. Map done in 4. Camera done in', 'Desk done in 2. Fire done in 3. Desk done in 4. Road done in 5. Watch done in', 'Queen done in 3. House done in 4. Flag done in 5. Tree done in 6. Ring done in', 'Desk done in 4. Window done in 5. Fish done in 6. Jar done in 7. Hat done in', 'Snake done in 5. Key done in 6. Glass done in 7. Van done in 8. House done in', 'Camera done in 6. Year done in 7. Rose done in 8. Map done in 9. Fire done in', 'Rose done in 7. Snake done in 8. Corn done in 9. Desk done in 10. Zip done in', 'Table done in 8. Key done in 9. Eye done in 10. Van done in 11. Key done in', 'Star done in 1. Hat done in 2. Star done in 3. Eye done in 4. Gold done in', 'Gate done in 2. Queen done in 3. Camera done in 4. Zip done in 5. Ocean done in', 'Fan done in 3. Corn done in 4. Apple done in 5. Van done in 6. Cat done in', 'Key done in 4. Gold done in 5. Star done in 6. Queen done in 7. Arm done in', 'Van done in 5. Snake done in 6. Car done in 7. Star done in 8. Watch done in', 'Jar done in 6. Flag done in 7. Window done in 8. Sun done in 9. Van done in', 'Wolf done in 7. Watch done in 8. Ring done in 9. Desk done in 10. Car done in', 'Cow done in 8. Tree done in 9. Book done in 10. Wall done in 11. Flag done in', 'Van done in 1. Wall done in 2. Ring done in 3. Car done in 4. Key done in', 'Jar done in 2. Wallet done in 3. Fish done in 4. Watch done in 5. Orange done in', 'Watch done in 3. Orange done in 4. Table done in 5. Corn done in 6. Van done in', 'Lake done in 4. Star done in 5. Fan done in 6. Fan done in 7. Night done in', 'Fan done in 5. Ball done in 6. Desk done in 7. Jam done in 8. Tree done in', 'Key done in 6. Jar done in 7. Night done in 8. Fish done in 9. Fan done in', 'Ice done in 7. Rose done in 8. Wall done in 9. Road done in 10. Window done in', 'Tree done in 8. Hill done in 9. Wall done in 10. Queen done in 11. Desk done in', 'Gold done in 1. King done in 2. Queen done in 3. Desk done in 4. Night done in', 'Van done in 2. Sand done in 3. Tree done in 4. Car done in 5. Cow done in', 'Pear done in 3. Eye done in 4. Lake done in 5. Fan done in 6. Light done in', 'Jam done in 4. Road done in 5. Corn done in 6. Ice done in 7. Pen done in', 'Iron done in 5. Ring done in 6. Star done in 7. Van done in 8. Star done in', 'Bird done in 6. Wheel done in 7. Cat done in 8. Desk done in 9. Desk done in', 'Fish done in 7. Gold done in 8. King done in 9. Ax done in 10. Jam done in', 'Desk done in 8. House done in 9. Cow done in 10. Hill done in 11. Gate done in', 'Nut done in 1. Light done in 2. Sand done in 3. Wall done in 4. Key done in', 'Tree done in 2. Fan done in 3. Car done in 4. Desk done in 5. Ring done in', 'Orange done in 3. Ocean done in 4. Sun done in 5. Bug done in 6. Queen done in', 'Orange done in 4. House done in 5. Watch done in 6. Corn done in 7. Ring done in', 'Wind done in 5. Orange done in 6. Rose done in 7. Zip done in 8. Queen done in', 'Iron done in 6. Cat done in 7. Glass done in 8. Box done in 9. Rose done in', 'Clock done in 7. Zip done in 8. Cat done in 9. Snow done in 10. Ant done in', 'Rose done in 8. Ear done in 9. Rose done in 10. Rat done in 11. Orange done in', 'Road done in 1. Ring done in 2. Pen done in 3. Oil done in 4. Camera done in', 'House done in 2. Zip done in 3. Queen done in 4. Snow done in 5. Lake done in', 'Van done in 3. Tree done in 4. Wall done in 5. Rat done in 6. King done in', 'Ball done in 4. Hill done in 5. Clock done in 6. Ear done in 7. Van done in', 'Night done in 5. Ice done in 6. Apple done in 7. Tree done in 8. House done in', 'Clock done in 6. Orange done in 7. Queen done in 8. Desk done in 9. Arm done in', 'Map done in 7. Sun done in 8. Clock done in 9. Cat done in 10. Ball done in', 'Desk done in 8. Key done in 9. Star done in 10. Cow done in 11. Rose done in', 'Ring done in 1. Fan done in 2. Window done in 3. Tree done in 4. Chair done in', 'House done in 2. Nut done in 3. Queen done in 4. Zip done in 5. Rose done in', 'Window done in 3. Gate done in 4. Zip done in 5. Key done in 6. Snow done in', 'Corn done in 4. Ice done in 5. Eye done in 6. Light done in 7. Desk done in', 'Hill done in 5. Snake done in 6. Orange done in 7. Road done in 8. Flag done in', 'Queen done in 6. Window done in 7. Zip done in 8. Cow done in 9. Fire done in', 'Star done in 7. Hat done in 8. Chair done in 9. Car done in 10. Ring done in', 'Desk done in 8. Watch done in 9. Desk done in 10. Ax done in 11. Queen done in', 'Train done in 1. Snow done in 2. Cow done in 3. Ring done in 4. Queen done in', 'Window done in 2. Pen done in 3. Ax done in 4. Iron done in 5. Nut done in', 'Desk done in 3. Flag done in 4. Fan done in 5. Hat done in 6. Chair done in', 'King done in 4. Star done in 5. Cow done in 6. Wallet done in 7. Fan done in', 'Star done in 5. Watch done in 6. Car done in 7. Van done in 8. Queen done in']
    if args.num_samples is not None and args.num_samples < len(texts_full):
        texts = random.sample(texts_full, args.num_samples)
        print(f"Randomly selected {args.num_samples} samples out of {len(texts_full)} using seed {args.seed}.")
    else:
        texts = texts_full
        print(f"Using all {len(texts)} samples.")

    next_tokens, target_tokens, distractor_tokens = infer_tokens(texts)
    texts_clm = [t + nxt for t, nxt in zip(texts, next_tokens)]
    
    texts_mlm = []
    if teacher_is_mlm or student_is_mlm:
        mask_token = tokenizer_s.mask_token if student_is_mlm else tokenizer_t.mask_token
        if mask_token is None: raise ValueError("MLM model detected but tokenizer has no mask_token.")
        texts_mlm = [t + " " + mask_token for t in texts]

    # --- Calculating Baseline Logit Differences ---
    print("\n--- Calculating Baseline Logit Differences ---")
    if teacher_is_mlm:
        teacher_outputs, teacher_mask_indices = get_logits_mlm(teacher, tokenizer_t, texts_mlm)
        teacher_baseline_diff = compute_logit_diff_mlm_dynamic(teacher_outputs.logits, teacher_mask_indices, tokenizer_t, target_tokens, distractor_tokens)
    else:
        teacher_baseline_diff = compute_logit_diff_causal_lm_smart(teacher, tokenizer_t, texts)
    print(f"Teacher ({args.teacher_model}) baseline logit difference: {teacher_baseline_diff:.4f}")

    if student_is_mlm:
        student_outputs, student_mask_indices = get_logits_mlm(student, tokenizer_s, texts_mlm)
        student_baseline_diff = compute_logit_diff_mlm_dynamic(student_outputs.logits, student_mask_indices, tokenizer_s, target_tokens, distractor_tokens)
    else:
        student_baseline_diff = compute_logit_diff_causal_lm_smart(student, tokenizer_s, texts)
    print(f"Student ({args.student_model}) baseline logit difference: {student_baseline_diff:.4f}")

    # --- Teacher Calculations ---
    print("\n--- Calculating impacts for teacher model ---")
    if teacher_is_mlm:
        teacher_heads = calculate_head_impacts_distilbert(teacher, tokenizer_t, texts_mlm, target_tokens, distractor_tokens, teacher_baseline_diff)
        teacher_mlp = calculate_mlp_impacts_distilbert(teacher, tokenizer_t, texts_mlm, target_tokens, distractor_tokens, teacher_baseline_diff)
        teacher_head_vecs = get_head_vectors_distilbert(teacher, tokenizer_t, texts_mlm, [h[0] for h in teacher_heads])
        teacher_mlp_vecs = get_mlp_vectors_distilbert(teacher, tokenizer_t, texts_mlm, [m[0] for m in teacher_mlp])
    else:
        teacher_heads = calculate_head_impacts_causal_lm(teacher, tokenizer_t, texts, teacher_baseline_diff)
        teacher_mlp = calculate_mlp_impacts_causal_lm(teacher, tokenizer_t, texts, teacher_baseline_diff)
        teacher_head_vecs = get_head_vectors_causal_lm(teacher, tokenizer_t, texts_clm, [h[0] for h in teacher_heads])
        teacher_mlp_vecs = get_mlp_vectors_causal_lm(teacher, tokenizer_t, texts_clm, [m[0] for m in teacher_mlp])

    # --- Student Calculations ---
    print("\n--- Calculating impacts for student model ---")
    if student_is_mlm:
        student_heads = calculate_head_impacts_distilbert(student, tokenizer_s, texts_mlm, target_tokens, distractor_tokens, student_baseline_diff)
        student_mlp = calculate_mlp_impacts_distilbert(student, tokenizer_s, texts_mlm, target_tokens, distractor_tokens, student_baseline_diff)
        student_head_vecs = get_head_vectors_distilbert(student, tokenizer_s, texts_mlm, [h[0] for h in student_heads])
        student_mlp_vecs = get_mlp_vectors_distilbert(student, tokenizer_s, texts_mlm, [m[0] for m in student_mlp])
    else:
        student_heads = calculate_head_impacts_causal_lm(student, tokenizer_s, texts, student_baseline_diff)
        student_mlp = calculate_mlp_impacts_causal_lm(student, tokenizer_s, texts, student_baseline_diff)
        student_head_vecs = get_head_vectors_causal_lm(student, tokenizer_s, texts_clm, [h[0] for h in student_heads])
        student_mlp_vecs = get_mlp_vectors_causal_lm(student, tokenizer_s, texts_clm, [m[0] for m in student_mlp])

    # --- Final Matching & Scoring ---
    influence_teacher = dict(teacher_heads + teacher_mlp)
    influence_student = dict(student_heads + student_mlp)

    print("\n--- Computing best matches ---")
    attn_matches = compute_best_matches_attn(teacher_head_vecs, student_head_vecs)
    mlp_matches = compute_best_matches_mlp(teacher_mlp_vecs, student_mlp_vecs)
    
    matches = {**attn_matches, **mlp_matches}
    all_pairs = []
    for teacher_comp, (student_comp, sim_score) in matches.items():
        I_T = influence_teacher.get(teacher_comp, 0.0)
        I_S = influence_student.get(student_comp, 0.0)
        all_pairs.append((I_S, I_T, sim_score))

    if not all_pairs:
        print("No matching pairs found. Alignment score is 0.")
        return

    sum_I_S = sum(p[0] for p in all_pairs) or 1.0
    sum_I_T = sum(p[1] for p in all_pairs) or 1.0
    normalized = [(I_S / sum_I_S, I_T / sum_I_T, S) for (I_S, I_T, S) in all_pairs]
    alignment_score = compute_alignment_score(normalized)
    print(f"\nAlignment score between {args.teacher_model} and {args.student_model}: {alignment_score:.4f}")

if __name__ == "__main__":
    main()
