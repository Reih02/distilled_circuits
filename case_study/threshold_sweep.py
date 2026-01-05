#!/usr/bin/env python3
"""
threshold_sweep.py

Runs a sweep over circuit threshold T_n for the GPT-2 numeral sequence case study.

Assumptions (matching your notebook):
- You have the repo folder: <project_root>/seqcont_circuits/
- Modules exist at:
    seqcont_circuits/src/iter_edge_pruning/
    seqcont_circuits/src/iter_node_pruning/
- Data pickles exist at:
    <data_root>/<task>/<task>_prompts_<ptype>.pkl
    <data_root>/<task>/randDS_<task>.pkl

Example:
  python threshold_sweep.py \
    --project_root . \
    --task numerals \
    --tn_values 0.10,0.15,0.20,0.25,0.30 \
    --prompt_types done,lost,names \
    --num_samps_per_ptype 128
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import pandas as pd
from transformer_lens import HookedTransformer

# -----------------------------
# Utilities
# -----------------------------

Head = Tuple[int, int]   # (layer, head)
Mlp = int               # layer index


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_csv_floats(s: str) -> List[float]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def all_heads_for_model(model: HookedTransformer) -> List[Head]:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    return [(l, h) for l in range(n_layers) for h in range(n_heads)]


def all_mlps_for_model(model: HookedTransformer) -> List[Mlp]:
    return list(range(model.cfg.n_layers))


def circuit_node_set(heads: Sequence[Head], mlps: Sequence[Mlp]) -> Set[Tuple]:
    """
    Normalized node-set representation so heads & mlps don't collide.
    """
    s: Set[Tuple] = set()
    for (l, h) in heads:
        s.add(("H", int(l), int(h)))
    for m in mlps:
        s.add(("M", int(m)))
    return s


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Best-effort edge extraction
# -----------------------------

def try_get_edges_from_modules(
    edge_pruning_mod: Any,
    viz_circuits_mod: Any,
) -> Optional[Set[Tuple]]:
    """
    Your notebook doesn't explicitly store a final edge-set in a variable, so this is best-effort:
    we look for common attribute/function names in imported modules.

    If you *do* have a canonical way of obtaining the final edge set (e.g. a function like
    `get_circuit_edges(...)`), replace this with that call.
    """
    candidate_attrs = [
        "curr_circ_edges",
        "circuit_edges",
        "circ_edges",
        "edges",
        "edge_set",
    ]

    for mod in [edge_pruning_mod, viz_circuits_mod]:
        for name in candidate_attrs:
            if hasattr(mod, name):
                obj = getattr(mod, name)
                edges = _normalize_edge_container(obj)
                if edges is not None:
                    return edges

    candidate_fns = [
        "get_circuit_edges",
        "get_edges",
        "extract_edges",
        "edges_from_circuit",
    ]
    for mod in [edge_pruning_mod, viz_circuits_mod]:
        for fn_name in candidate_fns:
            if hasattr(mod, fn_name) and callable(getattr(mod, fn_name)):
                try:
                    obj = getattr(mod, fn_name)()
                    edges = _normalize_edge_container(obj)
                    if edges is not None:
                        return edges
                except TypeError:
                    # function likely needs args; skip
                    pass
                except Exception:
                    pass

    return None


def _normalize_edge_container(obj: Any) -> Optional[Set[Tuple]]:
    """
    Normalize a variety of possible edge container formats into a set of tuples.
    """
    if obj is None:
        return None

    # Already a set/list/tuple of edges?
    if isinstance(obj, (set, list, tuple)):
        try:
            edges = set(tuple(e) for e in obj)  # type: ignore[arg-type]
            # sanity: edges should be tuples of len>=2
            if all(isinstance(e, tuple) and len(e) >= 2 for e in edges):
                return edges
        except Exception:
            return None

    # Dict mapping -> treat keys or values as edges if they look right
    if isinstance(obj, dict):
        for cand in [obj.keys(), obj.values()]:
            try:
                edges = set(tuple(e) for e in cand)  # type: ignore[arg-type]
                if all(isinstance(e, tuple) and len(e) >= 2 for e in edges):
                    return edges
            except Exception:
                continue

    return None


# -----------------------------
# Core sweep
# -----------------------------

@dataclass
class SweepResult:
    tn: float
    threshold_percent: float
    n_heads: int
    n_mlps: int
    n_nodes: int
    n_edges: Optional[int]
    completeness_pct: float
    completeness_drop_pct: float
    faithfulness_pct: float
    faithfulness_drop_pct: float
    jaccard_nodes_vs_base: Optional[float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=".", help="Directory containing seqcont_circuits/")
    parser.add_argument("--data_root", type=str, default=None, help="Defaults to <project_root>/seqcont_circuits/data")
    parser.add_argument("--task", type=str, default="numerals", help="numerals | numwords | months (as in notebook)")
    parser.add_argument("--prompt_types", type=str, default="done,lost,names")
    parser.add_argument("--num_samps_per_ptype", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="gpt2-small")
    parser.add_argument("--tn_values", type=str, default="0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--baseline_tn", type=float, default=0.20)
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="tn_sweep_out")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else (project_root / "seqcont_circuits" / "data")
    out_dir = Path(args.out_dir).resolve()
    safe_mkdir(out_dir)

    prompt_types = parse_csv_list(args.prompt_types)
    tn_values = parse_csv_floats(args.tn_values)

    # ---- Add repo src subdirs to sys.path (mirrors your %cd + from ... import ... pattern)
    src_root = project_root / "seqcont_circuits" / "src"
    iter_edge_dir = src_root / "iter_edge_pruning"
    iter_node_dir = src_root / "iter_node_pruning"

    if not iter_edge_dir.exists() or not iter_node_dir.exists():
        raise FileNotFoundError(
            f"Expected directories:\n  {iter_edge_dir}\n  {iter_node_dir}\n"
            f"Check --project_root (currently {project_root})."
        )

    sys.path.insert(0, str(iter_edge_dir))
    sys.path.insert(0, str(iter_node_dir))

    # ---- Import your project functions/classes (names match notebook usage)
    from dataset import Dataset  # type: ignore
    from metrics import get_logit_diff  # type: ignore
    from node_ablation_fns import add_ablation_hook_MLP_head  # type: ignore
    import edge_pruning_fns as edge_pruning_mod  # type: ignore
    import viz_circuits as viz_circuits_mod  # type: ignore

    find_circuit_backw = getattr(edge_pruning_mod, "find_circuit_backw")
    find_circuit_forw = getattr(edge_pruning_mod, "find_circuit_forw")

    # ---- Load model
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        args.model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    model.eval()

    # ---- Load prompts (as in notebook)
    prompts_list: List[Dict[str, Any]] = []
    for ptype in prompt_types:
        p = data_root / args.task / f"{args.task}_prompts_{ptype}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"Missing prompts pickle: {p}")
        with open(p, "rb") as f:
            filelist = pickle.load(f)
        prompts_list += filelist[: args.num_samps_per_ptype]

    if not prompts_list:
        raise RuntimeError("No prompts loaded; check prompt_types / num_samps_per_ptype / data_root.")

    # ---- Build pos_dict (as in notebook)
    n_pos = len(model.tokenizer.tokenize(prompts_list[0]["text"]))
    pos_dict = {f"S{i}": i for i in range(n_pos)}

    dataset = Dataset(prompts_list, pos_dict, model.tokenizer)

    rand_path = data_root / args.task / f"randDS_{args.task}.pkl"
    if not rand_path.exists():
        raise FileNotFoundError(f"Missing random/corrupted dataset pickle: {rand_path}")
    with open(rand_path, "rb") as f:
        prompts_list_2 = pickle.load(f)
    dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)

    # ---- Compute original score
    model.reset_hooks(including_permanent=True)
    logits_original = model(dataset.toks)
    orig_score = get_logit_diff(logits_original, dataset)
    del logits_original
    gc.collect()

    all_heads = all_heads_for_model(model)
    all_mlps = all_mlps_for_model(model)

    def score_percent_with_kept_components(heads_keep: Sequence[Head], mlps_keep: Sequence[Mlp]) -> float:
        """
        Keeps only heads_keep/mlps_keep "unablated" (everything else mean-ablated using dataset_2),
        then returns 100 * (logit_diff / orig_logit_diff).
        """
        model.reset_hooks(including_permanent=True)
        add_ablation_hook_MLP_head(model, dataset_2, list(heads_keep), list(mlps_keep))
        logits = model(dataset.toks)
        score = get_logit_diff(logits, dataset)
        pct = (100.0 * score / orig_score).item()
        del logits
        gc.collect()
        return float(pct)

    def extract_circuit_for_threshold(threshold_percent: float) -> Tuple[List[Head], List[Mlp], Optional[Set[Tuple]]]:
        """
        Runs your iterative backw/fwd circuit extraction loop (as in notebook cell with threshold=20).

        Returns:
          (curr_circ_heads, curr_circ_mlps, edge_set_or_None)
        """
        curr_circ_heads: List[Head] = []
        curr_circ_mlps: List[Mlp] = []
        comp_scores_history: List[Any] = []

        for it in range(1, args.max_iters + 1):
            # Backward prune step
            old_heads = curr_circ_heads.copy()
            old_mlps = curr_circ_mlps.copy()

            curr_circ_heads, curr_circ_mlps, new_score, comp_scores = find_circuit_backw(
                model, dataset, dataset_2, curr_circ_heads, curr_circ_mlps, orig_score, threshold_percent
            )
            comp_scores_history.append(comp_scores)

            if old_heads == curr_circ_heads and old_mlps == curr_circ_mlps:
                break

            # Forward prune step
            old_heads = curr_circ_heads.copy()
            old_mlps = curr_circ_mlps.copy()

            curr_circ_heads, curr_circ_mlps, new_score, comp_scores = find_circuit_forw(
                model, dataset, dataset_2, curr_circ_heads, curr_circ_mlps, orig_score, threshold_percent
            )
            comp_scores_history.append(comp_scores)

            if old_heads == curr_circ_heads and old_mlps == curr_circ_mlps:
                break

        # Save comp_scores history for debugging / appendix if desired
        hist_path = out_dir / f"comp_scores_history_T{threshold_percent:.0f}.pkl"
        with open(hist_path, "wb") as f:
            pickle.dump(comp_scores_history, f)

        # Best-effort: try to retrieve edge set if modules expose it
        edges = try_get_edges_from_modules(edge_pruning_mod, viz_circuits_mod)

        return curr_circ_heads, curr_circ_mlps, edges

    # ---- Run sweep (store circuits so we can compute overlap after)
    circuits_by_tn: Dict[float, Dict[str, Any]] = {}
    results: List[SweepResult] = []

    # First, ensure baseline is included for overlap computations
    if args.baseline_tn not in tn_values:
        tn_values = sorted(set(tn_values + [args.baseline_tn]))

    for tn in tn_values:
        threshold_percent = 100.0 * tn

        print(f"\n=== Running T_n={tn:.3f} (threshold_percent={threshold_percent:.1f}) ===")
        heads, mlps, edges = extract_circuit_for_threshold(threshold_percent)

        # Sizes
        n_heads = len(heads)
        n_mlps = len(mlps)
        n_nodes = n_heads + n_mlps
        n_edges = len(edges) if edges is not None else None

        # Completeness: keep only the circuit
        completeness_pct = score_percent_with_kept_components(heads, mlps)
        completeness_drop_pct = 100.0 - completeness_pct

        # Faithfulness: ablate the circuit (keep everything else)
        heads_keep = [h for h in all_heads if h not in set(heads)]
        mlps_keep = [m for m in all_mlps if m not in set(mlps)]
        faithfulness_pct = score_percent_with_kept_components(heads_keep, mlps_keep)
        faithfulness_drop_pct = 100.0 - faithfulness_pct

        circuits_by_tn[tn] = {
            "tn": tn,
            "threshold_percent": threshold_percent,
            "heads": heads,
            "mlps": mlps,
            "edges": edges,
            "n_heads": n_heads,
            "n_mlps": n_mlps,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "completeness_pct": completeness_pct,
            "faithfulness_pct": faithfulness_pct,
        }

        results.append(
            SweepResult(
                tn=tn,
                threshold_percent=threshold_percent,
                n_heads=n_heads,
                n_mlps=n_mlps,
                n_nodes=n_nodes,
                n_edges=n_edges,
                completeness_pct=completeness_pct,
                completeness_drop_pct=completeness_drop_pct,
                faithfulness_pct=faithfulness_pct,
                faithfulness_drop_pct=faithfulness_drop_pct,
                jaccard_nodes_vs_base=None,  # fill after baseline known
            )
        )

        # Persist each circuit immediately
        with open(out_dir / f"circuit_Tn{tn:.3f}.pkl", "wb") as f:
            pickle.dump(circuits_by_tn[tn], f)

    # ---- Compute overlaps vs baseline
    baseline_tn = args.baseline_tn
    if baseline_tn not in circuits_by_tn:
        raise RuntimeError(f"Baseline tn={baseline_tn} not found in computed circuits.")

    base_nodes = circuit_node_set(
        circuits_by_tn[baseline_tn]["heads"],
        circuits_by_tn[baseline_tn]["mlps"],
    )

    updated_results: List[Dict[str, Any]] = []
    for r in results:
        this_nodes = circuit_node_set(
            circuits_by_tn[r.tn]["heads"],
            circuits_by_tn[r.tn]["mlps"],
        )
        j = jaccard(this_nodes, base_nodes)

        updated_results.append(
            {
                "T_n": r.tn,
                "threshold_percent": r.threshold_percent,
                "n_heads": r.n_heads,
                "n_mlps": r.n_mlps,
                "n_nodes": r.n_nodes,
                "n_edges": r.n_edges,
                "completeness_pct": r.completeness_pct,
                "completeness_drop_pct": r.completeness_drop_pct,
                "faithfulness_pct": r.faithfulness_pct,
                "faithfulness_drop_pct": r.faithfulness_drop_pct,
                "jaccard_nodes_vs_base_Tn": baseline_tn,
                "jaccard_nodes_vs_base": j,
            }
        )

    df = pd.DataFrame(updated_results).sort_values("T_n").reset_index(drop=True)

    # Save outputs
    csv_path = out_dir / "threshold_sweep_results.csv"
    df.to_csv(csv_path, index=False)

    with open(out_dir / "all_circuits_by_tn.pkl", "wb") as f:
        pickle.dump(circuits_by_tn, f)

    print("\n=== Sweep summary ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {out_dir / 'all_circuits_by_tn.pkl'}")

    if df["n_edges"].isna().all():
        print(
            "\nNOTE: n_edges is empty (None) for all thresholds.\n"
            "This means your current code path does not expose a final edge-set via a simple module attribute/function.\n"
            "If you have a canonical function that returns the extracted edge list, plug it into try_get_edges_from_modules()."
        )


if __name__ == "__main__":
    main()
