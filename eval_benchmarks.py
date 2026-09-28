"""Unified Pure System 1 Benchmark Evaluator for MiniCPM5-2B-Jev:
Evaluates on pure decision benchmarks (zero browser coupling):
1. decision-v7 10-Source Stratified Eval (val_10source.jsonl, 300 states / 360 questions)
2. hard-v1 Held-Out Template 4 Eval (val_hard_v1.jsonl, 350 states across 7 reasoning families)
3. DecisionBench Medium (akhilaaa3/decision-bench medium.jsonl, 80 states / 293 questions)
4. DecisionBench Hard (akhilaaa3/decision-bench hard.jsonl, 80 states / 293 questions)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from model import (
    SERVE_MAX_BRANCH,
    SERVE_MAX_PACKED,
    SERVE_MAX_STATE,
    MiniCPMSystemOne,
    to_internal_record,
)
from train import compute_ece


def load_decision_bench_records(subset: str = "medium") -> list[dict[str, Any]]:
    """Load akhilaaa3/decision-bench (medium or hard) and convert into standard wire-format records."""
    fname = f"data/{subset}.jsonl"
    try:
        path = hf_hub_download(
            repo_id="akhilaaa3/decision-bench",
            filename=fname,
            repo_type="dataset",
            local_files_only=True,
        )
    except Exception:
        path = hf_hub_download(
            repo_id="akhilaaa3/decision-bench",
            filename=fname,
            repo_type="dataset",
        )

    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            st = r.get("state", "")
            if isinstance(st, str) and ("<image>" in st or "<audio>" in st or "<video>" in st):
                continue
            qs_raw = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
            ans_raw = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
            qs_with_labels: dict[str, Any] = {}
            for qk, qv in qs_raw.items():
                if qk not in ans_raw:
                    continue
                q_copy = dict(qv)
                q_copy["label"] = ans_raw[qk]
                qs_with_labels[qk] = q_copy
            if qs_with_labels:
                out.append(
                    {
                        "id": r.get("id", ""),
                        "source": f"db_{subset}_{r.get('family', 'general')}",
                        "state": st,
                        "questions": qs_with_labels,
                    }
                )
    return out


def load_jevbench_records(jevbench_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load official fstandhartinger/jevbench public dataset (easy 48 + original 72 + hard 111 = 231 decisions across 195 groups)."""
    d = jevbench_dir or (Path(__file__).resolve().parent / "data" / "jevbench")
    out: list[dict[str, Any]] = []
    for tier in ("easy", "original", "hard"):
        p = d / f"{tier}.jsonl"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                q = dict(r["question"])
                qt = q["type"]
                exp = r["expected"]
                if qt == "choice":
                    crit = q.get("criteria")
                    if isinstance(crit, dict) and r.get("labels"):
                        q["criteria"] = {k: crit.get(k, k) for k in r["labels"]}
                    q["label"] = str(exp)
                elif qt == "noul":
                    q["label"] = "yes" if str(exp).lower() in ("yes", "true", "1") else "no"
                elif qt == "score":
                    q["label"] = int(exp)
                fam = r.get("family", "general")
                out.append(
                    {
                        "id": r["id"],
                        "group": r.get("group") or r["id"],
                        "tier": tier,
                        "source": f"jevbench_{tier}",
                        "family": fam,
                        "state": r["state"],
                        "questions": {"decision": q},
                    }
                )
    return out


def load_jabr_records(version: str = "v2") -> list[dict[str, Any]]:
    """Load jabr/classifier-benchmark v1 (8 tasks / 78 cases) or v2 (49 tasks / 869 cases)."""
    from data.jabr.jabr_cases import TASKS as TASKS_V1
    from data.jabr.jabr_cases_v2 import TASKS_V2

    tasks = TASKS_V1 if version == "v1" else TASKS_V2
    out: list[dict[str, Any]] = []
    for t in tasks:
        qt = t.type
        for idx, c in enumerate(t.cases):
            if qt == "choice":
                q = {
                    "type": "choice",
                    "instructions": t.question.instructions,
                    "criteria": dict(t.question.criteria),
                    "label": str(c.expected),
                }
            elif qt == "noul":
                q = {
                    "type": "noul",
                    "instructions": t.question.instructions,
                    "criteria": getattr(t.question, "criteria", None),
                    "label": "yes" if bool(c.expected) else "no",
                }
            elif qt == "score":
                q = {
                    "type": "score",
                    "instructions": t.question.instructions,
                    "criteria": list(t.question.criteria),
                    "label": int(c.expected),
                }
            else:
                raise ValueError(f"Unknown jabr task type: {qt}")
            out.append(
                {
                    "id": f"{t.id}_{idx:02d}",
                    "group": t.id,
                    "task_type": qt,
                    "source": t.id,
                    "state": c.state,
                    "questions": {"decision": q},
                }
            )
    return out


@torch.no_grad()
def evaluate_suite(
    model: MiniCPMSystemOne,
    records: list[dict[str, Any]],
    suite_name: str,
    use_pride: bool = False,
) -> dict[str, Any]:
    """Run evaluation on a benchmark suite and compute Micro/Macro Accuracy, Choice/Noul/Score Acc, NLL, Brier, ECE."""
    model.eval()
    t0 = time.time()
    nlls, accs, confs, briers = [], [], [], []
    state_accs: list[float] = []
    by_group: dict[str, list[int]] = defaultdict(list)
    group_qtype: dict[str, str] = {}
    by_type: dict[str, list[int]] = defaultdict(list)
    by_source: dict[str, list[int]] = defaultdict(list)

    for idx, rec_w in enumerate(records):
        internal = to_internal_record(rec_w, add_date_facts=True)
        out_pairs = model.predict_record(
            internal,
            use_pride=use_pride,
            max_state=SERVE_MAX_STATE,
            max_branch=SERVE_MAX_BRANCH,
            max_packed=SERVE_MAX_PACKED,
        )
        st_oks: list[int] = []
        grp_key = str(rec_w.get("group") or rec_w.get("id") or f"s_{idx}")
        for q, (_, p_ten) in zip(internal["questions"], out_pairs):
            p = p_ten.numpy().astype(np.float64)
            p = np.clip(p, 1e-9, 1.0)
            p = p / p.sum()
            y = int(q["label"])
            ok = int(int(p.argmax()) == y)
            accs.append(ok)
            st_oks.append(ok)
            by_group[grp_key].append(ok)
            group_qtype[grp_key] = q["type"]
            confs.append(float(p.max()))
            nlls.append(float(-math.log(max(float(p[y]), 1e-9))))
            target = np.eye(len(p))[y]
            briers.append(float(((p - target) ** 2).sum()))
            by_type[q["type"]].append(ok)
            by_source[str(internal["source"])].append(ok)
        if st_oks:
            state_accs.append(float(np.mean(st_oks)))

    elapsed = time.time() - t0
    confs_arr = np.asarray(confs) if confs else np.zeros(1)
    accs_arr = np.asarray(accs, dtype=float) if accs else np.zeros(1)
    high = confs_arr >= 0.90
    group_means = [float(np.mean(v)) for v in by_group.values() if v]
    choice_grp_means = [float(np.mean(v)) for g, v in by_group.items() if v and group_qtype.get(g) == "choice"]
    noul_grp_means = [float(np.mean(v)) for g, v in by_group.items() if v and group_qtype.get(g) == "noul"]
    score_grp_means = [float(np.mean(v)) for g, v in by_group.items() if v and group_qtype.get(g) == "score"]

    res = {
        "suite": suite_name,
        "mode": "PriDe (cyclic)" if use_pride else "Single-Pass (K=1)",
        "n_states": len(state_accs),
        "n_groups": len(group_means),
        "n_questions": len(accs),
        "micro_acc": round(float(accs_arr.mean()), 4),
        "state_macro_acc": round(float(np.mean(state_accs)), 4) if state_accs else 0.0,
        "group_macro_acc": round(float(np.mean(group_means)), 4) if group_means else 0.0,
        "choice_macro_acc": round(float(np.mean(choice_grp_means)), 4) if choice_grp_means else 0.0,
        "noul_macro_acc": round(float(np.mean(noul_grp_means)), 4) if noul_grp_means else 0.0,
        "score_macro_acc": round(float(np.mean(score_grp_means)), 4) if score_grp_means else 0.0,
        "choice_acc": round(float(np.mean(by_type["choice"])), 4) if by_type["choice"] else 0.0,
        "noul_acc": round(float(np.mean(by_type["noul"])), 4) if by_type["noul"] else 0.0,
        "score_acc": round(float(np.mean(by_type["score"])), 4) if by_type["score"] else 0.0,
        "nll": round(float(np.mean(nlls)), 4) if nlls else 0.0,
        "brier": round(float(np.mean(briers)), 4) if briers else 0.0,
        "ece": round(compute_ece(confs_arr, accs_arr), 4),
        "mean_conf": round(float(confs_arr.mean()), 4),
        "confident_error_rate": round(float(np.mean(high & (accs_arr < 0.5))), 4),
        "elapsed_s": round(elapsed, 2),
        "ms_per_state": round((elapsed * 1000.0) / max(1, len(state_accs)), 1),
        "by_source": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_source.items())},
    }
    print(
        f"[{suite_name} | {res['mode']}] "
        f"Micro={res['micro_acc']*100:.2f}% | Task/GroupMacro={res['group_macro_acc']*100:.2f}% | "
        f"ChoiceMacro={res['choice_macro_acc']*100:.2f}% (Micro {res['choice_acc']*100:.2f}%) | "
        f"NoulMacro={res['noul_macro_acc']*100:.2f}% | ScoreMacro={res['score_macro_acc']*100:.2f}% | "
        f"ECE={res['ece']:.4f} | Brier={res['brier']:.4f} ({res['ms_per_state']} ms/state)",
        flush=True,
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent / "checkpoints" / "stage2"),
    )
    ap.add_argument(
        "--data_dir",
        default=str(Path(__file__).resolve().parent / "data" / "stage2"),
    )
    ap.add_argument("--pride", action="store_true", help="Also run cyclic permutation PriDe evaluation")
    ap.add_argument("--only", default="", help="Substring filter for suite name")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading checkpoint {args.checkpoint} on {device}...", flush=True)
    model, _ = MiniCPMSystemOne.load_checkpoint(args.checkpoint, device=device, dtype=torch.bfloat16)
    if hasattr(model.lm, "merge_adapter"):
        try:
            model.lm.merge_adapter()
        except Exception:
            pass

    data_dir = Path(args.data_dir)
    val_10source = [
        json.loads(line)
        for line in (data_dir / "val_10source.jsonl").open("r", encoding="utf-8")
        if line.strip()
    ]
    val_hard = [
        json.loads(line)
        for line in (data_dir / "val_hard_v1.jsonl").open("r", encoding="utf-8")
        if line.strip()
    ]
    db_medium = load_decision_bench_records("medium")
    db_hard = load_decision_bench_records("hard")
    jevbench = load_jevbench_records()
    jabr_v1 = load_jabr_records("v1")
    jabr_v2 = load_jabr_records("v2")

    suites = [
        ("decision-v7 (10-Source Stratified)", val_10source),
        ("hard-v1 (Held-Out Template 4)", val_hard),
        ("DecisionBench Medium", db_medium),
        ("DecisionBench Hard", db_hard),
        ("JevBench (195 groups / 231 decisions)", jevbench),
        ("jabr v1 (8 Tasks / 78 Cases)", jabr_v1),
        ("jabr v2 (49 Tasks / 869 Cases)", jabr_v2),
    ]
    if args.only:
        suites = [(n, r) for n, r in suites if args.only.lower() in n.lower()]

    out_path = Path(args.out_json) if args.out_json else (Path(args.checkpoint) / "benchmark_results.json")
    if out_path.exists():
        all_reports: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        all_reports = {"checkpoint": args.checkpoint, "single_pass": {}, "pride": {}}

    for name, recs in suites:
        all_reports["single_pass"][name] = evaluate_suite(model, recs, name, use_pride=False)
        if args.pride:
            all_reports["pride"][name] = evaluate_suite(model, recs, name, use_pride=True)

    out_path.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved complete benchmark report to {out_path}", flush=True)


if __name__ == "__main__":
    main()


