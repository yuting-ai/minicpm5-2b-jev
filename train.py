"""Full-Scale All-Layer Fine-Tuning & Class-Balanced Calibration for MiniCPM5-2B-Jev."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import signal
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from build_stage1_data import SCHEMA_VERSION, option_text, to_internal_record
from model import (
    DEFAULT_BASE_MODEL,
    MAX_BRANCH,
    MAX_STATE,
    MiniCPMSystemOne,
    encode_record,
    load_tokenizer,
)


def free_legacy_s1_ports(ports: tuple[int, ...] = (8009, 8010, 8011, 8012, 8013)) -> None:
    """Stop idle System 1 servers on ports 8009-8013 to free unified GPU memory before training."""
    for port in ports:
        try:
            out = subprocess.check_output(["lsof", "-ti", f"tcp:{port}"], text=True).strip()
            for pid_str in out.splitlines():
                if pid_str.strip().isdigit():
                    pid = int(pid_str.strip())
                    print(f"[Memory Cleanup] Stopping background server on port {port} (PID {pid})...", flush=True)
                    os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


def gaussian_targets(k: int, n: int = 11, sigma: float = 0.60, device: str = "cpu") -> torch.Tensor:
    xs = torch.arange(n, dtype=torch.float32, device=device)
    logits = -0.5 * ((xs - float(k)) / sigma) ** 2
    return F.softmax(logits, dim=-1)


def compute_question_loss(
    z: torch.Tensor,
    y: int,
    q_type: str,
    target_probs: list[float] | None,
    device: str,
    brier_weight: float = 0.25,
) -> torch.Tensor:
    """Composite proper-scoring-rule loss:
    - Unknowable / soft-target items: Soft Cross-Entropy toward target_probs (e.g. uniform 1/K)
    - Score items:
      * For 3..5 level discrete rubrics (n_opt <= 5): sharp sigma=0.38 Gaussian Soft CE (0.65) + gentle Ordinal CDF (0.15) + Brier (0.20)
        so boundary levels (0 and K-1) are not pulled toward the middle.
      * For 6..11 level numeric scales: sigma=0.60 Gaussian Soft CE (0.50) + Ordinal CDF (0.30) + Brier (0.20)
    - Choice / Noul items: Candidate Cross-Entropy + brier_weight * Brier Loss
    """
    n_opt = int(z.shape[0])
    log_p = F.log_softmax(z, dim=-1)
    p = torch.exp(log_p)

    if target_probs is not None and len(target_probs) == n_opt:
        tgt = torch.tensor(target_probs, dtype=torch.float32, device=device)
        tgt = tgt / tgt.sum().clamp_min(1e-8)
        soft_ce = -(tgt * log_p).sum()
        brier = ((p - tgt) ** 2).sum()
        return soft_ce + 0.20 * brier

    y_clamped = max(0, min(int(y), n_opt - 1))
    onehot = F.one_hot(torch.tensor(y_clamped, device=device), num_classes=n_opt).float()
    brier = ((p - onehot) ** 2).sum()

    if q_type == "score" and n_opt >= 2:
        sigma = 0.38 if n_opt <= 5 else 0.60
        w_gauss = 0.65 if n_opt <= 5 else 0.50
        w_cdf = 0.15 if n_opt <= 5 else 0.30
        tgt_gauss = gaussian_targets(y_clamped, n=n_opt, sigma=sigma, device=device)
        gauss_ce = -(tgt_gauss * log_p).sum()
        cdf_pred = torch.cumsum(p, dim=-1)[:-1]
        cdf_true = torch.cumsum(onehot, dim=-1)[:-1]
        cdf_emd = ((cdf_pred - cdf_true) ** 2).mean() * 2.0
        return w_gauss * gauss_ce + w_cdf * cdf_emd + 0.20 * brier

    ce = F.cross_entropy(z, torch.tensor(y_clamped, device=device))
    return ce + brier_weight * brier


_ABSTAIN_WORDS = ("none", "other", "unknown", "insufficient", "neither", "cannot", "no_match", "not_listed")


def augment_and_encode(
    rec_wire: dict[str, Any],
    tok,
    rng: random.Random,
    distractor_pool: list[str],
    p_none: float = 0.06,
    p_distract: float = 0.04,
) -> dict[str, Any]:
    """Apply view-augmentation (option order permutation on choice ONLY + none-of-the-above contrast) and encode.
    Note: noul options are NEVER permuted so '(A) no / (B) yes' remains 100% consistent with inference.
    """
    src = str(rec_wire.get("source", ""))
    add_df = src in ("hard_temporal_numeric", "hard_long_policy", "hard_ambiguous", "decider_custom_multi")
    rec = to_internal_record(copy.deepcopy(rec_wire), add_date_facts=add_df)

    for q in rec["questions"]:
        qtype = q["type"]
        n = len(q["options"])
        if qtype == "choice" and n >= 2:
            # View Augmentation: randomly permute choice option order so gold label is uniform across (A)..(Z)
            perm = list(range(n))
            rng.shuffle(perm)
            q["options"] = [q["options"][i] for i in perm]
            q["keys"] = [q["keys"][i] for i in perm]
            q["label"] = perm.index(q["label"])
            if q.get("target_probs") is not None and len(q["target_probs"]) == n:
                q["target_probs"] = [q["target_probs"][i] for i in perm]

            # None-of-the-above augmentation on standard choice questions without existing abstain options
            if (
                q.get("target_probs") is None
                and not src.startswith("hard_")
                and not src.startswith("pool5_")
                and not any(any(w in opt.lower() for w in _ABSTAIN_WORDS) for opt in q["options"])
            ):
                u = rng.random()
                if u < p_none:
                    gold = q["label"]
                    none_desc = rng.choice(
                        [
                            "none_of_the_above: none of the listed options apply",
                            "other: a category or answer not listed above",
                            "none: none of these options fits the context",
                        ]
                    )
                    kept_opts = [o for i, o in enumerate(q["options"]) if i != gold] + [none_desc]
                    perm2 = list(range(len(kept_opts)))
                    rng.shuffle(perm2)
                    q["options"] = [kept_opts[i] for i in perm2]
                    q["label"] = perm2.index(len(kept_opts) - 1)
                elif u < p_none + p_distract and distractor_pool and n < 12:
                    dist_opt = rng.choice(distractor_pool)
                    if dist_opt not in q["options"]:
                        q["options"] = q["options"] + [dist_opt]

    return encode_record(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False)


def stratify_training_subset(
    train_records: list[dict[str, Any]],
    target_count: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select a source-balanced subset of `target_count` records across all 5 pure decision pools."""
    if target_count <= 0 or target_count >= len(train_records):
        out = list(train_records)
        random.Random(seed).shuffle(out)
        return out

    rng = random.Random(seed)
    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in train_records:
        by_src[str(r.get("source", "general"))].append(r)

    for grp in by_src.values():
        rng.shuffle(grp)

    selected: list[dict[str, Any]] = []
    keys = sorted(by_src.keys())
    idx_map = {k: 0 for k in keys}
    boost_sources = {
        "decider_custom_multi",
        "decider_routing_terse",
        "decider_commands",
        "l1_sst5",
        "l1_yelp",
        "l1_legacy_policy",
        "l1_compositional",
        "hard_long_policy",
        "hard_multi_hop",
        "hard_temporal_numeric",
        "hard_tradeoff",
        "hard_probability",
        "hard_judge",
        "hard_ambiguous",
    }

    while len(selected) < target_count:
        added = False
        for k in keys:
            is_boost = (k in boost_sources) or k.startswith("pool5_")
            repeats = 2 if is_boost else 1
            for _ in range(repeats):
                if idx_map[k] < len(by_src[k]) and len(selected) < target_count:
                    selected.append(by_src[k][idx_map[k]])
                    idx_map[k] += 1
                    added = True
        if not added:
            break

    rng.shuffle(selected)
    return selected


def compute_ece(confs: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if len(confs) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece_val = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs >= lo) & (confs < hi) if hi < 1.0 else (confs >= lo) & (confs <= hi)
        if mask.any():
            ece_val += float(mask.mean() * abs(correct[mask].mean() - confs[mask].mean()))
    return ece_val


def read_mps_gpu_utilization() -> int:
    """Read live Apple Silicon GPU Device Utilization % from IOAccelerator without sudo."""
    try:
        out = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if '"Device Utilization %"' in line:
                parts = line.split("=")
                if len(parts) >= 2:
                    val = parts[-1].strip().rstrip(",")
                    if val.isdigit():
                        return int(val)
    except Exception:
        pass
    return -1


def _class_balanced_prior_bias(
    items: list[tuple[int, np.ndarray]],
    k_val: int,
    alpha: float,
    min_per_class: int = 3,
) -> np.ndarray | None:
    """Compute class-balanced marginal prior bias across true labels y in 0..k_val-1:
        bias = alpha * (1 / K) * sum_{y=0}^{K-1} E[centered_log_p | label = y]
    Returns None if any class y has fewer than `min_per_class` examples in dev, preventing
    label-frequency skew in dev.jsonl from introducing false positional/option biases.
    """
    by_y: dict[int, list[np.ndarray]] = defaultdict(list)
    for y, lp in items:
        if 0 <= y < k_val:
            by_y[y].append(lp)
    if any(len(by_y[y]) < min_per_class for y in range(k_val)):
        return None
    class_means = [np.mean(np.stack(by_y[y], axis=0), axis=0) for y in range(k_val)]
    macro_mean = np.mean(np.stack(class_means, axis=0), axis=0)
    macro_mean = macro_mean - np.mean(macro_mean)
    return macro_mean * alpha


@torch.no_grad()
def evaluate_and_calibrate(
    model: MiniCPMSystemOne,
    tok,
    dev_records: list[dict[str, Any]],
    fit_priors: bool = True,
    alpha_prior: float = 0.45,
    max_records: int = 0,
) -> dict[str, Any]:
    """Evaluate raw logits on dev split using length-bucketed batched causal forward passes,
    fit class-balanced marginal prior tables (noul_bias, choice_k_bias, score_k_bias)
    and per-type temperatures (choice, noul, score) minimizing NLL + 0.20 * Brier.
    """
    model.eval()
    if 0 < max_records < len(dev_records):
        v7_recs = [r for r in dev_records if str(r.get("source", "")).startswith("l1_")]
        hard_recs = [r for r in dev_records if str(r.get("source", "")).startswith("hard_")]
        p5_recs = [r for r in dev_records if str(r.get("source", "")).startswith("pool5_")]
        n_p5 = min(len(p5_recs), max(40, max_records // 3))
        rem = max(2, max_records - n_p5)
        n_v7 = rem // 2
        n_hd = rem - n_v7
        step_v7 = max(1, len(v7_recs) // max(1, n_v7))
        step_hd = max(1, len(hard_recs) // max(1, n_hd))
        step_p5 = max(1, len(p5_recs) // max(1, n_p5))
        active_records = (
            v7_recs[::step_v7][:n_v7]
            + hard_recs[::step_hd][:n_hd]
            + p5_recs[::step_p5][:n_p5]
        )
    else:
        active_records = dev_records

    pending_items: list[dict[str, Any]] = []
    for rec_idx, rec_w in enumerate(active_records):
        src = str(rec_w.get("source", ""))
        add_df = src in ("hard_temporal_numeric", "hard_long_policy", "hard_ambiguous", "decider_custom_multi")
        rec = to_internal_record(rec_w, add_date_facts=add_df)
        try:
            enc = encode_record(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False)
        except Exception:
            continue
        for q_idx, (q, row, n_opt, qt) in enumerate(
            zip(rec["questions"], enc["causal_rows"], enc["n_opts"], enc["q_types"])
        ):
            b_len = -(-len(row) // 64) * 64
            pending_items.append(
                {
                    "order_key": (rec_idx, q_idx),
                    "source": rec["source"],
                    "name": q["name"],
                    "type": q["type"],
                    "label": int(q["label"]),
                    "n_opt": int(n_opt),
                    "bucket_len": b_len,
                    "single_enc": {
                        "causal_rows": [row],
                        "n_opts": [int(n_opt)],
                        "q_types": [qt],
                    },
                }
            )

    pending_items.sort(key=lambda x: x["bucket_len"])
    eval_rows: list[dict[str, Any]] = []
    merged = False
    if hasattr(model.lm, "merge_adapter"):
        try:
            model.lm.merge_adapter()
            merged = True
        except Exception:
            merged = False

    try:
        idx = 0
        while idx < len(pending_items):
            b_len = pending_items[idx]["bucket_len"]
            max_b = max(1, min(8, 3072 // max(b_len, 64)))
            chunk = [pending_items[idx]]
            idx += 1
            while idx < len(pending_items) and len(chunk) < max_b and pending_items[idx]["bucket_len"] == b_len:
                chunk.append(pending_items[idx])
                idx += 1
            batch_zs = model.raw_logits_batch([it["single_enc"] for it in chunk])
            for it, zs in zip(chunk, batch_zs):
                z_raw = zs[0]
                eval_rows.append(
                    {
                        "order_key": it["order_key"],
                        "source": it["source"],
                        "name": it["name"],
                        "type": it["type"],
                        "label": it["label"],
                        "n_opt": len(z_raw),
                        "raw_logits": z_raw.numpy().astype(np.float64),
                    }
                )
    finally:
        if merged and hasattr(model.lm, "unmerge_adapter"):
            try:
                model.lm.unmerge_adapter()
            except Exception:
                pass

    eval_rows.sort(key=lambda x: x["order_key"])
    rows = eval_rows

    def _centered_log_p(z: np.ndarray) -> np.ndarray:
        z_c = z - np.max(z)
        lp = z_c - np.log(np.sum(np.exp(z_c)))
        return lp - np.mean(lp)

    if fit_priors and rows:
        # 1. Fit class-balanced noul_bias
        model.head.noul_bias.zero_()
        noul_items = [(r["label"], _centered_log_p(r["raw_logits"])) for r in rows if r["type"] == "noul" and r["n_opt"] == 2]
        noul_b_vec = _class_balanced_prior_bias(noul_items, k_val=2, alpha=alpha_prior, min_per_class=5)
        if noul_b_vec is not None:
            model.head.noul_bias.copy_(torch.tensor(noul_b_vec, dtype=torch.float32, device=model.device))

        # 2. Fit class-balanced choice_k_bias (gentle alpha=0.15 since choice options are uniformly shuffled)
        choice_by_k: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for r in rows:
            if r["type"] == "choice" and 2 <= r["n_opt"] <= model.head.MAX_K_TABLE:
                choice_by_k[r["n_opt"]].append((r["label"], _centered_log_p(r["raw_logits"])))
        model.head.choice_k_bias.zero_()
        for k_val, c_items in choice_by_k.items():
            c_bias = _class_balanced_prior_bias(c_items, k_val=k_val, alpha=0.15, min_per_class=4)
            if c_bias is not None:
                model.head.choice_k_bias[k_val, :k_val].copy_(
                    torch.tensor(c_bias, dtype=torch.float32, device=model.device)
                )

        # 3. Fit class-balanced score_k_bias for each K (only when all levels 0..K-1 are represented in dev)
        score_by_k: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for r in rows:
            if r["type"] == "score" and 2 <= r["n_opt"] <= model.head.MAX_K_TABLE:
                score_by_k[r["n_opt"]].append((r["label"], _centered_log_p(r["raw_logits"])))
        model.head.score_k_bias.zero_()
        for k_val, s_items in score_by_k.items():
            s_bias = _class_balanced_prior_bias(s_items, k_val=k_val, alpha=0.35, min_per_class=3)
            if s_bias is not None:
                model.head.score_k_bias[k_val, :k_val].copy_(
                    torch.tensor(s_bias, dtype=torch.float32, device=model.device)
                )

    noul_b = model.head.noul_bias.detach().cpu().numpy().astype(np.float64)
    choice_kb = model.head.choice_k_bias.detach().cpu().numpy().astype(np.float64)
    score_kb = model.head.score_k_bias.detach().cpu().numpy().astype(np.float64)

    for r in rows:
        z = r["raw_logits"].copy()
        k_val = r["n_opt"]
        if r["type"] == "noul" and k_val == 2:
            z = z - noul_b
        elif r["type"] == "score" and 2 <= k_val <= model.head.MAX_K_TABLE:
            z = z - score_kb[k_val, :k_val]
        elif r["type"] == "choice" and 2 <= k_val <= model.head.MAX_K_TABLE:
            z = z - choice_kb[k_val, :k_val]
        r["debiased_logits"] = z

    grid = np.exp(np.linspace(np.log(0.4), np.log(3.2), 121))
    fitted_temps: dict[str, float] = {"choice": 1.0, "noul": 1.0, "score": 1.0}

    for qtype in ("choice", "noul", "score"):
        sub = [r for r in rows if r["type"] == qtype]
        if not sub:
            continue
        best_t = 1.0
        best_obj = float("inf")
        for t in grid:
            objs = []
            for r in sub:
                z = (r["debiased_logits"] - np.max(r["debiased_logits"])) / float(t)
                exp_z = np.exp(z)
                p = exp_z / np.sum(exp_z)
                y = r["label"]
                nll = -math.log(max(float(p[y]), 1e-9))
                target = np.eye(len(p))[y]
                brier = float(((p - target) ** 2).sum())
                objs.append(nll + 0.20 * brier)
            mean_obj = float(np.mean(objs))
            if mean_obj < best_obj:
                best_obj = mean_obj
                best_t = float(t)
        fitted_temps[qtype] = round(best_t, 4)

    model.head.type_temps[0] = fitted_temps["choice"]
    model.head.type_temps[1] = fitted_temps["noul"]
    model.head.type_temps[2] = fitted_temps["score"]
    model.head.temperature = fitted_temps["choice"]

    def _summarize(logit_key: str, use_type_temps: bool) -> dict[str, Any]:
        nlls, accs, confs, briers = [], [], [], []
        by_type: dict[str, list[int]] = defaultdict(list)
        by_pool: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            t = fitted_temps.get(r["type"], 1.0) if use_type_temps else 1.0
            z = (r[logit_key] - np.max(r[logit_key])) / t
            exp_z = np.exp(z)
            p = exp_z / np.sum(exp_z)
            y = r["label"]
            nlls.append(float(-np.log(max(p[y], 1e-9))))
            ok = int(int(p.argmax()) == y)
            accs.append(ok)
            confs.append(float(p.max()))
            target = np.eye(len(p))[y]
            briers.append(float(((p - target) ** 2).sum()))
            by_type[r["type"]].append(ok)
            src_s = str(r["source"])
            pool_key = "hard_v1" if src_s.startswith("hard_") else ("pool5" if src_s.startswith("pool5_") else "v7_10source")
            by_pool[pool_key].append(ok)

        confs_arr = np.asarray(confs) if confs else np.zeros(1)
        accs_arr = np.asarray(accs, dtype=float) if accs else np.zeros(1)
        high = confs_arr >= 0.90
        return {
            "n_questions": len(accs),
            "acc": round(float(accs_arr.mean()), 4),
            "choice_acc": round(float(np.mean(by_type["choice"])), 4) if by_type["choice"] else 0.0,
            "noul_acc": round(float(np.mean(by_type["noul"])), 4) if by_type["noul"] else 0.0,
            "score_acc": round(float(np.mean(by_type["score"])), 4) if by_type["score"] else 0.0,
            "v7_10source_acc": round(float(np.mean(by_pool["v7_10source"])), 4) if by_pool["v7_10source"] else 0.0,
            "hard_v1_acc": round(float(np.mean(by_pool["hard_v1"])), 4) if by_pool["hard_v1"] else 0.0,
            "pool5_acc": round(float(np.mean(by_pool["pool5"])), 4) if by_pool["pool5"] else 0.0,
            "nll": round(float(np.mean(nlls)), 4) if nlls else 0.0,
            "brier": round(float(np.mean(briers)), 4) if briers else 0.0,
            "ece": round(compute_ece(confs_arr, accs_arr), 4),
            "mean_conf": round(float(confs_arr.mean()), 4),
            "confident_error_rate": round(float(np.mean(high & (accs_arr < 0.5))), 4),
        }

    raw_stats = _summarize("raw_logits", use_type_temps=False)
    cal_stats = _summarize("debiased_logits", use_type_temps=True)
    return {
        "fitted_temperature": fitted_temps["choice"],
        "type_temperatures": fitted_temps,
        "raw": raw_stats,
        "calibrated": cal_stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE_MODEL)
    ap.add_argument(
        "--data_dir",
        default=str(Path(__file__).resolve().parent / "data" / "stage2"),
    )
    ap.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parent / "checkpoints" / "stage2"),
    )
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default 16 across all 7 linear projections)")
    ap.add_argument("--lora_targets", default="all")
    ap.add_argument("--top_k_layers", type=int, default=0, help="Top K layers to transform (0 = all 42 layers = 25.3M params)")
    ap.add_argument("--head_dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument(
        "--max_steps",
        type=int,
        default=400,
        help="0 = use entire dataset per epoch, >0 = optimizer steps per epoch",
    )
    ap.add_argument("--lr", type=float, default=2.8e-5, help="Learning rate for LoRA parameters")
    ap.add_argument("--head_lr", type=float, default=1e-4, help="Separate learning rate for head parameters if any")
    ap.add_argument("--min_lr_ratio", type=float, default=0.15)
    ap.add_argument("--brier_weight", type=float, default=0.25)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    free_legacy_s1_ports((8009, 8010, 8011, 8012, 8013))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"
    stats_path = data_dir / "stats.json"

    needs_rebuild = not train_path.exists() or not dev_path.exists()
    if stats_path.exists():
        try:
            st_meta = json.loads(stats_path.read_text(encoding="utf-8"))
            if st_meta.get("schema_version") != SCHEMA_VERSION:
                needs_rebuild = True
        except Exception:
            needs_rebuild = True

    if needs_rebuild:
        print(f"Building High-Purity System 1 dataset in {data_dir} ({SCHEMA_VERSION})...", flush=True)
        import sys as _sys

        _old_argv = list(_sys.argv)
        _sys.argv = [_sys.argv[0], "--out_dir", str(data_dir)]
        from build_stage1_data import main as build_main

        build_main()
        _sys.argv = _old_argv

    with train_path.open("r", encoding="utf-8") as f:
        full_train_records = [json.loads(line) for line in f if line.strip()]
    with dev_path.open("r", encoding="utf-8") as f:
        dev_records = [json.loads(line) for line in f if line.strip()]

    eff_batch = args.batch * args.accum
    if args.lora_r <= 0 or args.epochs <= 0:
        train_records = []
    elif args.max_steps > 0:
        target_records = args.max_steps * eff_batch
        train_records = stratify_training_subset(full_train_records, target_records, seed=args.seed)
    else:
        train_records = stratify_training_subset(full_train_records, len(full_train_records), seed=args.seed)

    print(
        f"Loaded {len(full_train_records)} total pool records -> selected {len(train_records)} "
        f"records per epoch ({len(dev_records)} dev records).",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        f"Initializing MiniCPMSystemOne ({args.base}) on device={device} "
        f"(dtype=bf16, lora_r={args.lora_r}, top_k_layers={args.top_k_layers}, targets={args.lora_targets})...",
        flush=True,
    )
    tok = load_tokenizer(args.base)
    model = MiniCPMSystemOne(
        base_model=args.base,
        tok=tok,
        device=device,
        lora_r=args.lora_r,
        head_dim=args.head_dim,
        dtype=torch.bfloat16,
        lora_targets=args.lora_targets,
        top_k_layers=args.top_k_layers,
    )

    trainable_params = model.trainable_parameters()
    lora_params = model.lora_parameters()
    head_params = model.head_parameters()
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters (lora_r={args.lora_r}, top_k_layers={args.top_k_layers}): {n_trainable:,}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_dir = out_dir / "best_step"
    t_start = time.time()
    opt_step = 0
    history_log: list[dict[str, Any]] = []

    print("\n[Step 0] Evaluating pre-fine-tuning baseline on stratified dev subset (210 records)...", flush=True)
    step0_cal = evaluate_and_calibrate(model, tok, dev_records, max_records=210)
    best_val_acc = float(step0_cal["calibrated"]["acc"])
    print(
        f"  [Step 0 Baseline] calibrated_acc={best_val_acc:.4f} "
        f"(v7={step0_cal['calibrated']['v7_10source_acc']:.4f}, hard_v1={step0_cal['calibrated']['hard_v1_acc']:.4f}, "
        f"pool5={step0_cal['calibrated']['pool5_acc']:.4f}, choice={step0_cal['calibrated']['choice_acc']:.4f}, "
        f"noul={step0_cal['calibrated']['noul_acc']:.4f}, score={step0_cal['calibrated']['score_acc']:.4f}, "
        f"brier={step0_cal['calibrated']['brier']:.4f})",
        flush=True,
    )
    model.save_checkpoint(best_ckpt_dir, extra_meta={"opt_step": 0, "val_acc": best_val_acc})

    if n_trainable > 0 and len(train_records) > 0 and args.epochs > 0:
        distractor_pool: list[str] = []
        for r in full_train_records[:3000]:
            for q in r["questions"].values():
                if q["type"] == "choice" and isinstance(q.get("criteria"), dict):
                    for k, v in q["criteria"].items():
                        distractor_pool.append(option_text(str(k), v))

        steps_per_epoch = math.ceil(len(train_records) / eff_batch)
        total_opt_steps = steps_per_epoch * args.epochs
        warmup_steps = max(1, int(0.06 * total_opt_steps))
        min_ratio = float(args.min_lr_ratio)

        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": args.lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": args.head_lr})
        opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.01)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_ratio + (1.0 - min_ratio) * cosine

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        t_train_start = time.time()
        for ep in range(args.epochs):
            model.train()
            ep_rng = random.Random(args.seed + ep * 101)
            if args.max_steps > 0:
                ep_records = stratify_training_subset(
                    full_train_records, args.max_steps * eff_batch, seed=args.seed + ep
                )
            else:
                ep_records = list(train_records)
                ep_rng.shuffle(ep_records)

            print(
                f"\n[Epoch {ep + 1}/{args.epochs}] View-augmenting and 64-token bucketing {len(ep_records)} records...",
                flush=True,
            )
            by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for r in ep_records:
                try:
                    enc = augment_and_encode(r, tok, ep_rng, distractor_pool)
                except Exception:
                    continue
                q_indices = list(range(len(enc["causal_rows"])))
                if len(q_indices) > 3:
                    q_indices = ep_rng.sample(q_indices, 3)
                t_probs_all = enc.get("target_probs") or [None] * len(enc["causal_rows"])
                for qi in q_indices:
                    row = enc["causal_rows"][qi]
                    b_len = -(-len(row) // 64) * 64
                    by_bucket[b_len].append(
                        {
                            "causal_rows": [row],
                            "bucket_len": b_len,
                            "n_opts": [enc["n_opts"][qi]],
                            "q_types": [enc["q_types"][qi]],
                            "labels": [enc["labels"][qi]],
                            "target_probs": [t_probs_all[qi]],
                        }
                    )

            micro_batches: list[list[dict[str, Any]]] = []
            for b_len, items in by_bucket.items():
                ep_rng.shuffle(items)
                max_b = max(1, min(args.batch, 2560 // max(b_len, 64)))
                for i in range(0, len(items), max_b):
                    micro_batches.append(items[i : i + max_b])

            ep_rng.shuffle(micro_batches)
            target_mbs = steps_per_epoch * args.accum
            if len(micro_batches) > target_mbs:
                micro_batches = micro_batches[:target_mbs]
            print(
                f"Prepared {sum(len(mb) for mb in micro_batches)} question causal rows across "
                f"{len(micro_batches)} uniform-shape micro-batches ({steps_per_epoch} optimizer steps/epoch).",
                flush=True,
            )

            opt.zero_grad()
            running_loss = 0.0
            running_correct = 0
            running_questions = 0
            running_mbs = 0
            last_gpu_util = 100

            for m_idx, mb in enumerate(micro_batches, start=1):
                logits_batch = model.forward_batch(mb)
                losses = []
                for enc, zs in zip(mb, logits_batch):
                    q_types = enc["q_types"]
                    t_probs_list = enc.get("target_probs") or [None] * len(zs)
                    for z, y, qt, t_probs in zip(zs, enc["labels"], q_types, t_probs_list):
                        q_loss = compute_question_loss(
                            z,
                            int(y),
                            qt,
                            t_probs,
                            device=device,
                            brier_weight=args.brier_weight,
                        )
                        losses.append(q_loss)
                        running_correct += int(int(z.detach().argmax().item()) == int(y))
                        running_questions += 1

                loss = torch.stack(losses).mean()
                (loss / args.accum).backward()
                running_loss += float(loss.detach().item())
                running_mbs += 1

                if m_idx % args.accum == 0 or m_idx == len(micro_batches):
                    if opt_step in (0, 1, 2, 3, 4, 9) or (opt_step + 1) % 10 == 0:
                        g_u = read_mps_gpu_utilization()
                        if g_u > 0:
                            last_gpu_util = g_u

                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    opt_step += 1

                    if device == "mps" and opt_step % 10 == 0:
                        torch.mps.empty_cache()

                    if opt_step in (1, 2, 3, 4, 5, 10) or opt_step % 10 == 0 or opt_step == total_opt_steps:
                        elapsed = time.time() - t_train_start
                        rate = opt_step / max(elapsed, 1e-6)
                        eta_s = (total_opt_steps - opt_step) / max(rate, 1e-6)
                        avg_loss = running_loss / max(1, running_mbs)
                        acc = running_correct / max(1, running_questions)
                        cur_lrs = sched.get_last_lr()
                        lora_lr_cur = cur_lrs[0]
                        log_entry = {
                            "epoch": ep + 1,
                            "opt_step": opt_step,
                            "total_opt_steps": total_opt_steps,
                            "loss": round(avg_loss, 4),
                            "train_acc_window": round(acc, 4),
                            "lr": round(lora_lr_cur, 7),
                            "gpu_util_pct": last_gpu_util,
                            "elapsed_s": round(elapsed, 1),
                            "eta_s": round(eta_s, 1),
                        }
                        history_log.append(log_entry)
                        print(
                            f"  [Step {opt_step:4d}/{total_opt_steps}] "
                            f"loss={avg_loss:.4f} acc={acc:.3f} lr={lora_lr_cur:.2e} "
                            f"gpu={last_gpu_util}% elapsed={elapsed:.0f}s ETA={eta_s:.0f}s",
                            flush=True,
                        )
                        running_loss = 0.0
                        running_correct = 0
                        running_questions = 0
                        running_mbs = 0

                    if args.eval_every > 0 and opt_step % args.eval_every == 0 and opt_step < total_opt_steps:
                        val_rep = evaluate_and_calibrate(model, tok, dev_records, max_records=210)
                        val_acc = float(val_rep["calibrated"]["acc"])
                        print(
                            f"  --> [Val @ Step {opt_step}] acc={val_acc:.4f} "
                            f"(v7={val_rep['calibrated']['v7_10source_acc']:.4f}, hard_v1={val_rep['calibrated']['hard_v1_acc']:.4f}, "
                            f"pool5={val_rep['calibrated']['pool5_acc']:.4f}, choice={val_rep['calibrated']['choice_acc']:.4f}, "
                            f"noul={val_rep['calibrated']['noul_acc']:.4f}, score={val_rep['calibrated']['score_acc']:.4f}, "
                            f"brier={val_rep['calibrated']['brier']:.4f})",
                            flush=True,
                        )
                        if val_acc >= best_val_acc:
                            best_val_acc = val_acc
                            model.save_checkpoint(
                                best_ckpt_dir,
                                extra_meta={"opt_step": opt_step, "val_acc": best_val_acc, "calibration": val_rep},
                            )
                            print(f"  --> [New Best Checkpoint] saved to {best_ckpt_dir} (acc={best_val_acc:.4f})", flush=True)
                        model.train()

                    if args.save_every > 0 and opt_step % args.save_every == 0:
                        model.save_checkpoint(out_dir, extra_meta={"opt_step": opt_step, "status": "in_progress"})

    if n_trainable > 0 and (best_ckpt_dir / "adapter_model.safetensors").exists():
        final_sub_cal = evaluate_and_calibrate(model, tok, dev_records, max_records=210)
        final_sub_acc = float(final_sub_cal["calibrated"]["acc"])
        if best_val_acc > final_sub_acc + 0.005:
            print(
                f"\nRestoring best validation checkpoint from {best_ckpt_dir} "
                f"(best_val_acc={best_val_acc:.4f} > final_sub_acc={final_sub_acc:.4f})...",
                flush=True,
            )
            model, tok = MiniCPMSystemOne.load_checkpoint(best_ckpt_dir, device=device, dtype=torch.bfloat16)

    print(f"\nRunning final evaluation and class-balanced calibration on full dev.jsonl ({len(dev_records)} records)...", flush=True)
    final_cal = evaluate_and_calibrate(model, tok, dev_records)
    print(json.dumps(final_cal, indent=2, ensure_ascii=False), flush=True)

    final_meta = {
        "status": "completed",
        "schema_version": SCHEMA_VERSION,
        "epochs": args.epochs,
        "opt_steps": opt_step,
        "lora_r": args.lora_r,
        "top_k_layers": args.top_k_layers,
        "lr": args.lr,
        "train_records_per_epoch": len(train_records),
        "full_pool_records": len(full_train_records),
        "dev_records": len(dev_records),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "step0_baseline": step0_cal,
        "calibration": final_cal,
        "history": history_log,
    }
    model.save_checkpoint(out_dir, extra_meta=final_meta)
    with (out_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(final_meta, f, indent=2, ensure_ascii=False)
    print(
        f"Saved calibrated System 1 checkpoint to {out_dir} "
        f"(T_choice={model.head.type_temps[0]:.4f}, T_noul={model.head.type_temps[1]:.4f}, T_score={model.head.type_temps[2]:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
