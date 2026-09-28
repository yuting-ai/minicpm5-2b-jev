"""MiniCPM5-2B-Jev: Native causal FlashAttention multi-branch decision model with lm_head letter readout."""
from __future__ import annotations

import copy
from datetime import datetime
import json
import os
import re
import string
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "openbmb/MiniCPM5-2B"

MAX_OPTIONS = 255
MAX_STATE = 1152
MAX_BRANCH = 1664
MAX_PACKED = 2304
SERVE_MAX_STATE = 2560
SERVE_MAX_BRANCH = 3072
SERVE_MAX_PACKED = 3584

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
_TYPE_PREFIX_RE = re.compile(r"^\[(choice|noul|score)\]\s*", re.IGNORECASE)
_TOKENIZER_META_CACHE: dict[int, tuple[Any, dict[str, Any]]] = {}


class ContextOverflow(ValueError):
    """Raised when a record exceeds state, branch, or packed token limits."""


def load_tokenizer(name: str = DEFAULT_BASE_MODEL):
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(name)


def pad_id(tok) -> int:
    return tok.pad_token_id if tok.pad_token_id is not None else 0


def user_tokens(tok, text: str) -> list[int]:
    """Tokenize user text while preventing control-token injection."""
    safe_text = _SPECIAL_RE.sub(r"<¦\1¦>", str(text or ""))
    return tok(safe_text, add_special_tokens=False).input_ids


def tokenizer_decision_meta(tok) -> dict[str, Any]:
    """Build and cache option letter token IDs (A..Z, AA..) and chat template wrappers."""
    key = id(tok)
    if key in _TOKENIZER_META_CACHE:
        return _TOKENIZER_META_CACHE[key][1]

    upper = string.ascii_uppercase
    candidates = list(upper) + [a + b for a in upper for b in upper]
    ans_prefix = tok.encode("Answer: (", add_special_tokens=False)
    open_paren = tok.encode("\n(", add_special_tokens=False)

    label_names: list[str] = []
    label_ids: list[int] = []
    seen_ids: set[int] = set()
    for cand in candidates:
        enc = tok.encode(f"Answer: ({cand}", add_special_tokens=False)
        if len(enc) == len(ans_prefix) + 1 and enc[: len(ans_prefix)] == ans_prefix:
            tid = enc[-1]
            if tid not in seen_ids:
                seen_ids.add(tid)
                label_names.append(cand)
                label_ids.append(tid)
                if len(label_names) == MAX_OPTIONS:
                    break

    sentinel = "@@SYSTEMONE_USER_CONTENT@@"
    msgs = [{"role": "user", "content": sentinel}]
    try:
        rendered = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    head_str, tail_str = rendered.split(sentinel)
    for opened, closed in (("<think>\n", "</think>\n\n"), ("<think>", "</think>\n\n")):
        if tail_str.endswith(opened):
            tail_str += closed

    chat_head_ids = tok.encode(head_str + "Context:\n", add_special_tokens=False)
    chat_tail_ids = tok.encode(tail_str + "Answer: (", add_special_tokens=False)

    meta = {
        "label_names": label_names,
        "label_ids": label_ids,
        "open_paren": open_paren,
        "chat_head_ids": chat_head_ids,
        "chat_tail_ids": chat_tail_ids,
    }
    _TOKENIZER_META_CACHE[key] = (tok, meta)
    return meta


def _trim_state_tokens(state_tokens: list[int], keep_state: int) -> list[int]:
    """Preserve both the header/rules (first 60%) and most recent events/claim (last 40%) when trimming long states."""
    if len(state_tokens) <= keep_state:
        return state_tokens
    head_n = int(keep_state * 0.6)
    tail_n = max(0, keep_state - head_n)
    if tail_n == 0:
        return state_tokens[:keep_state]
    return state_tokens[:head_n] + state_tokens[-tail_n:]


def encode_record(
    tok,
    rec: dict[str, Any],
    max_state: int = MAX_STATE,
    max_branch: int = MAX_BRANCH,
    strict: bool = False,
) -> dict[str, Any]:
    """Encode a record into shared state tokens S and per-question branches.
    Each question branch forms an independent causal sequence S + br_k (mathematically identical to block-causal masking,
    while unlocking native hardware-fused is_causal=True FlashAttention on MPS/CUDA with zero 4D mask overhead).
    """
    meta = tokenizer_decision_meta(tok)
    label_names = meta["label_names"]
    label_ids = meta["label_ids"]
    open_paren = meta["open_paren"]
    chat_head_ids = meta["chat_head_ids"]
    chat_tail_ids = meta["chat_tail_ids"]

    state_tokens = user_tokens(tok, rec["state"])
    if strict and len(chat_head_ids) + len(state_tokens) > max_state:
        raise ContextOverflow(
            f"state exceeds {max_state} tokens: {len(chat_head_ids) + len(state_tokens)}"
        )

    keep_state = max(16, max_state - len(chat_head_ids))
    S = list(chat_head_ids) + _trim_state_tokens(state_tokens, keep_state)

    branches: list[tuple[list[int], int, str]] = []
    max_br_len = 0
    for q in rec["questions"]:
        raw_instr_full = str(q.get("instr", ""))
        q_type = str(q.get("type") or "").lower()
        if not q_type:
            m = _TYPE_PREFIX_RE.match(raw_instr_full)
            q_type = m.group(1).lower() if m else "choice"
        raw_instr = _TYPE_PREFIX_RE.sub("", raw_instr_full).strip()
        if q_type == "score":
            q_head = user_tokens(tok, f"\n\nQuestion (ordered rating scale, lowest to highest): {raw_instr}\nOptions:")
        elif q_type == "noul":
            q_head = user_tokens(tok, f"\n\nQuestion (yes/no judgement): {raw_instr}\nOptions:")
        else:
            q_head = user_tokens(tok, f"\n\nQuestion: {raw_instr}\nOptions:")

        opts = q["options"]
        if len(opts) > len(label_names):
            raise ContextOverflow(f"Too many options ({len(opts)} > {len(label_names)})")

        opt_tokens: list[int] = []
        for j, opt_str in enumerate(opts):
            clean_opt = str(opt_str).replace("\n", " ").strip()
            if j < 26:
                opt_tokens.extend(user_tokens(tok, f"\n({label_names[j]}) {clean_opt}"))
            else:
                opt_tokens.extend(open_paren + [label_ids[j]] + user_tokens(tok, f") {clean_opt}"))

        br = q_head + opt_tokens + list(chat_tail_ids)
        max_br_len = max(max_br_len, len(br))
        branches.append((br, len(opts), q_type))

    if len(S) + max_br_len > max_branch:
        if strict:
            raise ContextOverflow(
                f"branch too long: {max_br_len} tokens with {len(S)}-token state (limit {max_branch})"
            )
        allowed_state = max(64, max_branch - max_br_len - len(chat_head_ids))
        S = list(chat_head_ids) + _trim_state_tokens(state_tokens, allowed_state)

    ids = list(S)
    seg = [0] * len(S)
    pos = list(range(len(S)))
    decide_idx: list[int] = []
    opt_idx: list[list[int]] = []
    n_opts: list[int] = []
    q_types: list[str] = []
    causal_rows: list[list[int]] = []

    for k, (br, n_opt, q_type) in enumerate(branches, start=1):
        base = len(ids)
        p0 = len(S)
        br_pos = list(range(p0, p0 + len(br)))
        ids.extend(br)
        seg.extend([k] * len(br))
        pos.extend(br_pos)
        decide_idx.append(base + len(br) - 1)
        opt_idx.append(list(range(n_opt)))
        n_opts.append(n_opt)
        q_types.append(q_type)
        causal_rows.append(S + br)

    return {
        "ids": ids,
        "seg": seg,
        "pos": pos,
        "state_ids": S,
        "causal_rows": causal_rows,
        "decide_idx": decide_idx,
        "opt_idx": opt_idx,
        "n_opts": n_opts,
        "q_types": q_types,
        "labels": [int(q.get("label", 0)) for q in rec["questions"]],
        "target_probs": [q.get("target_probs") for q in rec["questions"]],
        "state_truncated": len(chat_head_ids) + len(state_tokens) > len(S),
    }


def record_fits(
    rec: dict[str, Any],
    tok,
    max_state: int = MAX_STATE,
    max_branch: int = MAX_BRANCH,
    max_packed: int = MAX_PACKED,
) -> bool:
    try:
        enc = encode_record(tok, rec, max_state=max_state, max_branch=max_branch, strict=True)
        return len(enc["ids"]) <= max_packed
    except Exception:
        return False


class LetterReadoutHead(nn.Module):
    """Native lm_head projection onto option letter tokens (A..Z, AA..) with PriDe prior debiasing and per-type temperature calibration."""

    MAX_K_TABLE = 32

    def __init__(self, letter_weight: torch.Tensor) -> None:
        super().__init__()
        n_max = int(letter_weight.shape[0])
        self.register_buffer("letter_weight", letter_weight.detach().clone().float())
        self.register_buffer("letter_bias", torch.zeros(n_max, dtype=torch.float32))
        self.register_buffer(
            "choice_k_bias",
            torch.zeros(self.MAX_K_TABLE + 1, self.MAX_K_TABLE, dtype=torch.float32),
        )
        self.register_buffer(
            "score_k_bias",
            torch.zeros(self.MAX_K_TABLE + 1, self.MAX_K_TABLE, dtype=torch.float32),
        )
        self.register_buffer("noul_bias", torch.zeros(2, dtype=torch.float32))
        self.register_buffer("score_bias", torch.zeros(11, dtype=torch.float32))
        self.register_buffer("type_temps", torch.ones(3, dtype=torch.float32))  # [choice, noul, score]
        self.temperature = 1.0
        self.use_prior = True

    def forward_raw(self, h_decide: torch.Tensor, n_opt: int) -> torch.Tensor:
        return F.linear(h_decide, self.letter_weight[:n_opt])

    def forward(self, h_decide: torch.Tensor, n_opt: int, q_type: str = "choice") -> torch.Tensor:
        z = F.linear(h_decide, self.letter_weight[:n_opt])
        if self.use_prior and not self.training:
            if q_type == "noul" and n_opt == 2:
                z = z - self.noul_bias
            elif q_type == "score":
                if (
                    2 <= n_opt <= self.MAX_K_TABLE
                    and torch.any(self.score_k_bias[n_opt, :n_opt] != 0)
                ):
                    z = z - self.score_k_bias[n_opt, :n_opt]
                elif n_opt == 11 and torch.any(self.score_bias != 0):
                    z = z - self.score_bias
            else:
                if (
                    2 <= n_opt <= self.MAX_K_TABLE
                    and torch.any(self.choice_k_bias[n_opt, :n_opt] != 0)
                ):
                    z = z - self.choice_k_bias[n_opt, :n_opt]
                else:
                    z = z - self.letter_bias[:n_opt]

        if self.training:
            return z

        type_idx = 1 if q_type == "noul" else (2 if q_type == "score" else 0)
        t_type = float(self.type_temps[type_idx].item())
        t = t_type if t_type != 1.0 else float(self.temperature)
        return z if t == 1.0 else z / t


class MiniCPMSystemOne(nn.Module):
    """MiniCPM5-2B backbone + All-Layer LoRA + native causal FlashAttention + LetterReadoutHead."""

    SHAPE_BUCKET = int(os.environ.get("MINICPM_S1_SHAPE_BUCKET", "64"))
    MAX_ROWS_PER_PASS = int(os.environ.get("MINICPM_S1_MAX_ROWS", "8"))

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        tok=None,
        device: str = "mps",
        lora_r: int = 0,
        head_dim: int = 256,
        dtype: torch.dtype = torch.bfloat16,
        adapter_dir: str | Path | None = None,
        lora_targets: str = "all",
        top_k_layers: int = 20,
        grad_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.tok = tok or load_tokenizer(base_model)
        self.pad_id = pad_id(self.tok)
        self.device = device
        self.lora_r = lora_r
        self.head_dim = head_dim
        self.lora_targets = lora_targets
        self.top_k_layers = top_k_layers

        meta = tokenizer_decision_meta(self.tok)
        label_ids = torch.tensor(meta["label_ids"], dtype=torch.long)

        try:
            full_lm = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=dtype,
                attn_implementation="sdpa",
                local_files_only=True,
            )
        except Exception:
            full_lm = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=dtype,
                attn_implementation="sdpa",
            )
        letter_weight = full_lm.lm_head.weight[label_ids].detach().clone().float()
        base_lm = full_lm.model
        del full_lm

        for p in base_lm.parameters():
            p.requires_grad_(False)

        has_adapter_files = (
            lora_r > 0
            and adapter_dir is not None
            and (Path(adapter_dir) / "adapter_config.json").exists()
            and (Path(adapter_dir) / "adapter_model.safetensors").exists()
        )
        if has_adapter_files:
            self.lm = PeftModel.from_pretrained(base_lm, str(adapter_dir), is_trainable=False)
        elif lora_r > 0:
            if grad_ckpt and hasattr(base_lm, "gradient_checkpointing_enable"):
                base_lm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                if hasattr(base_lm, "enable_input_require_grads"):
                    base_lm.enable_input_require_grads()

            target_map = {
                "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            }
            targets = target_map.get(lora_targets, target_map["all"])
            n_layers = len(base_lm.layers)
            layers_to_transform = (
                list(range(max(0, n_layers - top_k_layers), n_layers))
                if 0 < top_k_layers < n_layers
                else None
            )
            try:
                cfg = LoraConfig(
                    task_type="FEATURE_EXTRACTION",
                    r=lora_r,
                    lora_alpha=2 * lora_r,
                    lora_dropout=0.0,
                    target_modules=targets,
                    layers_to_transform=layers_to_transform,
                    layers_pattern="layers" if layers_to_transform else None,
                )
                self.lm = get_peft_model(base_lm, cfg)
            except Exception:
                cfg = LoraConfig(
                    task_type="FEATURE_EXTRACTION",
                    r=lora_r,
                    lora_alpha=2 * lora_r,
                    lora_dropout=0.0,
                    target_modules=targets,
                )
                self.lm = get_peft_model(base_lm, cfg)
            if 0 < top_k_layers < n_layers:
                for layer in base_lm.layers[:-top_k_layers]:
                    for p in layer.parameters():
                        p.requires_grad_(False)
        else:
            self.lm = base_lm

        self.head = LetterReadoutHead(letter_weight)
        self.to(device)
        self.head.float()

    def _extract_causal_rows(self, enc: dict[str, Any]) -> list[list[int]]:
        if "causal_rows" in enc:
            return enc["causal_rows"]
        # Fallback if a legacy encoded dict without causal_rows is passed
        ids = enc["ids"]
        seg = enc["seg"]
        state_ids = [tid for tid, s in zip(ids, seg) if s == 0]
        rows: list[list[int]] = []
        for k in range(1, len(enc["decide_idx"]) + 1):
            br_ids = [tid for tid, s in zip(ids, seg) if s == k]
            rows.append(state_ids + br_ids)
        return rows

    def _forward_causal_rows(self, flat_rows: list[list[int]]) -> torch.Tensor:
        """Run a list of causal rows [S + br_k] using native unmasked causal SDPA (is_causal=True)
        and return the hidden states at each row's terminal `Answer: (` decision token [N_rows, H].
        Right-padding after `len(row) - 1` cannot affect causal positions `< len(row)`.
        """
        max_len = max(len(r) for r in flat_rows)
        if max_len > 1536:
            max_r = 1
        elif max_len > 896:
            max_r = min(2, self.MAX_ROWS_PER_PASS)
        else:
            max_r = self.MAX_ROWS_PER_PASS

        if len(flat_rows) > max_r:
            chunks = [
                self._forward_causal_rows(flat_rows[i : i + max_r])
                for i in range(0, len(flat_rows), max_r)
            ]
            return torch.cat(chunks, dim=0)

        L = max_len
        if str(self.device) == "mps" and self.SHAPE_BUCKET > 1:
            L = -(-L // self.SHAPE_BUCKET) * self.SHAPE_BUCKET
        B = len(flat_rows)
        padded = [r + [self.pad_id] * (L - len(r)) for r in flat_rows]
        slot_idx = [len(r) - 1 for r in flat_rows]
        ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        out = self.lm(input_ids=ids, use_cache=False)
        h = out.last_hidden_state
        batch_idx = torch.arange(B, device=self.device)
        slot_ten = torch.tensor(slot_idx, dtype=torch.long, device=self.device)
        res = h[batch_idx, slot_ten].float()
        if str(self.device) == "mps" and max_len > 1536:
            del out, h, ids
            torch.mps.empty_cache()
        return res

    @torch.no_grad()
    def _forward_record_eval(self, enc: dict[str, Any]) -> torch.Tensor:
        """Fast inference for a single record: reuses the state prefix KV-cache when a record
        has multiple questions and a long state (`len(state_ids) >= 256`), avoiding redundant state encoding.
        """
        rows = self._extract_causal_rows(enc)
        state_ids = enc.get("state_ids")
        if state_ids is None:
            ids = enc["ids"]
            seg = enc["seg"]
            state_ids = [tid for tid, s in zip(ids, seg) if s == 0]
        L_s = len(state_ids)
        if len(rows) <= 1 or L_s < 256:
            return self._forward_causal_rows(rows)

        s_ten = torch.tensor([state_ids], dtype=torch.long, device=self.device)
        out_s = self.lm(input_ids=s_ten, use_cache=True)
        pkv = out_s.past_key_values
        h_list: list[torch.Tensor] = []
        for r in rows:
            br = r[L_s:]
            br_ten = torch.tensor([br], dtype=torch.long, device=self.device)
            out_b = self.lm(input_ids=br_ten, past_key_values=pkv, use_cache=True)
            h_list.append(out_b.last_hidden_state[0, -1].float())
            pkv.crop(-len(br))
        res = torch.stack(h_list, dim=0)
        del out_s, pkv, s_ten
        if str(self.device) == "mps":
            torch.mps.empty_cache()
        return res

    def forward_batch(self, encs: list[dict[str, Any]]) -> list[list[torch.Tensor]]:
        flat_rows: list[list[int]] = []
        counts: list[int] = []
        for e in encs:
            rows = self._extract_causal_rows(e)
            flat_rows.extend(rows)
            counts.append(len(rows))

        h_decides = self._forward_causal_rows(flat_rows)
        out: list[list[torch.Tensor]] = []
        offset = 0
        for e, c in zip(encs, counts):
            n_opts = e.get("n_opts") or [len(oi) for oi in e["opt_idx"]]
            q_types = e.get("q_types") or ["choice"] * len(n_opts)
            h_sub = h_decides[offset : offset + c]
            offset += c
            out.append(
                [
                    self.head(h_sub[j], n_opt, q_type=qt)
                    for j, (n_opt, qt) in enumerate(zip(n_opts, q_types))
                ]
            )
        return out

    def forward(self, enc: dict[str, Any]) -> list[torch.Tensor]:
        return self.forward_batch([enc])[0]

    @torch.no_grad()
    def raw_logits(self, enc: dict[str, Any]) -> list[torch.Tensor]:
        h_decides = self._forward_record_eval(enc)
        n_opts = enc.get("n_opts") or [len(oi) for oi in enc["opt_idx"]]
        return [self.head.forward_raw(h_decides[j], n_opt).cpu() for j, n_opt in enumerate(n_opts)]

    @torch.no_grad()
    def raw_logits_batch(self, encs: list[dict[str, Any]]) -> list[list[torch.Tensor]]:
        flat_rows: list[list[int]] = []
        counts: list[int] = []
        for e in encs:
            rows = self._extract_causal_rows(e)
            flat_rows.extend(rows)
            counts.append(len(rows))
        h_decides = self._forward_causal_rows(flat_rows)
        out: list[list[torch.Tensor]] = []
        offset = 0
        for e, c in zip(encs, counts):
            n_opts = e.get("n_opts") or [len(oi) for oi in e["opt_idx"]]
            h_sub = h_decides[offset : offset + c]
            offset += c
            out.append(
                [self.head.forward_raw(h_sub[j], n_opt).cpu() for j, n_opt in enumerate(n_opts)]
            )
        return out

    @torch.no_grad()
    def probs(self, enc: dict[str, Any]) -> list[torch.Tensor]:
        return [F.softmax(z, dim=-1).cpu() for z in self.forward(enc)]

    @torch.no_grad()
    def logits_and_probs(self, enc: dict[str, Any]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        h_decides = self._forward_record_eval(enc)
        n_opts = enc.get("n_opts") or [len(oi) for oi in enc["opt_idx"]]
        q_types = enc.get("q_types") or ["choice"] * len(n_opts)
        zs = [
            self.head(h_decides[j], n_opt, q_type=qt)
            for j, (n_opt, qt) in enumerate(zip(n_opts, q_types))
        ]
        return [(z.cpu(), F.softmax(z, dim=-1).cpu()) for z in zs]

    @torch.no_grad()
    def predict_record(
        self,
        rec_internal: dict[str, Any],
        use_pride: bool = False,
        max_state: int = SERVE_MAX_STATE,
        max_branch: int = SERVE_MAX_BRANCH,
        max_packed: int = SERVE_MAX_PACKED,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Predict (calibrated_logits, probs) for every question in rec_internal using fast native causal rows."""
        self.eval()
        qs = rec_internal["questions"]
        if not qs:
            return []

        if not use_pride:
            enc_full = encode_record(
                self.tok, rec_internal, max_state=max_state, max_branch=max_branch, strict=False
            )
            return self.logits_and_probs(enc_full)

        # Exact PriDe cyclic permutation across shifts for choice/noul (preserving ordered scale for score)
        all_sub_qs: list[dict[str, Any]] = []
        q_specs: list[tuple[str, int, float, list[list[int]] | None]] = []
        for q in qs:
            opts = list(q["options"])
            K = len(opts)
            qt = str(q.get("type") or "choice").lower()
            type_idx = 1 if qt == "noul" else (2 if qt == "score" else 0)
            t_type = float(self.head.type_temps[type_idx].item())
            temp = t_type if t_type != 1.0 else float(self.head.temperature)

            if qt == "score" or K <= 1 or K > 12:
                all_sub_qs.append(q)
                q_specs.append((qt, K, temp, None))
            else:
                perms: list[list[int]] = []
                for s in range(K):
                    perm = [(j + s) % K for j in range(K)]
                    perms.append(perm)
                    q_shift = copy.deepcopy(q)
                    q_shift["options"] = [opts[idx] for idx in perm]
                    all_sub_qs.append(q_shift)
                q_specs.append((qt, K, temp, perms))

        sub_enc = encode_record(
            self.tok,
            {**rec_internal, "questions": all_sub_qs},
            max_state=max_state,
            max_branch=max_branch,
            strict=False,
        )
        h_decides = self._forward_record_eval(sub_enc)

        results_pride: list[tuple[torch.Tensor, torch.Tensor]] = []
        offset = 0
        for qt, K, temp, perms in q_specs:
            if perms is None:
                z_cal = self.head(h_decides[offset], K, q_type=qt).cpu()
                offset += 1
                results_pride.append((z_cal, F.softmax(z_cal, dim=-1)))
            else:
                unpermed_log_probs = np.zeros((K, K), dtype=np.float64)
                for s, perm in enumerate(perms):
                    z_s = self.head.forward_raw(h_decides[offset + s], K).cpu().numpy().astype(np.float64)
                    z_c = z_s - np.max(z_s)
                    log_p = z_c - np.log(np.sum(np.exp(z_c)))
                    for pos_j, opt_idx in enumerate(perm):
                        unpermed_log_probs[s, opt_idx] = log_p[pos_j]
                offset += K
                debiased_logits = unpermed_log_probs.mean(axis=0)
                debiased_logits = debiased_logits - debiased_logits.mean()
                cal_logits = debiased_logits / max(temp, 1e-4)
                z_ten = torch.tensor(cal_logits, dtype=torch.float32)
                p_ten = F.softmax(z_ten, dim=-1)
                results_pride.append((z_ten, p_ten))

        return results_pride

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.head.parameters() if p.requires_grad]

    def lora_parameters(self) -> list[nn.Parameter]:
        head_ids = {id(p) for p in self.head.parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in head_ids]

    def save_checkpoint(self, out_dir: str | Path, extra_meta: dict[str, Any] | None = None) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if self.lora_r > 0 and hasattr(self.lm, "save_pretrained"):
            self.lm.save_pretrained(str(out_path))
        else:
            for stale in ("adapter_config.json", "adapter_model.safetensors"):
                sp = out_path / stale
                if sp.exists():
                    sp.unlink()
        type_temps_list = [round(float(x), 4) for x in self.head.type_temps.detach().cpu().tolist()]
        payload = {
            "state_dict": self.head.state_dict(),
            "base_model": self.base_model,
            "lora_r": self.lora_r,
            "lora_targets": self.lora_targets,
            "top_k_layers": self.top_k_layers,
            "head_dim": self.head_dim,
            "readout": "native_lm_head_letters_pride",
            "temperature": float(self.head.temperature),
            "type_temperatures": {
                "choice": type_temps_list[0],
                "noul": type_temps_list[1],
                "score": type_temps_list[2],
            },
            "meta": extra_meta or {},
        }
        torch.save(payload, out_path / "head.pt")
        with (out_path / "checkpoint_meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_model": self.base_model,
                    "lora_r": self.lora_r,
                    "lora_targets": self.lora_targets,
                    "top_k_layers": self.top_k_layers,
                    "head_dim": self.head_dim,
                    "readout": "native_lm_head_letters_pride",
                    "temperature": float(self.head.temperature),
                    "type_temperatures": {
                        "choice": type_temps_list[0],
                        "noul": type_temps_list[1],
                        "score": type_temps_list[2],
                    },
                    **(extra_meta or {}),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load_checkpoint(
        cls,
        ckpt_dir: str | Path,
        device: str = "mps",
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple["MiniCPMSystemOne", Any]:
        ckpt_path = Path(ckpt_dir)
        head_pt = ckpt_path / "head.pt"
        if head_pt.exists():
            head_data = torch.load(head_pt, map_location="cpu", weights_only=True)
        else:
            head_data = {}
        base_model = head_data.get("base_model", DEFAULT_BASE_MODEL)
        lora_r = int(head_data.get("lora_r", 0))
        lora_targets = str(head_data.get("lora_targets", "all"))
        top_k_layers = int(head_data.get("top_k_layers", 0))
        head_dim = int(head_data.get("head_dim", 256))
        tok = load_tokenizer(base_model)
        has_adapter = (
            lora_r > 0
            and (ckpt_path / "adapter_config.json").exists()
            and (ckpt_path / "adapter_model.safetensors").exists()
        )
        model = cls(
            base_model=base_model,
            tok=tok,
            device=device,
            lora_r=lora_r,
            head_dim=head_dim,
            dtype=dtype,
            adapter_dir=ckpt_path if has_adapter else None,
            lora_targets=lora_targets,
            top_k_layers=top_k_layers,
            grad_ckpt=False,
        )
        if "state_dict" in head_data and "letter_weight" in head_data["state_dict"]:
            model.head.load_state_dict(head_data["state_dict"], strict=False)
            model.head.to(device)
        model.head.temperature = float(
            os.environ.get("MINICPM_S1_TEMPERATURE") or head_data.get("temperature", 1.0)
        )
        if os.environ.get("MINICPM_S1_TEMPERATURE"):
            t_override = float(os.environ["MINICPM_S1_TEMPERATURE"])
            model.head.type_temps.fill_(t_override)
        model.eval()
        return model, tok


SCORE_OPTIONS = [f"{i}: score {i} out of 10" for i in range(11)]
_LEVEL_PREFIX_RE = re.compile(r"^\s*-?\d+\s*:")
_NORM_TOKEN_RE = re.compile(r"[\s_\.\-/]+")
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE_RE = re.compile(
    rf"\b(?:{_MONTHS}) \d{{1,2}}, \d{{4}}\b|\b\d{{1,2}} (?:{_MONTHS}) \d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b"
)


def _norm_key(s: str) -> str:
    return _NORM_TOKEN_RE.sub(" ", str(s or "").strip().lower()).strip()


def compute_date_facts(text: str, max_dates: int = 7) -> str:
    """Deterministic date difference helper for states containing 2..max_dates absolute dates."""
    found: list[tuple[str, datetime]] = []
    for m in _DATE_RE.finditer(text):
        raw = m.group(0)
        d = None
        for fmt in ("%B %d, %Y", "%d %B %Y", "%Y-%m-%d"):
            try:
                d = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        if d is not None and raw not in [r for r, _ in found]:
            found.append((raw, d))
            if len(found) >= max_dates:
                break
    if len(found) < 2:
        return ""
    facts: list[str] = []
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            n = (found[j][1] - found[i][1]).days
            if n == 0:
                facts.append(f"{found[j][0]} is the same day as {found[i][0]}.")
            else:
                direction = "after" if n > 0 else "before"
                facts.append(
                    f"{found[j][0]} is {abs(n)} day{'s' if abs(n) != 1 else ''} {direction} {found[i][0]}."
                )
    return " ".join(facts[:10])


def render_json_content(v: Any, indent: int = 0) -> str:
    """Flatten str | dict | list into clean text, unpacking JSON-serialized string objects when present."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, str) and indent == 0:
        s_strip = v.strip()
        if (s_strip.startswith("{") and s_strip.endswith("}")) or (
            s_strip.startswith("[") and s_strip.endswith("]")
        ):
            try:
                parsed = json.loads(s_strip)
                if isinstance(parsed, (dict, list)):
                    return render_json_content(parsed, indent=0)
            except Exception:
                pass
        return v
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render_json_content(x, indent + 1).lstrip()}" for x in v)
    if isinstance(v, dict):
        return "\n".join(
            f"{pad}{k}:\n{render_json_content(x, indent + 1)}"
            if isinstance(x, (dict, list))
            else f"{pad}{k}: {render_json_content(x)}"
            for k, x in v.items()
        )
    return str(v)


def option_text(name: str, desc: Any) -> str:
    """Render option name and optional description while suppressing redundant 'name: name' duplicates."""
    if desc is None or desc == "":
        return str(name)
    rendered = render_json_content(desc).strip()
    if not rendered or _norm_key(rendered) == _norm_key(name):
        return str(name)
    return f"{name}: {rendered}"


def to_internal_record(rec: dict[str, Any], add_date_facts: bool = False) -> dict[str, Any]:
    """Convert wire-format record (questions dict) into internal question list for encoding."""
    qs = []
    for name, q in rec["questions"].items():
        qtype = str(q["type"]).lower()
        instr_str = render_json_content(q.get("instructions"))
        if not instr_str and qtype == "noul":
            instr_str = "Which answer fits the context?"
        target_probs: list[float] | None = None

        if qtype == "choice":
            crit = q.get("criteria") or {}
            if isinstance(crit, list):
                crit = {str(x): None for x in crit}
            keys = list(crit.keys())
            opts = [option_text(str(k), crit[k]) for k in keys]
            lbl = q.get("label", q.get("answer", keys[0] if keys else 0))
            label = keys.index(lbl) if lbl in keys else int(lbl)
        elif qtype == "noul":
            crit = q.get("criteria") if isinstance(q.get("criteria"), dict) else {}
            keys = ["false", "true"]
            f_desc = crit.get("false", crit.get(False, "false, negative, does not hold"))
            t_desc = crit.get("true", crit.get(True, "true, affirmative, holds"))
            opts = [
                option_text("no", f_desc or "false, negative, does not hold"),
                option_text("yes", t_desc or "true, affirmative, holds"),
            ]
            lbl = q.get("label", q.get("answer", False))
            if isinstance(lbl, bool):
                label = int(lbl)
            elif str(lbl).lower() in ("true", "yes", "1"):
                label = 1
            else:
                label = 0
        elif qtype == "score":
            crit = q.get("criteria")
            if isinstance(crit, dict):
                crit = [crit[k] for k in sorted(crit, key=lambda x: float(x))]
            if isinstance(crit, list) and crit:
                opts = []
                for i, x in enumerate(crit):
                    rendered_x = render_json_content(x).strip()
                    if _LEVEL_PREFIX_RE.match(rendered_x):
                        opts.append(rendered_x)
                    else:
                        opts.append(f"{i}: {rendered_x}")
                keys = [str(i) for i in range(len(opts))]
            else:
                opts = list(SCORE_OPTIONS)
                keys = [str(i) for i in range(11)]
            label = int(q.get("label", q.get("answer", 0)))
        else:
            raise ValueError(f"Unknown question type: {qtype}")

        if isinstance(q.get("target_probs"), list) and len(q["target_probs"]) == len(opts):
            target_probs = [float(p) for p in q["target_probs"]]
        elif isinstance(q.get("target"), dict):
            tgt_map = q["target"]
            probs_raw = [float(tgt_map.get(k, 1.0 / len(keys))) for k in keys]
            s_tot = sum(probs_raw) or 1.0
            target_probs = [p / s_tot for p in probs_raw]

        qs.append(
            {
                "name": name,
                "type": qtype,
                "instr": f"[{qtype}] {instr_str}",
                "options": opts,
                "keys": keys,
                "label": max(0, min(label, len(opts) - 1)),
                "target_probs": target_probs,
            }
        )
    meta = rec.get("_meta") or {}
    state_str = render_json_content(rec["state"])
    if add_date_facts and "date_facts:" not in state_str:
        df = compute_date_facts(state_str)
        if df:
            state_str = f"{state_str}\n\ndate_facts: {df}"
    return {
        "id": rec.get("id") or meta.get("id", ""),
        "source": rec.get("source") or meta.get("source", "custom"),
        "state": state_str,
        "questions": qs,
    }

