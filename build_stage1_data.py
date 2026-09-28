"""Build High-Purity System 1 Decision Dataset (jev_s1_clean_v2) for MiniCPM5-2B-Jev:
- Pool 1: decision-v7 full balanced 12-source pool enriched with teacher label_descriptions.json (50% rich semantic criteria, 50% bare labels)
- Pool 2: hard-v1 7 programmatic reasoning families (Templates 0-3 train, Template 4 held-out eval)
- Pool 3: night2 calibration & robustness suite (dates with relational day counts, unknowable with uniform 1/K soft targets, statement-form noul assertions)
- Pool 4: decider strictly-filtered multi-branch teacher suite (custom_questions with teacher_p>=0.85, monotone & level-balanced score, situations, generic-vs-catchall routing + routing_terse, dual-question command safety)
- Pool 5: programmatic rule twins (decider.data.rules base + state_twin + rule_twin) & strictly 1:1 level-balanced 3/4/5-level ordinal score + multi-condition noul verification rubrics
Completely decoupled from any browser or DOM tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from model import DEFAULT_BASE_MODEL, MAX_BRANCH, MAX_PACKED, MAX_STATE, load_tokenizer, record_fits

SEED = 42
SCHEMA_VERSION = "jev_s1_clean_v2"
SCORE_OPTIONS = [f"{i}: score {i} out of 10" for i in range(11)]
NOUL_OPTIONS = ["no: false, negative, does not hold", "yes: true, affirmative, holds"]
SUITES_REVISION = "a88f56db5341397299137cb68775c2ea6e3f68cb"

KEV_ROOT = Path(__file__).resolve().parents[1] / "kev"
DECIDER_ROOT = Path(__file__).resolve().parents[1] / "decider"
DECIDER_TEACHER_DIR = DECIDER_ROOT / "teacher_data"

DBPEDIA14_DESCRIPTIONS = {
    "company": "Company, business enterprise, or corporation",
    "educationalinstitution": "Educational institution, school, college, or university",
    "artist": "Artist, musician, painter, author, or creator",
    "athlete": "Athlete, sports player, or competitor",
    "officeholder": "Political officeholder, government official, or leader",
    "meanoftransportation": "Means of transportation, vehicle, ship, train, or aircraft",
    "building": "Building, architectural structure, museum, or venue",
    "naturalplace": "Natural place, river, mountain, lake, or geographic feature",
    "village": "Village, town, settlement, or municipality",
    "animal": "Animal, mammal, bird, insect, or fauna species",
    "plant": "Plant, tree, flower, or flora species",
    "album": "Music album, record, or soundtrack release",
    "film": "Film, movie, documentary, or cinema work",
    "writtenwork": "Written work, book, novel, periodical, or publication",
}

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


def shuffle_choice_questions(questions: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Deterministically shuffle option order for all choice questions so gold positions are strictly 1/K uniform."""
    out: dict[str, Any] = {}
    for qk, qv in questions.items():
        q_copy = dict(qv)
        if q_copy.get("type") == "choice":
            crit = q_copy.get("criteria")
            if isinstance(crit, dict) and len(crit) >= 2:
                keys = list(crit.keys())
                rng.shuffle(keys)
                q_copy["criteria"] = {k: crit[k] for k in keys}
        out[qk] = q_copy
    return out


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


def _load_teacher_label_map() -> dict[str, dict[str, str]]:
    """Load flattened option descriptions from decider/teacher_data/label_descriptions.json."""
    ld_path = DECIDER_TEACHER_DIR / "label_descriptions.json"
    if not ld_path.exists():
        return {}
    raw = json.loads(ld_path.read_text(encoding="utf-8"))
    src_alias = {
        "agnews": "ag_news",
        "mnli": "mnli",
        "trec": "trec",
        "sst5": "sst5",
        "yelp": "yelp",
        "banking77": "banking77",
        "dbpedia14": "dbpedia",
        "imdb": "imdb",
        "boolq": "boolq",
    }
    out: dict[str, dict[str, str]] = {}
    for v7_src, ld_key in src_alias.items():
        if ld_key not in raw:
            continue
        norm_map: dict[str, str] = {}
        for _, opt_dict in raw[ld_key].items():
            if not isinstance(opt_dict, dict):
                continue
            for opt_name, spec in opt_dict.items():
                base_name = opt_name.split("(")[0].strip()
                if isinstance(spec, dict) and spec.get("what"):
                    what_s = str(spec["what"]).strip()
                    not_for_s = str(spec.get("not_for") or "").strip()
                    full_s = f"{what_s} (Not for: {not_for_s})" if not_for_s else what_s
                    norm_map[_norm_key(opt_name)] = full_s
                    norm_map[_norm_key(base_name)] = what_s
        out[v7_src] = norm_map
    return out


def _clean_v7_questions(
    src: str,
    questions: dict[str, Any],
    rng: random.Random,
    teacher_desc_map: dict[str, dict[str, str]],
    for_eval: bool = False,
) -> dict[str, Any]:
    """Ensure decision-v7 questions have clean criteria (no 'world: world' duplication),
    enriching 55% of training items with teacher semantic descriptions and keeping 45% bare/concise,
    and deterministically shuffling choice option order.
    """
    cleaned: dict[str, Any] = {}
    src_descs = teacher_desc_map.get(src, {})
    use_rich_desc = (not for_eval) and (rng.random() < 0.55)

    for qk, qv in questions.items():
        qtype = qv.get("type")
        q_copy = dict(qv)
        if qtype == "choice" and isinstance(q_copy.get("criteria"), dict):
            raw_crit = dict(q_copy["criteria"])
            gold_lbl = q_copy.get("label")
            if len(raw_crit) > 14 and gold_lbl in raw_crit:
                other_keys = [k for k in raw_crit.keys() if k != gold_lbl]
                rng.shuffle(other_keys)
                chosen_keys = [gold_lbl] + other_keys[:11]
                rng.shuffle(chosen_keys)
                raw_crit = {k: raw_crit[k] for k in chosen_keys}
            enriched_crit: dict[str, Any] = {}
            for ck, cv in raw_crit.items():
                nk = _norm_key(ck)
                has_real_cv = cv not in (None, "") and _norm_key(str(cv)) != nk
                if has_real_cv:
                    enriched_crit[ck] = cv
                elif use_rich_desc and nk in src_descs:
                    enriched_crit[ck] = src_descs[nk]
                elif src == "dbpedia14" and ck in DBPEDIA14_DESCRIPTIONS:
                    enriched_crit[ck] = DBPEDIA14_DESCRIPTIONS[ck] if (use_rich_desc or for_eval) else None
                elif src == "banking77":
                    enriched_crit[ck] = src_descs.get(nk) if use_rich_desc else None
                else:
                    enriched_crit[ck] = None
            keys = list(enriched_crit.keys())
            rng.shuffle(keys)
            q_copy["criteria"] = {k: enriched_crit[k] for k in keys}
        cleaned[qk] = q_copy
    return cleaned


def build_pool1_decision_v7(
    tok,
    train_target: int = 6500,
    per_source_eval: int = 30,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Load decision-v7 (all 12 sources including sst5, yelp, amazon, legacy_policy, compositional)."""
    print("[Pool 1] Loading jaredpalmer/kev-suites (v7/decision-v7/train.jsonl)...", flush=True)
    try:
        path = hf_hub_download(
            repo_id="jaredpalmer/kev-suites",
            filename="v7/decision-v7/train.jsonl",
            repo_type="dataset",
            revision=SUITES_REVISION,
            local_files_only=True,
        )
    except Exception:
        path = hf_hub_download(
            repo_id="jaredpalmer/kev-suites",
            filename="v7/decision-v7/train.jsonl",
            repo_type="dataset",
            revision=SUITES_REVISION,
        )

    rng = random.Random(SEED)
    teacher_desc_map = _load_teacher_label_map()
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            meta = rec.get("_meta") or {}
            variant = rec.get("variant") or meta.get("variant", "clean")
            src = rec.get("source") or meta.get("source", "general")
            if variant != "clean" or not isinstance(rec.get("questions"), dict):
                continue
            by_source[src].append(rec)

    for src in by_source:
        rng.shuffle(by_source[src])

    sources = sorted(by_source.keys())
    print(
        f"[Pool 1] Loaded {sum(len(v) for v in by_source.values())} records across {len(sources)} sources: {sources}",
        flush=True,
    )

    core_10_sources = [
        "agnews",
        "banking77",
        "boolq",
        "compositional",
        "dbpedia14",
        "imdb",
        "legacy_policy",
        "mnli",
        "sst5",
        "yelp",
    ]
    eval_10source: list[dict[str, Any]] = []
    idx_map = {s: 0 for s in sources}
    eval_hashes: set[str] = set()

    for s in core_10_sources:
        pool = by_source.get(s, [])
        collected = 0
        while idx_map[s] < len(pool) and collected < per_source_eval:
            cand = copy.deepcopy(pool[idx_map[s]]) if "copy" in globals() else json.loads(json.dumps(pool[idx_map[s]]))
            idx_map[s] += 1
            cand["questions"] = _clean_v7_questions(s, cand["questions"], rng, teacher_desc_map, for_eval=True)
            try:
                internal = to_internal_record(cand)
            except Exception:
                continue
            if record_fits(internal, tok):
                st_hash = hashlib.sha256(internal["state"].strip().lower().encode("utf-8")).hexdigest()
                if st_hash in eval_hashes:
                    continue
                eval_hashes.add(st_hash)
                meta = cand.get("_meta") or {}
                eval_10source.append(
                    {
                        "id": f"v7_eval_{s}_{collected:03d}_{meta.get('id', '')}",
                        "source": f"l1_{s}",
                        "variant": "clean",
                        "state": internal["state"],
                        "questions": cand["questions"],
                    }
                )
                collected += 1

    train_out: list[dict[str, Any]] = []
    active_sources = list(sources)
    while len(train_out) < train_target and active_sources:
        next_active = []
        for s in active_sources:
            pool = by_source[s]
            while idx_map[s] < len(pool):
                cand = json.loads(json.dumps(pool[idx_map[s]]))
                idx_map[s] += 1
                cand["questions"] = _clean_v7_questions(s, cand["questions"], rng, teacher_desc_map, for_eval=False)
                try:
                    internal = to_internal_record(cand)
                except Exception:
                    continue
                st_hash = hashlib.sha256(internal["state"].strip().lower().encode("utf-8")).hexdigest()
                if st_hash in eval_hashes:
                    continue
                if record_fits(internal, tok):
                    meta = cand.get("_meta") or {}
                    train_out.append(
                        {
                            "id": f"v7_train_{s}_{len(train_out):05d}_{meta.get('id', '')}",
                            "source": f"l1_{s}",
                            "variant": "clean",
                            "state": internal["state"],
                            "questions": cand["questions"],
                        }
                    )
                    break
            if idx_map[s] < len(pool) and len(train_out) < train_target:
                next_active.append(s)
        active_sources = next_active

    print(
        f"[Pool 1] Collected {len(train_out)} train records and {len(eval_10source)} stratified 10-source eval records.",
        flush=True,
    )
    return train_out, eval_10source, eval_hashes


def build_pool2_hard_v1(
    tok,
    train_per_family: int = 500,
    eval_per_family: int = 50,
    seen_hashes: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate hard-v1 7 families (Templates 0-3 for train, Held-out Template 4 for eval)."""
    print("[Pool 2] Generating hard-v1 7 reasoning families (Templates 0-3 train, Template 4 eval)...", flush=True)
    if str(KEV_ROOT) not in sys.path:
        sys.path.insert(0, str(KEV_ROOT))

    from scripts import hard_v1_policy
    from scripts.hard_v1_common import Ctx
    from scripts.hard_v1_families import FAMILIES, labels

    seen = set(seen_hashes or set())
    seed_base = "hard-v1-minicpm-20260927"
    templates_map = {"development": (4,), "train": (0, 1, 2, 3)}
    shuffle_rng = random.Random(SEED + 202)

    def _gen_partition(split_name: str, per_fam: int) -> list[dict[str, Any]]:
        out_recs: list[dict[str, Any]] = []
        templates = templates_map[split_name]
        for fam_name, (gen_fn, _) in FAMILIES.items():
            ctx = Ctx(f"{seed_base}:{fam_name}:{split_name}")
            orig_uniform = ctx.rng.uniform
            if fam_name == "long_policy":
                ctx.rng.uniform = lambda a, b, _u=orig_uniform: _u(450, 920) if (a == 850 and b == 4900) else _u(a, b)

            made: list[dict[str, Any]] = []
            attempts = 0
            while len(made) < per_fam and attempts < 80 * per_fam + 200:
                attempts += 1
                t = (
                    templates[len(made) % len(templates)]
                    if fam_name != "ambiguous"
                    else templates[(len(made) // 2) % len(templates)]
                )
                try:
                    item = gen_fn(ctx, t)
                except ValueError:
                    item = None
                if item is None:
                    continue
                items_list = item if isinstance(item, list) else [item]
                if len(made) + len(items_list) > per_fam:
                    continue

                batch_ok = True
                candidate_recs: list[dict[str, Any]] = []
                batch_hashes: set[str] = set()
                for sub_idx, it in enumerate(items_list):
                    try:
                        facts = json.loads(json.dumps(it["facts"]))
                        qs = {qid: dict(q) for qid, q in it["questions"].items()}
                        for qid, lab in labels(fam_name, facts, qs).items():
                            qs[qid]["label"] = lab
                    except Exception:
                        batch_ok = False
                        break

                    # Clean redundant criteria where value equals key and shuffle choice option order
                    qs = shuffle_choice_questions(qs, shuffle_rng)
                    for qv in qs.values():
                        if qv.get("type") == "choice" and isinstance(qv.get("criteria"), dict):
                            qv["criteria"] = {
                                k: (None if v is None or _norm_key(str(v)) == _norm_key(k) else v)
                                for k, v in qv["criteria"].items()
                            }

                    add_df = fam_name in ("temporal_numeric", "long_policy", "ambiguous")
                    wire_rec = {
                        "id": f"hard_{fam_name}_{split_name}_{len(made) + sub_idx:04d}",
                        "source": f"hard_{fam_name}",
                        "variant": "clean",
                        "state": it["state"],
                        "questions": qs,
                        "_meta": {"family": fam_name, "template": f"{fam_name}/t{t}", "split": split_name},
                    }
                    try:
                        internal = to_internal_record(wire_rec, add_date_facts=add_df)
                    except Exception:
                        batch_ok = False
                        break
                    st_hash = hashlib.sha256(internal["state"].strip().lower().encode("utf-8")).hexdigest()
                    if st_hash in seen or st_hash in batch_hashes:
                        batch_ok = False
                        break
                    if not record_fits(internal, tok):
                        batch_ok = False
                        break
                    batch_hashes.add(st_hash)
                    wire_rec["state"] = internal["state"]
                    candidate_recs.append(wire_rec)

                if batch_ok and candidate_recs:
                    seen.update(batch_hashes)
                    made.extend(candidate_recs)

            print(f"  [hard-v1/{split_name}] {fam_name}: {len(made)} records ({attempts} attempts)", flush=True)
            out_recs.extend(made)
        return out_recs

    eval_hard = _gen_partition("development", eval_per_family)
    train_hard = _gen_partition("train", train_per_family)
    print(
        f"[Pool 2] Collected {len(train_hard)} train and {len(eval_hard)} held-out Template-4 eval records.",
        flush=True,
    )
    return train_hard, eval_hard


def build_pool3_night2(
    tok,
    v7_train_pool: list[dict[str, Any]],
    seen_hashes: set[str],
) -> list[dict[str, Any]]:
    """Generate night2 calibration & robustness records: dates, unknowable (soft uniform 1/K targets), and assertions."""
    print("[Pool 3] Generating night2 dates, unknowable (soft targets), and assertion records...", flush=True)
    if str(KEV_ROOT) not in sys.path:
        sys.path.insert(0, str(KEV_ROOT))

    from kev import contrastive
    from kev.transfer_v9 import unknowable

    out: list[dict[str, Any]] = []
    seed_n2 = "night2-minicpm-20260927"
    role_map = {
        "purchase": "purchase",
        "request": "return request",
        "claim": "warranty claim",
        "promised": "promised delivery",
        "delivered": "delivery",
    }

    for family in ("return_window", "warranty_claim", "shipping_delay"):
        rng = random.Random(f"{seed_n2}:dates:{family}")
        kept, attempts = 0, 0
        while kept < 100 and attempts < 4000:
            attempts += 1
            a, b = contrastive.FAMILIES[family](rng)
            if contrastive.check_pair(a, b):
                continue
            pair_id = f"{seed_n2}-{family}-{kept:04d}"
            order_seed = rng.getrandbits(64)
            style = rng.choice(["plain", "relational", "facts"])
            pair_recs = []
            for sibling, item in (("a", a), ("b", b)):
                if style == "relational":
                    facts_d = {}
                    for _, f_dict in item["sentences"]:
                        facts_d.update(f_dict)
                    dates_list = [(k, v) for k, v in facts_d.items() if hasattr(v, "toordinal")]
                    if len(dates_list) == 2:
                        (ka, da), (kb, db) = sorted(dates_list, key=lambda kv: kv[1])
                        n_days = (db - da).days
                        sent = f"The {role_map.get(kb, kb)} date was {n_days} day{'s' if n_days != 1 else ''} after the {role_map.get(ka, ka)} date."
                        item = {**item, "sentences": item["sentences"] + [(sent, {})]}
                req = contrastive.to_request(item, family, pair_id, sibling, random.Random(order_seed))
                internal = to_internal_record(req, add_date_facts=(style == "facts"))
                if record_fits(internal, tok):
                    pair_recs.append(
                        {
                            "id": f"night2_dates_{family}_{kept:04d}_{sibling}",
                            "source": "night2_dates",
                            "variant": "clean",
                            "state": internal["state"],
                            "questions": req["questions"],
                        }
                    )
            if len(pair_recs) == 2:
                out.extend(pair_recs)
                kept += 1

    unknowable_fams = [f for f in contrastive.FAMILIES if f not in ("deadline", "authorization")]
    saved_fams = contrastive.FAMILIES
    try:
        contrastive.FAMILIES = {k: v for k, v in saved_fams.items() if k in unknowable_fams}
        unk_raw = unknowable(10, f"{seed_n2}:unknowable")
    finally:
        contrastive.FAMILIES = saved_fams

    for idx_u, r in enumerate(unk_raw):
        src_u = r.get("_meta", {}).get("source", "")
        if src_u == "unknowable":
            for q in r["questions"].values():
                if q["type"] == "choice":
                    n_k = len(q.get("criteria") or {})
                elif q["type"] == "noul":
                    n_k = 2
                else:
                    n_k = len(q.get("criteria") or SCORE_OPTIONS)
                if n_k > 0:
                    q["target_probs"] = [round(1.0 / n_k, 6)] * n_k
            src_tag = "night2_unknowable"
        else:
            src_tag = "night2_unknowable_control"
        internal = to_internal_record(r)
        if record_fits(internal, tok):
            out.append(
                {
                    "id": f"{src_tag}_{idx_u:04d}",
                    "source": src_tag,
                    "variant": "clean",
                    "state": internal["state"],
                    "questions": r["questions"],
                }
            )

    templates_assert = [
        "This text is about {x}.",
        "The right category for this is {x}.",
        "This should be filed under {x}.",
        "The topic here is {x}.",
    ]
    rng_a = random.Random(f"{seed_n2}:assertion")
    pool_a = [
        r
        for r in v7_train_pool
        if r["source"] in ("l1_agnews", "l1_dbpedia14", "l1_trec", "l1_banking77", "l1_yelp", "l1_imdb", "l1_boolq")
    ]
    rng_a.shuffle(pool_a)
    n_assert = 0
    for r in pool_a:
        if n_assert >= 440:
            break
        qs_new: dict[str, Any] = {}
        for qid, q in r["questions"].items():
            if q["type"] == "choice" and isinstance(q.get("criteria"), dict) and len(q["criteria"]) >= 3:
                truth = q["label"]
                want_true = rng_a.random() < 0.5
                key = truth if want_true else rng_a.choice([k for k in q["criteria"] if k != truth])
                desc = q["criteria"].get(key) or str(key).replace("_", " ")
                if not isinstance(desc, str):
                    continue
                desc = desc.split("(Not for:")[0].strip().rstrip("?. ")
                qs_new[f"{qid}_is"] = {
                    "type": "noul",
                    "instructions": rng_a.choice(templates_assert).format(x=desc),
                    "label": bool(key == truth),
                }
        if qs_new:
            cand_a = {
                "id": f"night2_assertion_{n_assert:04d}",
                "source": "night2_assertion",
                "variant": "clean",
                "state": r["state"],
                "questions": qs_new,
            }
            if record_fits(to_internal_record(cand_a), tok):
                out.append(cand_a)
                n_assert += 1

    print(f"[Pool 3] Collected {len(out)} night2 calibration & robustness records.", flush=True)
    return out


_SCORE_LOW_MARKERS = (
    "no ",
    "none",
    "low",
    "minimal",
    "minor",
    "calm",
    "routine",
    "normal",
    "standard",
    "trivial",
    "clear",
    "polite",
    "safe",
    "negligible",
    "vague",
    "incomplete",
    "basic",
    "mild",
    "informational",
)
_SCORE_HIGH_MARKERS = (
    "critical",
    "severe",
    "extreme",
    "catastrophic",
    "high",
    "urgent",
    "immediate",
    "hostile",
    "furious",
    "emergency",
    "major",
    "destructive",
    "complete",
    "full",
    "comprehensive",
    "legal action",
    "outage",
)
_FIXED_LEVEL_WORD_RE = re.compile(r"^\s*(low|medium|moderate|high|critical|none|mild|severe|extreme)\b", re.I)


def _clean_and_orient_score_question(
    q: dict[str, Any],
    state_lower: str,
) -> tuple[list[str], int, bool] | None:
    """Validate a score question from custom_questions.jsonl:
    - Require teacher_p >= 0.80
    - Reject questions referencing missing backtick paths
    - Auto-correct reversed scales (Critical -> None flipped to None -> Critical)
    - Reject non-monotone 'Goldilocks' choice-in-disguise scales where ans == 2
    Returns (cleaned_criteria_list, oriented_ans, is_monotone).
    """
    if float(q.get("teacher_p", 0.0)) < 0.80:
        return None
    ins = str(q.get("instructions") or "").strip()
    if len(ins) < 8:
        return None
    for path in re.findall(r"`([^`]+)`", ins):
        parts = [p for p in re.split(r"[\.\[\]]+", path.lower()) if p and not p.isdigit() and p != "state"]
        if parts and not any(p in state_lower for p in parts):
            return None

    crit = q.get("criteria")
    ans = q.get("answer")
    if (
        not isinstance(crit, list)
        or not (3 <= len(crit) <= 5)
        or not isinstance(ans, int)
        or isinstance(ans, bool)
        or not (0 <= ans < len(crit))
    ):
        return None

    crit_strs = [str(x).strip() for x in crit]
    c0_low = crit_strs[0].lower()
    clast_low = crit_strs[-1].lower()

    is_reversed = any(w in c0_low for w in ("critical", "severe", "extreme", "urgent", "immediate", "catastrophic")) and any(
        w in clast_low for w in ("none", "low", "no ", "routine", "minimal", "informational")
    )
    if is_reversed:
        crit_strs = list(reversed(crit_strs))
        ans = len(crit_strs) - 1 - ans
        c0_low = crit_strs[0].lower()
        clast_low = crit_strs[-1].lower()

    is_monotone = any(w in c0_low for w in _SCORE_LOW_MARKERS) and any(w in clast_low for w in _SCORE_HIGH_MARKERS)
    if not is_monotone and ans == 2 and float(q.get("teacher_p", 0.0)) < 0.94:
        return None
    return crit_strs, ans, is_monotone


def build_pool4_decider_teacher(
    tok,
    seen_hashes: set[str],
    custom_target: int = 2400,
    situations_target: int = 1000,
    routing_target: int = 800,
) -> list[dict[str, Any]]:
    """Load strictly filtered, level-balanced, and option-shuffled multi-branch records from decider/teacher_data."""
    print("[Pool 4] Loading strictly verified & level-balanced records from decider/teacher_data...", flush=True)
    rng = random.Random(SEED + 404)
    out: list[dict[str, Any]] = []

    # 1. custom_questions.jsonl with strict confidence gates, score level balancing, and choice option shuffling
    cq_path = DECIDER_TEACHER_DIR / "custom_questions.jsonl"
    if cq_path.exists():
        cq_lines = [json.loads(line) for line in cq_path.open("r", encoding="utf-8") if line.strip()]
        rng.shuffle(cq_lines)

        # Track score (K, level) counts so level=2 never dominates
        score_level_counts: dict[tuple[int, int], int] = Counter()
        score_level2_cap = {3: 90, 4: 420, 5: 120}
        cq_kept = 0

        for idx, row in enumerate(cq_lines):
            if cq_kept >= custom_target:
                break
            raw_qs = row.get("questions") or []
            state_lower = json.dumps(row.get("state", ""), ensure_ascii=False).lower()

            valid_score_qs: list[dict[str, Any]] = []
            valid_choice_qs: list[dict[str, Any]] = []
            valid_noul_qs: list[dict[str, Any]] = []

            for q in raw_qs:
                qt = q.get("type")
                ins = str(q.get("instructions") or "").strip()
                # Check backtick paths exist in state
                bt_bad = False
                for path in re.findall(r"`([^`]+)`", ins):
                    parts = [p for p in re.split(r"[\.\[\]]+", path.lower()) if p and not p.isdigit() and p != "state"]
                    if parts and not any(p in state_lower for p in parts):
                        bt_bad = True
                        break
                if bt_bad:
                    continue

                if qt == "score":
                    cleaned_sc = _clean_and_orient_score_question(q, state_lower)
                    if cleaned_sc is None:
                        continue
                    crit_strs, ans, is_mono = cleaned_sc
                    k_len = len(crit_strs)
                    # If ans == 2 is already at cap, try contiguous 3-level sub-window [L1, L2, L3] -> ans=1 (if no fixed level prefixes)
                    if ans == 2 and score_level_counts[(k_len, 2)] >= score_level2_cap.get(k_len, 200):
                        if (
                            k_len == 4
                            and is_mono
                            and not any(_FIXED_LEVEL_WORD_RE.match(x) for x in crit_strs)
                            and score_level_counts[(3, 1)] < 260
                        ):
                            crit_strs = crit_strs[1:4]
                            ans = 1
                            k_len = 3
                        else:
                            continue
                    score_level_counts[(k_len, ans)] += 1
                    valid_score_qs.append(
                        {
                            "type": "score",
                            "instructions": ins,
                            "criteria": crit_strs,
                            "label": int(ans),
                        }
                    )
                elif qt == "choice":
                    if float(q.get("teacher_p", 0.0)) < 0.85:
                        continue
                    crit = q.get("criteria")
                    ans = q.get("answer")
                    if not isinstance(crit, dict) or not (2 <= len(crit) <= 10) or ans not in crit:
                        continue
                    cleaned_crit = {
                        str(k): (None if v in (None, "") or _norm_key(str(v)) == _norm_key(k) else v)
                        for k, v in crit.items()
                    }
                    keys = list(cleaned_crit.keys())
                    rng.shuffle(keys)
                    valid_choice_qs.append(
                        {
                            "type": "choice",
                            "instructions": ins,
                            "criteria": {k: cleaned_crit[k] for k in keys},
                            "label": str(ans),
                        }
                    )
                elif qt == "noul":
                    if float(q.get("teacher_p", 0.0)) < 0.85:
                        continue
                    ans = q.get("answer")
                    if not isinstance(ans, bool):
                        continue
                    valid_noul_qs.append(
                        {
                            "type": "noul",
                            "instructions": ins,
                            "criteria": q.get("criteria") if isinstance(q.get("criteria"), dict) else None,
                            "label": ans,
                        }
                    )

            selected_qs = valid_score_qs[:2] + valid_choice_qs[:2] + valid_noul_qs[:2]
            if not selected_qs:
                continue

            q_dict = {f"q_{qi}_{q['type']}": q for qi, q in enumerate(selected_qs)}
            wire = {
                "id": f"decider_cq_{idx:05d}",
                "source": "decider_custom_multi",
                "variant": "clean",
                "state": row["state"],
                "questions": q_dict,
            }
            try:
                internal = to_internal_record(wire, add_date_facts=True)
            except Exception:
                continue
            st_hash = hashlib.sha256(internal["state"].strip().lower().encode("utf-8")).hexdigest()
            if st_hash in seen_hashes:
                continue
            if record_fits(internal, tok):
                seen_hashes.add(st_hash)
                wire["state"] = internal["state"]
                out.append(wire)
                cq_kept += 1

        print(f"  [Pool 4] custom_questions kept={cq_kept}, score (K, level) dist={dict(sorted(score_level_counts.items()))}", flush=True)

    # 2. situations.jsonl (situational action choice with shuffled options + immediate danger noul)
    sit_path = DECIDER_TEACHER_DIR / "situations.jsonl"
    if sit_path.exists():
        sit_lines = [json.loads(line) for line in sit_path.open("r", encoding="utf-8") if line.strip()]
        rng.shuffle(sit_lines)
        sit_kept = 0
        for idx, row in enumerate(sit_lines):
            if sit_kept >= situations_target:
                break
            opts = row.get("options") or []
            ans = row.get("answer")
            if not isinstance(opts, list) or len(opts) < 3 or not isinstance(ans, int) or not (0 <= ans < len(opts)):
                continue
            # Shuffle option order before assigning action_1..action_k keys so gold index is strictly uniform
            perm = list(range(len(opts)))
            rng.shuffle(perm)
            shuffled_opts = [str(opts[i]) for i in perm]
            new_ans = perm.index(ans)
            crit = {f"action_{i + 1}": o for i, o in enumerate(shuffled_opts)}
            q_dict: dict[str, Any] = {
                "best_action": {
                    "type": "choice",
                    "instructions": str(row.get("question") or "What is the best action to take right now?"),
                    "criteria": crit,
                    "label": f"action_{new_ans + 1}",
                }
            }
            if isinstance(row.get("danger"), bool):
                q_dict["immediate_danger"] = {
                    "type": "noul",
                    "instructions": "Is the agent or subject in immediate danger in this situation?",
                    "label": bool(row["danger"]),
                }
            wire = {
                "id": f"decider_sit_{idx:05d}",
                "source": "decider_situations",
                "variant": "clean",
                "state": str(row.get("situation") or ""),
                "questions": q_dict,
            }
            try:
                internal = to_internal_record(wire)
            except Exception:
                continue
            st_hash = hashlib.sha256(internal["state"].strip().lower().encode("utf-8")).hexdigest()
            if st_hash in seen_hashes:
                continue
            if record_fits(internal, tok):
                seen_hashes.add(st_hash)
                out.append(wire)
                sit_kept += 1

    # 3. routing_messages.jsonl, routing_terse.jsonl, and commands.jsonl (both risk choice and outside noul)
    for fn, src_name, cap in (
        ("routing_messages.jsonl", "decider_routing", routing_target // 3),
        ("routing_terse.jsonl", "decider_routing_terse", routing_target // 3),
        ("commands.jsonl", "decider_commands", routing_target - 2 * (routing_target // 3)),
    ):
        fpath = DECIDER_TEACHER_DIR / fn
        if not fpath.exists():
            continue
        lines = [json.loads(line) for line in fpath.open("r", encoding="utf-8") if line.strip()]
        rng.shuffle(lines)
        local_kept = 0
        for idx, row in enumerate(lines):
            if local_kept >= cap:
                break
            qs = row.get("questions") or []
            if not qs:
                continue
            q0 = qs[0]
            # Keep teacher-verified or generic-vs-catchall cases
            is_ok = bool(q0.get("teacher_ok", False)) or (
                row.get("group") == "generic" and q0.get("teacher_pred") == row.get("catchall")
            )
            if not is_ok:
                continue
            crit = q0.get("criteria")
            ans = q0.get("answer")
            if not isinstance(crit, dict) or ans not in crit:
                continue
            cleaned_crit = {
                str(k): (None if v in (None, "", "null") or _norm_key(str(v)) == _norm_key(k) else v)
                for k, v in crit.items()
            }
            keys = list(cleaned_crit.keys())
            rng.shuffle(keys)
            q_dict = {
                "decision": {
                    "type": "choice",
                    "instructions": str(q0.get("instructions") or ""),
                    "criteria": {k: cleaned_crit[k] for k in keys},
                    "label": str(ans),
                }
            }
            # For commands.jsonl, also include the 2nd question ("outside" noul)
            if src_name == "decider_commands" and len(qs) >= 2 and qs[1].get("type") == "noul":
                q1 = qs[1]
                if isinstance(q1.get("answer"), bool):
                    q_dict["outside_scope"] = {
                        "type": "noul",
                        "instructions": str(q1.get("instructions") or ""),
                        "criteria": None,
                        "label": bool(q1["answer"]),
                    }
            wire = {
                "id": f"{src_name}_{idx:05d}",
                "source": src_name,
                "variant": "clean",
                "state": str(row.get("state") or ""),
                "questions": q_dict,
            }
            try:
                internal = to_internal_record(wire)
            except Exception:
                continue
            st_hash = hashlib.sha256(internal["state"].strip().lower().encode("utf-8")).hexdigest()
            if st_hash in seen_hashes:
                continue
            if record_fits(internal, tok):
                seen_hashes.add(st_hash)
                out.append(wire)
                local_kept += 1

    print(f"[Pool 4] Collected {len(out)} verified teacher records.", flush=True)
    return out


def _generate_balanced_ordinal_rubrics(
    rng: random.Random,
    target_train: int = 1200,
    target_eval: int = 60,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate strictly 1:1:...:1 level-balanced ordinal score + multi-condition noul verification records
    across 8 realistic domains (frustration K=3, reading_complexity K=3, lead_qualification K=4,
    formality_level K=4, claim_priority K=4, symptom_triage K=4, incident_severity K=5, churn_risk K=5).
    Every item pairs an exact-level ordinal score question with a multi-condition verification noul question.
    """
    names = ["Alex Rivera", "Priya Patel", "Marcus Vance", "Elena Rostova", "Kenji Sato", "Sarah Jenkins", "Omar Haddad", "Hana Novak", "Devon Brooks", "Mei Lin", "Lucas Silva", "Nadia Okafor"]
    orgs = ["Apex Cloud", "Northwind Health", "Vanguard Logistics", "Meridian Bank", "Solstice Retail", "Kestrel Energy", "Cobalt SaaS", "Horizon Telecom"]
    channels = ["Ticket", "LiveChat", "Email", "Portal", "IncidentLog", "Review", "IntakeForm"]

    def _gen_frustration(idx: int, level: int) -> dict[str, Any]:
        person = rng.choice(names)
        org = rng.choice(orgs)
        ticket_id = rng.randint(10000, 99999)
        criteria = [
            "Calm and polite: Neutral inquiry or routine request with no annoyance.",
            "Mildly frustrated: Expresses impatience or disappointment about a delay or hiccup, but remains civil.",
            "Angry or hostile: Uses emphatic complaints, demands escalation, or threatens cancellation/bad reviews.",
        ]
        if level == 0:
            body = rng.choice([
                f"Hi {org} team, could you let me know when order #{ticket_id} is scheduled to ship? Thanks, {person}.",
                f"Hello, I would like to update the billing address on account #{ticket_id} before next month's invoice. Best regards, {person}.",
                f"Good morning, just checking whether the PDF export feature is included in the Starter tier for #{ticket_id}. Appreciate your help!",
            ])
            escalate = False
        elif level == 1:
            body = rng.choice([
                f"Hi, I submitted ticket #{ticket_id} three days ago and still haven't heard back. It's pretty frustrating waiting this long for a simple reset.",
                f"This is the second time the sync has stalled on #{ticket_id} this week. I'm disappointed because it's slowing down our afternoon reporting.",
                f"Still waiting on the callback promised yesterday for #{ticket_id}. Really hoping someone can look at this today so we aren't blocked.",
            ])
            escalate = False
        else:
            body = rng.choice([
                f"Unbelievable! Ticket #{ticket_id} has been broken for 48 hours and nobody replies! Connect me to a supervisor IMMEDIATELY or we are cancelling our {org} contract today!",
                f"This service is a complete joke. You charged #{ticket_id} twice and locked our workspace! Fix this right now or I'm filing a chargeback!",
                f"Third outage this month on #{ticket_id}!! I am done with {org}'s excuses—refund our annual fee immediately!",
            ])
            escalate = True
        return {
            "state": f"[{rng.choice(channels)} #{ticket_id} | Customer: {person} | Account: {org}]\n{body}",
            "questions": {
                "frustration_score": {
                    "type": "score",
                    "instructions": "Rate the customer's frustration level based on the message.",
                    "criteria": criteria,
                    "label": level,
                },
                "requires_escalation": {
                    "type": "noul",
                    "instructions": "Does the policy 'Escalate only if the customer explicitly threatens cancellation, a chargeback, or demands a supervisor' apply here?",
                    "criteria": None,
                    "label": escalate,
                },
            },
        }

    def _gen_incident_severity(idx: int, level: int) -> dict[str, Any]:
        svc = rng.choice(["auth-gateway", "checkout-api", "billing-worker", "search-indexer", "notification-hub", "analytics-etl"])
        region = rng.choice(["us-east-1", "eu-central-1", "ap-northeast-1", "us-west-2"])
        inc_id = f"INC-{rng.randint(2000, 9999)}"
        criteria = [
            "Cosmetic or informational: UI typo, log warning, or doc issue with zero functional impact.",
            "Minor degradation: Non-critical background delay or single-user glitch with an easy workaround.",
            "Moderate impact: Partial feature failure affecting a subset of users (<15%) with degraded performance.",
            "Major outage: Core production workflow failing for many users (>25%) with no immediate workaround.",
            "Critical catastrophe: Complete production outage, active data loss, or confirmed security breach.",
        ]
        if level == 0:
            state = f"{inc_id} | Service: {svc} ({region}) | Error rate: 0.0% | Detail: Deprecated config key warning logged during nightly lint check; tooltip alignment off by 2px on settings page."
            page_oncall = False
        elif level == 1:
            state = f"{inc_id} | Service: {svc} ({region}) | Error rate: 0.2% | Detail: CSV export background job delayed by 4 minutes for 2 internal users; manual refresh succeeds immediately."
            page_oncall = False
        elif level == 2:
            state = f"{inc_id} | Service: {svc} ({region}) | Error rate: 8.5% | Detail: P95 latency spiked to 2.4s in {region}; ~11% of search queries timing out, retry succeeds on second attempt."
            page_oncall = False
        elif level == 3:
            state = f"{inc_id} | Service: {svc} ({region}) | Error rate: 46.0% | Detail: Primary database connection pool exhausted in {region}; 46% of customer requests returning HTTP 503 with no workaround."
            page_oncall = True
        else:
            state = f"{inc_id} | Service: {svc} ({region}) | Error rate: 100.0% | Detail: Complete multi-region outage across all nodes PLUS unencrypted customer token table exposed in public dump."
            page_oncall = True
        return {
            "state": state,
            "questions": {
                "severity_level": {
                    "type": "score",
                    "instructions": "Classify the operational incident severity from lowest (0) to highest (4).",
                    "criteria": criteria,
                    "label": level,
                },
                "page_sev1": {
                    "type": "noul",
                    "instructions": "According to the rule 'Page the P0/P1 duty commander only when error rate exceeds 25% or a security/data breach occurs', should the duty commander be paged?",
                    "criteria": None,
                    "label": page_oncall,
                },
            },
        }

    def _gen_lead_qualification(idx: int, level: int) -> dict[str, Any]:
        person = rng.choice(names)
        org = rng.choice(orgs)
        criteria = [
            "Unqualified / spam: Student project, job seeker, competitor, or no relevant need.",
            "Early exploratory: Small team or individual researching options with no budget or timeline.",
            "Warm mid-market prospect: Defined use case and team size, evaluating within the next quarter.",
            "High-priority enterprise buyer: Decision-maker with approved budget, large seat count, and immediate timeline.",
        ]
        if level == 0:
            state = f"Lead: {person} | Email: {person.split()[0].lower()}@university.edu | Note: Undergraduate student asking for a free dataset sample for a class homework assignment. Seats: 1. Budget: $0."
            fast_track = False
        elif level == 1:
            state = f"Lead: {person} ({org}) | Role: Junior Analyst | Note: Browsing documentation to see how the API works; no approved budget yet, maybe revisiting next year. Seats: 3."
            fast_track = False
        elif level == 2:
            state = f"Lead: {person} ({org}) | Role: Engineering Manager | Note: Comparing 3 vendors for Q3 team rollout; needs SSO and audit logs for 45 seats, budget review scheduled next month."
            fast_track = False
        else:
            state = f"Lead: {person} ({org}) | Role: VP of Infrastructure | Note: Approved $180k budget to replace legacy vendor within 14 days; needs enterprise contract for 650 seats immediately."
            fast_track = True
        return {
            "state": state,
            "questions": {
                "lead_score": {
                    "type": "score",
                    "instructions": "Rate the sales lead qualification level based on the intake record.",
                    "criteria": criteria,
                    "label": level,
                },
                "enterprise_fast_track": {
                    "type": "noul",
                    "instructions": "Does this lead meet all requirements for Enterprise Fast-Track (VP/Director decision-maker, at least 200 seats, and approved budget)?",
                    "criteria": None,
                    "label": fast_track,
                },
            },
        }

    def _gen_triage_urgency(idx: int, level: int) -> dict[str, Any]:
        person = rng.choice(names)
        age = rng.randint(22, 74)
        criteria = [
            "Routine / non-urgent: Prescription refill, wellness check, or mild chronic question with no acute symptoms.",
            "Low urgency (within 48-72h): Mild localized discomfort or minor rash with stable vitals and no fever.",
            "Urgent same-day evaluation: Moderate pain, persistent high fever, or worsening symptoms needing prompt care.",
            "Immediate emergency (ER / 911): Severe chest pain, difficulty breathing, stroke signs, or uncontrolled bleeding.",
        ]
        if level == 0:
            state = f"Patient: {person} ({age}y) | Vitals: BP 118/76, HR 68, Temp 98.4°F | Intake: Requesting routine annual bloodwork lab slip and refill of daily vitamin D."
            er_redirect = False
        elif level == 1:
            state = f"Patient: {person} ({age}y) | Vitals: BP 122/78, HR 72, Temp 98.7°F | Intake: Mild itchy rash on left forearm for 2 days after gardening; no swelling, breathing normal."
            er_redirect = False
        elif level == 2:
            state = f"Patient: {person} ({age}y) | Vitals: BP 134/86, HR 96, Temp 102.6°F | Intake: Ear pain and fever lasting 36 hours, unable to sleep; able to drink fluids."
            er_redirect = False
        else:
            state = f"Patient: {person} ({age}y) | Vitals: BP 168/104, HR 118, SpO2 88% | Intake: Sudden crushing substernal chest pressure radiating to left jaw and shortness of breath starting 20 minutes ago."
            er_redirect = True
        return {
            "state": state,
            "questions": {
                "triage_level": {
                    "type": "score",
                    "instructions": "Assess the clinical triage urgency level from lowest (0) to highest (3).",
                    "criteria": criteria,
                    "label": level,
                },
                "immediate_er": {
                    "type": "noul",
                    "instructions": "Must the patient be directed to emergency services immediately due to chest pain, hypoxia (SpO2 < 92%), or acute respiratory distress?",
                    "criteria": None,
                    "label": er_redirect,
                },
            },
        }

    def _gen_review_sentiment(idx: int, level: int) -> dict[str, Any]:
        prod = rng.choice(["Noise-Cancelling Headphones", "Espresso Machine", "Standing Desk", "Ergonomic Mesh Chair", "4K Portable Monitor", "Smart Thermostat"])
        criteria = [
            "1 star (Very negative): Product arrived broken, unusable, or caused a major complaint.",
            "2 stars (Mostly negative): Disappointing quality or missing key features despite a minor positive.",
            "3 stars (Mixed / neutral): Works adequately for basic use, with clear pros and cons balancing out.",
            "4 stars (Mostly positive): Very satisfied overall, with only a small nitpick or minor suggestion.",
            "5 stars (Glowing praise): Exceeded expectations in every way; enthusiastic recommendation.",
        ]
        if level == 0:
            state = f"Product: {prod} | Review: Dead on arrival. The power unit sparked the first time I plugged it in and customer service refused to replace it. Total waste of money."
            verified_recommend = False
        elif level == 1:
            state = f"Product: {prod} | Review: The exterior finish looks nice, but the battery dies in 40 minutes and the hinge is already wobbling after four days. Would not buy again."
            verified_recommend = False
        elif level == 2:
            state = f"Product: {prod} | Review: It does the basic job fine for the price, though the setup manual is confusing and the plastic buttons feel a bit cheap. Average overall."
            verified_recommend = False
        elif level == 3:
            state = f"Product: {prod} | Review: Really happy with the build quality and daily performance! Only docking one point because the included cable is a little short."
            verified_recommend = True
        else:
            state = f"Product: {prod} | Review: Hands down the best purchase I've made all year! Flawless performance, whisper quiet, and setup took two minutes. Highly recommend to everyone!"
            verified_recommend = True
        return {
            "state": state,
            "questions": {
                "sentiment_rating": {
                    "type": "score",
                    "instructions": "Rate the reviewer's sentiment on the 5-level scale from most negative (0) to most positive (4).",
                    "criteria": criteria,
                    "label": level,
                },
                "positive_endorsement": {
                    "type": "noul",
                    "instructions": "Is the reviewer overall satisfied and endorsing the product?",
                    "criteria": None,
                    "label": verified_recommend,
                },
            },
        }

    def _gen_response_adequacy(idx: int, level: int) -> dict[str, Any]:
        order_id = f"ORD-{rng.randint(1000, 9999)}"
        fee = rng.choice(["$15", "$25", "$40"])
        days = rng.choice([14, 30, 45])
        criteria = [
            "Completely inadequate: Ignores the user's question or gives contradicted/fabricated facts.",
            "Partially helpful but incomplete: Answers only one part of the request and omits required details.",
            "Mostly complete with minor omission: Covers the main questions accurately but misses one secondary detail.",
            "Fully adequate and accurate: Directly and accurately addresses every part of the user's request.",
        ]
        req = f"User Request: Required facts to include for {order_id}: (1) return window ({days} days), (2) restocking fee ({fee}), and (3) prepaid label link."
        if level == 0:
            resp = f"Agent Draft: Thanks for contacting us about {order_id}! Our physical store in Denver is open 9am-5pm Monday through Friday."
            all_met = False
        elif level == 1:
            resp = f"Agent Draft: For {order_id}, you have {days} days to return your item. Let us know if you need anything else!"
            all_met = False
        elif level == 2:
            resp = f"Agent Draft: For {order_id}, our return window is {days} days and a {fee} restocking fee applies. (No link attached)."
            all_met = False
        else:
            resp = f"Agent Draft: For {order_id}, you can return the item within {days} days. Please note a {fee} restocking fee applies, and you can download your prepaid label at returns.example.com/{order_id}."
            all_met = True
        return {
            "state": f"{req}\n{resp}",
            "questions": {
                "adequacy_score": {
                    "type": "score",
                    "instructions": "Evaluate how completely and accurately the Agent Draft addresses the User Request.",
                    "criteria": criteria,
                    "label": level,
                },
                "meets_all_three": {
                    "type": "noul",
                    "instructions": "Does the Agent Draft explicitly include all three required items (return window days, restocking fee amount, and prepaid label link)?",
                    "criteria": None,
                    "label": all_met,
                },
            },
        }

    generators = [
        ("rubric_frustration", 3, _gen_frustration),
        ("rubric_severity", 5, _gen_incident_severity),
        ("rubric_lead", 4, _gen_lead_qualification),
        ("rubric_triage", 4, _gen_triage_urgency),
        ("rubric_sentiment", 5, _gen_review_sentiment),
        ("rubric_adequacy", 4, _gen_response_adequacy),
    ]

    train_recs: list[dict[str, Any]] = []
    eval_recs: list[dict[str, Any]] = []

    per_gen_train = target_train // len(generators)
    per_gen_eval = target_eval // len(generators)

    for fam_tag, k_levels, gen_fn in generators:
        for split_tag, count_target, dest in (("eval", per_gen_eval, eval_recs), ("train", per_gen_train, train_recs)):
            for i in range(count_target):
                lvl = i % k_levels
                item = gen_fn(i, lvl)
                # Ensure uniqueness by appending a deterministic timestamp/log reference
                ref_tag = f"Ref: {fam_tag.upper()}-{split_tag[0].upper()}{i:04d}-L{lvl}"
                dest.append(
                    {
                        "id": f"pool5_{fam_tag}_{split_tag}_{i:04d}",
                        "source": f"pool5_{fam_tag}",
                        "variant": "clean",
                        "state": f"{item['state']} ({ref_tag})",
                        "questions": item["questions"],
                    }
                )

    return train_recs, eval_recs


def build_pool5_rules_and_rubrics(
    tok,
    seen_hashes: set[str],
    rules_train_target: int = 1800,
    rules_eval_target: int = 60,
    rubrics_train_target: int = 1200,
    rubrics_eval_target: int = 60,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build Pool 5:
    1. Programmatic Rule Twins from decider/decider/data/rules.py (base + rule_twin packed + state_twin)
    2. Strictly 1:1:...:1 level-balanced 3/4/5-level ordinal score & multi-condition noul rubrics
    """
    print("[Pool 5] Generating programmatic rule twins (decider.data.rules) & balanced ordinal rubrics...", flush=True)
    if str(DECIDER_ROOT) not in sys.path:
        sys.path.insert(0, str(DECIDER_ROOT))

    from decider.data import rules as R

    rng = random.Random(SEED + 505)
    train_out: list[dict[str, Any]] = []
    eval_out: list[dict[str, Any]] = []

    def _example_to_qdict(q_obj, q_key: str) -> dict[str, Any]:
        opts = list(q_obj.options)
        gold_idx = int(q_obj.gold)
        if opts == ["no", "yes"]:
            return {
                q_key: {
                    "type": "noul",
                    "instructions": str(q_obj.text),
                    "criteria": None,
                    "label": bool(gold_idx == 1),
                }
            }
        # Choice question: shuffle option order deterministically
        perm = list(range(len(opts)))
        rng.shuffle(perm)
        shuffled_opts = [str(opts[i]) for i in perm]
        new_gold_opt = shuffled_opts[perm.index(gold_idx)]
        return {
            q_key: {
                "type": "choice",
                "instructions": str(q_obj.text),
                "criteria": {opt_s: None for opt_s in shuffled_opts},
                "label": new_gold_opt,
            }
        }

    # Generate raw rule examples from decider.data.rules
    raw_rules = R.build(rules_train_target * 2 + rules_eval_target * 2, seed=SEED + 55)
    # Group by identical context so base + rule_twin on the same context are packed into a multi-question record!
    by_ctx: dict[str, list[Any]] = defaultdict(list)
    for ex in raw_rules:
        if ex.task == "rules_form":
            continue
        by_ctx[ex.context].append(ex)

    grouped_items = list(by_ctx.items())
    rng.shuffle(grouped_items)

    for ctx_str, ex_list in grouped_items:
        is_eval = len(eval_out) < rules_eval_target
        if not is_eval and len(train_out) >= rules_train_target:
            break
        q_merged: dict[str, Any] = {}
        task_name = ex_list[0].task
        for qi, ex in enumerate(ex_list[:2]):
            q_merged.update(_example_to_qdict(ex.qs[0], f"q_{qi}"))
        if not q_merged:
            continue
        split_name = "eval" if is_eval else "train"
        rec_idx = len(eval_out) if is_eval else len(train_out)
        wire = {
            "id": f"pool5_{task_name}_{split_name}_{rec_idx:05d}",
            "source": f"pool5_{task_name}",
            "variant": "clean",
            "state": ctx_str,
            "questions": q_merged,
        }
        try:
            internal = to_internal_record(wire)
        except Exception:
            continue
        st_hash = hashlib.sha256(internal["state"].strip().lower().encode("utf-8")).hexdigest()
        if st_hash in seen_hashes:
            continue
        if record_fits(internal, tok):
            seen_hashes.add(st_hash)
            if is_eval:
                eval_out.append(wire)
            else:
                train_out.append(wire)

    rub_train, rub_eval = _generate_balanced_ordinal_rubrics(
        rng, target_train=rubrics_train_target, target_eval=rubrics_eval_target
    )
    for r in rub_eval:
        internal = to_internal_record(r)
        st_hash = hashlib.sha256(internal["state"].strip().lower().encode("utf-8")).hexdigest()
        if st_hash not in seen_hashes and record_fits(internal, tok):
            seen_hashes.add(st_hash)
            eval_out.append(r)

    for r in rub_train:
        internal = to_internal_record(r)
        st_hash = hashlib.sha256(internal["state"].strip().lower().encode("utf-8")).hexdigest()
        if st_hash not in seen_hashes and record_fits(internal, tok):
            seen_hashes.add(st_hash)
            train_out.append(r)

    print(
        f"[Pool 5] Collected {len(train_out)} train records and {len(eval_out)} dev records (rules + balanced ordinal rubrics).",
        flush=True,
    )
    return train_out, eval_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE_MODEL)
    ap.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parent / "data" / "stage2"),
    )
    ap.add_argument("--v7_train", type=int, default=6500)
    ap.add_argument("--hard_train_per_family", type=int, default=500)
    ap.add_argument("--hard_eval_per_family", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer {args.base}...", flush=True)
    tok = load_tokenizer(args.base)

    pool1_train, val_10source, seen_hashes = build_pool1_decision_v7(
        tok, train_target=args.v7_train, per_source_eval=30
    )
    pool2_train, val_hard_v1 = build_pool2_hard_v1(
        tok,
        train_per_family=args.hard_train_per_family,
        eval_per_family=args.hard_eval_per_family,
        seen_hashes=seen_hashes,
    )
    pool3_train = build_pool3_night2(tok, pool1_train, seen_hashes)
    pool4_train = build_pool4_decider_teacher(tok, seen_hashes)
    pool5_train, val_pool5 = build_pool5_rules_and_rubrics(tok, seen_hashes)

    all_train = pool1_train + pool2_train + pool3_train + pool4_train + pool5_train
    rng = random.Random(SEED)
    rng.shuffle(all_train)

    dev_combined = val_10source + val_hard_v1 + val_pool5

    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"
    val_10source_path = out_dir / "val_10source.jsonl"
    val_hard_path = out_dir / "val_hard_v1.jsonl"
    stats_path = out_dir / "stats.json"

    for path, records in (
        (train_path, all_train),
        (dev_path, dev_combined),
        (val_10source_path, val_10source),
        (val_hard_path, val_hard_v1),
    ):
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    src_counts = Counter(r["source"] for r in all_train)
    qtype_counts = Counter()
    for r in all_train:
        for q in r["questions"].values():
            qtype_counts[q["type"]] += 1

    dev_qtype_counts = Counter()
    for r in dev_combined:
        for q in r["questions"].values():
            dev_qtype_counts[q["type"]] += 1

    stats = {
        "schema_version": SCHEMA_VERSION,
        "base_model": args.base,
        "train_total_records": len(all_train),
        "train_total_questions": sum(qtype_counts.values()),
        "train_question_types": dict(qtype_counts),
        "pool_counts": {
            "pool1_decision_v7": len(pool1_train),
            "pool2_hard_v1": len(pool2_train),
            "pool3_night2": len(pool3_train),
            "pool4_decider_teacher": len(pool4_train),
            "pool5_rules_and_rubrics": len(pool5_train),
        },
        "dev_total_records": len(dev_combined),
        "dev_total_questions": sum(dev_qtype_counts.values()),
        "dev_question_types": dict(dev_qtype_counts),
        "val_10source_records": len(val_10source),
        "val_hard_v1_records": len(val_hard_v1),
        "val_pool5_records": len(val_pool5),
        "max_state": MAX_STATE,
        "max_branch": MAX_BRANCH,
        "max_packed": MAX_PACKED,
        "train_sources": dict(sorted(src_counts.items())),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved dataset to {out_dir}:", flush=True)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
