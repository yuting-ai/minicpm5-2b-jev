"""FastAPI System 1 server for MiniCPM5-2B-Jev (/v1/systemone on port 8013)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import (
    SERVE_MAX_BRANCH,
    SERVE_MAX_PACKED,
    SERVE_MAX_STATE,
    MiniCPMSystemOne,
    to_internal_record,
)

app = FastAPI(title="MiniCPM5-2B-Jev Server")

STATE: dict[str, Any] = {}


class QuestionSpec(BaseModel):
    type: str = "choice"
    instructions: Any = ""
    criteria: Any = None


class SystemOneRequest(BaseModel):
    model: str | None = None
    state: Any
    questions: dict[str, QuestionSpec]


@app.get("/health")
def health() -> dict[str, Any]:
    model: MiniCPMSystemOne | None = STATE.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t_list = [round(float(x), 4) for x in model.head.type_temps.detach().cpu().tolist()]
    return {
        "status": "ok",
        "model": "MiniCPM5-2B-Jev",
        "base_model": model.base_model,
        "checkpoint": str(STATE.get("ckpt_dir", "")),
        "device": str(model.device),
        "temperature": round(float(model.head.temperature), 4),
        "type_temperatures": {"choice": t_list[0], "noul": t_list[1], "score": t_list[2]},
        "pride_enabled": bool(torch.any(model.head.noul_bias != 0).item()),
    }


@app.post("/v1/systemone")
@app.post("/v1/evaluate")
def evaluate(req: SystemOneRequest) -> dict[str, Any]:
    model: MiniCPMSystemOne | None = STATE.get("model")
    tok = STATE.get("tok")
    if model is None or tok is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.monotonic()
    wire_rec = {
        "state": req.state,
        "questions": {k: q.model_dump() for k, q in req.questions.items()},
    }
    try:
        internal = to_internal_record(wire_rec, add_date_facts=True)
        out_pairs = model.predict_record(
            internal,
            use_pride=bool(STATE.get("use_pride", False)),
            max_state=SERVE_MAX_STATE,
            max_branch=SERVE_MAX_BRANCH,
            max_packed=SERVE_MAX_PACKED,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    answers: dict[str, Any] = {}
    for q, (_, p_ten) in zip(internal["questions"], out_pairs):
        probs_list = [float(x) for x in p_ten.tolist()]
        keys = q["keys"]
        dist = {str(k): round(p, 6) for k, p in zip(keys, probs_list)}
        best_idx = int(p_ten.argmax().item())
        best_key = str(keys[best_idx])
        best_prob = round(probs_list[best_idx], 6)
        answers[q["name"]] = {
            "choice": best_key,
            "confidence": best_prob,
            "probabilities": dist,
        }

    latency_ms = round((time.monotonic() - t0) * 1000.0, 2)
    return {
        "model": "MiniCPM5-2B-Jev",
        "answers": answers,
        "usage": {
            "latency_ms": latency_ms,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent / "checkpoints" / "stage2"),
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--pride", action="store_true", help="Enable cyclic permutation PriDe inference")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading MiniCPM5-2B-Jev checkpoint from {args.checkpoint} on {device}...")
    model, tok = MiniCPMSystemOne.load_checkpoint(args.checkpoint, device=device, dtype=torch.bfloat16)
    STATE["model"] = model
    STATE["tok"] = tok
    STATE["ckpt_dir"] = args.checkpoint
    STATE["use_pride"] = args.pride

    if device == "mps":
        dummy_req = SystemOneRequest(
            state="Ticket #104: Customer requests a refund for order placed 5 days ago (policy window: 14 days).",
            questions={
                "outcome": QuestionSpec(
                    type="choice",
                    instructions="What is the appropriate decision?",
                    criteria={"approve": "Approve the refund", "decline": "Decline the refund"},
                ),
                "within_window": QuestionSpec(
                    type="noul",
                    instructions="Is the request within the 14-day policy window?",
                ),
            },
        )
        evaluate(dummy_req)
        print("MPS warmup complete.")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
