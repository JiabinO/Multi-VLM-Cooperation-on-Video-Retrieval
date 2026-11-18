#!/usr/bin/env python3
"""Compare single-view and multi-view reasoning on one MVVBench question.

Runs the same multiple-choice question with View 0 only, View 3 only, and both
views. The model is asked for concise, image-grounded evidence and a rationale
summary, not hidden chain-of-thought. Raw responses are checkpointed to JSON.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def load_image(path: Path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def build_prompt(question: str, choices: dict[str, str]) -> str:
    choice_text = "\n".join(f"{key}. {value}" for key, value in choices.items())
    return f"""Answer the multiple-choice question using only the supplied image(s).

Question: {question}

Choices:
{choice_text}

Inspect each image independently before combining evidence. Treat image-left/right as positions in the image; distinguish them from a person's anatomical left/right and from a camera's location. Do not assume that two images show the same person, object, or scene unless visible evidence supports that link. For a camera-direction question, state which visible cues support the relative direction; do not infer a camera position from a person's facing direction alone. If the images do not support a choice, say so clearly.

Return valid JSON only, using this schema:
{{
  "per_image_observations": [
    {{"image": "View 0", "directly_visible_facts": ["..."], "relevant_spatial_cues": ["..."], "uncertainty": "..."}}
  ],
  "cross_view_links": ["State which entities or landmarks are matched across views and the visible basis; use an empty array if no reliable match is possible."],
  "reasoning_summary": ["Give a short, checkable sequence from visible cues to the selected direction. Separate observation from inference."],
  "eliminated_choices": [{{"choice": "A", "reason": "..."}}],
  "answer": "A",
  "confidence": "low|medium|high",
  "limitations": ["..."]
}}

Keep each observation concrete and concise. Do not invent details, and do not report private internal chain-of-thought; provide only the evidence and a brief rationale summary."""


class QwenRuntime:
    def __init__(self, model_path: Path):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", local_files_only=True,
        ).eval()
        self.model.generation_config.do_sample = False
        for key in ("temperature", "top_p", "top_k"):
            setattr(self.model.generation_config, key, None)
        self.metadata = {
            "model_path": str(model_path.resolve()),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16",
            "attention": "sdpa", "do_sample": False,
        }

    def generate(self, prompt: str, images: list[tuple[str, Path]], max_new_tokens: int) -> dict:
        opened = [(label, load_image(path)) for label, path in images]
        try:
            content = [{"type": "text", "text": prompt}]
            for label, image in opened:
                content.extend([
                    {"type": "text", "text": f"\n{label}:"},
                    {"type": "image", "image": image},
                ])
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt",
            ).to(self.model.device)
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                )
            generated = outputs[:, inputs["input_ids"].shape[1]:]
            raw = self.processor.batch_decode(
                generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return {
                "content": raw,
                "finish_reason": "length" if generated.shape[1] >= max_new_tokens else "stop",
                "completion_tokens": int(generated.shape[1]),
            }
        finally:
            for _, image in opened:
                image.close()


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view0", action="append", required=True, type=Path,
                        help="View 0 frame path; repeat once per sampled frame, in temporal order")
    parser.add_argument("--view3", action="append", required=True, type=Path,
                        help="View 3 frame path; repeat once per sampled frame, in temporal order")
    parser.add_argument("--question", required=True)
    parser.add_argument("--choice", action="append", required=True, metavar="LETTER=TEXT",
                        help="Choice text, e.g. A=Left; provide each option once")
    parser.add_argument("--gold", choices=("A", "B", "C", "D"), default=None,
                        help="Optional gold answer, stored only for post-run comparison; never sent in the prompt")
    parser.add_argument("--query-id", default="64pm_industry_safety_4-q01281")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for label, paths in (("View 0", args.view0), ("View 3", args.view3)):
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            parser.error(f"{label} image path(s) do not exist: {missing}")
    if not args.model_path.is_dir():
        parser.error(f"Model directory does not exist: {args.model_path}")
    if args.max_new_tokens < 128:
        parser.error("--max-new-tokens must be at least 128 for structured evidence output")

    choices: dict[str, str] = {}
    for entry in args.choice:
        if "=" not in entry:
            parser.error(f"Invalid --choice {entry!r}; expected LETTER=TEXT")
        key, value = entry.split("=", 1)
        key, value = key.strip().upper(), value.strip()
        if key not in ("A", "B", "C", "D") or not value or key in choices:
            parser.error(f"Invalid or duplicate choice: {entry!r}")
        choices[key] = value
    if list(choices) != ["A", "B", "C", "D"]:
        parser.error("Provide choices exactly once each in order A, B, C, D")

    views = {
        "view0_only": [(f"View 0 frame {i + 1}", p) for i, p in enumerate(args.view0)],
        "view3_only": [(f"View 3 frame {i + 1}", p) for i, p in enumerate(args.view3)],
        "both_views": ([(f"View 0 frame {i + 1}", p) for i, p in enumerate(args.view0)] +
                       [(f"View 3 frame {i + 1}", p) for i, p in enumerate(args.view3)]),
    }
    basis = {
        "query_id": args.query_id, "question": args.question, "choices": choices,
        "gold_answer_for_post_run_only": args.gold,
        "images": {name: [{"label": label, "path": str(path.resolve()),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for label, path in selected]
                   for name, selected in views.items()},
        "model_path": str(args.model_path.resolve()), "max_new_tokens": args.max_new_tokens,
        "conditions": list(views),
    }
    signature = hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if args.output.exists():
        if not args.resume:
            parser.error(f"Output exists; choose a new path or pass --resume: {args.output}")
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            parser.error("Existing output belongs to different question/images/model/settings")
        results = saved.get("results", [])

    done = {row.get("condition") for row in results if row.get("status") == "completed"}
    runtime = QwenRuntime(args.model_path)
    print(f"Model: {runtime.metadata['model_path']}; calls remaining: {len(views) - len(done)}/3", flush=True)
    prompt = build_prompt(args.question, choices)
    for condition, selected in views.items():
        if condition in done:
            continue
        print(f"[{len(done) + 1}/3] {condition} ({len(selected)} frame(s))", flush=True)
        row = {"condition": condition,
               "input_images": [{"label": label, "path": str(path.resolve())} for label, path in selected],
               "prompt": prompt, "status": "error"}
        started = time.time()
        try:
            response = runtime.generate(prompt, selected, args.max_new_tokens)
            row.update(response)
            row["status"] = "truncated" if response["finish_reason"] == "length" else "completed"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_seconds"] = round(time.time() - started, 2)
        results.append(row)
        out = {
            "created_at": datetime.now(timezone.utc).isoformat(), "signature": signature,
            "settings": basis, "runtime": runtime.metadata,
            "interpretation": [
                "Gold answer is metadata only and was not included in model prompts.",
                "Model outputs are evidence summaries, not guaranteed faithful explanations.",
                "Compare conditions qualitatively; this single example is diagnostic only.",
            ], "results": results,
        }
        write_json(args.output, out)
        print(json.dumps({k: row.get(k) for k in
                          ("condition", "status", "finish_reason", "content", "error", "elapsed_seconds")},
                         ensure_ascii=False), flush=True)
        if row["status"] != "completed":
            return 1
        done.add(condition)
    print(f"Saved {len(results)} records to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

