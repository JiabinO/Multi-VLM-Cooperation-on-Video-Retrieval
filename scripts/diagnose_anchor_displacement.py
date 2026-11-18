#!/usr/bin/env python3
"""Check whether a VLM can localize shared anchors and report image-plane shifts.

Runs the same image pair in both input orders with a local Qwen3-VL model.
The task is limited to visible anchor locations and their 2D image displacement;
it does not ask the model to infer real-world camera pose.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def make_prompt(anchors: list[str]) -> str:
    anchor_lines = "\n".join(f"- {i}: {value}" for i, value in enumerate(anchors, 1))
    return f"""Two images show a scene. The candidate anchors to inspect are:
{anchor_lines}

For each listed anchor, report only what is visibly supported:
1. Whether the corresponding physical anchor is clearly the same in both images
   (yes / uncertain / no). Similar-looking repeated objects are not enough for yes.
2. Its visibility in each image (clear / partial / occluded / not_visible / uncertain).
3. The location of the same reference point in each image using this 3x3 image grid:
   top_left, top_center, top_right, middle_left, center, middle_right,
   bottom_left, bottom_center, bottom_right, or unknown.
   Use the same physical point where possible (for a stand, its base; for a floor
   marking, a distinctive endpoint or intersection). If no unique point matches,
   use unknown.
4. The change in that reference point's image location from Image A to Image B:
   left, right, up, down, up_left, up_right, down_left, down_right,
   no_clear_shift, or unknown. This is strictly a screen-coordinate change.

Do not describe real-world direction or say that the camera moved left/right/up/down.
Do not use moving people as fixed landmarks. Do not force a match. If the anchor is
ambiguous, say uncertain/unknown and explain briefly.

Return only compact JSON with this structure:
{{"anchors":[{{"anchor_id":1,"same_physical_anchor":"yes|uncertain|no",
"image_A":{{"visibility":"clear|partial|occluded|not_visible|uncertain",
"location":"one grid value or unknown"}},
"image_B":{{"visibility":"clear|partial|occluded|not_visible|uncertain",
"location":"one grid value or unknown"}},
"reference_point":"short description",
"shift_A_to_B":"one allowed shift value",
"evidence":"short visual evidence","uncertainty":"short string"}}]}}.
Return one row for every anchor, preserving its anchor_id. Do not add a camera-pose guess."""


class QwenRuntime:
    def __init__(self, model_path: Path):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Local inference requires a CUDA GPU with BF16 support")
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
            "model_path": str(model_path.resolve()), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16",
            "attention": "sdpa", "do_sample": False,
        }

    def generate(self, prompt: str, ordered_paths: list[Path], max_new_tokens: int) -> dict:
        from PIL import Image

        images = []
        try:
            for path in ordered_paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            content = [{"type": "text", "text": prompt}]
            for label, image in zip(("Image A", "Image B"), images):
                content.extend([
                    {"type": "text", "text": f"\n{label}:"},
                    {"type": "image", "image": image},
                ])
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True,
                add_generation_prompt=True, return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                )
            generated = output[:, inputs["input_ids"].shape[1]:]
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
            for image in images:
                image.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-a", type=Path, required=True,
                        help="First image, e.g. View 0 frame_000")
    parser.add_argument("--image-b", type=Path, required=True,
                        help="Second image, e.g. View 3 frame_000")
    parser.add_argument("--anchor", action="append", required=True,
                        help="Candidate anchor description; repeat once per anchor")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for path in (args.image_a, args.image_b):
        if not path.is_file():
            parser.error(f"Image does not exist: {path}")
    if not args.model_path.is_dir():
        parser.error(f"Model directory does not exist: {args.model_path}")
    if any(not value.strip() for value in args.anchor):
        parser.error("Anchor descriptions cannot be empty")
    if args.max_new_tokens < 64:
        parser.error("--max-new-tokens must be at least 64")

    settings = {
        "image_a": str(args.image_a.resolve()), "image_b": str(args.image_b.resolve()),
        "image_a_bytes": args.image_a.stat().st_size,
        "image_b_bytes": args.image_b.stat().st_size,
        "anchors": args.anchor, "model_path": str(args.model_path.resolve()),
        "max_new_tokens": args.max_new_tokens,
        "orders": ["A_then_B", "B_then_A"],
    }
    signature = hashlib.sha256(json.dumps(
        settings, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if args.output.exists():
        if not args.resume:
            parser.error(f"Output exists; choose a new path or pass --resume: {args.output}")
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            parser.error("Existing output belongs to different images/model/anchors/settings")
        results = saved.get("results", [])

    done = {r.get("order") for r in results if r.get("status") == "completed"}
    orders = [order for order in ("A_then_B", "B_then_A") if order not in done]
    runtime = QwenRuntime(args.model_path)
    print(f"Anchor displacement check: {len(orders)} calls remaining; {len(args.anchor)} anchors", flush=True)
    for index, order in enumerate(orders, 1):
        paths = ([args.image_a, args.image_b] if order == "A_then_B"
                 else [args.image_b, args.image_a])
        prompt = make_prompt(args.anchor)
        record = {
            "order": order, "image_files": [str(x.resolve()) for x in paths],
            "prompt": prompt, "status": "error",
        }
        print(f"[{index}/{len(orders)}] {order}", flush=True)
        started = time.time()
        try:
            response = runtime.generate(prompt, paths, args.max_new_tokens)
            record.update(response)
            if response["finish_reason"] == "length":
                record["status"] = "truncated"
            else:
                try:
                    parsed = json.loads(response["content"])
                    record["parsed"] = parsed
                    record["status"] = "completed"
                except (TypeError, json.JSONDecodeError):
                    record["status"] = "invalid_json"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = round(time.time() - started, 2)
        old_index = next((i for i, old in enumerate(results)
                          if old.get("order") == order), None)
        if old_index is None:
            results.append(record)
        else:
            results[old_index] = record
        output = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "signature": signature, "settings": settings,
            "runtime": runtime.metadata,
            "interpretation_notes": [
                "This checks image-plane anchor localization and displacement only, not world or camera pose.",
                "The two input orders are a consistency check, not independent samples.",
                "Anchor identity remains a human-reviewed assumption; repeated similar objects may not be the same physical object.",
                "Model-reported confidence is not calibrated and is not requested in this probe.",
            ],
            "results": results,
        }
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: record.get(k) for k in
                          ("order", "status", "finish_reason", "content", "error", "elapsed_seconds")},
                         ensure_ascii=False), flush=True)
    print(f"Saved {len(results)} records to {args.output}", flush=True)
    return 0 if len(results) == 2 and all(r.get("status") == "completed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

