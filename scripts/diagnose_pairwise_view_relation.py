#!/usr/bin/env python3
"""Qualitatively diagnose cross-view relation reasoning on a pair of images.

Runs three prompt conditions in both image orders with a local Qwen3-VL model:
1) blind pair, 2) known same scene/different cameras, 3) human-confirmed anchors.
This is a small diagnostic, not a metric camera-pose benchmark.
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


def build_prompt(condition: str, anchors: str | None) -> str:
    output = (
        "Return compact JSON with keys: shared_entities (array of short strings), "
        "visual_evidence (array of short strings), viewpoint_relation (short qualitative "
        "description or 'unknown'), relation_confidence (low/medium/high), uncertainty "
        "(short string). Keep image-left/right separate from a person's anatomical left/right. "
        "Do not guess exact camera coordinates or directions unsupported by visible evidence."
    )
    if condition == "blind_pair":
        return (
            "Compare Image A and Image B. Without assuming they depict the same scene, "
            "identify any people, objects, or landmarks that plausibly appear in both. "
            "Describe only relationships supported by visible evidence; say unknown if you "
            "cannot establish a shared entity or spatial relationship. Do not answer any "
            "question about an external task. " + output
        )
    if condition == "same_scene_known":
        return (
            "Image A and Image B show the same scene from different camera positions. "
            "Identify likely shared people, objects, and especially stable scene landmarks. "
            "Using their relative image positions, occlusion, and visible background changes, "
            "describe the broad viewpoint relationship between the cameras if supported. "
            "Do not treat a person's movement as camera movement. If the pair does not support "
            "a relation, say unknown. " + output
        )
    if condition == "verified_anchors":
        assert anchors
        return (
            "Image A and Image B show the same scene from different camera positions. "
            "The following cross-image correspondences have been verified by a human: "
            f"{anchors}. Treat these correspondences as given, not as new visual discoveries. "
            "Use only visible stable landmarks, relative image positions, and occlusion changes "
            "to describe the broad viewpoint relationship between the cameras. Do not infer "
            "metric camera pose, and do not mistake subject motion for camera motion. If the "
            "images do not determine a relation, say unknown. " + output
        )
    raise ValueError(f"Unknown condition: {condition}")


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

    def generate(self, prompt: str, image_paths: list[Path], max_new_tokens: int) -> dict:
        images = [load_image(path) for path in image_paths]
        try:
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
            for image in images:
                image.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-a", required=True, type=Path)
    parser.add_argument("--image-b", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--anchors", default=None,
        help="Human-verified cross-image correspondences for the anchored condition, e.g. 'the same man in the black #7 jersey; the same yellow goal'",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.image_a.is_file() or not args.image_b.is_file():
        parser.error("Both image paths must exist")
    if not args.model_path.is_dir():
        parser.error(f"Model directory does not exist: {args.model_path}")
    if not args.anchors or not args.anchors.strip():
        parser.error("--anchors is required so the verified-anchor condition is explicit")
    if args.max_new_tokens < 32:
        parser.error("--max-new-tokens must be at least 32")

    signature_basis = {
        "images": [str(args.image_a.resolve()), str(args.image_b.resolve())],
        "sizes": [args.image_a.stat().st_size, args.image_b.stat().st_size],
        "model": str(args.model_path.resolve()), "anchors": args.anchors,
        "conditions": ["blind_pair", "same_scene_known", "verified_anchors"],
        "orders": ["A_then_B", "B_then_A"], "max_new_tokens": args.max_new_tokens,
    }
    signature = hashlib.sha256(json.dumps(
        signature_basis, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if args.output.exists():
        if not args.resume:
            parser.error(f"Output exists; choose a new path or pass --resume: {args.output}")
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            parser.error("Existing output belongs to different images/model/settings")
        results = saved.get("results", [])

    done = {(x.get("condition"), x.get("order")) for x in results
            if x.get("status") == "completed"}
    tasks = [(condition, order) for condition in
             ("blind_pair", "same_scene_known", "verified_anchors") for order in
             ("A_then_B", "B_then_A") if (condition, order) not in done]
    runtime = QwenRuntime(args.model_path)
    print(f"Model: {runtime.metadata['model_path']}; calls remaining: {len(tasks)}/6", flush=True)
    for i, (condition, order) in enumerate(tasks, start=1):
        ordered_paths = ([args.image_a, args.image_b] if order == "A_then_B"
                         else [args.image_b, args.image_a])
        prompt = build_prompt(condition, args.anchors)
        print(f"[{i}/{len(tasks)}] {condition} | {order}", flush=True)
        row = {
            "condition": condition, "order": order,
            "input_images": [str(x.resolve()) for x in ordered_paths],
            "prompt": prompt, "status": "error",
        }
        started = time.time()
        try:
            response = runtime.generate(prompt, ordered_paths, args.max_new_tokens)
            row.update(response)
            row["status"] = "truncated" if response["finish_reason"] == "length" else "completed"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_seconds"] = round(time.time() - started, 2)
        results.append(row)
        data = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "signature": signature, "settings": signature_basis,
            "runtime": runtime.metadata,
            "interpretation": [
                "Qualitative diagnostic only; no ground-truth camera pose is assumed.",
                "Image-order swaps are a consistency check, not independent samples.",
                "Self-reported confidence is not calibrated probability.",
            ],
            "results": results,
        }
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: row.get(k) for k in
                          ("condition", "order", "status", "finish_reason", "content", "error", "elapsed_seconds")},
                         ensure_ascii=False), flush=True)
    print(f"Saved {len(results)} records to {args.output}", flush=True)
    return 0 if all(r.get("status") == "completed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

