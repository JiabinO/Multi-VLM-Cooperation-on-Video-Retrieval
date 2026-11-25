"""Single-video sampled-image smoke test; run from the repository root."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def frame_indices(fps, total, start, end, count):
    if not math.isfinite(fps) or fps <= 0 or total <= 0:
        raise ValueError("Invalid video FPS or frame count")
    if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end <= total / fps:
        raise ValueError("Require 0 <= start < end <= video duration")
    first, last = math.ceil(start * fps), math.ceil(end * fps) - 1
    if count < 1 or last - first + 1 < count:
        raise ValueError("Not enough distinct frames in the requested interval")
    if count == 1:
        return [first]
    return [round(first + i * (last - first) / (count - 1)) for i in range(count)]


def parse_evidence(raw, frames):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    result = json.loads(text)
    if not isinstance(result, dict) or not isinstance(result.get("observations"), list):
        raise ValueError("Expected an object with observations list")
    times = {f["frame_id"]: f["nominal_time_sec"] for f in frames}
    for obs in result["observations"]:
        if not isinstance(obs, dict) or not isinstance(obs.get("claim"), str) or not obs["claim"].strip():
            raise ValueError("Each observation needs a nonempty claim")
        ids = obs.get("frame_ids")
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i not in times for i in ids):
            raise ValueError("Observation contains missing or invalid frame references")
        if not isinstance(obs.get("uncertainty"), str):
            raise ValueError("Each observation needs an uncertainty string")
        obs["sampled_time_bounds_sec"] = [min(times[i] for i in ids), max(times[i] for i in ids)]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--end", type=float, default=30)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--extract-only", action="store_true")
    args = p.parse_args()
    if not args.extract_only and args.model is None:
        p.error("--model is required for inference")
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    run = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
           "command": sys.argv, "python": sys.version, "experiment_id": "SMOKE-001",
           "run_id": out.name, "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    repo = Path(__file__).resolve().parents[1]
    def git(*parts):
        r = subprocess.run(["git", "-C", str(repo), *parts], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    try:
        run["code_commit"] = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
        run["git_dirty"] = None if status is None else bool(status)
        write_json(out / "run.json", run)
        import cv2
        from PIL import Image, ImageDraw
        run["opencv"] = cv2.__version__
        if not args.video.is_file():
            raise FileNotFoundError(args.video)
        cap = cv2.VideoCapture(str(args.video))
        try:
            if not cap.isOpened():
                raise RuntimeError("Cannot open video")
            fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = frame_indices(fps, total, args.start, args.end, args.frames)
            images, frames = [], []
            (out / "frames").mkdir()
            for frame_id, idx in enumerate(indices):
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
                    raise RuntimeError(f"Seek failed: {idx}")
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Decode failed: {idx}")
                decoded = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                if decoded != idx:
                    raise RuntimeError(f"Unexpected decoded frame: requested={idx}, reported={decoded}")
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                file = f"frames/{frame_id:03d}.png"
                image.save(out / file)
                images.append(image)
                frames.append({"frame_id": frame_id, "frame_index": decoded,
                               "nominal_time_sec": decoded / fps,
                               "backend_pos_msec": cap.get(cv2.CAP_PROP_POS_MSEC), "file": file,
                               "sha256": hashlib.sha256((out / file).read_bytes()).hexdigest()})
        finally:
            cap.release()
        write_json(out / "frames.json", {"fps": fps, "total_frames": total, "frames": frames,
                   "timestamp_basis": "frame index / reported FPS; nominal CFR timing, not verified presentation timestamps"})
        sheet = Image.new("RGB", (4 * 320, math.ceil(len(images) / 4) * 210), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (image, info) in enumerate(zip(images, frames)):
            thumb = image.copy()
            thumb.thumbnail((320, 180))
            x, y = (i % 4) * 320, (i // 4) * 210
            sheet.paste(thumb, (x, y))
            draw.text((x + 3, y + 182), f"Frame {i} | {info['nominal_time_sec']:.3f}s", fill="black")
        sheet.save(out / "preview.jpg")
        if args.extract_only:
            run["status"] = "extracted"
            return
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requires CUDA GPU with BF16 support")
        torch.manual_seed(0)
        run.update(torch=torch.__version__, transformers=transformers.__version__,
                   gpu=torch.cuda.get_device_name(0), seed=0)
        processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
        print("Loading model...", flush=True)
        started = time.perf_counter()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.model), dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", local_files_only=True).eval()
        torch.cuda.synchronize()
        run["load_seconds"] = time.perf_counter() - started
        content = []
        for image, f in zip(images, frames):
            content.extend([{"type": "text", "text": f"Frame {f['frame_id']}, time={f['nominal_time_sec']:.3f}s"},
                            {"type": "image", "image": image}])
        prompt = ('只根据这些按时间排列的采样帧，输出可直接看见的事实，不猜测帧间动作、动机或身份。'
                  '不能把未采样时段当作已观察。用中文，仅输出 JSON：'
                  '{"observations":[{"claim":"事实", "frame_ids":[0], "uncertainty":"不确定处；没有则填空字符串"}]}。'
                  'frame_ids 必须引用支持事实的输入帧编号。没有可靠事实时 observations 可为空。')
        (out / "prompt.txt").write_text(prompt, encoding="utf-8")
        content.append({"type": "text", "text": prompt})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        model.generation_config.do_sample = False
        for key in ("temperature", "top_p", "top_k"):
            setattr(model.generation_config, key, None)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        (out / "raw_response.txt").write_text(raw, encoding="utf-8")
        write_json(out / "metrics.json", {"generate_seconds": time.perf_counter() - started,
            "input_tokens": inputs["input_ids"].shape[1], "generated_tokens": generated.shape[1],
            "reached_token_limit": generated.shape[1] >= args.max_new_tokens,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "response_utf8_bytes": len(raw.encode("utf-8")),
            "note": "Single-process input/output sizes, not measured network traffic; first generation, no warmup."})
        try:
            evidence = parse_evidence(raw, frames)
            write_json(out / "evidence.json", evidence)
            run["status"] = "completed"
        except (ValueError, TypeError) as exc:
            run.update(status="parse_failed", parse_error=str(exc))
            raise
        print(raw, flush=True)
    except BaseException as exc:
        if run["status"] != "parse_failed":
            run["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        run["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "run.json", run)
        print(f"Status: {run['status']} | Output: {out}", flush=True)


if __name__ == "__main__":
    main()
