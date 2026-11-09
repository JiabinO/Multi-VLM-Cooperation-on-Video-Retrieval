"""Assembly101 W02 sampled-image multi-view QA; no environment installation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from video_smoke import write_json

VIEWS = ("C10095", "C10115", "C10118")
GROUPS = {"full": VIEWS, **{v: (v,) for v in VIEWS},
          "C10095+C10115": VIEWS[:2]}
QUESTION = "操作者从推土机模型上拆下了哪个部件？"
REFERENCE = "铲斗部件"
PROMPT = ("这些图片来自同一段视频的不同摄像机，标签给出了视角与采样时间。"
          "只根据可见证据，用中文简短回答；无法判断时明确说明。问题：" + QUESTION)


def extract(video, out, count, clip_start=None):
    """Use the existing decoder, sampling validation, PNG hashes and preview."""
    command = [sys.executable, str(Path(__file__).with_name("video_smoke.py")),
               "--video", str(video), "--start", "60" if clip_start is None else "0",
               "--end", "65" if clip_start is None else "5",
               "--frames", str(count), "--output", str(out), "--extract-only"]
    result = subprocess.run(command, capture_output=True, text=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "extract.stdout.log").write_text(result.stdout, encoding="utf-8")
    (out / "extract.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Extraction failed for {video}; see {out}/extract.stderr.log")
    data = json.loads((out / "frames.json").read_text(encoding="utf-8"))
    if clip_start is not None:
        for frame in data["frames"]:
            frame["clip_time_sec"] = frame["nominal_time_sec"]
            frame["nominal_time_sec"] += clip_start
        data["source_time_offset_sec"] = clip_start
        data["timestamp_basis"] += "; source time = clip time + user-declared offset"
        write_json(out / "frames.json", data)
    return data


class QwenRuntime:
    """Same local-only BF16/SDPA loading and sampled-image API as video_smoke."""
    def __init__(self, model_path):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requires CUDA GPU with BF16 support")
        self.torch = torch
        torch.manual_seed(0)
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", local_files_only=True).eval()
        torch.cuda.synchronize()
        self.metadata = dict(torch=torch.__version__, transformers=transformers.__version__,
                             gpu=torch.cuda.get_device_name(0), seed=0,
                             load_seconds=time.perf_counter() - started)
        self.model.generation_config.do_sample = False
        for key in ("temperature", "top_p", "top_k"):
            setattr(self.model.generation_config, key, None)

    def __call__(self, records, max_tokens, out, prompt=PROMPT):
        from PIL import Image
        torch, model, processor = self.torch, self.model, self.processor
        content = []
        for record in records:
            with Image.open(record["absolute_file"]) as image:
                content.extend([{"type": "text", "text": record["label"]},
                                {"type": "image", "image": image.convert("RGB")}])
        content.append({"type": "text", "text": prompt})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        metrics = {"input_tokens": inputs["input_ids"].shape[1]}
        write_json(out / "metrics.json", metrics)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
        metrics.update(generate_seconds=elapsed, generated_tokens=generated.shape[1],
                       reached_token_limit=generated.shape[1] >= max_tokens,
                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                       note="Sequential fixed order; first group cold, later groups warm; not network traffic.")
        return raw, metrics


def git_info():
    def git(*args):
        result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), *args],
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    status = git("status", "--porcelain")
    return {"code_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


def run(args, extractor=extract, runtime_factory=QwenRuntime):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta = dict(status="running", experiment_id="Assembly101-W02-QA", question=QUESTION,
                reference_answer=REFERENCE, config=config, command=sys.argv, python=sys.version,
                started_at=datetime.now(timezone.utc).isoformat(), **git_info())
    (out / "error.log").write_text("", encoding="utf-8")
    write_json(out / "run.json", meta)
    summaries = []
    try:
        sampled = {}
        errors = {}
        for view in VIEWS:
            video = getattr(args, view)
            try:
                if getattr(args, "w02_clips", False):
                    data = extractor(video, out / "views" / view, args.frames, clip_start=60)
                else:
                    data = extractor(video, out / "views" / view, args.frames)
                sampled[view] = []
                for frame in data["frames"]:
                    sampled[view].append(dict(frame, view=view,
                        absolute_file=str((out / "views" / view / frame["file"]).resolve()),
                        label=f"View {view}, frame {frame['frame_id']}, time={frame['nominal_time_sec']:.3f}s"))
            except Exception:
                errors[view] = traceback.format_exc()
        runtime = None
        load_error = None
        if not args.extract_only and sampled:
            try:
                runtime = runtime_factory(args.model)
                meta["runtime"] = runtime.metadata
            except Exception:
                load_error = traceback.format_exc()
        for name, views in GROUPS.items():
            group_out = out / name
            group_out.mkdir()
            started = time.perf_counter()
            group = dict(name=name, views=views, status="running", config=config,
                         question=QUESTION, reference_answer=REFERENCE)
            write_json(group_out / "run.json", group)
            write_json(group_out / "metrics.json", {"input_tokens": None, "generate_seconds": None})
            (group_out / "error.log").write_text("", encoding="utf-8")
            (group_out / "prompt.txt").write_text(PROMPT, encoding="utf-8")
            try:
                missing = {v: errors[v] for v in views if v in errors}
                if missing:
                    raise RuntimeError(json.dumps(missing, ensure_ascii=False))
                records = [f for v in views for f in sampled[v]]
                write_json(group_out / "frames.json", records)
                if args.extract_only:
                    group["status"] = "extracted"
                else:
                    if load_error:
                        raise RuntimeError(load_error)
                    raw, metrics = runtime(records, args.max_new_tokens, group_out)
                    (group_out / "raw_response.txt").write_text(raw, encoding="utf-8")
                    write_json(group_out / "metrics.json", metrics)
                    group["status"] = "completed"
            except Exception:
                group["status"] = "failed"
                (group_out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                group["total_seconds"] = time.perf_counter() - started
                write_json(group_out / "run.json", group)
                summaries.append(group)
                print(f"{name}: {group['status']}", flush=True)
        meta["status"] = "failed" if any(g["status"] == "failed" for g in summaries) else (
            "extracted" if args.extract_only else "completed")
    except BaseException:
        meta["status"] = "failed"
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        meta["groups"] = summaries
        write_json(out / "run.json", meta)
    return 1 if meta["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for view in VIEWS:
        parser.add_argument("--" + view, required=True, type=Path,
                            help="Full-length synchronized Assembly101 video, not a pre-trimmed clip")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--frames", type=int, default=8, help="Frames per view, identical across groups")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--w02-clips", action="store_true",
                        help="Inputs are W02 clips cut from source 60-65s; sample clip 0-5s")
    args = parser.parse_args()
    if args.frames < 1 or args.max_new_tokens < 1:
        parser.error("Frame and token counts must be positive")
    if not args.extract_only and args.model is None:
        parser.error("--model required for inference")
    if len({getattr(args, v).resolve() for v in VIEWS}) != 3:
        parser.error("Three distinct camera files are required")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
