"""Reuse smoke-test PNGs to read scoreboard text without re-extracting video."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from scoreboard_eval import evaluate, parse_readings
from video_smoke import write_json


PROMPT = '''逐帧读取图片上的比分牌和比赛计时，仅依据各帧中实际可辨认的文字。
不要根据其它帧、足球常识或片头字幕补全。不要描述广告、球员或比赛过程。
使用原始全局 frame_id，每帧恰好返回一条，按输入顺序输出，仅输出 JSON：
{"frames":[{"frame_id":0,"status":"readable|unreadable|not_visible",
"eng_score":null,"den_score":null,"match_clock":null,"uncertainty":"不确定之处"}]}。
能同时读出两队比分和计时时用 readable；存在但部分看不清时用 unreadable，
保留能读出的字段，其余填 null；看不到比分牌和计时用 not_visible，三个值均填 null。
比分为非负整数，match_clock 为 M:SS 字符串，是画面比赛计时而非视频时间。
uncertainty 用中文，没有不确定处则填空字符串。'''


def load_source(source, reference):
    manifest = json.loads((source / "frames.json").read_text(encoding="utf-8"))
    old_run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if Path(old_run["config"]["video"]).name != reference["video_filename"]:
        raise ValueError("Reference video does not match source run")
    frames = manifest["frames"]
    if [f["frame_id"] for f in frames] != [f["frame_id"] for f in reference["frames"]]:
        raise ValueError("Reference frame IDs do not match manifest")
    for frame, ref in zip(frames, reference["frames"]):
        if frame["frame_index"] != ref["frame_index"]:
            raise ValueError("Reference frame index does not match source sampling")
        path = (source / frame["file"]).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("Frame file escapes source directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != frame["sha256"]:
            raise ValueError(f"Frame checksum mismatch: {frame['frame_id']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "data/manifests/video_mme_scoreboard_review.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and args.model is None:
        parser.error("--model is required for inference")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    run = {"experiment_id": "PROBE-001", "run_id": out.name, "status": "running",
           "started_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
           "python": sys.version, "seed": 0,
           "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    try:
        repo = Path(__file__).resolve().parents[1]
        for key, cmd in [("code_commit", ["rev-parse", "HEAD"]), ("git_status", ["status", "--porcelain"])]:
            try:
                result = subprocess.run(["git", "-C", str(repo), *cmd], capture_output=True, text=True)
                run[key] = result.stdout.strip() if result.returncode == 0 else None
            except OSError:
                run[key] = None
        run["git_dirty"] = None if run["git_status"] is None else bool(run["git_status"])
        write_json(out / "run.json", run)
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        manifest = load_source(args.source, reference)
        write_json(out / "frames.json", manifest)
        write_json(out / "reference.json", reference)
        run["source_run_sha256"] = hashlib.sha256((args.source / "run.json").read_bytes()).hexdigest()
        (out / "prompt.txt").write_text(PROMPT, encoding="utf-8")
        if args.validate_only:
            run["status"] = "validated"
            return
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA GPU with BF16 support required")
        run.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__)
        run["packages"] = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate", "pillow")}
        config_file = args.model / "config.json"
        run["model_config_sha256"] = hashlib.sha256(config_file.read_bytes()).hexdigest()
        run["model_identity_note"] = "Local model path and config hash; weight hashes are not computed."
        torch.manual_seed(0)
        processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
        print("Loading model...", flush=True)
        started = time.perf_counter()
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(args.model),
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa", local_files_only=True).eval()
        torch.cuda.synchronize()
        run["load_seconds"] = time.perf_counter() - started
        content = []
        for f in manifest["frames"]:
            with Image.open(args.source / f["file"]) as image:
                rgb = image.convert("RGB")
            content.extend([{"type": "text", "text": f"Frame {f['frame_id']}; video time {f['nominal_time_sec']:.3f}s"},
                            {"type": "image", "image": rgb}])
        # Human reference is deliberately never included in the model input.
        content.append({"type": "text", "text": PROMPT})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        model.generation_config.do_sample = False
        for key in ("temperature", "top_p", "top_k"):
            setattr(model.generation_config, key, None)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        (out / "raw_response.txt").write_text(raw, encoding="utf-8")
        write_json(out / "metrics.json", {"generate_seconds": elapsed, "input_tokens": inputs["input_ids"].shape[1],
            "generated_tokens": generated.shape[1], "reached_token_limit": generated.shape[1] >= args.max_new_tokens,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "response_utf8_bytes": len(raw.encode("utf-8")),
            "note": "First generation, no warmup; multi-image inputs, not network traffic or native video inference."})
        baseline_path = args.source / "metrics.json"
        if baseline_path.is_file():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            current = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            keys = ("generate_seconds", "input_tokens", "generated_tokens", "peak_allocated_gib", "response_utf8_bytes")
            write_json(out / "comparison.json", {
                "note": "Different prompts/tasks, one run each. Descriptive comparison only; no speedup or accuracy claim.",
                "fields": {k: {"baseline": baseline.get(k), "current": current[k],
                    "difference": current[k] - baseline[k] if isinstance(baseline.get(k), (int, float)) else None} for k in keys}})
        try:
            readings = parse_readings(raw, [f["frame_id"] for f in manifest["frames"]])
        except (ValueError, TypeError) as exc:
            run.update(status="parse_failed", parse_error=str(exc))
            raise
        write_json(out / "readings.json", readings)
        evaluation = evaluate(readings, reference)
        write_json(out / "evaluation.json", evaluation)
        lines = ["# 比分读取诊断", "", "仅在人工确认的帧 1–3 上评分；其它帧未评分，不视为负例。",
                 "样本仅一段视频，不能代表总体准确率。", "", f"逐帧全字段正确：{evaluation['exact_match_frames']}/{evaluation['scored_frames']}", ""]
        for row in evaluation["details"]:
            lines.append(f"- 帧 {row['frame_id']}：{'匹配' if row['exact_match'] else '不匹配'}；字段检查 {row['checks']}")
        lines.extend(["", "原始输出见 raw_response.txt；未评分帧须对照原运行的 preview.jpg 人工回看。"])
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        run["status"] = "completed"
        print(raw, flush=True)
        print(f"Confirmed frames exact match: {evaluation['exact_match_frames']}/{evaluation['scored_frames']}", flush=True)
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
