#!/usr/bin/env python3
"""Download MVVBench annotations and the source-video mapping from Hugging Face.

Videos are not redistributed with MVVBench. Download the source datasets under
their own terms, then use the upstream prepare_videos.py described in the
generated README.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO = "everex/MVVBench"
FILES = (
    "README.md",
    "all_questions.json",
    "all_questions.parquet",
    "video_mapping.csv",
    "prepare_videos.py",
)


def download_file(repo: str, name: str, target: Path, token: str | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{name}?download=true"
    headers = {"User-Agent": "mvvbench-downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    temp = target.with_name(target.name + ".part")
    for attempt in range(4):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if temp.stat().st_size == 0:
                raise RuntimeError("downloaded file is empty")
            os.replace(temp, target)
            print(f"Downloaded {name} ({target.stat().st_size:,} bytes)")
            return
        except urllib.error.HTTPError as exc:
            temp.unlink(missing_ok=True)
            if exc.code in (401, 403):
                raise RuntimeError(
                    f"Hugging Face denied access to {name} (HTTP {exc.code}). "
                    "The repository requires non-commercial research use and compliance "
                    "with the source dataset licenses. After accepting those conditions "
                    "on the dataset page, set HF_TOKEN and retry."
                ) from exc
            if exc.code == 404:
                raise RuntimeError(
                    f"Hugging Face returned 404 for {name}. MVVBench gates its annotation "
                    "files; accept the dataset conditions in your account, set HF_TOKEN, "
                    "and retry. If already authorized, verify the repository file path."
                ) from exc
            if attempt == 3:
                raise
            time.sleep(2**attempt)
        except Exception:
            temp.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def validate(output: Path) -> tuple[int, int]:
    questions_path = output / "all_questions.json"
    mapping_path = output / "video_mapping.csv"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if isinstance(questions, dict):
        if "data" in questions or "questions" in questions:
            questions = questions.get("data", questions.get("questions", []))
        else:
            questions = [row for sample_rows in questions.values() for row in sample_rows]
    if not isinstance(questions, list):
        raise RuntimeError("Unexpected question JSON format; expected a list or sample-keyed object")
    with mapping_path.open(encoding="utf-8-sig", newline="") as stream:
        mappings = list(csv.DictReader(stream))
    if not mappings:
        raise RuntimeError("Video mapping CSV contains no rows")
    return len(questions), len(mappings)


def write_readme(output: Path, repo: str, q_count: int, map_count: int) -> None:
    text = f"""# MVVBench local data

Downloaded from [Hugging Face: {repo}](https://huggingface.co/datasets/{repo}).

- Question records: {q_count}
- Source-video mapping rows: {map_count}
- Upstream repository: `prepare_videos.py`

## Videos

MVVBench does not include video files. Its mapping refers to source datasets:

- Ego-Exo4D: download the required `takes/.../frame_aligned_videos/downscaled/448/` files from its official access portal.
- MMPTrack: download the train and validation videos from its official dataset page.
- CMU Panoptic Studio: download the sequences containing `hdVideos_down/` from its official dataset page.

After downloading, arrange them under `dataset/egoexo`, `dataset/mmptrack`, and
`dataset/panoptic` following the source paths in `video_mapping.csv`, then run
from this directory:

```bash
python prepare_videos.py --dataset-root dataset --output-dir videos --mapping-file video_mapping.csv --dry-run
python prepare_videos.py --dataset-root dataset --output-dir videos --mapping-file video_mapping.csv --symlink
```

The upstream preparation helper needs `pandas` and `tqdm`.

Check each source dataset's current access conditions, license, and attribution
requirements before downloading or using its videos. This folder contains only
MVVBench's metadata and preparation helper; the source videos remain governed
by their original dataset terms.
"""
    (output / "DOWNLOAD_README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mvvbench"))
    parser.add_argument("--repo", default=REPO, help="Hugging Face dataset repository")
    parser.add_argument(
        "--files", nargs="+", default=list(FILES),
        help="Repository-relative files to download (default: questions, mapping, helper, README)",
    )
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token  # type: ignore[import-not-found]

            token = get_token()
        except ImportError:
            pass
    try:
        for name in args.files:
            download_file(args.repo, name, args.output_dir / name, token)
        q_count, map_count = validate(args.output_dir)
        write_readme(args.output_dir, args.repo, q_count, map_count)
        print(f"Validated {q_count} questions and {map_count} video mappings.")
        print(f"Local metadata directory: {args.output_dir.resolve()}")
        return 0
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
