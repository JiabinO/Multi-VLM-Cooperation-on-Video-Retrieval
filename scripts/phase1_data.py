"""Bounded, explicit-file HF downloads and local clip inventory (stdlib only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import urllib.request

REPOS = {'assembly101': 'cvml-nus/assembly101', 'cvbench': 'Dongyh35/CVBench'}


def relative(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or '..' in p.parts or '\\' in value or ':' in value:
        raise ValueError('Expected safe relative POSIX path')
    return p.as_posix()


def fetch(plan, root, execute=False, limit=50 * 1024 * 1024):
    """No wildcards, archive extraction, implicit dataset loads or full snapshots."""
    repo = REPOS[plan['dataset']]
    revision = plan['revision']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Pin revision to full 40-character HF commit SHA')
    root = Path(root).resolve()
    files = plan['files']
    names = [relative(f['path']) for f in files]
    if len(set(names)) != len(names) or not files:
        raise ValueError('Empty or duplicate file list')
    total = 0
    for f in files:
        if type(f['bytes']) is not int or f['bytes'] <= 0:
            raise ValueError('Positive expected bytes required')
        if not re.fullmatch(r'[0-9a-f]{64}', f['sha256']):
            raise ValueError('Expected SHA256 required')
        total += f['bytes']
        target = (root / relative(f['path'])).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Path escapes destination')
        if target.exists() or target.with_name(target.name + '.part').exists():
            raise ValueError('Refusing overwrite: ' + str(target))
    if total > limit:
        raise ValueError('Plan exceeds total byte budget')
    if not execute:
        return {'mode': 'dry-run', 'repo': repo, 'revision': revision, 'files': names, 'bytes': total}
    receipts = []
    for f in files:
        target = root / relative(f['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + '.part')
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{relative(f['path'])}"
        headers = {'Authorization': 'Bearer ' + os.environ['HF_TOKEN']} if os.environ.get('HF_TOKEN') else {}
        digest, size = hashlib.sha256(), 0
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response, part.open('xb') as out:
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > f['bytes']:
                        raise ValueError('Response exceeds declared size')
                    digest.update(block)
                    out.write(block)
            if size != f['bytes'] or digest.hexdigest() != f['sha256']:
                raise ValueError('Size or SHA256 mismatch')
            # Exclusive creation prevents replacing a file created during transfer.
            with target.open('xb') as out, part.open('rb') as inp:
                import shutil
                shutil.copyfileobj(inp, out)
            receipts.append({'path': f['path'], 'bytes': size, 'sha256': digest.hexdigest()})
        finally:
            part.unlink(missing_ok=True)
    return {'repo': repo, 'revision': revision, 'files': receipts}


def inventory(rows, root):
    """Explicit, manually checked source mapping; never infer camera synchrony."""
    root = Path(root).resolve()
    seen = set()
    for row in rows:
        key = row['clip_id']
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError('Duplicate/empty clip ID')
        seen.add(key)
        path = (root / relative(row['path'])).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing/unsafe source file')
        for name in ('dataset', 'session_id', 'view_id', 'split'):
            if not isinstance(row[name], str) or not row[name]:
                raise ValueError('Missing ' + name)
        start, end = row['start_s'], row['end_s']
        import math
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)) or not 0 <= start < end:
            raise ValueError('Invalid clip interval')
        if row['sync_status'] not in ('verified', 'asynchronous', 'unknown'):
            raise ValueError('Invalid sync status')
        if row['sync_status'] == 'verified' and not row.get('sync_note'):
            raise ValueError('Verified synchrony requires provenance')
        digest = hashlib.sha256()
        with path.open('rb') as inp:
            while block := inp.read(1024 * 1024):
                digest.update(block)
        yield {**row, 'sha256': digest.hexdigest(), 'source_bytes': path.stat().st_size}


def cvbench_tasks(rows):
    """Import official video_N QA JSON as pending tasks, never as evidence gold."""
    seen = set()
    for row in rows:
        qid = 'cvbench-' + str(row['id'])
        if qid in seen or not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError('Invalid/duplicate CVBench question')
        seen.add(qid)
        sources = [{'view_id': key, 'path': relative(value)} for key, value in sorted(row.items())
                   if re.fullmatch(r'video_[1-9][0-9]*', key) and value]
        if len(sources) < 2:
            raise ValueError('Expected at least two source videos')
        yield {'schema_version': 1, 'query_id': qid, 'dataset': 'cvbench',
               'query': row['question'], 'options': row.get('options', []),
               'source_answer_unverified': row['answer'], 'task_type': row.get('task_type'),
               'review_status': 'pending', 'sync_status': 'unknown',
               'sources': sources, 'split': 'unassigned', 'clips': [], 'required_units': []}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('download')
    d.add_argument('--plan', required=True)
    d.add_argument('--root', required=True)
    d.add_argument('--execute', action='store_true')
    d.add_argument('--max-bytes', type=int, default=50 * 1024 * 1024)
    i = sub.add_parser('inventory')
    i.add_argument('--input', required=True)
    i.add_argument('--root', required=True)
    i.add_argument('--output', required=True)
    c = sub.add_parser('cvbench-tasks')
    c.add_argument('--input', required=True)
    c.add_argument('--output', required=True)
    a = p.parse_args()
    if a.command == 'download':
        print(json.dumps(fetch(json.loads(Path(a.plan).read_text(encoding='utf-8')), a.root, a.execute, a.max_bytes), indent=2))
    else:
        source = json.loads(Path(a.input).read_text(encoding='utf-8'))
        rows = list(cvbench_tasks(source) if a.command == 'cvbench-tasks' else inventory(source, a.root))
        with Path(a.output).open('x', encoding='utf-8') as out:
            json.dump(rows, out, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
