"""Strict paired evidence-selection evaluation; no model or tokenizer dependency."""
import argparse
import json
import math
from pathlib import Path


def unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if not isinstance(value, str) or not value or value in result:
            raise ValueError('Duplicate/empty ' + key)
        result[value] = row
    return result


def validate(ref):
    if ref['schema_version'] != 1 or ref['review_status'] != 'adjudicated':
        raise ValueError('Only adjudicated v1 references are scoreable')
    for field in ('query', 'dataset', 'split'):
        if not isinstance(ref[field], str) or not ref[field].strip():
            raise ValueError('Missing reference ' + field)
    clips = unique(ref['clips'], 'clip_id')
    for c in clips.values():
        for field in ('view_id', 'session_id', 'path'):
            if not isinstance(c[field], str) or not c[field]:
                raise ValueError('Missing clip provenance')
        a, b = c['start_s'], c['end_s']
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (a, b)) or not 0 <= a < b:
            raise ValueError('Invalid interval')
    units = unique(ref['required_units'], 'unit_id')
    if not units:
        raise ValueError('At least one required evidence unit needed')
    for u in units.values():
        if not u['alternatives']:
            raise ValueError('Empty evidence alternatives')
        for bundle in u['alternatives']:
            if not bundle or len(set(bundle)) != len(bundle) or not set(bundle) <= clips.keys():
                raise ValueError('Invalid evidence bundle')
    if not isinstance(ref['accepted_answers'], list) or not ref['accepted_answers'] or not all(isinstance(a, str) and a.strip() for a in ref['accepted_answers']):
        raise ValueError('Accepted answers required')
    return clips, units


def score(ref, pred):
    clips, units = validate(ref)
    selected = pred['selected_clip_ids']
    if not isinstance(selected, list) or not all(isinstance(x, str) for x in selected) or len(set(selected)) != len(selected) or not set(selected) <= clips.keys():
        raise ValueError('Unknown or duplicate selected clip')
    kept = set(selected)
    coverage = sum(any(set(bundle) <= kept for bundle in u['alternatives']) for u in units.values()) / len(units)
    total_duration = sum(c['end_s'] - c['start_s'] for c in clips.values())
    retained_duration = sum(clips[x]['end_s'] - clips[x]['start_s'] for x in kept)
    if not isinstance(pred['answer'], str):
        raise ValueError('Answer must be string')
    normalize = lambda x: ' '.join(x.casefold().split())
    row = {'query_id': ref['query_id'], 'answer_exact_match': normalize(pred['answer']) in {normalize(a) for a in ref['accepted_answers']},
           'evidence_unit_recall': coverage, 'all_required_preserved': coverage == 1,
           'clip_reduction': 1 - len(kept) / len(clips),
           'duration_reduction': 1 - retained_duration / total_duration}
    # Measured counters only. No fabricated token-to-byte conversion.
    for field in ('uplink_bytes', 'context_tokens'):
        if field in pred:
            if type(pred[field]) is not int or pred[field] < 0:
                raise ValueError('Invalid measured counter')
            row[field] = pred[field]
    return row


def evaluate(references, predictions):
    refs = unique(references, 'query_id')
    if not refs:
        raise ValueError('Empty evaluation')
    by_condition = {}
    for p in predictions:
        condition = p['condition']
        if not isinstance(condition, str) or not condition:
            raise ValueError('Condition required')
        by_condition.setdefault(condition, []).append(p)
    if 'all' not in by_condition:
        raise ValueError('Paired all baseline required')
    scored = {}
    for condition, rows in by_condition.items():
        preds = unique(rows, 'query_id')
        if preds.keys() != refs.keys():
            raise ValueError('Missing/extra query IDs; incomplete runs cannot be silently scored')
        if condition == 'all':
            for q, p in preds.items():
                if set(p['selected_clip_ids']) != {c['clip_id'] for c in refs[q]['clips']}:
                    raise ValueError('all must retain every candidate')
        scored[condition] = [score(refs[q], preds[q]) for q in sorted(refs)]
    report = {'n_queries': len(refs), 'conditions': {}, 'details': scored}
    for condition, rows in scored.items():
        summary = {k: sum(r[k] for r in rows) / len(rows) for k in ('answer_exact_match', 'evidence_unit_recall', 'all_required_preserved', 'clip_reduction', 'duration_reduction')}
        summary['answer_delta_vs_all'] = summary['answer_exact_match'] - sum(r['answer_exact_match'] for r in scored['all']) / len(refs)
        for field in ('uplink_bytes', 'context_tokens'):
            base = scored['all']
            summary[field + '_reduction_vs_all'] = (1 - sum(r[field] for r in rows) / sum(r[field] for r in base)
                if all(field in r for r in rows + base) and sum(r[field] for r in base) else None)
        report['conditions'][condition] = summary
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    report = evaluate(load(a.reference), load(a.predictions))
    with Path(a.output).open('x', encoding='utf-8') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
