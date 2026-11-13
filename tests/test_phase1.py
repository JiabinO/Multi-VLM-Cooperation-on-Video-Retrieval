import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from phase1_eval import evaluate
from phase1_data import fetch, inventory, cvbench_tasks


class Phase1Tests(unittest.TestCase):
    def setUp(self):
        self.refs = json.loads((ROOT / 'data/manifests/phase1_synthetic_reference.json').read_text(encoding='utf-8'))
        self.preds = json.loads((ROOT / 'data/manifests/phase1_synthetic_predictions.json').read_text(encoding='utf-8'))

    def test_redundant_removal_and_critical_loss(self):
        result = evaluate(self.refs, self.preds)['conditions']
        self.assertEqual(result['dedup']['evidence_unit_recall'], 1)
        self.assertAlmostEqual(result['dedup']['clip_reduction'], 1 / 3)
        self.assertEqual(result['bad_drop']['evidence_unit_recall'], .5)
        self.assertEqual(result['bad_drop']['answer_delta_vs_all'], -1)
        self.assertIsNone(result['dedup']['uplink_bytes_reduction_vs_all'])

    def test_joint_bundle_is_conjunctive(self):
        self.refs[0]['required_units'] = [{'unit_id': 'joint', 'alternatives': [['c1', 'c2']]}]
        self.assertEqual(evaluate(self.refs, self.preds)['conditions']['dedup']['evidence_unit_recall'], 0)

    def test_bad_predictions_rejected(self):
        mutations = [lambda p: p[1].update(query_id='missing'), lambda p: p.append(p[0]),
                     lambda p: p[1].update(selected_clip_ids=['unknown']),
                     lambda p: p[1].update(selected_clip_ids=['c1', 'c1']),
                     lambda p: p[0].update(selected_clip_ids=['c1']),
                     lambda p: p[1].update(uplink_bytes=-1)]
        for change in mutations:
            p = copy.deepcopy(self.preds)
            change(p)
            with self.assertRaises(ValueError):
                evaluate(self.refs, p)

    def test_unreviewed_and_bad_gold_rejected(self):
        for field, value in [('review_status', 'pending'), ('required_units', []), ('accepted_answers', [])]:
            r = copy.deepcopy(self.refs)
            r[0][field] = value
            with self.assertRaises(ValueError):
                evaluate(r, self.preds)

    def plan(self):
        return {'dataset': 'cvbench', 'revision': 'a' * 40,
                'files': [{'path': 'small.json', 'bytes': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}]}

    def test_dry_run_no_network_and_budget(self):
        with tempfile.TemporaryDirectory() as d, patch('urllib.request.urlopen') as net:
            self.assertEqual(fetch(self.plan(), d)['mode'], 'dry-run')
            net.assert_not_called()
            with self.assertRaises(ValueError):
                fetch(self.plan(), d, limit=1)

    def test_download_integrity_and_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'{}')):
                fetch(self.plan(), d, True)
            self.assertEqual((Path(d) / 'small.json').read_bytes(), b'{}')
            with self.assertRaises(ValueError):
                fetch(self.plan(), d, True)
        for body in (b'XX', b'{}oversize'):
            with tempfile.TemporaryDirectory() as d, patch('urllib.request.urlopen', return_value=io.BytesIO(body)):
                with self.assertRaises(ValueError):
                    fetch(self.plan(), d, True)
                self.assertEqual(list(Path(d).iterdir()), [])

    def test_unsafe_paths_and_unpinned_revision(self):
        for path in ('../escape', '/absolute', 'C:/file', 'a\\b'):
            plan = self.plan()
            plan['files'][0]['path'] = path
            with self.assertRaises(ValueError):
                fetch(plan, '.')
        plan = self.plan()
        plan['revision'] = 'main'
        with self.assertRaises(ValueError):
            fetch(plan, '.')

    def test_inventory_hash_and_bad_times(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'source.mp4').write_bytes(b'synthetic, not real video')
            row = {'clip_id': 'c', 'dataset': 'synthetic', 'session_id': 's', 'view_id': 'v',
                   'split': 'pilot', 'path': 'source.mp4', 'start_s': 0, 'end_s': 2, 'sync_status': 'unknown'}
            self.assertEqual(len(list(inventory([row], d))[0]['sha256']), 64)
            row['end_s'] = float('nan')
            with self.assertRaises(ValueError):
                list(inventory([row], d))

    def test_cvbench_import_is_pending_not_gold(self):
        rows = [{'id': 7, 'question': 'Synthetic?', 'answer': 'A',
                 'video_1': '7/a.mp4', 'video_2': '7/b.mp4', 'video_3': None}]
        task = list(cvbench_tasks(rows))[0]
        self.assertEqual(task['review_status'], 'pending')
        self.assertEqual(len(task['sources']), 2)
        with self.assertRaises(ValueError):
            evaluate([task], [])
        with self.assertRaises(ValueError):
            list(cvbench_tasks(rows + rows))


if __name__ == '__main__':
    unittest.main()
