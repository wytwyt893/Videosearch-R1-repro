"""Small fixtures target candidate/answer pairing and scoring denominators."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/score_didemo_protocol.py'
spec = importlib.util.spec_from_file_location('didemo_protocol', path)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def example(index=0, answer='matched', checked='GT', retrieved='GT', iou=0.6):
    turn = {'turn': 1, 'answer': answer, 'top1_doc_id': checked,
            'top1_is_gt': checked == 'GT', 'policy_top1_doc_id': retrieved,
            'policy_rank': 1 if retrieved == 'GT' else 2,
            'time_parse_ok': answer == 'matched', 'iou_r1': iou,
            'refine_applied': False}
    return {'index': index, 'qid': 'shared-qid', 'max_turn': 2, 'gt_pos_doc_id': 'GT',
            'turns': [turn], 'final_answer': answer,
            'final_top1_doc_id': retrieved, 'final_top1_is_gt': retrieved == 'GT',
            'final_rank': turn['policy_rank'], 'final_iou_r1': iou}


class ProtocolTests(unittest.TestCase):
    def test_cli_preserves_official_query_whitespace_normalization(self):
        annotations = [json.loads(s) for s in (Path(__file__).resolve().parents[1] /
            'data/didemo/raw_annotation/test.jsonl').read_text().splitlines() if s.strip()][:432]
        self.assertNotEqual(annotations[431]['fig_desc'], annotations[431]['fig_desc'].strip())
        rows = []
        for i, annotation in enumerate(annotations):
            row = example(index=i)
            video = annotation['video']
            row.update(query=annotation['fig_desc'].strip(), gt_pos_doc_id=video,
                       gt_start=annotation['time'][0], gt_end=annotation['time'][1], final_top1_doc_id=video)
            row['turns'][0].update(top1_doc_id=video, policy_top1_doc_id=video)
            rows.append(row)
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp)/'synthetic_fixture.jsonl', Path(tmp)/'fixture_score.json'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            process = subprocess.run([sys.executable, '-B', scorer.__file__, str(path), str(output),
                                      '--expected-count', '432'], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(output.read_text())['num_examples'], 432)

    def test_didemo_corpus_upper_bound_is_1002(self):
        row = example(answer='not_matched', checked='WRONG', retrieved='WRONG', iou=0)
        row['final_rank'] = row['turns'][0]['policy_rank'] = 1002
        self.assertEqual(scorer.score_rows([row], 1)['primary']['VR']['mean_rank'], 1002)
        row['final_rank'] = row['turns'][0]['policy_rank'] = 1003
        with self.assertRaises(ValueError):
            scorer.score_rows([row], 1)

    def test_wrong_checked_video_iou_remains_zero_when_final_ranking_is_correct(self):
        row = example(answer='not_matched', checked='WRONG', retrieved='GT', iou=0)
        row['turns'][0].update(turn=2, refine_applied=True)
        first = copy.deepcopy(row['turns'][0])
        first.update(turn=1, policy_top1_doc_id='WRONG', policy_rank=2)
        row['turns'].insert(0, first)
        result = scorer.score_rows([row], 1)
        self.assertEqual(result['primary']['VR']['R@1'], 1)
        self.assertEqual(result['primary']['VCMR']['IoU@0.3@R1'], 0)

    def test_early_stop_counts_as_last_and_has_vcmr_credit(self):
        result = scorer.score_rows([example()], 1)
        self.assertEqual(result['primary']['VER']['accuracy'], 1)
        self.assertEqual(result['primary']['VCMR']['IoU@0.5@R1'], 1)
        self.assertNotIn('supplementary', result)

    def test_final_rerank_does_not_change_the_video_being_verified(self):
        row = example(answer='not_matched', checked='WRONG', retrieved='GT', iou=0)
        second = copy.deepcopy(row['turns'][0])
        second.update(turn=2, refine_applied=True)
        row['turns'][0].update(policy_top1_doc_id='WRONG', policy_rank=2)
        row['turns'].append(second)
        result = scorer.score_rows([row], 1)
        self.assertEqual(result['primary']['VER']['accuracy'], 1)
        self.assertEqual(result['diagnostics']['official_final_field_pairing_VER']['accuracy'], 0)
        self.assertEqual(result['primary']['VR']['R@1'], 1)
        self.assertEqual(result['primary']['VCMR']['IoU@0.3@R1'], 0)

    def test_unknown_answer_keeps_coverage_visible(self):
        row = example(answer='unknown', checked='WRONG', retrieved='WRONG', iou=0)
        metric = scorer.score_rows([row], 1)['primary']['VER']
        self.assertIsNone(metric['accuracy'])
        self.assertEqual((metric['N'], metric['invalid_answers'], metric['coverage']), (0, 1, 0))
        self.assertEqual(metric['accuracy_all_examples'], 0)

    def test_qid_collision_does_not_delete_an_example(self):
        self.assertEqual(scorer.score_rows([example(0), example(1)], 2)['num_examples'], 2)
        with self.assertRaises(ValueError):
            scorer.score_rows([example(0), example(0)], 2)
        with self.assertRaises(ValueError):
            scorer.score_rows([example(0)], 2)

    def test_wrong_video_cannot_receive_temporal_credit(self):
        with self.assertRaises(ValueError):
            scorer.score_rows([example(checked='WRONG', retrieved='WRONG')], 1)

    def test_first_turn_is_a_supplement_and_does_not_replace_primary(self):
        row = example(answer='matched', checked='WRONG', retrieved='WRONG', iou=0)
        first = copy.deepcopy(row['turns'][0])
        first.update(answer='not_matched', time_parse_ok=False)
        row['turns'][0]['turn'] = 2
        row['turns'].insert(0, first)
        result = scorer.score_rows([row], 1, include_first_turn=True)
        self.assertEqual(result['primary']['VER']['accuracy'], 0)
        self.assertEqual(result['supplementary']['first_turn_VER']['accuracy'], 1)


if __name__ == '__main__':
    unittest.main()
