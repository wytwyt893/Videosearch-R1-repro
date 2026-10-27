"""Small fixtures target candidate/answer pairing and scoring denominators."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/score_charades_protocol.py'
spec = importlib.util.spec_from_file_location('charades_protocol', path)
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
