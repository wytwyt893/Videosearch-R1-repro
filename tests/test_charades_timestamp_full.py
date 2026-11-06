"""Guard full selection, paired gains/losses, and GT/initial-retrieval alignment."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_charades_timestamp_full import full_arguments,validate_rows
from summarize_charades_timestamp_full import paired_statistics
import summarize_charades_timestamp_full as summary_module
from inspect_charades_localization_cases import sha


def row(index,iou,start=0,end=7):
    turn={'turn':1,'top1_doc_id':'GT','top1_is_gt':True,'answer':'matched','time_parse_ok':True,
          'policy_top1_doc_id':'GT','policy_rank':1,'iou_r1':iou,'refine_applied':False}
    return {'index':index,'qid':str(index),'query':f'q{index}','gt_pos_doc_id':'GT','gt_start':0,'gt_end':10,
        'orig_rank':1,'turn1_top1_doc_id':'GT','final_pred_start':start,'final_pred_end':end,
        'final_iou_r1':iou,'final_answer':'matched','final_rank':1,'final_top1_doc_id':'GT',
        'final_top1_is_gt':True,'refine_applied_turns':0,'final_time_parse_ok':True,'max_turn':2,'turns':[turn]}


class FullTimestampTests(unittest.TestCase):
    def test_full_plan_removes_only_sample_exclusions(self):
        root=Path(__file__).resolve().parents[1]
        args=json.loads((root/'reports/40_charades_timestamp_cpu.json').read_text())['arguments']
        old=copy.deepcopy(args);full=full_arguments(args)
        self.assertEqual(args,old)
        self.assertEqual([k for k in args if args[k]!=full[k]],['--resume_skip_jsonl'])
        self.assertEqual(full['--resume_skip_jsonl'],'')
        with self.assertRaises(ValueError):full_arguments({**args,'--max_samples':'70'})
        with self.assertRaises(ValueError):full_arguments({**args,'--vllm_tensor_parallel_size':'2'})

    def test_paired_thresholds_count_both_gains_and_losses(self):
        a=[row(0,.6),row(1,.8)];b=[row(0,.8),row(1,.4)]
        s,p=paired_statistics(a,b)
        self.assertEqual((s['thresholds']['0.7']['gained'],s['thresholds']['0.7']['lost']), (1,1))
        self.assertEqual(s['thresholds']['0.7']['net'],0)
        self.assertEqual(s['thresholds']['0.5']['net'],-1)
        self.assertEqual((s['iou_improved'],s['iou_worsened']),(1,1))

    def test_dropped_or_changed_question_is_rejected(self):
        a=[row(0,.6),row(1,.8)]
        with self.assertRaises(ValueError):paired_statistics(a,a[:1])
        changed=copy.deepcopy(a);changed[0]['gt_end']=12
        with self.assertRaises(ValueError):paired_statistics(a,changed)
        changed=copy.deepcopy(a);changed[0]['turn1_top1_doc_id']='OTHER'
        with self.assertRaises(ValueError):paired_statistics(a,changed)

    def test_invalid_answer_is_counted_without_dropping_question(self):
        r=row(0,0);r['final_answer']='unknown';r['turns'][0]['answer']='unknown'
        r['turns'][0]['time_parse_ok']=False
        score=validate_rows([r],expected=1)
        self.assertEqual(score['primary']['VER']['invalid_answers'],1)
        self.assertEqual(score['primary']['VER']['N_expected'],1)
        with self.assertRaises(ValueError):validate_rows([r,r],expected=2)

    def test_full_postprocess_writes_complete_paired_artifacts_without_gpu(self):
        # Synthetic records live only in a temporary directory. This exercises
        # the overnight CPU finish path; it is not an inference/benchmark run.
        with tempfile.TemporaryDirectory(prefix='charades_full_finish_test_') as temporary:
            root=Path(temporary);(root/'doc').mkdir();(root/'reports').mkdir()
            output=root/'outputs'/'fixture';output.mkdir(parents=True)
            baseline=[row(i,.6,end=6) for i in range(3720)]
            source=copy.deepcopy(baseline)
            source[0]['final_iou_r1']=.8;source[0]['turns'][0]['iou_r1']=.8;source[0]['final_pred_end']=8
            plan=root/'plan.json';plan.write_text(json.dumps({'time_semantics':'synthetic unit fixture'}))
            run={'status':'inference_complete','output_dir':str(output),'gpu_index':1,'gpu_uuid':'UNIT-TEST-NO-GPU',
                 'plan':str(plan),'conditions':{}}
            for clock,records in [('nominal',baseline),('source',source)]:
                d=output/clock;d.mkdir();pred=d/'predictions.jsonl'
                pred.write_text(''.join(json.dumps(x)+'\n' for x in records))
                primary=d/'primary.json';primary.write_text('{}\n')
                run['conditions'][clock]={'status':'complete','resource_release':{'passed':True},
                    'predictions':str(pred),'prediction_sha256':sha(pred),
                    'primary_result':str(primary),'primary_result_sha256':sha(primary),
                    'arguments':{'--output_json':str(d/'summary.json'),'--max_samples':'0'},
                    'elapsed_seconds':0,'peak_gpu_used_mib':0}
            run_path=root/'run.json';run_path.write_text(json.dumps(run))
            destination=root/'reports'/'result.json'
            baseline_path=Path(run['conditions']['nominal']['predictions'])
            with mock.patch.object(summary_module,'ROOT',root),mock.patch.object(summary_module,'LOCAL',baseline_path),\
                 mock.patch.object(summary_module,'REFERENCE',baseline_path),\
                 mock.patch.object(summary_module,'audit_inputs',return_value={'passed':True,'seen_videos':1}),\
                 mock.patch('builtins.print'):
                summary_module.main(run_path,destination)
            result=json.loads(destination.read_text())
            self.assertEqual(result['paired_statistics']['thresholds']['0.7']['net'],1)
            self.assertTrue(Path(result['document']).exists())
            self.assertEqual(len((output/'paired_examples.csv').read_text().splitlines()),3721)


if __name__=='__main__':unittest.main()
