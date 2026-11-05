#!/usr/bin/env python3
"""Run the smoke-qualified repaired full evaluation, then compare Table 2."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from run_activitynet_full import write_json
from run_activitynet_repair_smoke import ROOT, digest, execute, verify_inputs, verify_qualified_smoke

REPORT = ROOT/'reports/36_activitynet_repair_full.json'
VARIANT = 'source_time_repaired_subset'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--report', type=Path, default=REPORT)
    args = parser.parse_args()
    if not args.execute:
        _, plans = verify_inputs()
        selected = plans['plans'][VARIANT+'_full']
        qualification = verify_qualified_smoke(VARIANT, selected, plans)
        print(json.dumps({'passed': True, 'model_execution': False, 'gpu_access': False,
                          'qualified_smoke': qualification, 'expected_queries':16966,
                          'expected_videos':4866, 'full_arguments':selected['arguments'],
                          'automatic_postprocessing':'strict completeness, primary scoring, first-turn diagnostic, Table 2 comparison JSON/CSV and experiment Markdown'},
                         ensure_ascii=False, indent=2))
        return
    if args.gpu is None or args.gpu<0:
        parser.error('--execute requires a nonnegative --gpu')
    execute(args.gpu, VARIANT, args.report, full=True)
    manifest = json.loads(args.report.read_text())
    try:
        from compare_activitynet_full_to_paper import compare
        manifest['paper_comparison_outputs'] = compare(args.report)
    except Exception as error:
        manifest['paper_comparison_outputs'] = {'status':'failed', 'error':str(error),
            'recovery':'Full predictions are preserved. Re-run offline comparison after fixing the error; do not rerun inference.'}
        raise
    finally:
        manifest['full_launcher_sha256'] = digest(Path(__file__))
        manifest['comparison_script_sha256'] = digest(ROOT/'scripts/compare_activitynet_full_to_paper.py')
        write_json(args.report, manifest)
        write_json(Path(manifest['output_dir'])/'run_manifest.json', manifest)
    print('[full] Inference, scoring and paper comparison completed.', flush=True)


if __name__ == '__main__':
    main()
