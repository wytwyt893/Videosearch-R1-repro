#!/usr/bin/env python3
"""Read-only provenance, structural checks and frozen text-review selection.

Action regexes are triage rules, not error labels. No model execution, inference
configuration changes, relabeling, test exclusions or Markdown output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/charades-sta'
CACHE = ROOT / 'cache/annotation_audit/20261106'
SEED = 20261106
# Inflected words intentionally remain transparent. They do not distinguish
# main verbs from attributes, support verbs, negation, or nested descriptions.
ACTION_RULES = {
    'sit': r'sit|sits|sitting|sat|seated', 'stand': r'stand|stands|standing|stood',
    'walk': r'walk|walks|walking|walked', 'run': r'run|runs|running|ran|jog|jogging',
    'drink': r'drink|drinks|drinking|drank|sip|sips|sipping',
    'pour': r'pour|pours|pouring|poured', 'eat': r'eat|eats|eating|ate|chew|chewing',
    'open': r'open|opens|opening|opened', 'close': r'close|closes|closing|closed|shut|shuts|shutting',
    'hold': r'hold|holds|holding|held|carry|carries|carrying',
    'put': r'put|puts|putting|place|places|placing|placed',
    'take': r'take|takes|taking|took|grab|grabs|grabbing|pick|picks|picking',
    'look': r'look|looks|looking|looked|watch|watches|watching|stare|stares|staring',
    'read': r'read|reads|reading', 'talk': r'talk|talks|talking|speak|speaks|speaking',
    'wash': r'wash|washes|washing|rinse|rinses|rinsing',
    'clean': r'clean|cleans|cleaning|wipe|wipes|wiping|scrub|scrubs|scrubbing',
    'cook': r'cook|cooks|cooking|stir|stirs|stirring',
    'undress': r'undress|undresses|undressing', 'dress': r'dress|dresses|dressing',
    'sneeze': r'sneeze|sneezes|sneezing', 'laugh': r'laugh|laughs|laughing',
    'smile': r'smile|smiles|smiling', 'cry': r'cry|cries|crying',
    'lie': r'lie|lies|lying|lay|lays|laying', 'sleep': r'sleep|sleeps|sleeping|asleep',
    'throw': r'throw|throws|throwing|threw|toss|tosses|tossing|discard|discards|discarding',
    'tidy': r'tidy|tidies|tidying|organize|organizes|organizing|arrange|arranges|arranging',
    'fold': r'fold|folds|folding', 'fix': r'fix|fixes|fixing|repair|repairs|repairing',
    'turn': r'turn|turns|turning|turned|switch|switches|switching',
    'touch': r'touch|touches|touching', 'reach': r'reach|reaches|reaching',
    'bend': r'bend|bends|bending|bent|lean|leans|leaning',
    'stretch': r'stretch|stretches|stretching', 'exercise': r'exercise|exercises|exercising',
    'play': r'play|plays|playing', 'brush': r'brush|brushes|brushing|comb|combs|combing',
    'write': r'write|writes|writing|wrote', 'type': r'type|types|typing',
    'push': r'push|pushes|pushing', 'pull': r'pull|pulls|pulling',
    'lift': r'lift|lifts|lifting|raise|raises|raising',
    'drop': r'drop|drops|dropping|dropped', 'fall': r'fall|falls|falling|fell',
    'enter': r'enter|enters|entering|entered', 'leave': r'leave|leaves|leaving|left|exit|exits|exiting',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def tags(text):
    return {name for name, rule in ACTION_RULES.items()
            if re.search(r'\b(?:' + rule + r')\b', text.lower())}


def prepare(output):
    source = DATA / 'raw_annotation/test.jsonl'
    upstream = CACHE / 'verified_charades_fig_test.jsonl'
    rows, official = load(source), load(upstream)
    assert source.read_bytes() == upstream.read_bytes()
    assert len(rows) == 3720
    metadata = {r['video_id']: r for r in load(DATA / 'test/video_npy_with_meta/meta.jsonl')}
    query_meta = {str(r['qid']): r for r in load(DATA / 'test/query_embedding/query_meta.test.jsonl')}
    random_indices = sorted(random.Random(SEED).sample(range(len(rows)), 100))
    candidates = []
    overruns = []
    for i, row in enumerate(rows):
        assert row == official[i]
        before, after = tags(row['cog_desc']), tags(row['fig_desc'])
        if before and after and not before.intersection(after):
            words_a = set(re.findall(r'[a-z]+', row['cog_desc'].lower()))
            words_b = set(re.findall(r'[a-z]+', row['fig_desc'].lower()))
            jaccard = len(words_a & words_b) / max(1, len(words_a | words_b))
            candidates.append(dict(index=i, coarse_tags=sorted(before), fine_tags=sorted(after),
                                   lexical_jaccard=jaccard))
        start, end = row['time']
        delta = end - row['duration']
        if delta > 0:
            overruns.append(dict(index=i, video=row['video'], gt=[start, end],
                                 annotation_duration=row['duration'],
                                 metadata_source_duration=metadata[row['video']]['source_duration_sec'],
                                 annotation_overrun_sec=delta,
                                 source_overrun_sec=end - metadata[row['video']]['source_duration_sec']))
    candidates.sort(key=lambda c: (c['lexical_jaccard'], c['index']))
    risk = [c for c in candidates if c['index'] not in random_indices and c['index'] != 81][:20]
    selection = dict(seed=SEED, random_indices=random_indices, random_count=len(random_indices),
                     known_diagnostic_indices=[81], risk_candidates=risk,
                     risk_rule='Both descriptions have recognized action words but their regex tag sets are disjoint; ascending lexical Jaccard, then original index; exclude random/known case.',
                     all_rule_candidate_count=len(candidates), selected_before_manual_review=True,
                     selection_is_not_model_prediction_dependent=True,
                     limits='Triage only, not automatic semantic labels or confirmed annotation errors. Random and risk samples must be reported separately.')
    output.mkdir(parents=True)
    save(output / 'selection.json', selection)
    structural = dict(
        records=len(rows), videos=len(metadata),
        both_descriptions=sum(bool(r.get('cog_desc')) and bool(r.get('fig_desc')) for r in rows),
        qid_query_text_mismatches=sum(query_meta.get(str(r['desc_id']) + '_fig', {}).get('query') != r['fig_desc'] for r in rows),
        nonfinite_or_invalid_spans=sum(not all(math.isfinite(float(v)) for v in r['time']) or not 0 <= r['time'][0] < r['time'][1] for r in rows),
        gt_end_beyond_annotation_duration=len(overruns),
        overrun_videos=len({r['video'] for r in overruns}),
        overrun_count_by_threshold={str(t): sum(r['annotation_overrun_sec'] > t for r in overruns) for t in [0, .1, .5, 1, 2]},
        max_annotation_overrun_sec=max(r['annotation_overrun_sec'] for r in overruns),
        note='Duration checks identify source-label inconsistencies, not semantic errors; original boundary rounding/annotation policy may contribute.')
    save(output / 'structural.json', structural)
    save(output / 'duration_overruns.json', overruns)
    review_rows = []
    for group, indices in [('random', random_indices), ('risk', [r['index'] for r in risk]), ('known', [81])]:
        for i in indices:
            review_rows.append(dict(group=group, index=i, **rows[i],
                                    coarse_tags=sorted(tags(rows[i]['cog_desc'])), fine_tags=sorted(tags(rows[i]['fig_desc']))))
    (output / 'review_rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in review_rows))
    report = dict(status='PREPARED_FOR_MANUAL_REVIEW', output=str(output), structural=structural,
                  local_equals_upstream_bytes=True, verified_revision='a22cc63c6f6bd26493ff2b35718bb93e81817271',
                  bindings={str(p.relative_to(ROOT)): sha(p) for p in [source, upstream, Path(__file__), ROOT / 'reports/43_charades_timestamp_full_cpu.json', ROOT / 'reports/44_charades_timestamp_full.json']},
                  model_execution=False, gpu_access=False, official_data_modified=False)
    save(output / 'prepare.json', report)
    print(json.dumps(dict(output=str(output), structural=structural, risk_selected=[r['index'] for r in risk],
                          rule_candidates=len(candidates)), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / ('outputs/charades_annotation_audit_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f'))
    prepare(output)


if __name__ == '__main__':
    main()
