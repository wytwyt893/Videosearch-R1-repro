#!/usr/bin/env python3
"""Package an assistant qualitative text pass and published-NPY review."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw

from audit_charades_annotations import ROOT, DATA, CACHE, load, save, sha
from inspect_charades_localization_cases import font

# All 100 pairs below were read before this classification was recorded.
EXPECTED_RANDOM = [10,22,31,47,82,99,102,130,179,197,225,236,266,292,317,338,361,366,446,463,546,547,562,574,650,701,716,794,823,856,857,866,880,942,948,950,962,991,1116,1140,1153,1182,1266,1298,1331,1388,1476,1490,1535,1606,1641,1662,1696,1709,1751,1765,1782,1847,1851,1865,1932,1970,2058,2073,2113,2189,2233,2270,2320,2350,2369,2449,2455,2529,2545,2570,2657,2713,2744,2784,2838,2842,2864,2895,2924,2939,2990,2996,3110,3126,3168,3192,3208,3257,3416,3480,3599,3610,3646,3707]
RANDOM_NOTES = {
    47: ('apparent_target_change', 'Closing laptop becomes moving an open laptop onto sofa; closure not explicit.'),
    1782: ('apparent_target_change', 'Putting a picture back becomes lifting a picture; opposite manipulation phase.'),
    236: ('related_action_or_boundary_risk', 'Adds finishing drink before placing cup and interaction with companion after it.'),
    856: ('related_action_or_boundary_risk', 'Awakening expands to twitching, sitting up and reaching after awakening.'),
    1662: ('related_action_or_boundary_risk', 'Holding shoes expands to putting on shoe and holding a different object.'),
    1847: ('related_action_or_boundary_risk', 'Eating remains, but reading and sharing food with cat add interactions to verify.'),
    2113: ('related_action_or_boundary_risk', 'Adds seated scene and standing-up precursor to running away.'),
    2545: ('related_action_or_boundary_risk', 'Adds washing hands after looking out window; potential temporal expansion.'),
    2842: ('related_action_or_boundary_risk', 'Putting bag over head becomes holding it overhead; transition versus state.'),
    2864: ('related_action_or_boundary_risk', 'Opening closet adds extracting a bag; possible extension beyond opening.'),
    3110: ('related_action_or_boundary_risk', 'Holding book specializes into opening/reading it; requires interval verification.'),
    3126: ('related_action_or_boundary_risk', 'Running into bedroom adds stopping and unpacking afterwards.'),
    3707: ('related_action_or_boundary_risk', 'Turning on light adds remaining stationary afterwards.'),
    2058: ('wording_ambiguity', 'Closing a closed door is awkward wording; target closure retained, visual interpretation unresolved.'),
}
RISK_NOTES = {
    336: ('apparent_target_change', 'Running past someone becomes entering/exploring closet while boy stays seated.'),
    677: ('same_target_in_text', 'Rising from seated couch position paraphrases standing up; regex false alarm.'),
    1616: ('related_action_or_boundary_risk', 'Eating becomes sandwich held at mouth; state may support same event, not proven shift.'),
    1628: ('apparent_target_change', 'Eating sandwich becomes sipping water.'),
    2293: ('apparent_target_change', 'Walking out becomes entering bathroom and preparing to clean.'),
    2571: ('apparent_target_change', 'Running away becomes throwing paper, bending, standing and laughing.'),
    2630: ('related_action_or_boundary_risk', 'Rising paraphrases standing up but adds grabbing bottle afterwards.'),
    2696: ('apparent_target_change', 'Laughing becomes focused eating.'),
    2724: ('apparent_target_change', 'Removing phone disappears, replaced by sitting/sipping/observing; coarse pronouns also ambiguous.'),
    2882: ('same_target_in_text', 'Savoring sandwich/bringing it to mouth paraphrases eating; regex false alarm.'),
    3003: ('apparent_target_change', 'Sitting on floor becomes crouching and attempting to open door.'),
    3202: ('same_target_in_text', 'Consuming flatbread preserves eating; writing is added concurrent detail.'),
    3555: ('same_target_in_text', 'Capturing image preserves taking picture; opening is less explicit.'),
    309: ('same_target_in_text', 'Taking bites paraphrases eating; regex false alarm.'),
    1697: ('apparent_target_change', 'Opening door becomes typing on laptop.'),
    811: ('same_target_in_text', 'Activating light bulb paraphrases turning on light; regex false alarm.'),
    1257: ('same_target_in_text', 'Laughter paraphrases laughing; regex false alarm.'),
    2115: ('related_action_or_boundary_risk', 'Watching NBA remains implied, but eventually leaving may extend event boundary.'),
    3362: ('same_target_in_text', 'Rising from seated position paraphrases standing up; regex false alarm.'),
    204: ('apparent_target_change', 'Beginning eating becomes interacting with tablet/book.'),
}
VISUAL_SELECTION = [47, 1782, 1628, 1697, 2293]

VISUAL_FINDINGS = {
    47: dict(category='compatible_event_expansion',
             observation='Published frames show laptop upright around 10.55s, lowered/closed around 11.59s, then placed on sofa around 13.72s, all within/near the original 6.7-16.4s label.',
             conclusion='Text difference alone is not enough to call an annotation error; both captions describe a connected sequence.'),
    1782: dict(category='strong_fine_query_temporal_mismatch',
               observation='The picture is taken from the table around 15.54-16.58s, held afterwards, and moved back toward the table around 29.04-30.08s. GT 25.7-32.0s is the return phase, but FIG asks about lifting.',
               conclusion='Strong evidence of a manipulation-phase reversal in the fine query paired with inherited return-phase GT. No replacement interval is assigned.'),
    1628: dict(category='cooccurring_actions_not_resolved_by_text',
               observation='Inside GT 13.8-23.5s, a cup is at the mouth around 14.17s and food is brought to the mouth around 18.33-20.33s.',
               conclusion='Eating and drinking coexist. The action-word difference does not establish that the fine query is false in the labeled clip.'),
    1697: dict(category='coarse_caption_or_source_alignment_suspect',
               observation='Door opening appears near 0.98-1.96s; the late GT 16.8-30.4s mostly shows laptop interaction. All three original coarse rows for this video refer to door opening under that same late interval.',
               conclusion='Fine caption fits the published late frames better than coarse caption. Could reflect original coarse/time annotation or source alignment problems; raw video has not been independently verified. Do not force FIG back to coarse wording.'),
    2293: dict(category='strong_fine_query_direction_mismatch',
               observation='At the end of the clip, the person rises from the toilet around 43.83s and moves toward/out of the doorway around 44.83-47.0s. GT 43.3-48.0s is labeled walking out in original STA; fine query says entering and preparing to clean.',
               conclusion='Strong evidence of exit/entry reversal and unsupported preparation-to-clean detail in the fine query; preserve official labels for evaluation.'),
    81: dict(category='target_change_with_temporal_scope_risk',
             observation='Original STA has sitting at table at 6.4-11.1s and pouring at 8.9-23.2s. Existing processor-RGB review shows pouring posture at 15.43s, beyond the sitting label. Some pouring also overlaps the sitting label.',
             conclusion='The sitting-row FIG query foregrounds pouring but inherits the sitting interval. Strong scope-mismatch risk, not proof that every frame inside GT lacks pouring.'),
}


def finalize(directory):
    summary = json.loads((directory / 'text_review_summary.json').read_text())
    annotations = load(DATA / 'raw_annotation/test.jsonl')
    upper_bounds = []
    for i, row in enumerate(annotations):
        start, end = row['time']
        maximum = max(0., min(end, row['duration']) - max(start, 0.)) / (end - start)
        if maximum < .7:
            upper_bounds.append(dict(index=i, video=row['video'], gt=row['time'],
                                     duration=row['duration'], maximum_possible_iou=maximum))
    save(directory / 'duration_iou_upper_bounds.json', dict(
        condition='Official parser clamps both predicted endpoints to [0, annotation_duration], while GT remains unchanged.',
        formula='length(GT intersect [0,duration]) / length(GT)',
        count_below_point7=len(upper_bounds), rows=upper_bounds,
        note='Structural bound only, not an observed score improvement or a proposed metric change.'))
    save(directory / 'visual_findings.json', VISUAL_FINDINGS)
    # Compact, static evidence; actual published pixels, display resizing only.
    width, row_h = 1440, 550
    sheet = Image.new('RGB', (width, row_h * 2 + 65), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 10), 'Annotation audit | published NPY frames | source times', fill='black', font=font(24))
    draw.text((15, 42), 'Diagnostic evidence only. Original GT and all official evaluation results are unchanged.', fill='#555555', font=font(16))
    display_cases = [
        (1782, 'URAIM', [16, 27, 29], ['Picture held after pickup', 'Picture still held', 'Picture returned to table']),
        (1697, 'W8CWW', [1, 18, 27], ['Door interaction near start', 'Laptop interaction inside GT', 'Laptop interaction inside GT']),
    ]
    evidence = []
    for row_number, (i, video, indices, notes) in enumerate(display_cases):
        folder = directory / f'case_{i:06d}_{video}'
        case = json.loads((folder / 'case.json').read_text())
        array = np.load(DATA / case['metadata']['output_npy_path'], allow_pickle=False)
        y = 70 + row_number * row_h
        draw.text((15, y), f'Index {i} / {video} / GT {annotations[i]["time"]}', fill='black', font=font(22))
        title = 'COARSE: return picture | FINE: lift picture' if i == 1782 else 'COARSE: open door | FINE: use laptop'
        draw.text((15, y + 32), title, fill='#a137aa', font=font(20))
        for j, (frame_index, note) in enumerate(zip(indices, notes)):
            x = 15 + j * 475
            im = Image.fromarray(array[frame_index])
            im.thumbnail((450, 390), Image.Resampling.LANCZOS)
            sheet.paste(im, (x + (450 - im.width) // 2, y + 70))
            time = case['metadata']['sampled_timestamps_sec'][frame_index]
            draw.text((x, y + 470), f'frame {frame_index} / source {time:.2f}s', fill='black', font=font(18))
            draw.text((x, y + 500), note, fill='black', font=font(17))
            evidence.append(dict(index=i, frame_index=frame_index, source_seconds=time, manual_observation=note))
    figure = directory / 'audit_evidence.png'
    sheet.save(figure)
    # Bind existing full predictions without rescoring or rewriting them.
    run = json.loads((ROOT / 'reports/44_charades_timestamp_full.json').read_text())
    for name, condition in run['conditions'].items():
        assert sha(condition['predictions']) == condition['prediction_sha256'], name
    assert sha(DATA / 'raw_annotation/test.jsonl') == '6b7c4bf0b63048a6cf1dffac5a2e1cd18af7df154bead0b2b99ec72311155fb0'
    from run_charades_timestamp_full import verify
    verify(ROOT / 'reports/43_charades_timestamp_full_cpu.json')
    structural = json.loads((directory / 'structural.json').read_text())
    report = dict(status='INITIAL_ANNOTATION_AUDIT_COMPLETE', output=str(directory),
                  provenance=summary['provenance'], structural=structural,
                  text_review_counts=summary['counts'], visual_findings=VISUAL_FINDINGS,
                  duration_iou_upper_bound_below_point7=len(upper_bounds),
                  evidence_figure=str(figure), evidence_frames=evidence,
                  evidence_figure_sha256=sha(figure),
                  model_execution=False, gpu_access=False, official_data_modified=False,
                  frozen_full_inputs_and_prediction_hashes_unchanged=True,
                  limits=summary['limits'] + [
                      'Visual review covers five newly rendered cases across four videos plus existing case 81, not all 100 random cases.',
                      'Strong visual findings are based on published sampled NPYs, not independent original-video/PTS decoding.',
                      'Original TALL Drive source is unavailable; primary 2D-TAN preserved copy and original Charades CSV provide cross-checks.',
                      'Risk sample is purposive and cannot estimate population annotation error prevalence.',
                  ],
                  review_script_sha256=sha(Path(__file__)),
                  artifact_sha256={p.name: sha(p) for p in directory.glob('*.json') if p.name != 'summary.json'})
    save(directory / 'summary.json', report)
    save(ROOT / 'reports/46_charades_annotation_audit.json', report)
    print(json.dumps(dict(status=report['status'], report='reports/46_charades_annotation_audit.json',
                          source_video_time_match=report['provenance']['original_video_and_time_multiset_equal'],
                          theoretical_point7_impossible=len(upper_bounds)), ensure_ascii=False))


def original_rows():
    result = []
    for number, line in enumerate((CACHE / 'charades_sta_test_2dtan.txt').read_text().splitlines(), 1):
        header, text = line.split('##', 1)
        video, start, end = header.split()
        result.append(dict(line=number, video=video, time=[float(start), float(end)], cog_desc=text))
    return result


def key(row):
    return row['video'], *row['time'], row['cog_desc']


def render(row, meta, folder):
    folder.mkdir(exist_ok=True)
    path = DATA / meta['output_npy_path']
    before = sha(path)
    array = np.load(path, allow_pickle=False)
    assert array.dtype == np.uint8 and array.shape[-1] == 3
    times = meta['sampled_timestamps_sec']
    assert len(array) == len(times)
    width, cell_w, cell_h, header = 1560, 260, 190, 215
    sheet = Image.new('RGB', (width, header + math.ceil(len(array) / 6) * cell_h), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f'Index {row["index"]} / {row["video"]} | released NPY RGB | source times', fill='black', font=font(23))
    for y, label, text in [(45, 'COARSE', row['cog_desc']), (90, 'FINE', row['fig_desc'])]:
        for j, line in enumerate(textwrap.wrap(label + ': ' + text, 140)):
            draw.text((12, y + 22 * j), line, fill='black', font=font(16))
    start, end = row['time']
    draw.text((12, 155), f'Official GT {start:g}-{end:g}s; green captions mark sampled frames inside GT.', fill='#258445', font=font(18))
    draw.text((12, 184), 'No model generation. Pixels resized for display only; no new labels.', fill='#555555', font=font(15))
    for i, frame in enumerate(array):
        x, y = (i % 6) * cell_w, header + (i // 6) * cell_h
        im = Image.fromarray(frame)
        im.thumbnail((cell_w - 8, 140), Image.Resampling.LANCZOS)
        sheet.paste(im, (x + (cell_w - im.width) // 2, y))
        color = '#258445' if start <= times[i] <= end else '#444444'
        draw.text((x + 5, y + 145), f'i={i:02d} / {times[i]:.2f}s', fill=color, font=font(16))
        if start <= times[i] <= end:
            draw.line((x + 5, y + 173, x + cell_w - 5, y + 173), fill='#258445', width=4)
    sheet.save(folder / 'all_frames.png')
    assert sha(path) == before
    evidence = dict(index=row['index'], annotation=row, metadata=meta, npy_sha256=before,
                    image=str(folder / 'all_frames.png'), image_sha256=sha(folder / 'all_frames.png'),
                    display='Published uint8 NPY RGB, resized only; source timestamps from released metadata. Not a reconstruction of the inference processor patch tensor.')
    save(folder / 'case.json', evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.finalize:
        finalize(directory)
        return
    selection = json.loads((directory / 'selection.json').read_text())
    assert selection['random_indices'] == EXPECTED_RANDOM
    assert set(r['index'] for r in selection['risk_candidates']) == set(RISK_NOTES)
    review_rows = load(directory / 'review_rows.jsonl')
    reviews = []
    for row in review_rows:
        if row['group'] == 'random':
            category, note = RANDOM_NOTES.get(row['index'], ('same_target_in_text', 'Main target event is retained/paraphrased; added visual details not verified by text review.'))
        elif row['group'] == 'risk':
            category, note = RISK_NOTES[row['index']]
        else:
            category, note = 'apparent_target_change', 'Known case: sitting at table becomes pouring; shared original GT.'
        reviews.append(dict(group=row['group'], index=row['index'], category=category, note=note,
                            scope='Assistant qualitative text comparison; not a human annotation panel, confirmed visual error or alternative GT.'))
    save(directory / 'text_reviews.json', reviews)
    visual_selection = directory / 'visual_selection.json'
    if not visual_selection.exists():
        save(visual_selection, dict(indices=VISUAL_SELECTION[:4], selected_before_visual_review=True,
             rule='Both apparent target changes from the random sample, plus eating-to-drinking and door-to-laptop changes from the independent risk sample; no predictions used.'))
    save(directory / 'visual_extension.json', dict(indices=[2293],
         selection_after_initial_visual_review=True,
         reason='Initial VAV4C sheet also exposes the already text-flagged entering/leaving reversal at the end; additional diagnostic case, not a random sample.'))
    annotations = load(DATA / 'raw_annotation/test.jsonl')
    originals = original_rows()
    counts = Counter(key(r) for r in originals)
    local_counts = Counter(key(r) for r in annotations)
    missing = list((local_counts - counts).elements())
    extra = list((counts - local_counts).elements())
    metadata = {r['video_id']: r for r in load(DATA / 'test/video_npy_with_meta/meta.jsonl')}
    visual = []
    by_index = {r['index']: r for r in review_rows}
    for i in VISUAL_SELECTION:
        row = by_index[i]
        visual.append(render(row, metadata[row['video']], directory / f'case_{i:06d}_{row["video"]}'))
    with (CACHE / 'charades_v1_test_2dtan.csv').open() as handle:
        csv_rows = {r['id']: r for r in csv.DictReader(handle)}
    original_span_counts = Counter((r['video'], *r['time']) for r in originals)
    local_span_counts = Counter((r['video'], *r['time']) for r in annotations)
    def without_parentheses(k):
        return *k[:3], k[3].replace('(', '').replace(')', '')
    normalized_match = Counter(without_parentheses(key(r)) for r in originals) == Counter(without_parentheses(key(r)) for r in annotations)
    provenance = dict(
        original_records=len(originals), original_match_key='(video_id, gt_start, gt_end, exact coarse text); multiset comparison, not row number/desc_id',
        original_multiset_equal=not missing and not extra, missing_local_tuples=missing,
        extra_original_tuples=extra,
        original_video_and_time_multiset_equal=original_span_counts == local_span_counts,
        original_exact_tuple_matches=len(annotations) - len(missing),
        original_tuples_match_after_parenthesis_removal=normalized_match,
        normalization_note='Comparison only: one coarse sentence loses an opening parenthesis in VERIFIED; source files are unchanged.',
        original_text_sha256=sha(CACHE / 'charades_sta_test_2dtan.txt'),
        original_mirror_revision='1744b2159c3d11694f94dab4fcde8660c685e79a',
        original_mirror_note='2D-TAN official implementation copy; TALL official Drive URL returned 404. Not a fresh direct download from TALL.',
        known_case_81_original_records=[r for r in originals if r['video'] == 'EYZXC'],
        known_case_81_charades_csv=csv_rows['EYZXC'],
        csv_sha256=sha(CACHE / 'charades_v1_test_2dtan.csv'))
    save(directory / 'provenance.json', provenance)
    report = dict(status='TEXT_REVIEW_COMPLETE_VISUAL_REVIEW_PENDING',
                  counts={g: dict(Counter(r['category'] for r in reviews if r['group'] == g)) for g in ['random', 'risk', 'known']},
                  provenance=provenance, visual=visual,
                  selection_sha256=sha(directory / 'selection.json'),
                  limits=['Text flags are not confirmed annotation errors.', 'No prevalence estimate from risk-selected cases.',
                          'No added visual detail was validated for all 100 random cases.',
                          'Known 2233 independently appears in the frozen random sample and is retained.',
                          'No test GT, scores, model weights or configurations changed; no GPU.'])
    save(directory / 'text_review_summary.json', report)
    print(json.dumps(dict(counts=report['counts'], original_multiset_equal=provenance['original_multiset_equal'],
                          rendered=[v['index'] for v in visual], output=str(directory)), ensure_ascii=False))


if __name__ == '__main__':
    main()
