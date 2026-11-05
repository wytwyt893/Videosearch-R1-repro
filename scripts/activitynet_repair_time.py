"""Explicit timestamp transport for the isolated repaired-data evaluation.

Qwen3-VL 4.57.1 and vLLM 0.11 both construct time labels as index/fps.
Their metadata schema has no explicit per-frame PTS field. Encode source PTS as
integer microsecond clock ticks, with a 1 MHz timebase, only for already sampled
NPYs with do_sample_frames=False. Physical source FPS/counts stay in the dataset
receipts; this transport is never represented as a physical video frame rate.
"""
from __future__ import annotations
import math

CLOCK_HZ = 1_000_000
FIELD = 'videosearch_source_timestamps_sec'


def timestamp_payload(row):
    values = row[FIELD]
    if not isinstance(values, list) or not values or any(
            not math.isfinite(float(x)) or x < 0 for x in values):
        raise ValueError('Invalid source timestamp list')
    if len(values) != row['output_num_frames'] or any(b < a for a, b in zip(values, values[1:])):
        raise ValueError('Timestamp length/order does not match the sampled NPY')
    ticks = [int(round(float(x)*CLOCK_HZ)) for x in values]
    duration = float(row['source_duration_sec'])
    if not math.isfinite(duration) or duration <= 0 or values[-1] > duration+1e-3:
        raise ValueError('Timestamp outside recorded video duration')
    return {
        # fps is only the sampled-array path's nominal sample_fps return value.
        # raw_fps and integer indices form the actual source-time clock.
        'fps': 1.0, 'raw_fps': float(CLOCK_HZ), 'frames_indices': ticks,
        'total_num_frames': max(ticks[-1]+1, int(math.ceil(duration*CLOCK_HZ))),
    }


def install():
    """Patch only the imported metadata parser in this dedicated process."""
    import utils.video_metadata as video_metadata
    original = video_metadata._extract_meta_payload
    if getattr(original, '_source_time_adapter', False):
        return

    def extract(row):
        if FIELD in row:
            return timestamp_payload(row)
        return original(row)

    extract._source_time_adapter = True
    video_metadata._extract_meta_payload = extract
