#!/usr/bin/env bash
set -euo pipefail
cd /data/wyt/videosearch_r1_repro
source scripts/env.sh
exec python -B scripts/run_charades_localization_probe.py --gpu 1
