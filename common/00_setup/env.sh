#!/usr/bin/env bash
# Source bridge for the unchanged historical installer reached via a symlink.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)/scripts/env.sh"
