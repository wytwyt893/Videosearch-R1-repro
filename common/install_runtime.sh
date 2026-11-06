#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/env.sh"
# env.sh selects the validated project-local proxy; direct/inherit remain explicit options.
PY="$VIDEOSEARCH_WORKSPACE/envs/videosearchr1/bin/python"
export PIP_FIND_LINKS="$VIDEOSEARCH_WORKSPACE/cache/wheels"
[[ "$($PY -c 'import sys;print(sys.prefix)')" == "$VIDEOSEARCH_WORKSPACE/envs/videosearchr1" ]]
[[ "${CUDA_VISIBLE_DEVICES}" == '' ]]
cd "$VIDEOSEARCH_WORKSPACE/upstream"
FA_WHEEL="$VIDEOSEARCH_WORKSPACE/cache/wheels/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
if [[ ! -f "$FA_WHEEL" ]]; then
  echo 'STOP: obtain the complete official FlashAttention wheel first; .partial is not installable. No automatic source compilation.' >&2
  exit 4
fi
printf '%s  %s\n' '3807faf34d263ad0c3dff6f07c7ebdd27aef4a2614ef124db2afee4578ef7232' "$FA_WHEEL" | sha256sum -c -
printf '[stage] %s bootstrap\n' "$(date -Iseconds)"
"$PY" -m pip install --index-url https://pypi.org/simple --timeout 30 --retries 2 setuptools==80.9.0 wheel==0.45.1 packaging==25.0 ninja==1.13.0 psutil==7.2.1
printf '[stage] %s torch\n' "$(date -Iseconds)"
# The official Linux PyPI torch 2.8.0 wheel is CUDA 12.8 too. The assertion
# below validates it; avoiding an extra index prevents choosing the slow
# +cu128 download again instead of reusing official PyPI / cached wheels.
TORCH_SOURCE='torch==2.8.0'
TORCH_WHEEL="$VIDEOSEARCH_WORKSPACE/cache/wheels/torch-2.8.0-cp311-cp311-manylinux_2_28_x86_64.whl"
if [[ -f "$TORCH_WHEEL" ]]; then
  printf '%s  %s\n' 'c12fa219f51a933d5f80eeb3a7a5d0cbe9168c0a14bbb4055f1979431660879b' "$TORCH_WHEEL" | sha256sum -c -
  TORCH_SOURCE="$TORCH_WHEEL"
fi
"$PY" -m pip install --index-url https://pypi.org/simple --timeout 30 --retries 2 -c requirements.txt "$TORCH_SOURCE" torchvision==0.23.0 torchaudio==2.8.0
"$PY" -c 'import torch; assert torch.__version__.split("+")[0] == "2.8.0" and torch.version.cuda == "12.8" and torch._C._GLIBCXX_USE_CXX11_ABI and not torch.cuda.is_initialized()'
"$PY" -m pip install --no-deps "$FA_WHEEL"
printf '[stage] %s official_requirements\n' "$(date -Iseconds)"
"$PY" -m pip install --index-url https://pypi.org/simple --timeout 30 --retries 2 --no-build-isolation -c "$VIDEOSEARCH_WORKSPACE/scripts/runtime_constraints.txt" -r requirements.txt
"$PY" -m pip install --no-build-isolation --no-deps -e .
"$PY" -B "$VIDEOSEARCH_WORKSPACE/scripts/repair_decord.py"
"$PY" -m pip check
"$PY" -m pip freeze --all > "$VIDEOSEARCH_WORKSPACE/reports/environment.pip.freeze.txt"
printf '[complete] %s runtime_installed_no_gpu_model_test\n' "$(date -Iseconds)"
