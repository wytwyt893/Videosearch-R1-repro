#!/usr/bin/env python3
"""Repair the upstream universal wheel's incorrect internal cp36 tag, payload unchanged."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
with urllib.request.urlopen('https://pypi.org/pypi/decord/0.6.0/json', timeout=30) as response:
    metadata = json.load(response)
filename = 'decord-0.6.0-py3-none-manylinux2010_x86_64.whl'
official = next(row for row in metadata['urls'] if row['filename'] == filename)
original = ROOT/'cache/wheels'/filename
if not original.exists():
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--silent', '--show-error',
                    '--output', str(original), official['url']], check=True)
with original.open('rb') as handle:
    assert hashlib.file_digest(handle, 'sha256').hexdigest() == official['digests']['sha256']
repaired = ROOT/'cache/wheels/repacked/decord-0.6.0-1py311fix-py3-none-manylinux2010_x86_64.whl'
repaired.parent.mkdir(exist_ok=True)
wheel_path = 'decord-0.6.0.dist-info/WHEEL'
record_path = 'decord-0.6.0.dist-info/RECORD'
changed = []
with zipfile.ZipFile(original) as source, zipfile.ZipFile(repaired, 'w', zipfile.ZIP_DEFLATED) as output:
    original_wheel = source.read(wheel_path).decode()
    assert 'Tag: cp36-cp36m-manylinux2010_x86_64' in original_wheel
    fixed_wheel = original_wheel.replace('Tag: cp36-cp36m-manylinux2010_x86_64',
                                         'Tag: py3-none-manylinux2010_x86_64').encode()
    record = []
    for info in source.infolist():
        if info.filename == record_path:
            continue
        data = fixed_wheel if info.filename == wheel_path else source.read(info.filename)
        output.writestr(info, data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        record.append([info.filename, 'sha256='+digest, str(len(data))])
    record.append([record_path, '', ''])
    text = io.StringIO()
    csv.writer(text, lineterminator='\n').writerows(record)
    output.writestr(record_path, text.getvalue())
with zipfile.ZipFile(original) as source, zipfile.ZipFile(repaired) as output:
    assert set(source.namelist()) == set(output.namelist())
    for name in source.namelist():
        if source.read(name) != output.read(name):
            changed.append(name)
    assert sorted(changed) == sorted([wheel_path, record_path])
    native = [name for name in source.namelist() if name.endswith('.so')]
    assert native == ['decord/libdecord.so'], native
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', str(repaired)], check=True)
fixture = ROOT/'cache/tmp/decord_cpu_fixture.mp4'
subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                'color=c=blue:s=64x64:r=2:d=1', '-c:v', 'mpeg4', str(fixture)], check=True)
import decord
reader = decord.VideoReader(str(fixture), ctx=decord.cpu(0), num_threads=1)
frames = reader.get_batch([0, 1]).asnumpy()
assert frames.shape == (2, 64, 64, 3) and str(frames.dtype) == 'uint8'
result = {'official_url': official['url'], 'original_sha256': official['digests']['sha256'],
          'repacked_wheel': str(repaired), 'changed_members': changed,
          'native_payload_sha256': hashlib.sha256(zipfile.ZipFile(original).read(native[0])).hexdigest(),
          'binary_and_python_payloads_unchanged': True, 'decord_version': decord.__version__,
          'cpu_decode_shape': list(frames.shape), 'cpu_decode_passed': True,
          'reason': 'Published filename is py3-none, but internal WHEEL incorrectly says cp36-cp36m; only libdecord.so is native, loaded through ctypes.'}
(ROOT/'reports/03_decord_repair.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
