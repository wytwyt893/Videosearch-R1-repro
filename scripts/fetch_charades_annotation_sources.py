#!/usr/bin/env python3
"""Download small public annotation evidence into an isolated cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'cache/annotation_audit/20261106'
REV = 'a22cc63c6f6bd26493ff2b35718bb93e81817271'
SOURCES = [
    ('verified_charades_fig_test.jsonl',
     f'https://raw.githubusercontent.com/hlchen23/VERIFIED/{REV}/fine-grained-anno/charades-fig/charades_fig_test.jsonl',
     'ffff62d5a17927b0db75d1f6829f7aa1e9d060d1'),
    ('verified_dataset_document.pdf',
     f'https://raw.githubusercontent.com/hlchen23/VERIFIED/{REV}/fine-grained-anno/dataset_document.pdf',
     '6f5010442e2a507ca594f9b51a9e59f772122148'),
    ('charades_sta_test_original.txt',
     'https://drive.google.com/uc?export=download&id=1QG4MXFkoj6JFU0YK5olTY75xTARKSW5e', None),
    ('charades_sta_test_2dtan.txt',
     'https://raw.githubusercontent.com/microsoft/VideoX/1744b2159c3d11694f94dab4fcde8660c685e79a/2D-TAN/data/Charades-STA/charades_sta_test.txt',
     '9268d0e7cfeebea4927903a8692fe86fbd5c3f1c'),
    ('charades_v1_test_2dtan.csv',
     'https://raw.githubusercontent.com/microsoft/VideoX/1744b2159c3d11694f94dab4fcde8660c685e79a/2D-TAN/data/Charades-STA/Charades_v1_test.csv',
     '664527bd18e7f68258703e634855631efb31ecbe'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', action='append', choices=[s[0] for s in SOURCES])
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    receipt_path = CACHE / 'receipts.json'
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else []
    for name, url, expected in SOURCES:
        if args.only and name not in args.only:
            continue
        path = CACHE / name
        try:
            if path.exists():
                data = path.read_bytes()
                final_url = url
            else:
                response = requests.get(url, timeout=(15, 60))
                response.raise_for_status()
                data, final_url = response.content, response.url
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if expected and blob != expected:
                raise ValueError('Public Git blob hash mismatch')
            if name.endswith('.txt') and (b'##' not in data or b'<html' in data[:1000].lower()):
                raise ValueError('Response is not a Charades-STA annotation text')
            if name.endswith('.pdf') and not data.startswith(b'%PDF'):
                raise ValueError('Response is not PDF')
            path.write_bytes(data)
            row = dict(path=str(path), url=url, final_url=final_url, size=len(data),
                       sha256=hashlib.sha256(data).hexdigest(), git_blob_sha1=blob,
                       expected_blob_sha1=expected, status='verified' if expected else 'public_original')
        except Exception as error:
            row = dict(path=str(path), url=url, status='failed', error=str(error))
        receipts = [r for r in receipts if r['path'] != str(path) and r['path'] != str(path.relative_to(ROOT))]
        receipts.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (CACHE / 'receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
