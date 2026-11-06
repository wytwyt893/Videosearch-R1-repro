#!/usr/bin/env python3
"""Restore an exact upstream shallow checkout when Git smart HTTPS is blocked.

Only GitHub's fixed-commit archive and Git Data API are accepted. This does NOT
create a research commit: imported commit bytes must hash to the published SHA.
"""
import datetime
import email.utils
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SHA = '69234fd9783b7235164d9f92b8903d9781dc284e'
REPO = ROOT / 'upstream'


def git(*args, data=None):
    return subprocess.check_output(['git', '-C', str(REPO), *args], input=data).decode().strip()


def commit_bytes(meta):
    # Git Data API normalizes dates to UTC and omits their serialized offset.
    # Accept reconstruction ONLY when the entire object hashes to the official
    # SHA. Both observed author/committer offsets resolve uniquely to +0900.
    def identity(role, zone):
        person = meta[role]
        epoch = int(datetime.datetime.fromisoformat(person['date'].replace('Z', '+00:00')).timestamp())
        return f"{role} {person['name']} <{person['email']}> {epoch} {zone}"
    for author_zone in ['+0900', '+0000', '-0000']:
        for committer_zone in ['+0900', '+0000', '-0000']:
            header = ['tree ' + meta['tree']['sha']]
            header += ['parent ' + p['sha'] for p in meta['parents']]
            header += [identity('author', author_zone), identity('committer', committer_zone)]
            for suffix in ['', '\n']:
                raw = ('\n'.join(header) + '\n\n' + meta['message'] + suffix).encode()
                digest = hashlib.sha1(b'commit ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
                if digest == SHA:
                    return raw
    raise ValueError('Cannot reconstruct exact published commit; do not invent a substitute SHA')


def main():
    meta = json.loads((ROOT / 'logs/01_upstream_git_commit.json').read_text())
    if meta['sha'] != SHA or meta['verification']['reason'] != 'unsigned':
        raise ValueError('Unexpected upstream commit metadata')
    raw = commit_bytes(meta)
    if REPO.exists():
        raise ValueError('Refusing to overwrite an existing checkout')
    archive = ROOT / 'cache/upstream_69234fd.tar.gz'
    with tarfile.open(archive) as package:
        for member in package.getmembers():
            parts = Path(member.name).parts
            if not parts or parts[0] != 'VideoSearch-R1-' + SHA or '..' in parts or Path(member.name).is_absolute():
                raise ValueError('Unsafe/unexpected archive member')
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError('Unexpected linked/special archive member')
    REPO.mkdir()
    subprocess.run(['tar', '-xzf', str(archive), '--strip-components=1', '--no-same-owner', '-C', str(REPO)], check=True)
    subprocess.run(['git', 'init', '--template=', '--initial-branch=main', str(REPO)], check=True)
    git('-c', 'core.autocrlf=false', '-c', 'core.attributesFile=/dev/null', 'add', '--force', '.')
    tree = git('write-tree')
    if tree != meta['tree']['sha']:
        raise ValueError('Archive worktree does not match published Git tree')
    if git('hash-object', '-t', 'commit', '-w', '--stdin', data=raw) != SHA:
        raise ValueError('Imported commit hash mismatch')
    # Explicit shallow boundary, exactly as a depth=1 fetch would provide.
    (REPO / '.git/shallow').write_text(SHA + '\n')
    git('update-ref', 'refs/heads/main', SHA)
    git('remote', 'add', 'origin', 'https://github.com/mlvlab/VideoSearch-R1.git')
    git('fsck', '--connectivity-only')
    if git('rev-parse', 'HEAD') != SHA or git('status', '--porcelain'):
        raise ValueError('Restored checkout is not clean')
    result = {'method': 'official_archive_plus_exact_git_object_restore_not_standard_clone',
              'commit_sha': SHA, 'tree_sha': tree, 'worktree_matches_official_tree': True,
              'commit_bytes_match_official_sha': True, 'clean': True,
              'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'new_research_commit_created': False, 'network_issue': 'Git smart HTTPS TLS termination; official API/codeload work'}
    (ROOT / 'logs/01_upstream_provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
