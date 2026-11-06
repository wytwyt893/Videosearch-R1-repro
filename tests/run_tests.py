#!/usr/bin/env python3
"""Run classified regression tests once, without collecting legacy aliases."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('common', 'reproduction', 'optimization')
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=GROUPS, help='Run one classified group')
    args = parser.parse_args()
    suite = unittest.TestSuite()
    modules = []
    for group in (args.group,) if args.group else GROUPS:
        for path in sorted((ROOT / 'tests' / group).rglob('test_*.py')):
            name = '_classified_' + '_'.join(path.relative_to(ROOT / 'tests').with_suffix('').parts)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
            modules.append(str(path.relative_to(ROOT)))
    if not modules:
        raise ValueError('No classified tests found')
    print(f'Classified discovery: {len(modules)} modules, {suite.countTestCases()} test cases', flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
