"""A full benchmark must retain invalid answers under the frozen VER protocol."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_activitynet_repair_smoke import validate_ver_coverage


class FullCoverageTests(unittest.TestCase):
    def test_full_retains_one_invalid_answer(self):
        validate_ver_coverage({'N':16965,'N_expected':16966,'invalid_answers':1,
                               'coverage':16965/16966}, smoke=False)

    def test_smoke_still_requires_all_answers_parseable(self):
        with self.assertRaisesRegex(ValueError, 'Unparseable'):
            validate_ver_coverage({'N':9,'N_expected':10,'invalid_answers':1,'coverage':.9}, smoke=True)

    def test_complete_smoke_passes(self):
        validate_ver_coverage({'N':10,'N_expected':10,'invalid_answers':0,'coverage':1}, smoke=True)

    def test_inconsistent_denominator_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_ver_coverage({'N':16965,'N_expected':16966,'invalid_answers':1,'coverage':1}, smoke=False)


if __name__ == '__main__':
    unittest.main()
