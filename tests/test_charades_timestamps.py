"""The clock intervention must retain source time and fixed paired cohorts."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from charades_timestamp_eval import source_payload
from check_charades_timestamps import select_indices


class CharadesTimestampTests(unittest.TestCase):
    def row(self):
        return {'sampled_timestamps_sec':[0,31/30,748/30,1029/30],
                'sampled_indices_src':[0,31,748,1029],'raw_fps':30,
                'output_num_frames':4,'source_duration_sec':1030/30,'source_total_frames':1030}

    def test_original_source_frame_time_survives_transport(self):
        r=self.row();p=source_payload(r)
        for tick,t in zip(p['frames_indices'],r['sampled_timestamps_sec']):
            self.assertAlmostEqual(tick/p['raw_fps'],t,places=6)
        self.assertEqual(r['sampled_indices_src'],[0,31,748,1029])

    def test_rejects_missing_or_inconsistent_time_information(self):
        for field,value in [('sampled_timestamps_sec',[0,1,24,33]),('raw_fps',0),('output_num_frames',5)]:
            row=self.row();row[field]=value
            with self.assertRaises(ValueError):source_payload(row)

    def test_diagnostic_and_random_cohorts_are_frozen_and_disjoint(self):
        c=select_indices();self.assertEqual(c,select_indices())
        a,b=map(set,c.values())
        self.assertEqual((len(a),len(b),len(a|b)),(6,64,70))
        self.assertFalse(a&b)


if __name__=='__main__':unittest.main()
