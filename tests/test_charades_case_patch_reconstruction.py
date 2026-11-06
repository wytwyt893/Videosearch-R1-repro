"""Check displayed frames against the real pinned HF patch processor."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_charades_localization_cases import recover_normalized_frames,repack_frames


class PatchReconstructionTests(unittest.TestCase):
    def test_inverse_recovers_actual_processor_frames_without_normalization(self):
        import torch
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        processor=Qwen3VLVideoProcessor(do_resize=False,do_rescale=False,do_normalize=False,
                                       patch_size=2,temporal_patch_size=2,merge_size=2)
        frames=(torch.arange(4*3*8*12).reshape(4,3,8,12)%251).to(torch.uint8)
        batch=processor(videos=[frames],do_sample_frames=False,return_tensors='pt')
        recovered=recover_normalized_frames(batch['pixel_values_videos'],batch['video_grid_thw'][0],2,2,2)
        self.assertTrue(torch.equal(frames,recovered))
        self.assertTrue(torch.equal(repack_frames(recovered,2,2,2),batch['pixel_values_videos']))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    unittest.main()
