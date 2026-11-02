"""Safety checks for contact moves and human frame precedence."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # prototype root, after move to tests/
import unittest
from unittest.mock import patch
import numpy as np

import contact_timing
import hit_refinement


class ContactFrameTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(90)/30
        self.features = np.zeros((90, 2, 55))
        self.features[:, :, 22] = .8
        self.features[:, :, 27] = 1
        self.event = {'id': 'event', 'player': 'near', 'peak_time': 1., 'frame_index': 30}
        self.dense = {'id': 'job', 'source_sha256': 'source'}

    def select(self, frame, **kwargs):
        _, pool, _ = contact_timing.event_inputs(self.times, self.features, self.event)
        values = np.zeros(len(pool)); values[np.flatnonzero(pool == frame)[0]] = 1
        with patch.object(contact_timing, 'score', return_value=values):
            return contact_timing.choose_frame(self.times, self.features, self.event, {}, **kwargs)

    def test_supported_small_move(self):
        result = self.select(33)
        self.assertTrue(result['applied'])
        self.assertEqual(result['suggested_frame'], 33)

    def test_unsupported_motion_peak_keeps_current_frame(self):
        self.features[33, 0, 22] = 0
        result = self.select(33)
        self.assertFalse(result['applied'])
        self.assertEqual(result['suggested_frame'], 30)
        self.assertEqual(result['alternative_frame'], 33)

    def test_large_and_boundary_moves_are_review_only(self):
        result = self.select(42)
        self.assertFalse(result['applied'])
        self.assertEqual(result['suggested_frame'], 30)
        _, pool, _ = contact_timing.event_inputs(self.times, self.features, self.event)
        result = self.select(int(pool[-1]))
        self.assertTrue(result['boundary'])
        self.assertFalse(result['applied'])

    def test_confirmed_frame_and_no_hit_are_not_overridden(self):
        suggestion = {'player': 'near', 'applied': True, 'suggested_frame': 33, 'suggested_time': 1.1}
        with patch.object(hit_refinement, 'read', return_value={'suggestions': {'event': suggestion}}):
            for annotation in [{'contact_confirmed': True}, {'stroke_type': 'no_stroke'},
                               {'center_override': True}, {'window_basis': 'manual'}, {'player': 'far'}]:
                actual, _ = hit_refinement.effective_event(self.dense, self.event, annotation)
                self.assertEqual(actual['frame_index'], 30)
            actual, _ = hit_refinement.effective_event(self.dense, self.event)
            self.assertEqual(actual['frame_index'], 33)


if __name__ == '__main__':
    unittest.main()
