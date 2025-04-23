import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segmentpy.shared_state import SharedState

class TestSharedState(unittest.TestCase):
    def test_singleton_instance(self):
        # Ensure that only one instance of SharedState is created
        instance1 = SharedState()
        instance2 = SharedState()
        self.assertIs(instance1, instance2, "SharedState is not a singleton")

    def test_initial_attributes(self):
        # Ensure all attributes are initialized to None
        instance = SharedState()
        attributes = [
            "hdu_cube", "full_image", "original_image", "masked_image", "markers",
            "old_labels", "new_labels", "new_labels2", "inos", "tags", "csiz",
            "ftag", "inom", "fmex", "ceco", "fseg", "feco", "tmpo", "perc",
            "classes", "scale", "hmin", "radius", "lmer", "ltop", "imex",
            "ntop", "nmer", "lcut", "igpx", "lcut2"
        ]
        for attr in attributes:
            with self.subTest(attr=attr):
                self.assertIsNone(getattr(instance, attr), f"Attribute {attr} is not initialized to None")

if __name__ == "__main__":
    unittest.main()