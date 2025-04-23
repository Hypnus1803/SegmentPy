import unittest
import numpy as np
from skimage.util import invert
from skimage.morphology import square, disk
from skimage.filters.rank import equalize
from skimage.segmentation import watershed
from skimage.measure import label
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segmentpy.utilities import imhmin, labels_hmin, normalize_labels, merge_labels

class TestUtilities(unittest.TestCase):

    def test_imhmin(self):
        arr = 10 * np.ones([10, 10])
        arr[1:4, 1:4] = 7
        arr[5:8, 5:8] = 2
        arr[0:3, 6:9] = 13
        arr[1, 7] = 10
        h = 4
        output = imhmin(arr, h)
        expected_output = np.array([
            [10., 10., 10., 10., 10., 10., 13., 13., 13., 10.],
            [10., 10., 10., 10., 10., 10., 13., 13., 13., 10.],
            [10., 10., 10., 10., 10., 10., 13., 13., 13., 10.],
            [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
            [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
            [10., 10., 10., 10., 10., 6., 6., 6., 10., 10.],
            [10., 10., 10., 10., 10., 6., 6., 6., 10., 10.],
            [10., 10., 10., 10., 10., 6., 6., 6., 10., 10.],
            [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
            [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.]
        ])
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_labels_hmin(self):
        image = np.array([
            [10, 10, 10, 10, 10],
            [10, 7, 7, 7, 10],
            [10, 7, 7, 7, 10],
            [10, 7, 7, 7, 10],
            [10, 10, 10, 10, 10]
        ], dtype=np.uint8)
        radius = 1
        percentile = 50
        labels = labels_hmin(image, radius, percentile)
        self.assertEqual(labels.max(), 1)  # Expecting one region

    def test_normalize_labels(self):
        image = np.array([
            [10, 10, 10],
            [10, 7, 10],
            [10, 10, 10]
        ], dtype=np.uint8)
        labels = np.array([
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1]
        ], dtype=np.int32)
        normalized = normalize_labels(image, labels)
        self.assertEqual(normalized.max(), 100)  # Normalized values should not exceed 100

    def test_merge_labels(self):
        image = np.array([
            [10, 10, 10],
            [10, 7, 10],
            [10, 10, 10]
        ], dtype=np.uint8)
        normed_labels = np.array([
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1]
        ], dtype=np.int32)
        labels = normed_labels.copy()
        ref_level = 50
        top_thresh = 80
        bright = 5
        px_thresh = 2
        ref_thresh = 1
        merged_labels, _, _, _ = merge_labels(image, normed_labels, labels, ref_level, top_thresh, bright, px_thresh, ref_thresh)
        self.assertEqual(merged_labels.max(), 1)  # Expecting merged regions

if __name__ == '__main__':
    unittest.main()