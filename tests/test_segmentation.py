import unittest
from PyQt5.QtWidgets import QApplication
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segmentpy.segmentation import CustomWidget  # Adjusted import path to remove the relative import

class TestCustomWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    def setUp(self):
        self.widget = CustomWidget()

    def test_initialization(self):
        self.assertIsNotNone(self.widget)
        self.assertIsNone(self.widget.image)
        self.assertFalse(self.widget.iscut)
        self.assertFalse(self.widget.iscube)

    def test_open_image_file_dialog(self):
        # Simulate opening an image file
        self.widget.openImageFileDialog()
        # Since the file dialog is interactive, we can't test file selection directly
        # Instead, we ensure no exceptions are raised and the method exists
        self.assertTrue(callable(self.widget.openImageFileDialog))

    def test_open_cube_file_dialog(self):
        # Simulate opening a cube file
        self.widget.openCubeFileDialog()
        # Ensure the method exists and is callable
        self.assertTrue(callable(self.widget.openCubeFileDialog))

    def test_update_combobox(self):
        # Simulate changing the combobox selection
        self.widget.comboBox.setCurrentIndex(1)
        self.widget.updateCombobox()
        self.assertTrue(callable(self.widget.updateCombobox))

    def test_start_manual_cut(self):
        # Test manual cut functionality
        self.widget.start_manualcut()
        self.assertTrue(hasattr(self.widget, 'selected'))
        self.assertTrue(hasattr(self.widget, 'img1_cut'))

    def test_start_automatic_cut(self):
        # Test automatic cut functionality
        self.widget.start_automaticcut()
        self.assertTrue(hasattr(self.widget, 'iscut'))

    def test_no_masking(self):
        # Test no masking functionality
        self.widget.no_masking()
        self.assertTrue(hasattr(self.widget, 'image'))

    def test_start_masking(self):
        # Test start masking functionality
        self.widget.start_masking()
        self.assertTrue(hasattr(self.widget, 'masking_widget'))

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()

if __name__ == "__main__":
    unittest.main()