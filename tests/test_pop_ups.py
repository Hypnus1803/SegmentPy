import unittest
from PyQt5.QtWidgets import QApplication
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segmentpy.pop_ups import ProgressDialog, Slider, SliderOdd, ThresholdMultiOtsuWidget, ThresholdNiblack

app = QApplication([])  # Required for PyQt5 widgets

class TestPopUps(unittest.TestCase):

    def test_progress_dialog(self):
        dialog = ProgressDialog()
        dialog.setProgress(50)
        self.assertEqual(dialog.progressBar.value(), 50)
        dialog.cancel()
        self.assertFalse(dialog.isVisible())

    def test_slider(self):
        slider = Slider(0, 100, "Test Slider", "int")
        slider.slider.setValue(50)
        self.assertEqual(slider.slider.value(), 50)
        slider.setLabelValue(50)
        self.assertEqual(slider.x, 50)

    def test_slider_odd(self):
        slider_odd = SliderOdd(1, 99, "Odd Slider", "int")
        slider_odd.slider.setValue(51)
        self.assertEqual(slider_odd.slider.value(), 51)
        slider_odd.setLabelValue(51)
        self.assertEqual(slider_odd.x, 51)

    def test_threshold_multiotsu_widget(self):
        widget = ThresholdMultiOtsuWidget()
        widget.classesSpinBox.setValue(3)
        widget.nbinsSlider.slider.setValue(128)
        self.assertEqual(widget.classesSpinBox.value(), 3)
        self.assertEqual(widget.nbinsSlider.slider.value(), 128)

    def test_threshold_niblack(self):
        widget = ThresholdNiblack()
        widget.windowSlider.slider.setValue(5)
        widget.kSpinBox.setValue(1.0)
        widget.qSpinBox.setValue(3)
        self.assertEqual(widget.windowSlider.slider.value(), 5)
        self.assertEqual(widget.kSpinBox.value(), 1.0)
        self.assertEqual(widget.qSpinBox.value(), 3)

if __name__ == "__main__":
    unittest.main()