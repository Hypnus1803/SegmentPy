from PyQt5.QtWidgets import QPushButton, QWidget, QVBoxLayout, QSpinBox, QGroupBox, QRadioButton, QLineEdit, QDialog
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSlider, QDoubleSpinBox, QComboBox,QFileDialog, QProgressBar
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QLocale, QThread
from shared_state import SharedState

import numpy as np
from skimage.util import img_as_ubyte
from skimage.filters import threshold_multiotsu

import pyqtgraph as pg


class ProgressDialog(QDialog):
    def __init__(self, parent=None):
        super(ProgressDialog, self).__init__(parent)

        self.setWindowTitle("Progress")

        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0, 100)

        self.cancelButton = QPushButton("Cancel", self)
        self.cancelButton.clicked.connect(self.cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(self.progressBar)
        layout.addWidget(self.cancelButton)

    def setProgress(self, value):
        self.progressBar.setValue(value)

    def cancel(self):
        self.reject()  # Close the dialog

class CalculationThread(QThread):
    progressChanged = pyqtSignal(int)  # Signal to update the progress bar

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shared_state = SharedState()

    def run(self):
        from skimage.exposure import rescale_intensity
        from utilities import apply_config_file
        # from tqdm import tqdm
        percentile = self.shared_state.perc
        scale = self.shared_state.scale
        hmin = self.shared_state.hmin
        lmer = self.shared_state.lmer
        ltop = self.shared_state.ltop
        imex = self.shared_state.imex
        ntop = self.shared_state.ntop
        nmer = self.shared_state.nmer
        lcut = self.shared_state.lcut
        igpx = self.shared_state.igpx
        lcut2 = self.shared_state.lcut2
        params = [scale, hmin, lmer, ltop, imex, ntop, nmer, lcut, igpx, lcut2]
        hdu = self.shared_state.hdu_cube

        total_images = hdu[0].data.shape[0]
        fp = np.memmap('cube_masked.dat', dtype='uint8', mode='w+', shape=hdu[0].data.shape)
        for i in range(total_images):
            fov = rescale_intensity(hdu[0].data[i, :, :], out_range=(0, 1.))
            thresh1 = np.percentile(fov, percentile)
            masked_im = fov.copy()  # self.shared_state.original_image.copy()
            masked_im[masked_im < thresh1] = 0.0
            masked_im = img_as_ubyte(masked_im)
            fseg, _, _ = apply_config_file(masked_im, params)
            new_mask = fseg.copy()
            vals0 = np.unique(new_mask)
            new_mask[new_mask == vals0[-1]] = vals0[-2]
            new_mask[new_mask == vals0[0]] = vals0[1]
            fp[i, ...] = new_mask[...]
            del fov, masked_im, fseg, new_mask
            # Emit signal with the current progress
            self.progressChanged.emit((i + 1) / total_images * 100)
        fp.flush()
        del fp




class SliderOdd(QWidget):
    def __init__(self, minimum, maximum, slider_label, dtype, parent=None):
        self.dtype = dtype
        super(SliderOdd, self).__init__(parent=parent)

        self.horizontalLayout = QHBoxLayout()

        self.sliderLabel = QLabel(slider_label, self)
        self.sliderLabel.setAlignment(Qt.AlignCenter)
        self.horizontalLayout.addWidget(self.sliderLabel)

        self.label = QLabel(self)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.horizontalLayout.addWidget(self.slider)
        self.horizontalLayout.addWidget(self.label)

        self.minimum = minimum
        self.maximum = maximum
        self.slider.setMinimum(self.minimum)
        self.slider.setMaximum(self.maximum)

        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = 0
        self.setLabelValue(self.slider.value())

        self.setLayout(self.horizontalLayout)  # Set the layout to the widget

    def setLabelValue(self, value):
        if self.dtype == 'int':
            self.x = int(value)
            # Ensure the value is odd
            if self.x % 2 == 0:
                self.x += 1
            self.label.setText("{}".format(self.x))

class Slider(QWidget):
    def __init__(self, minimum, maximum, slider_label, dtype, parent=None):
        self.dtype = dtype
        super(Slider, self).__init__(parent=parent)

        self.horizontalLayout = QHBoxLayout()

        self.sliderLabel = QLabel(slider_label, self)
        self.sliderLabel.setAlignment(Qt.AlignCenter)
        self.horizontalLayout.addWidget(self.sliderLabel)

        self.label = QLabel(self)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.horizontalLayout.addWidget(self.slider)
        self.horizontalLayout.addWidget(self.label)

        self.minimum = minimum
        self.maximum = maximum
        self.slider.setMinimum(self.minimum)
        self.slider.setMaximum(self.maximum)
        #self.slider.setSingleStep(1)

        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = 0
        self.setLabelValue(self.slider.value())

        self.setLayout(self.horizontalLayout)  # Set the layout to the widget

    def setLabelValue(self, value):
        if self.dtype == 'int':
            self.x = int(value)#int(self.minimum + value)
            self.label.setText("{}".format(self.x))
        if self.dtype == 'float':
            #self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (self.maximum - self.minimum)
            self.x = (float(value) / (self.slider.maximum() - self.slider.minimum())) * (self.maximum - self.minimum)
            self.label.setText("{0:.2f}".format(self.x))


class ThresholdMultiOtsuWidget(QWidget):
    valuesChanged = pyqtSignal(int, int)
    def __init__(self, parent=None):
        super(ThresholdMultiOtsuWidget, self).__init__(parent=parent)
        self.shared_state = SharedState()

        # Create a QVBoxLayout instance
        layout = QVBoxLayout()

        # Create a QSpinBox instance for classes
        layout2 = QHBoxLayout()
        label_classes = QLabel('classes', self)
        self.classesSpinBox = QSpinBox()
        self.classesSpinBox.setMinimum(2)  # Set minimum value
        self.classesSpinBox.setMaximum(5)  # Set maximum value

        layout2.addWidget(label_classes)
        layout2.addWidget(self.classesSpinBox)
        layout.addLayout(layout2)

        # Create a Slider instance for nbins
        self.nbinsSlider = Slider(1, 256, 'nbins', 'int')
        self.nbinsSlider.slider.setValue(256)
        layout.addWidget(self.nbinsSlider)




        self.pushButton = QPushButton("Finish", self)
        self.pushButton.clicked.connect(self.close_widget)
        layout.addWidget(self.pushButton)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_signal)

        # ...
        self.old_slider_x = self.nbinsSlider.x
        self.old_spinbox_value = self.classesSpinBox.value()
        # ...

        self.nbinsSlider.slider.valueChanged.connect(self.start_timer)
        self.classesSpinBox.valueChanged.connect(self.start_timer)


        self.setLayout(layout)

    def update_signal(self):
        if self.nbinsSlider.x != self.old_slider_x or self.classesSpinBox.value() != self.old_spinbox_value:
            self.valuesChanged.emit(self.classesSpinBox.value(), self.nbinsSlider.x)
        #self.close()
    def start_timer(self):
        self.timer.start(200)

    def close_widget(self):
        self.close()


class ThresholdNiblack(QWidget):
    valuesChanged = pyqtSignal(int, float,int)
    def __init__(self, parent=None):
        super(ThresholdNiblack, self).__init__(parent=parent)
        self.shared_state = SharedState()

        layout = QVBoxLayout()

        self.windowSlider = SliderOdd(1, 50, 'window', 'int')
        self.windowSlider.slider.setValue(1)
        layout.addWidget(self.windowSlider)

        layout2 = QHBoxLayout()
        label_k = QLabel('k', self)
        label_k.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        self.kSpinBox = QDoubleSpinBox()
        self.kSpinBox.setMinimum(0.0)  # Set minimum value
        self.kSpinBox.setMaximum(2.0)  # Set maximum value
        self.kSpinBox.setSingleStep(0.05)

        layout2.addWidget(label_k)
        layout2.addWidget(self.kSpinBox)
        layout.addLayout(layout2)

        layout3 = QHBoxLayout()
        label_q = QLabel('q', self)
        label_q.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        self.qSpinBox = QSpinBox()
        self.qSpinBox.setMinimum(1)  # Set minimum value
        self.qSpinBox.setMaximum(5)  # Set maximum value
        layout3.addWidget(label_q)
        layout3.addWidget(self.qSpinBox)
        layout.addLayout(layout3)


        self.pushButton = QPushButton("Finish", self)
        self.pushButton.clicked.connect(self.close_widget)
        layout.addWidget(self.pushButton)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_signal)

        self.old_slider_x = self.windowSlider.x
        self.windowSlider.slider.valueChanged.connect(self.start_timer)

        self.old_spinbox_value = self.kSpinBox.value()
        self.kSpinBox.valueChanged.connect(self.start_timer)

        self.old_qspinbox_value = self.qSpinBox.value()
        self.qSpinBox.valueChanged.connect(self.start_timer)

        self.setLayout(layout)

    def update_signal(self):
        if self.windowSlider.x != self.old_slider_x or self.kSpinBox.value() != self.old_spinbox_value or self.qSpinBox.value() != self.old_qspinbox_value:
            self.valuesChanged.emit(self.windowSlider.x, self.kSpinBox.value(),self.qSpinBox.value())

    def start_timer(self):
        self.timer.start(200)

    def close_widget(self):
        self.close()


class ThresholdSauvola(QWidget):
    valuesChanged = pyqtSignal(int, float)
    def __init__(self, parent=None):
        super(ThresholdSauvola, self).__init__(parent=parent)
        self.shared_state = SharedState()

        layout = QVBoxLayout()

        self.windowSlider = SliderOdd(1, 50, 'window', 'int')
        self.windowSlider.slider.setValue(1)
        layout.addWidget(self.windowSlider)

        layout2 = QHBoxLayout()
        label_k = QLabel('k', self)
        label_k.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        self.kSpinBox = QDoubleSpinBox()
        self.kSpinBox.setMinimum(0.0)  # Set minimum value
        self.kSpinBox.setMaximum(2.0)  # Set maximum value
        self.kSpinBox.setSingleStep(0.05)

        layout2.addWidget(label_k)
        layout2.addWidget(self.kSpinBox)
        layout.addLayout(layout2)



        self.pushButton = QPushButton("Finish", self)
        self.pushButton.clicked.connect(self.close_widget)
        layout.addWidget(self.pushButton)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_signal)

        self.old_slider_x = self.windowSlider.x
        self.windowSlider.slider.valueChanged.connect(self.start_timer)

        self.old_spinbox_value = self.kSpinBox.value()
        self.kSpinBox.valueChanged.connect(self.start_timer)


        self.setLayout(layout)

    def update_signal(self):
        if self.windowSlider.x != self.old_slider_x or self.kSpinBox.value() != self.old_spinbox_value:
            self.valuesChanged.emit(self.windowSlider.x, self.kSpinBox.value())

    def start_timer(self):
        self.timer.start(200)

    def close_widget(self):
        self.close()


class ThresholdLi(QWidget):
    valuesChanged = pyqtSignal(int)
    def __init__(self, parent=None):
        super(ThresholdLi, self).__init__(parent=parent)
        self.shared_state = SharedState()

        layout = QVBoxLayout()

        self.initSlider = Slider(1, 255, 'initial guess', 'int')
        self.initSlider.slider.setValue(1)
        layout.addWidget(self.initSlider)

        self.pushButton = QPushButton("Finish", self)
        self.pushButton.clicked.connect(self.close_widget)
        layout.addWidget(self.pushButton)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_signal)

        self.old_slider_x = self.initSlider.x
        self.initSlider.slider.valueChanged.connect(self.start_timer)

        # self.old_spinbox_value = self.kSpinBox.value()
        # self.kSpinBox.valueChanged.connect(self.start_timer)


        self.setLayout(layout)

    def update_signal(self):
        if self.initSlider.x != self.old_slider_x:
            self.valuesChanged.emit(self.initSlider.x)

    def start_timer(self):
        self.timer.start(200)

    def close_widget(self):
        self.close()

class ThresholdOtsu(QWidget):
    valuesChanged = pyqtSignal(int)
    def __init__(self, parent=None):
        super(ThresholdOtsu, self).__init__(parent=parent)
        self.shared_state = SharedState()

        layout = QVBoxLayout()

        self.nbinsSlider = Slider(1, 512, 'nbins', 'int')
        self.nbinsSlider.slider.setValue(1)
        layout.addWidget(self.nbinsSlider)

        # layout2 = QHBoxLayout()
        # label_k = QLabel('k', self)
        # label_k.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        # self.kSpinBox = QDoubleSpinBox()
        # self.kSpinBox.setMinimum(0.0)  # Set minimum value
        # self.kSpinBox.setMaximum(2.0)  # Set maximum value
        # self.kSpinBox.setSingleStep(0.05)
        #
        # layout2.addWidget(label_k)
        # layout2.addWidget(self.kSpinBox)
        # layout.addLayout(layout2)



        self.pushButton = QPushButton("Finish", self)
        self.pushButton.clicked.connect(self.close_widget)
        layout.addWidget(self.pushButton)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_signal)

        self.old_slider_x = self.nbinsSlider.x
        self.nbinsSlider.slider.valueChanged.connect(self.start_timer)

        # self.old_spinbox_value = self.kSpinBox.value()
        # self.kSpinBox.valueChanged.connect(self.start_timer)


        self.setLayout(layout)

    def update_signal(self):
        if self.nbinsSlider.x != self.old_slider_x:
            self.valuesChanged.emit(self.nbinsSlider.x)

    def start_timer(self):
        self.timer.start(200)

    def close_widget(self):
        self.close()


class ThresholdYen(QWidget):
    valuesChanged = pyqtSignal(int)
    def __init__(self, parent=None):
        super(ThresholdYen, self).__init__(parent=parent)
        self.shared_state = SharedState()

        layout = QVBoxLayout()

        self.nbinsSlider = Slider(1, 512, 'nbins', 'int')
        self.nbinsSlider.slider.setValue(1)
        layout.addWidget(self.nbinsSlider)

        # layout2 = QHBoxLayout()
        # label_k = QLabel('k', self)
        # label_k.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        # self.kSpinBox = QDoubleSpinBox()
        # self.kSpinBox.setMinimum(0.0)  # Set minimum value
        # self.kSpinBox.setMaximum(2.0)  # Set maximum value
        # self.kSpinBox.setSingleStep(0.05)
        #
        # layout2.addWidget(label_k)
        # layout2.addWidget(self.kSpinBox)
        # layout.addLayout(layout2)



        self.pushButton = QPushButton("Finish", self)
        self.pushButton.clicked.connect(self.close_widget)
        layout.addWidget(self.pushButton)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_signal)

        self.old_slider_x = self.nbinsSlider.x
        self.nbinsSlider.slider.valueChanged.connect(self.start_timer)

        # self.old_spinbox_value = self.kSpinBox.value()
        # self.kSpinBox.valueChanged.connect(self.start_timer)


        self.setLayout(layout)

    def update_signal(self):
        if self.nbinsSlider.x != self.old_slider_x:
            self.valuesChanged.emit(self.nbinsSlider.x)

    def start_timer(self):
        self.timer.start(200)

    def close_widget(self):
        self.close()

class ThresholdMinimum(QWidget):
    valuesChanged = pyqtSignal(int)
    def __init__(self, parent=None):
        super(ThresholdMinimum, self).__init__(parent=parent)
        self.shared_state = SharedState()

        layout = QVBoxLayout()

        self.nbinsSlider = Slider(1, 512, 'nbins', 'int')
        self.nbinsSlider.slider.setValue(1)
        layout.addWidget(self.nbinsSlider)

        # layout2 = QHBoxLayout()
        # label_k = QLabel('k', self)
        # label_k.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        # self.kSpinBox = QDoubleSpinBox()
        # self.kSpinBox.setMinimum(0.0)  # Set minimum value
        # self.kSpinBox.setMaximum(2.0)  # Set maximum value
        # self.kSpinBox.setSingleStep(0.05)
        #
        # layout2.addWidget(label_k)
        # layout2.addWidget(self.kSpinBox)
        # layout.addLayout(layout2)



        self.pushButton = QPushButton("Finish", self)
        self.pushButton.clicked.connect(self.close_widget)
        layout.addWidget(self.pushButton)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_signal)

        self.old_slider_x = self.nbinsSlider.x
        self.nbinsSlider.slider.valueChanged.connect(self.start_timer)

        # self.old_spinbox_value = self.kSpinBox.value()
        # self.kSpinBox.valueChanged.connect(self.start_timer)


        self.setLayout(layout)

    def update_signal(self):
        if self.nbinsSlider.x != self.old_slider_x:
            self.valuesChanged.emit(self.nbinsSlider.x)

    def start_timer(self):
        self.timer.start(200)

    def close_widget(self):
        self.close()


class ThresholdTriangle(QWidget):
    valuesChanged = pyqtSignal(int)
    def __init__(self, parent=None):
        super(ThresholdTriangle, self).__init__(parent=parent)
        self.shared_state = SharedState()

        layout = QVBoxLayout()

        self.nbinsSlider = Slider(1, 512, 'nbins', 'int')
        self.nbinsSlider.slider.setValue(1)
        layout.addWidget(self.nbinsSlider)

        # layout2 = QHBoxLayout()
        # label_k = QLabel('k', self)
        # label_k.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        # self.kSpinBox = QDoubleSpinBox()
        # self.kSpinBox.setMinimum(0.0)  # Set minimum value
        # self.kSpinBox.setMaximum(2.0)  # Set maximum value
        # self.kSpinBox.setSingleStep(0.05)
        #
        # layout2.addWidget(label_k)
        # layout2.addWidget(self.kSpinBox)
        # layout.addLayout(layout2)



        self.pushButton = QPushButton("Finish", self)
        self.pushButton.clicked.connect(self.close_widget)
        layout.addWidget(self.pushButton)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_signal)

        self.old_slider_x = self.nbinsSlider.x
        self.nbinsSlider.slider.valueChanged.connect(self.start_timer)

        # self.old_spinbox_value = self.kSpinBox.value()
        # self.kSpinBox.valueChanged.connect(self.start_timer)


        self.setLayout(layout)

    def update_signal(self):
        if self.nbinsSlider.x != self.old_slider_x:
            self.valuesChanged.emit(self.nbinsSlider.x)

    def start_timer(self):
        self.timer.start(200)

    def close_widget(self):
        self.close()


class ThresholdIsodata(QWidget):
    valuesChanged = pyqtSignal(int,str)
    def __init__(self, parent=None):
        super(ThresholdIsodata, self).__init__(parent=parent)
        self.shared_state = SharedState()

        layout = QVBoxLayout()

        self.nbinsSlider = Slider(1, 512, 'nbins', 'int')
        self.nbinsSlider.slider.setValue(1)
        layout.addWidget(self.nbinsSlider)

        layout2 = QHBoxLayout()
        label_r = QLabel('Return All', self)
        label_r.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        self.comboBox = QComboBox()
        self.comboBox.addItems(["False","True"])

        layout2.addWidget(label_r)
        layout2.addWidget(self.comboBox)
        layout.addLayout(layout2)

        self.pushButton = QPushButton("Finish", self)
        self.pushButton.clicked.connect(self.close_widget)
        layout.addWidget(self.pushButton)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_signal)

        self.old_slider_x = self.nbinsSlider.x
        self.nbinsSlider.slider.valueChanged.connect(self.start_timer)

        self.old_combo_x = self.comboBox.currentText()
        self.comboBox.currentIndexChanged.connect(self.start_timer)
        # self.comboBox.currentIndexChanged.connect(self.update_signal)

        # self.old_spinbox_value = self.kSpinBox.value()
        # self.kSpinBox.valueChanged.connect(self.start_timer)


        self.setLayout(layout)

    def update_signal(self):
        if self.nbinsSlider.x != self.old_slider_x or self.comboBox.currentText() != self.old_combo_x:
            self.valuesChanged.emit(self.nbinsSlider.x,self.comboBox.currentText())

    def start_timer(self):
        self.timer.start(200)

    def close_widget(self):
        self.close()


class ThresholdLocal(QWidget):
    valuesChanged = pyqtSignal(int,str,str)
    def __init__(self, parent=None):
        super(ThresholdLocal, self).__init__(parent=parent)
        self.shared_state = SharedState()

        layout = QVBoxLayout()

        self.blockSlider = SliderOdd(1, 50, 'Block Size', 'int')
        self.blockSlider.slider.setValue(1)
        layout.addWidget(self.blockSlider)

        layout2 = QHBoxLayout()
        label_m = QLabel('Method', self)
        label_m.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        self.comboBox1 = QComboBox()
        self.comboBox1.addItems(["gaussian","mean","median"])

        layout2.addWidget(label_m)
        layout2.addWidget(self.comboBox1)
        layout.addLayout(layout2)

        layout3 = QHBoxLayout()
        label_mo = QLabel('Mode', self)
        label_mo.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        self.comboBox2 = QComboBox()
        self.comboBox2.addItems(["reflect","constant","nearest","mirror","wrap"])

        layout3.addWidget(label_mo)
        layout3.addWidget(self.comboBox2)
        layout.addLayout(layout3)

        self.pushButton = QPushButton("Finish", self)
        self.pushButton.clicked.connect(self.close_widget)
        layout.addWidget(self.pushButton)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_signal)

        self.old_slider_x = self.blockSlider.x
        self.blockSlider.slider.valueChanged.connect(self.start_timer)

        self.old_combo1_x = self.comboBox1.currentText()
        self.comboBox1.currentIndexChanged.connect(self.start_timer)

        self.old_combo2_x = self.comboBox2.currentText()
        self.comboBox2.currentIndexChanged.connect(self.start_timer)
        # self.comboBox.currentIndexChanged.connect(self.update_signal)

        # self.old_spinbox_value = self.kSpinBox.value()
        # self.kSpinBox.valueChanged.connect(self.start_timer)


        self.setLayout(layout)

    def update_signal(self):
        if self.blockSlider.x != self.old_slider_x or self.comboBox1.currentText() != self.old_combo1_x or self.comboBox2.currentText() != self.old_combo2_x:
            self.valuesChanged.emit(self.blockSlider.x,self.comboBox1.currentText(),self.comboBox2.currentText())

    def start_timer(self):
        self.timer.start(200)

    def close_widget(self):
        self.close()

class ThresholdImhmin(QWidget):
    valuesChanged = pyqtSignal(int,int)
    valuesChanged2 = pyqtSignal(int)
    valuesChanged3 = pyqtSignal(int,int,str)
    paramsChanged = pyqtSignal(list)
    fullChanged = pyqtSignal(list)
    progressChanged = pyqtSignal(int)
    def __init__(self, parent=None):
        super(ThresholdImhmin, self).__init__(parent=parent)
        self.shared_state = SharedState()
        self.ws2 = 0
        self.labels = 0

        mainLayout = QVBoxLayout()

        bigGroupBox = QGroupBox("Image thresholding")
        big_layout = QVBoxLayout()


        # =============== Section thresholding (Im H-min) =============== #
        groupBox1 = QGroupBox("Params h-min")
        layout1 = QHBoxLayout()
        layout1_1 = QHBoxLayout()
        label_line = QLabel('Scale:', self)
        self.line_scale = QLineEdit()
        self.line_scale.setText('0.058')
        self.line_scale.setFixedWidth(100)
        self.line_scale.setAlignment(Qt.AlignRight)
        layout1_1.addWidget(label_line)
        layout1_1.addWidget(self.line_scale)
        layout1.addLayout(layout1_1)
        self.radiusSlider = Slider(1, 50, 'Radius Eq.', 'int')
        self.radiusSlider.slider.setFixedWidth(100)
        layout1.addWidget(self.radiusSlider)
        self.percSlider = Slider(1, 100, 'Percentile H-min', 'int')
        self.percSlider.slider.setValue(1)
        self.percSlider.slider.setFixedWidth(100)
        layout1.addWidget(self.percSlider)
        groupBox1.setLayout(layout1)
        # =============================================================== #





        # ====================== Section MLT ============================ #

        groupBox2 = QGroupBox("Normalize Image")
        layout2 = QVBoxLayout()
        self.norm_button = QPushButton("Normalize Image",self)
        self.norm_button.clicked.connect(self.normalize_image)
        layout2.addWidget(self.norm_button)
        groupBox2.setLayout(layout2)

        groupBox3 = QGroupBox("Merging Contours")
        layout3 = QVBoxLayout()
        layout3_1 = QHBoxLayout()
        self.refSlider = Slider(1, 100, 'Reference', 'int')
        self.refSlider.slider.setValue(1)
        layout3_1.addWidget(self.refSlider)
        self.topSlider = Slider(1, 100, 'Top Thresh', 'int')
        self.topSlider.slider.setValue(1)
        layout3_1.addWidget(self.topSlider)

        layout3_2 = QHBoxLayout()
        layout3_2_1 = QHBoxLayout()
        self.brlabel = QLabel('Brightness', self)
        self.brightspin = QDoubleSpinBox()
        locale = QLocale(QLocale.C)
        self.brightspin.setLocale(locale)
        self.brightspin.setMinimum(0.0)  # Set minimum value
        self.brightspin.setMaximum(10.0)  # Set maximum value
        self.brightspin.setSingleStep(0.05)
        self.brightspin.setValue(1.15)
        layout3_2_1.addWidget(self.brlabel)
        layout3_2_1.addWidget(self.brightspin)
        layout3_2_2 = QHBoxLayout()
        self.pxlabel = QLabel('px>thresh', self)
        self.pxspin = QSpinBox()
        self.pxspin.setMinimum(1)  # Set minimum value
        self.pxspin.setMaximum(100)  # Set maximum value
        self.pxspin.setValue(20)
        layout3_2_2.addWidget(self.pxlabel)
        layout3_2_2.addWidget(self.pxspin)
        layout3_2_3 = QHBoxLayout()
        self.refthlabel = QLabel('Ref. Thresh', self)
        self.refthspin = QSpinBox()
        self.refthspin.setMinimum(1)  # Set minimum value
        self.refthspin.setMaximum(100)  # Set maximum value
        self.refthspin.setValue(4)
        layout3_2_3.addWidget(self.refthlabel)
        layout3_2_3.addWidget(self.refthspin)
        layout3_2.addLayout(layout3_2_1)
        layout3_2.addLayout(layout3_2_2)
        layout3_2.addLayout(layout3_2_3)
        layout3.addLayout(layout3_1)
        layout3.addLayout(layout3_2)
        groupBox3.setLayout(layout3)

        groupBox4 = QGroupBox("Shrinking Labels")
        layout4 = QHBoxLayout()
        self.cutSlider = Slider(1, 100, 'Cut-off threshold', 'int')
        self.cutSlider.slider.setValue(1)
        layout4.addWidget(self.cutSlider)
        groupBox4.setLayout(layout4)

        groupBox5 = QGroupBox("Intergranular Labels")
        layout5 = QVBoxLayout()
        layout5_1 = QHBoxLayout()
        self.max_intergranular = threshold_multiotsu(self.shared_state.masked_image, classes=4)[1]
        self.itgSlider = Slider(1, self.max_intergranular , 'Intergranular Level', 'int')
        self.itgSlider.slider.setValue(1)
        layout5_1.addWidget(self.itgSlider)
        self.offSlider = Slider(1, 100, '       Cut-off Level', 'int')
        self.offSlider.slider.setValue(1)
        layout5_1.addWidget(self.offSlider)

        layout5_2 = QHBoxLayout()
        label_plots = QLabel('Plot Options', self)
        label_plots.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        self.extended = QRadioButton('Extended Markers')
        self.extended.setChecked(True)
        self.colorcontours = QRadioButton('Contours Granules')
        self.colorlabels = QRadioButton('Final Labels')
        layout5_2.addWidget(label_plots)
        layout5_2.addWidget(self.extended)
        layout5_2.addWidget(self.colorcontours)
        layout5_2.addWidget(self.colorlabels)
        layout5.addLayout(layout5_1)
        layout5.addLayout(layout5_2)
        groupBox5.setLayout(layout5)

        big_layout.addWidget(groupBox1)
        big_layout.addWidget(groupBox2)
        big_layout.addWidget(groupBox3)
        big_layout.addWidget(groupBox4)
        big_layout.addWidget(groupBox5)
        bigGroupBox.setLayout(big_layout)

        # ====================== Section Files ============================ #

        groupBox10 = QGroupBox("Close and Save Properties")
        layout10 = QHBoxLayout()
        self.endButton = QPushButton("Finish", self)
        self.endButton.clicked.connect(self.close_widget)
        self.saveButton = QPushButton("Save Props", self)
        self.saveButton.clicked.connect(self.save_props)
        self.configButton = QPushButton("Save Config File", self)
        self.configButton.clicked.connect(self.openfile_dialog_config)
        self.loadconfButton = QPushButton("Load Config File", self)
        self.loadconfButton.clicked.connect(self.openfile_load_config)
        layout10.addWidget(self.loadconfButton)
        layout10.addWidget(self.configButton)
        layout10.addWidget(self.saveButton)
        layout10.addWidget(self.endButton)
        groupBox10.setLayout(layout10)

        self.finalGroupBox = QGroupBox("Final Steps")
        self.final_layout = QHBoxLayout()
        self.applyButton = QPushButton("Apply to full Image", self)
        self.applyButton.setEnabled(False)
        self.final_layout.addWidget(self.applyButton)
        self.applyButton.clicked.connect(self.apply_full)
        self.apply_cubebutton = QPushButton("Apply config to full cube", self)
        self.apply_cubebutton.setEnabled(False)
        self.final_layout.addWidget(self.apply_cubebutton)
        self.apply_cubebutton.clicked.connect(self.apply_full_cube)
        self.finalGroupBox.setLayout(self.final_layout)
        # =============================================================== #



        mainLayout.addWidget(bigGroupBox)
        mainLayout.addWidget(self.finalGroupBox)
        mainLayout.addWidget(groupBox10)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_signal)
        self.timer.timeout.connect(self.update_signal2)
        self.timer.timeout.connect(self.update_signal3)

        self.old_slider0_x = self.percSlider.x
        self.percSlider.slider.valueChanged.connect(self.start_timer)
        self.scale = float(self.line_scale.text())
        self.radiusSlider.slider.setValue(1/self.scale)
        self.old_slider1_x = self.radiusSlider.x
        self.radiusSlider.slider.valueChanged.connect(self.start_timer)

        self.old_slider2_x = self.cutSlider.x
        self.cutSlider.slider.valueChanged.connect(self.start_timer)

        self.old_slider3_x = self.itgSlider.x
        self.itgSlider.slider.valueChanged.connect(self.start_timer)

        self.old_slider4_x = self.offSlider.x
        self.offSlider.slider.valueChanged.connect(self.start_timer)

        self.setLayout(mainLayout)

        self.refSlider.slider.valueChanged.connect(self.plot_merge)
        self.topSlider.slider.valueChanged.connect(self.plot_merge)
        self.brightspin.valueChanged.connect(self.plot_merge)
        self.pxspin.valueChanged.connect(self.plot_merge)
        self.refthspin.valueChanged.connect(self.plot_merge)

        self.graphNorm = pg.GraphicsLayoutWidget( size=(12, 12))
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.graphNorm.resize(500, 500)
        self.p1 = self.graphNorm.addPlot(title="Normalized image with Merging contours")
        self.thread = None
        self.dialog = None

    # def set_classes(self):
    #     from skimage.measure import label, regionprops
    #     self.graphClasses = pg.GraphicsLayoutWidget()
    #     pg.setConfigOptions(imageAxisOrder='row-major')
    #     self.graphClasses.resize(500, 500)
    #     self.p11 = self.graphClasses.addPlot(title="Segmented image with classes\n0=mask,  75=granules, 200=brightpoints ")
    #     self.new_mask = self.shared_state.fseg.copy()
    #     vals0 = np.unique(self.new_mask)
    #     self.new_mask[self.new_mask== vals0[-1]] = vals0[-2]
    #     self.new_mask[self.new_mask== vals0[0]] = vals0[1]
    #
    #     self.tmp_labels = label(self.new_mask)
    #     self.image = self.shared_state.masked_image.copy()
    #     # eq_im = equalize(invert(self.image),disk(1/self.shared_state.scale))
    #     # bp_min_flux = np.nanmean(eq_im) + 1.25*np.nanstd(eq_im)
    #     bp_min_flux = np.nanmean(self.image) + 1.25 * np.nanstd(self.image)
    #     for reg in regionprops(self.tmp_labels,intensity_image=self.image):
    #         if reg.intensity_mean >= bp_min_flux:
    #             self.new_mask[self.tmp_labels == reg.label] = 200
    #
    #     self.shared_state.classes = self.new_mask
    #
    #     self.image_item = pg.ImageItem()
    #     self.p11.addItem(self.image_item)
    #     self.image_item.setImage(self.new_mask)
    #     self.graphClasses.show()

    # def simple_ml(self):
    #     from functools import partial
    #     from skimage import feature, future
    #     from sklearn.ensemble import RandomForestClassifier
    #     import time
    #     t0 = time.time()
    #
    #     image = self.shared_state.masked_image.copy()
    #     training_labels = self.shared_state.classes.copy()
    #     sigma_min = 1
    #     sigma_max = 5
    #     features_func = partial(
    #         feature.multiscale_basic_features,
    #         intensity=True,
    #         edges=False,
    #         texture=False,
    #         sigma_min=sigma_min,
    #         sigma_max=sigma_max,
    #         channel_axis=None,
    #     )
    #     features = features_func(image)
    #     clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
    #     clf = future.fit_segmenter(training_labels, features, clf)
    #     result = future.predict_segmenter(features, clf)
    #     print('time elapsed doing prediction->', time.time() - t0, 'seconds')
    #
    #     self.graph_ml = pg.GraphicsLayoutWidget()
    #     pg.setConfigOptions(imageAxisOrder='row-major')
    #     self.graph_ml.resize(1000, 500)
    #     self.pl11 = self.graph_ml.addPlot(title="Original Segmentation")
    #     self.image_item1 = pg.ImageItem()
    #     self.pl11.addItem(self.image_item1)
    #     self.image_item1.setImage(training_labels)
    #
    #     self.pl22 = self.graph_ml.addPlot(title="ML Segmentation")
    #     self.image_item2 = pg.ImageItem()
    #     self.pl22.addItem(self.image_item2)
    #     self.image_item2.setImage(result)
    #     print(np.unique(result))


        # self.graph_ml.show()
        #
        #
        #
        # pass

    
    def apply_full_cube(self):
        # Create a ProgressDialog instance
        self.dialog = ProgressDialog()
        self.dialog.show()

        # Create a CalculationThread instance
        self.thread = CalculationThread()
        self.thread.progressChanged.connect(self.dialog.setProgress)
        self.thread.start()



    def plot(self, data):
        # Set the image data of the ImageItem
        self.image_item = pg.ImageItem()
        self.p1.addItem(self.image_item)
        self.image_item.setImage(data)
        self.graphNorm.show()

    def update_signal(self):
        if self.percSlider.x != self.old_slider0_x or self.radiusSlider.x != self.old_slider1_x:
            self.valuesChanged.emit(self.percSlider.x,self.radiusSlider.x)
            self.shared_state.scale = self.scale
            self.shared_state.hmin = self.percSlider.x
            self.shared_state.radius = self.radiusSlider.x

    def update_signal2(self):
        from utilities import shrinking_labels
        if self.cutSlider.x != self.old_slider2_x:
            # image,normed_labels,new_labels,tags,cutoff
            self.image = self.shared_state.masked_image.copy()
            self.inos = self.shared_state.inos.copy()
            self.new_labels = self.shared_state.new_labels.copy()
            self.tags = self.shared_state.tags.copy()
            self.ftag, self.csiz, self.inom = shrinking_labels(self.image, self.inos, self.new_labels, self.tags, self.cutSlider.x)
            self.shared_state.ftag = self.ftag
            self.shared_state.csiz = self.csiz
            self.shared_state.inom = self.inom
            self.shared_state.lcut = self.cutSlider.x
            self.valuesChanged2.emit(self.cutSlider.x)

    def update_signal3(self):
        from utilities import intergranular_levels
        if self.itgSlider.x != self.old_slider3_x or self.offSlider.x != self.old_slider4_x:
            #image,normed_labels,ntop,fmex,ceco,csiz,ftag
            self.image = self.shared_state.masked_image.copy()
            self.normed_labels = self.shared_state.inom.copy()
            self.ntop = self.shared_state.ntop
            self.fmex = self.shared_state.fmex.copy()
            self.ceco = self.shared_state.ceco.copy()
            self.csiz = self.shared_state.csiz.copy()
            self.ftag = self.shared_state.ftag.copy()


            #fseg, feco, self.tmp = self.IntergranularLabels(self.image, self.normed_labels, self.ntop, self.fmex,self.ceco,self.csiz,self.ftag)#self.itgSlider.x, self.offSlider.x)
            fseg, feco, self.tmp = intergranular_levels(self.image, self.normed_labels, self.ntop, self.fmex,
                                                        self.ceco,self.csiz,self.ftag,self.itgSlider.x, self.offSlider.x)
            self.shared_state.fseg = fseg
            self.shared_state.feco = feco
            self.shared_state.tmpo = self.tmp
            self.shared_state.igpx = self.itgSlider.x
            self.shared_state.lcut2 = self.offSlider.x

            option_plot = ''
            if self.extended.isChecked():
                option_plot = 'extended'
            if self.colorcontours.isChecked():
                option_plot = 'contours'
            if self.colorlabels.isChecked():
                option_plot = 'labels'

            self.valuesChanged3.emit(self.itgSlider.x,self.offSlider.x,option_plot)
    
    def plot_merge(self):
        from utilities import merge_labels
        self.image = self.shared_state.masked_image.copy()
        # eq_im = equalize(invert(img_as_ubyte(self.image)), disk(1 / 0.06))
        self.labels = self.shared_state.old_labels.copy()
        self.inos = self.shared_state.inos.copy()

        self.ceco, self.fmex, self.tg, self.new_labels = merge_labels(self.image, self.inos, self.labels,
                                                                          self.refSlider.x,
                                                                          self.topSlider.x, self.brightspin.value(),
                                                                          self.pxspin.value(), self.refthspin.value())


        self.ceco = self.ceco.astype('float32')
        self.ceco[self.ceco == 0] = None

        self.plot(self.ceco)
        self.shared_state.ceco = self.ceco
        self.shared_state.fmex = self.fmex
        self.shared_state.tags = self.tg
        self.shared_state.new_labels = self.new_labels

        self.shared_state.ntop = self.pxspin.value()
        self.shared_state.lmer = self.refSlider.x
        self.shared_state.ltop = self.topSlider.x
        self.shared_state.imex = self.brightspin.value()
        self.shared_state.nmer = self.refthspin.value()


    def start_timer(self):
        self.timer.start(200)

    def normalize_image(self):
        from utilities import normalize_labels
        self.image = self.shared_state.masked_image.astype('float32')
        self.labels = self.shared_state.old_labels.copy()
        self.inos = normalize_labels(self.image, self.labels)
        self.shared_state.inos = self.inos
        self.plot(self.shared_state.masked_image)

    def close_widget(self):
        self.close()

    def save_props(self):
        pass

    def openfile_load_config(self):
        # Open a file dialog and get the selected file name
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Config File", "", "Config Files (*.ini)")

        # Check if a file name was selected
        if file_name:
            # Load the image from the file
            self.load_config(file_name)
            
    def load_config(self,filename):
        import configparser
        from utilities import apply_config_file
        self.image = self.shared_state.masked_image.copy()
        config = configparser.ConfigParser()
        config.read(filename)
        hmin = int(config['Param h-minima']['Percentile h-min'])
        radius = float(config['Param h-minima']['radius'])
        lmer = int(config['Merging']['Reference'])
        ltop = int(config['Merging']['Top threshold'])
        imex = float(config['Merging']['Brightness'])
        ntop = int(config['Merging']['px>thresh'])
        nmer = int(config['Merging']['Reference threshold'])
        lcut = int(config['Shrinking']['Cut-off threshold'])
        igpx = int(config['Intergranular']['Intergranular Level'])
        lcut2 = int(config['Intergranular']['Cut-off Level'])
        self.percSlider.slider.setValue(int(hmin))
        self.refSlider.slider.setValue(int(lmer))
        self.topSlider.slider.setValue(int(ltop))
        self.brightspin.setValue(float(imex))
        self.pxspin.setValue(int(ntop))
        self.refthspin.setValue(int(nmer))
        self.cutSlider.slider.setValue(int(lcut))
        self.itgSlider.slider.setValue(int(igpx))
        self.offSlider.slider.setValue(int(lcut2))
        self.shared_state.scale = float(radius)
        self.shared_state.hmin = int(hmin)
        self.shared_state.lmer = int(lmer)
        self.shared_state.ltop = int(ltop)
        self.shared_state.imex = float(imex)
        self.shared_state.ntop = int(ntop)
        self.shared_state.nmer = int(nmer)
        self.shared_state.lcut = int(lcut)
        self.shared_state.igpx = int(igpx)
        self.shared_state.lcut2 = int(lcut2)
        params = [radius,hmin,lmer,ltop,imex,ntop,nmer,lcut,igpx,lcut2]
        self.paramsChanged.emit(params)

    def apply_full(self):
        scale = self.shared_state.scale
        radius = self.shared_state.radius
        hmin = self.shared_state.hmin
        lmer = self.shared_state.lmer
        ltop = self.shared_state.ltop
        imex = self.shared_state.imex
        ntop = self.shared_state.ntop
        nmer = self.shared_state.nmer
        lcut = self.shared_state.lcut
        igpx = self.shared_state.igpx
        lcut2 = self.shared_state.lcut2
        params = [radius, hmin,lmer,ltop,imex,ntop,nmer,lcut,igpx,lcut2]
        self.fullChanged.emit(params)



    def openfile_dialog_config(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(self, "Save Config", "untitle.ini", "Config Files (*.ini);;All Files (*)",
                                                  options=options)
        if filename:
            self.save_config(filename)

        pass
    def save_config(self,filename):
        import configparser
        config = configparser.ConfigParser()
        hmin = str(self.shared_state.hmin)
        radius = str(self.shared_state.radius)
        lmer = str(self.shared_state.lmer)
        ltop = str(self.shared_state.ltop)
        imex = str(self.shared_state.imex)
        ntop = str(self.shared_state.ntop)
        nmer = str(self.shared_state.nmer)
        lcut = str(self.shared_state.lcut)
        igpx = str(self.shared_state.igpx)
        lcut2 =str(self.shared_state.lcut2)
        config['Param h-minima'] = {'Percentile h-min': hmin,"radius":radius}
        config['Merging'] = {'Reference': lmer, 'Top threshold': ltop, 'Brightness': imex, 'px>thresh': ntop, 'Reference threshold': nmer}
        config['Shrinking'] = {'Cut-off threshold': lcut}
        config['Intergranular'] = {'Intergranular Level': igpx, 'Cut-off Level': lcut2}
        with open(filename, 'w') as configfile:
            config.write(configfile)



class MaskWidget(QWidget):
    finished = pyqtSignal()
    def __init__(self, parent=None):
        super(MaskWidget, self).__init__(parent=parent)
        self.shared_state = SharedState()
        self.image = self.shared_state.original_image.copy()

        mainLayout = QVBoxLayout()

        groupBox2 = QGroupBox('Percentage Intensity')
        layout2 = QVBoxLayout()
        self.thresh1Slider = Slider(1, 100, 'Threshold (%)', 'int')
        self.thresh1Slider.slider.setValue(1)
        layout2.addWidget(self.thresh1Slider)

        groupBox2.setLayout(layout2)

        self.endButton = QPushButton("Finish", self)
        self.endButton.clicked.connect(self.close_widget_mask)

        mainLayout.addWidget(groupBox2)
        mainLayout.addWidget(self.endButton)
        self.setLayout(mainLayout)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.func_contours)

        self.old_slider3_x = self.thresh1Slider.x
        self.thresh1Slider.slider.valueChanged.connect(self.start_timer)

        self.graphNorm = pg.GraphicsLayoutWidget(size=(12, 12))
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.graphNorm.resize(500, 500)
        self.p1 = self.graphNorm.addPlot(title="Masking Image")
        self.thresh2 = None
        self.fov = None


    def func_contours(self):
        from skimage.exposure import rescale_intensity
        from skimage.measure import find_contours
        from skimage.color import label2rgb
        slices = np.s_[0:-1,0:-1]
        self.fov = rescale_intensity(self.shared_state.original_image[slices],out_range=(0,1.))
        self.plot(self.fov)
        self.thresh1 = np.percentile(self.fov, self.thresh1Slider.x)
        contours1 = find_contours(self.fov, self.thresh1)
        mask_contours = np.zeros_like(self.fov)
        for contour in contours1:
            mask_contours[contour[:, 0].astype(int), contour[:, 1].astype(int)] = 25
        self.plot(label2rgb(mask_contours, image=self.fov, bg_label=0, alpha=0.5))
        self.shared_state.perc = self.thresh1Slider.x



    def close_widget_mask(self):
        self.masked = self.fov.copy()#self.shared_state.original_image.copy()
        self.masked[self.masked < self.thresh1] = 0.0
        self.shared_state.masked_image = img_as_ubyte(self.masked)
        self.finished.emit()
        self.close()
        self.graphNorm.close()



    def start_timer(self):
        self.timer.start(200)

    def plot(self, data):
        # Set the image data of the ImageItem
        self.image_item = pg.ImageItem()
        self.p1.addItem(self.image_item)
        self.image_item.setImage(data)
        self.graphNorm.show()
