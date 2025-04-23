import sys

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QFileDialog
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QComboBox, QPushButton,QMessageBox

#from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

from pop_ups import ThresholdMultiOtsuWidget, ThresholdNiblack, ThresholdSauvola, ThresholdLi, ThresholdOtsu, ThresholdYen
from pop_ups import ThresholdMinimum, ThresholdTriangle, ThresholdIsodata, ThresholdLocal, ThresholdImhmin, MaskWidget
from shared_state import SharedState

import numpy as np
from astropy.io import fits

from skimage.filters.rank import equalize
from skimage.filters import threshold_multiotsu, threshold_niblack, threshold_sauvola
from skimage.filters import threshold_li,threshold_otsu,threshold_yen,threshold_mean
from skimage.filters import threshold_minimum,threshold_triangle,threshold_isodata
from skimage.filters import threshold_local
from skimage.util import img_as_ubyte, invert
from skimage.morphology import disk,square,reconstruction
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.segmentation import watershed
from skimage.exposure import rescale_intensity


#from sklearn.ensemble import RandomForestClassifier
#from functools import partial
#import time

pg.setConfigOptions(imageAxisOrder='row-major')

list_thrs = ['threshold_multiotsu', 'threshold_niblack', 'threshold_sauvola', 'threshold_li', 'threshold_otsu',
             'threshold_yen', 'threshold_mean', 'threshold_minimum', 'threshold_triangle', 'threshold_isodata',
             'threshold_local','imhmin']

filepath = '/home/hypnus1803/Downloads/MLT_4/IMAGE.BYT'


def imhmin(arr, h):
    im = invert(arr)
    seed = im - h
    suppress = reconstruction(seed, im, footprint=square(3))
    return invert(suppress)

class MessageMask(QMessageBox):
    yesClicked = pyqtSignal()
    noClicked = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        #self.setText("If the current FOV contains pores or sunspots, do you want to mask them?")
        self.setIcon(QMessageBox.Information)
        self.setWindowTitle("Masking Image")

        # Create a new QPushButton and set it as the accept button
        self.yes_button = QPushButton('Mask', self)
        self.no_button = QPushButton('No mask', self)
        self.addButton(self.yes_button, QMessageBox.AcceptRole)
        self.addButton(self.no_button, QMessageBox.RejectRole)

        self.yes_button.clicked.connect(self.emit_yes_clicked)
        self.no_button.clicked.connect(self.emit_no_clicked)

    def emit_yes_clicked(self):
        # Emit the continueClicked signal when the Continue button is clicked
        self.yesClicked.emit()

    def emit_no_clicked(self):
        # Emit the continueClicked signal when the Continue button is clicked
        self.noClicked.emit()

    def show_popup(self,message):
        # Add any custom logic here
        self.setText(message)
        self.show()

class MessageCut(QMessageBox):
    yesClicked = pyqtSignal()
    noClicked = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        #self.setText("If the current FOV contains pores or sunspots, do you want to mask them?")
        self.setIcon(QMessageBox.Information)
        self.setWindowTitle("Cutting Image")

        # Create a new QPushButton and set it as the accept button
        self.yes_button = QPushButton('Automatic ROI', self)
        self.no_button = QPushButton('Manual ROI', self)
        self.addButton(self.yes_button, QMessageBox.AcceptRole)
        self.addButton(self.no_button, QMessageBox.RejectRole)

        self.yes_button.clicked.connect(self.emit_yes_clicked)
        self.no_button.clicked.connect(self.emit_no_clicked)

    def emit_yes_clicked(self):
        # Emit the continueClicked signal when the Continue button is clicked
        self.yesClicked.emit()

    def emit_no_clicked(self):
        # Emit the continueClicked signal when the Continue button is clicked
        self.noClicked.emit()

    def show_popup(self,message):
        # Add any custom logic here
        self.setText(message)
        self.show()



class CustomGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    sigClose = pyqtSignal()

    def closeEvent(self, event):
        self.sigClose.emit()
        super().closeEvent(event)


class CustomWidget(QWidget):
    def __init__(self, parent=None):
        super(CustomWidget, self).__init__(parent=parent)
        self.image = None #img_as_float64(np.fromfile(filepath,dtype=np.uint8).reshape([401,399]))
        self.markers = None #np.zeros_like(self.image, dtype=np.uint8)
        self.screen_geometry = QApplication.desktop().screenGeometry()


        # Create a QVBoxLayout instance
        layout = QVBoxLayout()

        # Create a GraphicsLayoutWidget instance
        self.graphWidget = pg.GraphicsLayoutWidget(size=(10,10))
        p1 = self.graphWidget.addPlot(title="Intensity Image")
        self.img1 = pg.ImageItem()
        p1.addItem(self.img1)
        #img1.setImage(self.image)

        p2 = self.graphWidget.addPlot(title="Markers Image")
        p2.setXLink(p1)
        p2.setYLink(p1)
        self.img2 = pg.ImageItem()
        p2.addItem(self.img2)
        #self.img2.setImage(self.markers)


        # Create a QSlider instance
        layout2 = QHBoxLayout()
        # self.slider = Slider(1, 50,'var','float')
        self.open_ima = QPushButton('Open Image')
        self.open_ima.clicked.connect(self.openImageFileDialog)
        self.open_cube = QPushButton('Open Cube')
        self.open_cube.clicked.connect(self.openCubeFileDialog)

        self.label_opts = QLabel('Thresholding Options')
        self.label_opts.setAlignment(Qt.AlignRight | Qt.AlignCenter)
        # Create a QComboBox instance
        self.comboBox = QComboBox()
        self.comboBox.addItems([""]+list_thrs)
        for i in range(1,self.comboBox.count()):
            if self.comboBox.itemText(i) != 'imhmin':
                self.comboBox.model().item(i).setEnabled(False)

        # Connect the currentIndexChanged signal to a slot
        self.comboBox.currentIndexChanged.connect(self.updateCombobox)

        # Define a dictionary that maps each option to a tuple of minimum and maximum values


        # Add the GraphicsLayoutWidget and QSlider to the QVBoxLayout
        layout.addWidget(self.graphWidget)
        # layout2.addWidget(self.slider)
        #layout2.addStretch(1)
        layout2.addWidget(self.open_ima)
        layout2.addWidget(self.open_cube)
        layout2.addWidget(self.label_opts)
        layout2.addWidget(self.comboBox)
        layout.addLayout(layout2)

        # Set the layout of the CustomWidget to the QVBoxLayout
        self.setLayout(layout)

        self.shared_state = SharedState()

        self.win_cut = CustomGraphicsLayoutWidget()#pg.GraphicsLayoutWidget()
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.win_cut.resize(800, 400)
        self.p1_cut = self.win_cut.addPlot(title="Choose Region of Interest")
        self.p2_cut = self.win_cut.addPlot(title="Region of Interest")
        self.selected = 0
        #print(dir(self.win_cut))
        self.win_cut.sigClose.connect(self.on_win_cut_close)

        self.iscut = False
        self.iscube = False



        #print(np.histogram_bin_edges(self.image.ravel(),bins='fd').size)

    def openImageFileDialog(self):
        # Open a file dialog and get the selected file name
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.fits *.fit *.fts)")

        # Check if a file name was selected
        if file_name:
            # Load the image from the file
            self.loadImage(file_name)

    def loadImage(self, file_name):
        hdu = fits.open(file_name)
        self.data = hdu[0].data
        #clip_min,clip_max = np.percentile([np.min(self.data),np.max(self.data)],(30,80))
        self.data = np.clip(self.data,0.4,1.4) #np.clip(self.data,clip_min,clip_max)
        shape = self.data.shape
        if len(shape) == 3:
            QMessageBox.warning(self, "Data Cube Detected", "You are loading a data cube, use instead the button Open Cube")
        else:
            self.shared_state.full_image = self.data.astype('float32')
            self.shared_state.original_image = self.data
            if self.data.size > 500**2:

                self.popup2 = MessageCut()
                self.popup2.show_popup("Image is too large, maximum optimal size is (500 x 500 pixels).\n Do you want to select the region of interest?")
                self.popup2.yesClicked.connect(self.start_automaticcut)
                self.popup2.noClicked.connect(self.start_manualcut)
            else:
                self.popup1 = MessageMask()
                self.popup1.show_popup("If the current FOV contains pores or sunspots, \ndo you want to mask them?")
                self.popup1.yesClicked.connect(self.start_masking)
                self.popup1.noClicked.connect(self.no_masking)
                #slc = np.s_[0:-1,0:-1]
                self.image = img_as_ubyte(rescale_intensity(self.data,out_range=(0,1)))
                self.shared_state.original_image = self.image

            # self.shared_state.original_image = self.image
            #self.image += 1
            # print(self.shared_state.original_image.shape)
            # print(self.shared_state.masked_image.shape)
            self.markers = np.zeros_like(self.shared_state.original_image, dtype=np.uint8)

            # width = int(np.round(2 * self.image.shape[1]))
            # height = int(np.round(self.image.shape[0]))
            # aspect = height / ( width)
            #
            # if width > self.screen_geometry.width():
            #     self.setMinimumSize(width//2,int(aspect * width)//2)
            #     #self.setMaximumWidth(self.screen_geometry.width())
            # else:
            #     self.setMinimumSize(width, height)

            self.img1.setImage(self.image)

    def openCubeFileDialog(self):
        # Open a file dialog and get the selected file name
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Cube", "", "Cube Files (*.fits *.fit *.fts)")

        # Check if a file name was selected
        if file_name:
            # Load the image from the file
            self.loadCube(file_name)

    def loadCube(self, file_name):
        from utilities import best_contrast_image
        hdu = fits.open(file_name)
        #self.data = hdu[0].data
        shape = hdu[0].data.shape
        if len(shape) < 3:
            QMessageBox.warning(self, "Wrong Data Detected", f"You are loading data with {len(shape)} dimmensions. Data cube must have 3 dimensions.")
        else:
            self.shared_state.hdu_cube = hdu
            self.iscube = True
            index = best_contrast_image(hdu)
            self.frame0 = hdu[0].data[index,:,:]
            self.shared_state.full_image = self.frame0.astype('float64')
            self.shared_state.original_image = self.frame0
            if self.frame0.size > 500 ** 2:
                self.popup2 = MessageCut()
                self.popup2.show_popup(
                    "Initial frame is too large, maximum optimal size is (500 x 500 pixels).\n Do you want to select the region of interest?")
                self.popup2.yesClicked.connect(self.start_automaticcut)
                self.popup2.noClicked.connect(self.start_manualcut)
            else:
                self.popup1 = MessageMask()
                self.popup1.show_popup("If the current FOV contains pores or sunspots, \ndo you want to mask them?")
                self.popup1.yesClicked.connect(self.start_masking)
                self.popup1.noClicked.connect(self.no_masking)
                #slc = np.s_[0:-1,0:-1]
                self.image = img_as_ubyte(rescale_intensity(self.frame0,out_range=(0,1)))
                self.shared_state.original_image = self.image




    def updateFigure(self):
        self.image = img_as_ubyte(self.shared_state.masked_image)
        self.img1.setImage(self.image)

    def no_masking(self):
        self.shared_state.masked_image = self.shared_state.original_image
        self.image = img_as_ubyte(self.shared_state.masked_image)
        self.img1.setImage(self.image)
    def start_masking(self):

        self.masking_widget = MaskWidget()
        #self.masking_widget.valuesChanged.connect(self.functionMask)
        # Plot the result of self.widget.inos
        #self.plot_popup_merge.plot(self.inos)
        # Show the PlotPopup
        self.masking_widget.show()
        self.masking_widget.finished.connect(self.updateFigure)

        # print('Updating FIGURE')
        # print(self.shared_state.masked_image)
        #self.img1.setImage(self.shared_state.masked_image.T)

    def start_automaticcut(self):
        self.image = self.shared_state.original_image.copy()
        shape_y, shape_x = self.image.shape
        center_y, center_x = shape_y//2, shape_x//2
        if shape_y < 500:
            new_shape_x = (500**2)//shape_y
            self.image = self.image[center_y-shape_y//2:center_y+shape_y//2,center_x-new_shape_x//2:center_x+new_shape_x//2]
        if shape_x < 500:
            new_shape_y = (500**2)//shape_x
            self.image = self.image[center_y-new_shape_y//2:center_y+new_shape_y//2,center_x-shape_x//2:center_x+shape_x//2]
        else:
            self.image = self.image[center_y-250:center_y+250,center_x-250:center_x+250]

        self.image = img_as_ubyte(rescale_intensity(self.image))
        self.img1.setImage(self.image)
        self.shared_state.original_image = self.image

        self.popup1 = MessageMask()
        self.popup1.show_popup("If the current FOV contains pores or sunspots, do you want to mask them?")
        self.popup1.yesClicked.connect(self.start_masking)
        self.popup1.noClicked.connect(self.no_masking)
        self.iscut = True

    def start_manualcut(self):
        self.selected = 0
        self.image = self.shared_state.original_image.copy()
        self.img1_cut = pg.ImageItem()
        roi = pg.ROI([0, 0], [500, 500],pen="r")
        roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.p1_cut.addItem(roi)
        roi.setZValue(10)

        self.p1_cut.addItem(self.img1_cut)
        self.img1_cut.setImage(self.image)

        self.img2_cut = pg.ImageItem()
        self.p2_cut.addItem(self.img2_cut)

        def updatePlot():
            #global img, roi, data, self.p2_cut
            self.selected = roi.getArrayRegion(self.image, self.img1_cut)
            slices_roi = roi.getArraySlice(self.image, self.img1_cut)
            #print(self.selected.shape)
            if self.selected.size > 500**2:
                self.selected = np.ones([500,500])
                self.img2_cut.setImage(self.selected)
                potencia_str = "<math>500<sup>2</sup></math>"
                self.p2_cut.setTitle(f"ROI is larger than {potencia_str} pixels")
            else:
                self.img2_cut.setImage(self.selected)
                self.p2_cut.setTitle(
                    f"Bottom left pixel: ({slices_roi[0][0].start},{slices_roi[0][1].start}): Top right pixel: ({slices_roi[0][0].stop-1},{slices_roi[0][1].stop-1})")

        roi.sigRegionChanged.connect(updatePlot)
        updatePlot()
        self.win_cut.show()

    def on_win_cut_close(self):

        # This function will be called when the win_cut window is closed
        self.image = img_as_ubyte(rescale_intensity(self.selected))
        self.img1.setImage(self.image)
        self.shared_state.original_image = self.image
        self.popup1 = MessageMask()
        self.popup1.show_popup("If the current FOV contains pores or sunspots, do you want to mask them?")
        self.popup1.yesClicked.connect(self.start_masking)
        self.popup1.noClicked.connect(self.no_masking)
        self.iscut = True




    def updateCombobox(self):
        # Get the current text of the QComboBox
        current_text = self.comboBox.currentText()
        print(current_text)
        if current_text == '':
            print('Empty')

        if current_text == 'threshold_multiotsu':
            # Create an instance of ThresholdMultiOtsuWidget and show it
            self.threshold_multiotsu_widget = ThresholdMultiOtsuWidget()
            self.threshold_multiotsu_widget.valuesChanged.connect(self.function01)
            self.threshold_multiotsu_widget.show()

        if current_text == 'threshold_niblack':
            # Create an instance of ThresholdNiblack and show it
            self.threshold_niblack = ThresholdNiblack()
            self.threshold_niblack.valuesChanged.connect(self.function02)
            self.threshold_niblack.show()

        if current_text == 'threshold_sauvola':
            # Create an instance of ThresholdSauvola and show it
            self.threshold_sauvola = ThresholdSauvola()
            self.threshold_sauvola.valuesChanged.connect(self.function03)
            self.threshold_sauvola.show()

        if current_text == 'threshold_li':
            # Create an instance of ThresholdLI and show it
            self.threshold_li = ThresholdLi()
            self.threshold_li.valuesChanged.connect(self.function04)
            self.threshold_li.show()

        if current_text == 'threshold_otsu':
            # Create an instance of ThresholdOtsu and show it
            if np.issubdtype(self.image.dtype, np.floating):
                self.threshold_otsu = ThresholdOtsu()
                self.threshold_otsu.valuesChanged.connect(self.function05_1)
                self.threshold_otsu.show()
            else:
                self.function05_2()

        if current_text == 'threshold_yen':
            # Create an instance of ThresholdYen and show it
            if np.issubdtype(self.image.dtype, np.floating):
                self.threshold_yen = ThresholdYen()
                self.threshold_yen.valuesChanged.connect(self.function06_1)
                self.threshold_yen.show()
            else:
                self.function06_2()

        if current_text == 'threshold_mean':
            self.function07()

        if current_text == 'threshold_minimum':
            # Create an instance of ThresholdYen and show it
            if np.issubdtype(self.image.dtype, np.floating):
                self.threshold_minimum = ThresholdMinimum()
                self.threshold_minimum.valuesChanged.connect(self.function08_1)
                self.threshold_minimum.show()
            else:
                self.function08_2()

        if current_text == 'threshold_triangle':
            # Create an instance of ThresholdYen and show it
            if np.issubdtype(self.image.dtype, np.floating):
                self.threshold_triangle = ThresholdTriangle()
                self.threshold_triangle.valuesChanged.connect(self.function09_1)
                self.threshold_triangle.show()
            else:
                self.function09_2()

        if current_text == 'threshold_isodata':
            # Create an instance of ThresholdLI and show it
            self.threshold_isodata = ThresholdIsodata()
            self.threshold_isodata.valuesChanged.connect(self.function10)
            self.threshold_isodata.show()

        if current_text == 'threshold_local':
            # Create an instance of ThresholdLI and show it
            self.threshold_local = ThresholdLocal()
            self.threshold_local.valuesChanged.connect(self.function11)
            self.threshold_local.show()

        if current_text == 'imhmin':
            self.threshold_imhmin = ThresholdImhmin()
            self.threshold_imhmin.valuesChanged.connect(self.function12)
            self.threshold_imhmin.show()
            self.threshold_imhmin.valuesChanged2.connect(self.function13)
            self.threshold_imhmin.valuesChanged3.connect(self.function14)
            self.threshold_imhmin.paramsChanged.connect(self.function15)
            self.threshold_imhmin.fullChanged.connect(self.function16)
            # self.threshold_imhmin.progressChanged.connect(self.function17)

            if self.iscut:
                self.threshold_imhmin.applyButton.setEnabled(True)
            if self.iscube:
                self.threshold_imhmin.apply_cubebutton.setEnabled(True)



    #
    def function01(self, classes, nbins):
 
        self.img2.setImage(self.shared_state.markers)
        #self.image = self.shared_state.original_image

        thrs = threshold_multiotsu(self.image, classes=classes, nbins=nbins)
        self.markers = np.digitize(self.image, thrs)
        self.img2.setImage(self.markers)
        print('threshold_multiotsu')
    def function02(self,window,k,q):
        self.img2.setImage(self.shared_state.markers)
        thrs = threshold_niblack(self.image, window_size=window, k=k)*q
        self.markers = self.image > thrs
        self.img2.setImage(self.markers)
        #print('threshold_niblack')
    def function03(self,window,k):
        self.img2.setImage(self.shared_state.markers)
        thrs = threshold_sauvola(self.image, window_size=window, k=k)
        self.markers = self.image > thrs
        self.img2.setImage(self.markers)

    def function04(self,init_guess):
        self.img2.setImage(self.shared_state.markers)
        ubyt_im = img_as_ubyte(self.image)
        iter_thresholds2 = []
        opt_threshold2 = threshold_li(ubyt_im, initial_guess=init_guess, iter_callback=iter_thresholds2.append)
        self.markers = ubyt_im > opt_threshold2
        self.img2.setImage(self.markers)

    def function05_1(self,nbins):
        self.img2.setImage(self.markers)
        thrs = threshold_otsu(self.image, nbins=nbins)
        self.markers = self.image > thrs
        self.img2.setImage(self.markers)

    def function05_2(self):
        self.img2.setImage(self.markers)
        thrs = threshold_otsu(self.image)
        self.markers = self.image > thrs
        self.img2.setImage(self.markers)

    def function06_1(self, nbins):
        self.img2.setImage(self.markers)
        thrs = threshold_yen(self.image, nbins=nbins)
        self.markers = self.image > thrs
        self.img2.setImage(self.markers)

    def function06_2(self):
        self.img2.setImage(self.markers)
        thrs = threshold_yen(self.image)
        self.markers = self.image > thrs
        self.img2.setImage(self.markers)

    def function07(self):
        self.img2.setImage(self.markers)
        thrs = threshold_mean(self.image)
        self.markers = self.image > thrs
        self.img2.setImage(self.markers)
        print('threshold_mean')

    def function08_1(self, nbins):
        self.img2.setImage(self.markers)
        thrs = threshold_minimum(self.image, nbins=nbins)
        self.markers = self.image > thrs
        self.img2.setImage(self.markers)

    def function08_2(self):
        self.img2.setImage(self.markers)
        thrs = threshold_minimum(self.image)
        self.markers = self.image > thrs
        self.img2.setImage(self.markers)

    def function09_1(self, nbins):
        self.img2.setImage(self.markers)
        thrs = threshold_triangle(self.image, nbins=nbins)
        self.markers = self.image > thrs
        self.img2.setImage(self.markers)

    def function09_2(self):
        self.img2.setImage(self.markers)
        thrs = threshold_triangle(self.image)
        self.markers = self.image > thrs
        self.img2.setImage(self.markers)

    def function10(self,nbins,returning):
        self.img2.setImage(self.markers)
        if eval(returning):
            thrs = threshold_isodata(self.image, nbins=nbins,return_all=eval(returning))
            thr = np.mean(thrs)
        else:
            thr = threshold_isodata(self.image, nbins=nbins,return_all=eval(returning))

        self.markers = self.image > thr
        self.img2.setImage(self.markers)
        print('threshold_isodata')

    def function11(self,block,method,mode):
        self.img2.setImage(self.markers)
        thrs = threshold_local(self.image, block_size=block, method=method, mode=mode)
        self.markers = self.image > thrs
        self.img2.setImage(self.markers)
        print('threshold_local')

    def function12(self,percentile,radius):
        self.img2.setImage(self.markers)

        #from sklearn.cluster import KMeans #5
        #kmeans = KMeans(n_clusters=5, random_state=0).fit(self.image.reshape(self.image.size,1))
        #im_classes = kmeans.cluster_centers_[kmeans.labels_].reshape(self.image.shape)
        #im_classes = img_as_ubyte(rescale_intensity(im_classes,out_range=(0,1)))

        eq_im = equalize(invert(self.image),disk(radius))
        #eq_im = equalize(invert(im_classes),disk(radius))
        self.markers = imhmin(eq_im,np.percentile(eq_im,percentile))
        self.ws2 = watershed(self.markers, connectivity=2)
        self.labels = label(self.ws2)
        self.labels[self.image == 0] = 0
        self.shared_state.old_labels = self.labels
        self.image_label_overlay2 = label2rgb(self.labels, image=self.image, bg_label=0, alpha=0.6)
        self.img2.setImage(self.image_label_overlay2)
        #print('threshold_ihmin')

    def function13(self):
        self.img2.setImage(self.markers)
        self.image = self.shared_state.masked_image
        self.ftag = self.shared_state.ftag
        self.img2.setImage(label2rgb(self.ftag,image=self.image, bg_label=0,alpha=0.6))
        print('cut-off')

    def function14(self,a,b,option_plot):
        self.img2.setImage(self.markers)
        self.image = self.shared_state.masked_image
        self.tmp = self.shared_state.tmpo
        self.feco = self.shared_state.feco # contours of the granules
        self.fseg = self.shared_state.fseg # granules
        if option_plot == "extended":
            self.img2.setImage(self.fseg)
        if option_plot == "contours":
            self.img2.setImage(label2rgb(self.feco,image=self.image, bg_label=0,alpha=0.6))
        if option_plot == "labels":
            self.img2.setImage(label2rgb(self.tmp,image=self.image, bg_label=0,alpha=0.7))

        print('Intergranular')

    def function15(self,params):
        from utilities import apply_config_file
        self.image = self.shared_state.masked_image
        self.fseg, _, _ = apply_config_file(self.image, params)
        self.img2.setImage(self.fseg)

    def function16(self,params):
        from utilities import apply_config_file

        self.image = self.shared_state.full_image
        self.shared_state.original_image = self.image
        fov = rescale_intensity(self.shared_state.original_image, out_range=(0, 1.))
        if self.shared_state.perc != None:
            thresh1 = np.percentile(fov, self.shared_state.perc)
        else:
            thresh1 = 0.0
        masked = fov.copy()  # self.shared_state.original_image.copy()
        masked[masked < thresh1] = 0.0
        self.shared_state.masked_image = img_as_ubyte(masked)

        self.fseg, _, _ = apply_config_file(self.shared_state.masked_image, params)
        self.new_mask = self.fseg.copy()
        vals0 = np.unique(self.new_mask)
        self.new_mask[self.new_mask == vals0[-1]] = vals0[-2]
        self.new_mask[self.new_mask == vals0[0]] = vals0[1]

        self.tmp_labels = label(self.new_mask)
        self.image = self.shared_state.masked_image.copy()
        # eq_im = equalize(invert(self.image),disk(1/self.shared_state.scale))
        # bp_min_flux = np.nanmean(eq_im) + 1.25*np.nanstd(eq_im)
        bp_min_flux = np.nanmean(self.image) + 1.25 * np.nanstd(self.image)
        for reg in regionprops(self.tmp_labels,intensity_image=self.image):
            if reg.intensity_mean >= bp_min_flux:
                self.new_mask[self.tmp_labels == reg.label] = 200
        self.shared_state.classes = self.new_mask
        self.img1.setImage(self.shared_state.masked_image)
        self.img2.setImage(self.new_mask)

    # def function17(self,progress):
    #     self.progress = progress
    #     self.progress_dialog = ProgressDialog()
    #     self.progress_dialog.show()
    #     self.progress_dialog.setProgress(self.progress)
    #     if self.progress == 100:
    #         self.progress_dialog.accept()



if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create a CustomWidget instance and show it
    widget = CustomWidget()
    widget.show()

    sys.exit(app.exec_())
