# QT modules
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class PanoramaStitcherWidget(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self)
        self._parent = parent

        # Arrange layout
        pano_vb = QVBoxLayout(self)

        # Resultant image scale slider
        self.pano_scale_slider = QSlider(Qt.Horizontal)
        self.pano_scale_slider.setValue(30)
        self.pano_scale_slider.setMinimum(5)
        self.pano_scale_slider.setMaximum(100)
        self.pano_scale_slider.valueChanged.connect(self.refresh_parameters)
        self.pano_scale_label = QLabel("Output Scale: {0}%".format(self.pano_scale_slider.value()))
        self.pano_scale_hb = QHBoxLayout(self)
        self.pano_scale_hb.addWidget(self.pano_scale_slider)
        self.pano_scale_hb.addWidget(self.pano_scale_label)

        # Warping Mode
        self.cboWarpingMode = QComboBox(self)
        self.cboWarpingMode.addItem("spherical")
        self.cboWarpingMode.addItem("plane")
        self.cboWarpingMode.addItem("affine")
        self.cboWarpingMode.addItem("cylindrical")
        self.cboWarpingMode.addItem("fisheye")
        self.cboWarpingMode.addItem("stereographic")
        self.cboWarpingMode.addItem("compressedPlaneA2B1")
        self.cboWarpingMode.addItem("compressedPlaneA1")
        self.cboWarpingMode.addItem("compressedPlanePortraitA2B1")
        self.cboWarpingMode.addItem("compressedPlanePortraitA1.5B1")
        self.cboWarpingMode.addItem("paniniA2B1")
        self.cboWarpingMode.addItem("paniniA1.5B1")
        self.cboWarpingMode.addItem("paniniPortraitA2B1")
        self.cboWarpingMode.addItem("paniniPortraitA1.5B1")
        self.cboWarpingMode.addItem("mercator")
        self.cboWarpingMode.addItem("transverseMercator")
        self.cboWarpingMode.currentIndexChanged.connect(self.refresh_parameters)
        self.lblWarpingMode = QLabel("Warping Mode")

        self.warp_mode_hb = QHBoxLayout(self)
        self.warp_mode_hb.addWidget(self.cboWarpingMode)
        self.warp_mode_hb.addWidget(self.lblWarpingMode)

        # Blend Mode
        self.cboBlendMode = QComboBox(self)
        self.cboBlendMode.addItem("multiband")
        self.cboBlendMode.addItem("feather")
        self.cboBlendMode.currentIndexChanged.connect(self.refresh_parameters)
        self.lblBlendMode = QLabel("Blend Type")

        self.blend_mode_hb = QHBoxLayout(self)
        self.blend_mode_hb.addWidget(self.cboBlendMode)
        self.blend_mode_hb.addWidget(self.lblBlendMode)

        # Blending strength
        self.blend_strength_slider = QSlider(Qt.Horizontal)
        self.blend_strength_slider.setValue(5)
        self.blend_strength_slider.setMinimum(0)
        self.blend_strength_slider.setMaximum(10)
        self.blend_strength_slider.valueChanged.connect(self.refresh_parameters)
        self.blend_strength_label = QLabel("Blend Strength {0}".format(self.blend_strength_slider.value()))
        self.blend_strength_hb = QHBoxLayout(self)
        self.blend_strength_hb.addWidget(self.blend_strength_slider)
        self.blend_strength_hb.addWidget(self.blend_strength_label)

        # Refresh the pano
        self.btnMakePano = QToolButton(self)
        self.btnMakePano.setText('Refresh Panorama')
        self.btnMakePano.clicked.connect(self.refresh_panorama)

        pano_vb.addLayout(self.pano_scale_hb)
        pano_vb.addLayout(self.warp_mode_hb)
        pano_vb.addLayout(self.blend_mode_hb)
        pano_vb.addLayout(self.blend_strength_hb)
        pano_vb.addWidget(self.btnMakePano)

        self.setLayout(pano_vb)
        self.setWindowTitle("Stitching Parameters")

    def refresh_parameters(self):
        # Update values and slider labels.
        self.pano_scale_label.setText("Output Scale: {0}%".format(self.pano_scale_slider.value()))
        self.blend_strength_label.setText("Blend Strength {0}".format(self.blend_strength_slider.value()))

    def refresh_panorama(self):
        self._parent.update_panorama_params()
