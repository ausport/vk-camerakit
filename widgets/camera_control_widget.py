# QT modules
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class CameraControllerWidget(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self)
        self._parent = parent

        # Arrange layout
        camera_vb = QVBoxLayout(self)

        # # Resultant image scale slider
        # self.pano_scale_slider = QSlider(Qt.Horizontal)
        # self.pano_scale_slider.setValue(30)
        # self.pano_scale_slider.setMinimum(5)
        # self.pano_scale_slider.setMaximum(100)
        # self.pano_scale_slider.valueChanged.connect(self.refresh_parameters)
        # self.pano_scale_label = QLabel("Output Scale: {0}%".format(self.pano_scale_slider.value()))
        # self.pano_scale_hb = QHBoxLayout(self)
        # self.pano_scale_hb.addWidget(self.pano_scale_slider)
        # self.pano_scale_hb.addWidget(self.pano_scale_label)

        # Warping Mode
        self.cboImageResolutionMode = QComboBox(self)
        self.cboImageResolutionMode.addItem("400x400")
        self.cboImageResolutionMode.addItem("320x240")
        self.cboImageResolutionMode.addItem("640x480")
        self.cboImageResolutionMode.addItem("1280x720")
        self.cboImageResolutionMode.addItem("1620x1080")
        self.cboImageResolutionMode.addItem("1620x1220")
        self.cboImageResolutionMode.addItem("1920x1440")

        self.cboImageResolutionMode.currentIndexChanged.connect(self.refresh_widget)
        self.lblImageResolutionMode = QLabel("Image Resolution Mode")

        self.image_resolution_mode_hb = QHBoxLayout(self)
        self.image_resolution_mode_hb.addWidget(self.cboImageResolutionMode)
        self.image_resolution_mode_hb.addWidget(self.lblImageResolutionMode)

        # # Blend Mode
        # self.cboBlendMode = QComboBox(self)
        # self.cboBlendMode.addItem("multiband")
        # self.cboBlendMode.addItem("feather")
        # self.cboBlendMode.currentIndexChanged.connect(self.refresh_widget)
        # self.lblBlendMode = QLabel("Blend Type")
        #
        # self.blend_mode_hb = QHBoxLayout(self)
        # self.blend_mode_hb.addWidget(self.cboBlendMode)
        # self.blend_mode_hb.addWidget(self.lblBlendMode)
        #
        # # Blending strength
        # self.blend_strength_slider = QSlider(Qt.Horizontal)
        # self.blend_strength_slider.setValue(25)
        # self.blend_strength_slider.setMinimum(0)
        # self.blend_strength_slider.setMaximum(50)
        # self.blend_strength_slider.valueChanged.connect(self.refresh_widget)
        # self.blend_strength_label = QLabel("Blend Strength {0}".format(self.blend_strength_slider.value()))
        # self.blend_strength_hb = QHBoxLayout(self)
        # self.blend_strength_hb.addWidget(self.blend_strength_slider)
        # self.blend_strength_hb.addWidget(self.blend_strength_label)

        # Update the camera
        self.btnApplySettings = QToolButton(self)
        self.btnApplySettings.setText('Apply Settings')
        self.btnApplySettings.clicked.connect(self.apply_parameters)

        camera_vb.addLayout(self.image_resolution_mode_hb)
        # camera_vb.addLayout(self.warp_mode_hb)
        # camera_vb.addLayout(self.blend_mode_hb)
        # camera_vb.addLayout(self.blend_strength_hb)
        camera_vb.addWidget(self.btnApplySettings)

        self.setLayout(camera_vb)
        self.setWindowTitle("Camera Control Parameters")

    def refresh_widget(self):
        # Update values and slider labels.
        # self.pano_scale_label.setText("Output Scale: {0}%".format(self.pano_scale_slider.value()))
        # self.blend_strength_label.setText("Blend Strength {0}".format(self.blend_strength_slider.value()))
        pass

    def apply_parameters(self):
        """Applies current capture parameters to the capture device.
        This method assumes that the parent class has a function <parent.update_capture_params(params)>
        """

        if not hasattr(self._parent, "update_capture_params"):
            raise NotImplementedError

        # Parse image resolution
        print(self.cboImageResolutionMode.currentText())
        config = {"CAP_PROP_FRAME_WIDTH": self.cboImageResolutionMode.currentText().split("x")[0],
                  "CAP_PROP_FRAME_HEIGHT": self.cboImageResolutionMode.currentText().split("x")[1]}

        self._parent.update_capture_params(params=config)
