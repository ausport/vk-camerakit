import os

# QT modules
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# VK modules
import cameras
SUPPORTED_DEVICES = ["VKCameraGenericDevice", "VKCameraVimbaDevice"]


class CameraControllerWidget(QWidget):
    def __init__(self, parent, devices):

        # The camera control widget should be initialised with a known camera device.
        # The device should be a VKCamera - either VKCameraGenericDevice, or VKCameraVimbaDevice
        for device in devices:
            assert device.__class__.__name__ in SUPPORTED_DEVICES, "A supported VKCamera class is required."

        QWidget.__init__(self)

        # The parent class should implement a callback <update_current_camera_device>
        self._parent = parent
        if not hasattr(parent, "update_current_camera_device"):
            print("Implement <update_current_camera_device> "
                  "in the parent instance to receive device selection callbacks.")
            raise NotImplementedError

        self._devices = devices.copy()
        self._current_active_device = None
        if len(self._devices) > 0:
            self._current_active_device = self._devices[0]

        # Arrange layout
        camera_vb = QVBoxLayout(self)

        # Device selection
        self.cboDeviceSelection = QComboBox(self)
        for device in devices:
            self.cboDeviceSelection.addItem(device.name())

        self.cboDeviceSelection.setCurrentText(devices[0].name())
        self.lblDeviceSelection = QLabel("Available Devices")
        self.cboDeviceSelection.currentIndexChanged.connect(self.refresh_device)
        self.device_selection_hb = QHBoxLayout(self)
        self.device_selection_hb.addWidget(self.cboDeviceSelection)
        self.device_selection_hb.addWidget(self.lblDeviceSelection)

        # Resolution Mode
        self.cboImageResolutionMode = QComboBox(self)
        self.cboImageResolutionMode.addItem("400x400")
        self.cboImageResolutionMode.addItem("320x240")
        self.cboImageResolutionMode.addItem("640x480")
        self.cboImageResolutionMode.addItem("1280x720")
        self.cboImageResolutionMode.addItem("1620x1080")
        self.cboImageResolutionMode.addItem("1620x1220")
        self.cboImageResolutionMode.addItem("1920x1440")
        self.lblImageResolutionMode = QLabel("Image Resolution Mode")
        self.image_resolution_mode_hb = QHBoxLayout(self)
        self.image_resolution_mode_hb.addWidget(self.cboImageResolutionMode)
        self.image_resolution_mode_hb.addWidget(self.lblImageResolutionMode)

        # Exposure
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setValue(1)
        self.exposure_slider.setMinimum(1)
        self.exposure_slider.setMaximum(100000)
        self.exposure_slider.valueChanged.connect(self.update_slider_controls)
        self.exposure_label = QLabel("Exposure (μs): {0}".format(self.exposure_slider.value()))
        self.exposure_hb = QHBoxLayout(self)
        self.exposure_hb.addWidget(self.exposure_slider)
        self.exposure_hb.addWidget(self.exposure_label)

        # FPS
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setValue(25)
        self.fps_slider.setMinimum(1)
        self.fps_slider.setMaximum(60)
        self.fps_slider.valueChanged.connect(self.update_slider_controls)
        self.fps_label = QLabel("F.P.S.: {0}".format(self.fps_slider.value()))
        self.fps_hb = QHBoxLayout(self)
        self.fps_hb.addWidget(self.fps_slider)
        self.fps_hb.addWidget(self.fps_label)

        # Update the camera
        self.btnApplySettings = QToolButton(self)
        self.btnApplySettings.setText('Apply Settings')
        self.btnApplySettings.clicked.connect(self.apply_parameters)

        # Save a video from the camera
        self.btnSaveVideo = QToolButton(self)
        self.btnSaveVideo.setText('Save Video')
        self.btnSaveVideo.clicked.connect(self.save_video)

        # Layout widgets
        camera_vb.addLayout(self.device_selection_hb)
        camera_vb.addLayout(self.image_resolution_mode_hb)
        camera_vb.addLayout(self.exposure_hb)
        camera_vb.addLayout(self.fps_hb)
        camera_vb.addWidget(self.btnApplySettings)
        camera_vb.addWidget(self.btnSaveVideo)

        self.setLayout(camera_vb)
        self.setWindowTitle("Camera Control Parameters")

        # Refresh components
        self.refresh_widget_with_camera_properties()

    def refresh_widget_with_camera_properties(self):
        # Update values and slider labels.
        # Get resolution for current device..
        if self._devices is not None:
            _w = self._current_active_device.width()
            _h = self._current_active_device.height()
            self.cboImageResolutionMode.setCurrentText("{0}x{1}".format(_w, _h))

        # Sliders
        self.fps_label.setText("F.P.S.: {0}".format(int(self._current_active_device.fps())))
        self.fps_slider.setValue(self._current_active_device.fps())
        self.exposure_label.setText("Exposure (μs): {0}".format(self.exposure_slider.value()))
        self.exposure_slider.setValue(self._current_active_device.exposure_time() * 1e3)

    def update_slider_controls(self):
        self.fps_label.setText("F.P.S.: {0}".format(int(self.fps_slider.value())))
        self.exposure_label.setText("Exposure (μs): {0}".format(self.exposure_slider.value()))

    def refresh_device(self):
        _i = self.cboDeviceSelection.currentIndex()
        self._current_active_device = self._devices[_i]
        print(self._current_active_device)
        self.refresh_widget_with_camera_properties()

    def apply_parameters(self):
        """Applies current capture parameters to the capture device.
        This method assumes that the parent class has a function <parent.update_capture_params(params)>
        """

        if not hasattr(self._current_active_device, "set_capture_parameters"):
            raise NotImplementedError

        # Parse image resolution
        config = {"CAP_PROP_FRAME_WIDTH": self.cboImageResolutionMode.currentText().split("x")[0],
                  "CAP_PROP_FRAME_HEIGHT": self.cboImageResolutionMode.currentText().split("x")[1],
                  "CAP_PROP_EXPOSURE": self.exposure_slider.value(),
                  "CAP_PROP_FPS": self.fps_slider.value(),
                  }

        self._current_active_device.set_capture_parameters(configs=config)
        self.refresh_widget_with_camera_properties()
        self._parent.update_current_camera_device(camera=self._current_active_device)

    def save_video(self):

        # path = QFileDialog.getSaveFileName(self, 'Export Panorama Composite',
        #                                    self._current_active_device.name(),
        #                                    "mp4(*.mp4)")

        self._current_active_device.save_video(video_export_path="/home/stuart/Desktop/test.mp4",
                                               size=(self._current_active_device.width(), self._current_active_device.height()),
                                               fps=self._current_active_device.fps())


        # import threading
        # th = threading.Thread(target=self._current_active_device.save_video, args=(path[0],
        #                                                                            (self._current_active_device.width(), self._current_active_device.height()),
        #                                                                            self._current_active_device.fps()))
        #
        # th.start()
