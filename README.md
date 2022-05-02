# CameraKit
A generic toolkit for camera control and camera-world models.

Essential parts of the image calibration routines used in [VisionKit 2.0](git@github.com:ausport/visionkit.git "VisionKit 2.0 Github repository").

Interface to support:
* Image to model correspondences
* Image to model warping
* Estimate camera extrinsics
* Perspective-free image crops.

### CameraKit Modules

#### VKCamera:

Cameras can be instantiated using device-specific `VKCamera` subclasses.  

Details on usage [here](cameras/README.md).


#### VKWorldModel:

World models provides the coordinate translations between image and world spaces.

World models are typically used as a property of a device-specific `VKCamera` class.

![](models/surfaces/hockey.png)


![](images/markers.png)

![](images/verticals.png)

### Installation Notes:

Pip install the headless `opencv-python` package (i.e. `opencv-python-headless`), since the 
versions later than **4.2.0.34** are not compatible with Qt GUI functionality in PyQt5.

Otherwise, `pip install opencv_python==4.2.0.34` still seems to be a safe option.

#### VIMBA Cameras:

Allied Vision (Prosilica) imaging devices are supported by the VIMBA SDK.

Download the SDK [here](https://www.alliedvision.com/en/products/vimba-sdk/#c1497).

Note that the installer requires Python 3.8+