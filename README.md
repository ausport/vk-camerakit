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

An error seems to emanate on some systems:

```bash
Failed to load platform plugin "xcb". Available platforms are:
```

This can be resolved by reinstalling the `xcb` library:
```bash
sudo apt-get install --reinstall libxcb-xinerama0
```
or, 

```bash
sudo apt-get install libqt5x11extras5
```

#### VIMBA Cameras:

Allied Vision (Prosilica) imaging devices are supported by the VIMBA SDK.

Download the SDK [here](https://www.alliedvision.com/en/products/vimba-sdk/#c1497).

Note that the installer requires Python 3.8+