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

## Installation Notes:

### Vimba SDK

If using the Vimba cameras, the Vimba SDK (see below) is required to install `vimbapython`.  This requires a pip environment.

The easiest method is to first install `pipenv` to create an environment in which to install the
`vimbapython` libraries.

A further note - the pip install for `PyQt5` is not compatible with the typical `opencv-python`, since the pip installation
of the latter installs it's own Qt libraries by default (for Linux, but not for OSX).  The easiest workaround here is to
install `opencv-python-headless`, which doesn't install Qt overheads, and avoids this conflict.

On occasion, depending on the order of dependency installation, the following error can occur mixing the pip version of PyQt5, and OpenCV.

```bash
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.

This application failed to start because no Qt platform plugin could be initialized. 
Reinstalling the application may fix this problem.
```

Resolve this by re-installing `xcb`:

```bash
sudo apt install libxcb-xinerama0 
```

### BlackMagic BRAW API

[Aiden Nibali](https://github.com/anibali/pybraw) has developed a set of python wrappers exposing the important functions in the BlackMagic API.

He recommends setting up a Conda environment - follow his instructions if you prefer a Conda.

Alternately, you can install the pybraw package in a pip environment (preserving the Vimba SDK requirements).

You will need to copy all of the Blackmagic libraries to somewhere accessible..  eg.

```text
cd [path_to_downloaded_libraries]/BlackmagicRAW/BlackmagicRawAPI/
sudo cp *.* /usr/lib/x86_64-linux-gnu/
```

Clone the pybraw package from Aiden's github repository:

```text
git clone https://github.com/anibali/pybraw
```
Activate the pip venv, and navigate to the cloned repository.

Install the pybraw package:

```text
pip install --no-build-isolation -U .
```

Test the package has installed correctly:

```python
from pybraw import PixelFormat, ResolutionScale
from pybraw.torch.reader import FrameImageReader
file_name = [path_to_sample_footage]
reader = FrameImageReader(file_name, processing_device='cuda')
# Get the number of frames in the video.
frame_count = reader.frame_count()
print(frame_count)
```


#### Requirements:
```bash
[packages]
filetype = "==1.0.7"
numpy = "==1.19.2"
opencv-python = "==4.5.5.64"
pyqt = "==5.9.2"
vimbapython = "==1.1.0"
Pillow = "==9.1.0"
```
#### VIMBA Cameras:

Allied Vision (Prosilica) imaging devices are supported by the VIMBA SDK.

Download the SDK [here](https://www.alliedvision.com/en/products/vimba-sdk/#c1497).

Note that the installer requires Python 3.8+