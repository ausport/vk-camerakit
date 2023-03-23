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

![](surfaces/hockey.png)


![](images/markers.png)

![](images/verticals.png)

# Setup

VK-CameraKit supports a range of camera classes, some of which include their own sets of 
dependencies.

The following instructions support generic installation, but additional dependencies 
may be required to enable specific devices.

Activate a Conda environment with the generically-required dependencies:

```shell
conda env create -f environment.yml
conda activate vkcamerakit
```


### BlackMagic (braw format) Camera Support - (braw format) - [Optional]

[Aiden Nibali](https://github.com/anibali/pybraw) has developed a set of python wrappers exposing the important functions in the BlackMagic API.

If these are desired, the `vkcamerakit` Conda environment should include all of the required dependencies for the
`pybraw` library, so you may ignore the steps for creating the `pybraw` environment.

You will, however, still need to download the [Blackmagic RAW 2.1 SDK for Linux](https://www.blackmagicdesign.com/support/download/ea11ce9660c642879612f363ca387c7f/Linux)
and copy `libBlackmagicRawAPI.so` into your Conda environment's library path.

```shell
cd [path_to_downloaded_libraries]/BlackmagicRAW/BlackmagicRawAPI/
sudo cp *.so "$CONDA_PREFIX/lib/"
```

Clone the pybraw package from Aiden's GitHub repository:

```shell
git clone https://github.com/anibali/pybraw
```
Ensure the current `vkcamerakit` Conda venv is activated, and navigate to the cloned repository.

Install the pybraw package:

```shell
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

Now, continue with the setup for the `vk-camerakit` package.

Again, ensure that the `vkcamerakit` Conda venv is activated, and navigate to
the root directory in the vk-camerakit repository.

Install the `vk-camerakit` package:

```shell
pip install .
```

If you plan on developing the `vk-camerakit` package from the source code, install the 
package in develop mode:

```shell
pip install -e .
```

Any changes made to the `vk-camerakit` repository will be automatically mapped to your 
application without needing to build or re-install the package.

### Allied Vision Camera Support [optional]

Allied Vision (Prosilica) imaging devices are supported by the VIMBA SDK.

Download the SDK [here](https://www.alliedvision.com/en/products/vimba-sdk/#c1497) and install `vimbapython`.  

This requires a pip version of Python 3.8+.

A further note - the pip install for `PyQt5` is not compatible with the typical `opencv-python`, since the pip installation
of the latter installs it's own Qt libraries by default (for Linux, but not for OSX).  The easiest workaround here is to
install `opencv-python-headless`, which doesn't install Qt overheads, and avoids this conflict.

On occasion, depending on the order of dependency installation, the following error can occur mixing the pip version of PyQt5, and OpenCV.

```shell
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.

This application failed to start because no Qt platform plugin could be initialized. 
Reinstalling the application may fix this problem.
```

Resolve this by re-installing `xcb`:

```shell
sudo apt install libxcb-xinerama0 
```


