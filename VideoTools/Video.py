import numpy as np
from PIL import Image
from . import ffmpeg_reader as vid

class VideoObject():
    def __init__(self, video_path,
                target_resolution= (128, 128),
                 spatial_transform=None,
                 temporal_transform=None,
                 verbose_mode = False):

        if verbose_mode:
            print("Loading ", video_path)

        self.__videoObject = vid.FFMPEG_VideoReader(video_path, target_resolution=target_resolution)

        self.fps = self.__videoObject.fps
        self.filename = video_path
        self.target_resolution = target_resolution
        self.verbose_mode = verbose_mode
        self.video_duration = self.__videoObject.duration
        self.frameCount = self.__videoObject.nframes

        if verbose_mode:
            self.showDetails()

    def showFrame(self, frame=1):
        img = Image.fromarray(self.__videoObject.frameAtFrameNumber(frame))
        img.show()

    def getFrame(self, frame=1):
        img = Image.fromarray(self.__videoObject.frameAtFrameNumber(frame))
        return img

    def getFrameAsNumpy(self, frame=1):
        return self.__videoObject.frameAtFrameNumber(frame)

    def nextFrame(self):
        img = Image.fromarray(self.__videoObject.read_frame())
        return img

    def frameCount(self):
        return self.frameCount

    def video_width(self):
        return self.target_resolution[1]

    def video_height(self):
        return self.target_resolution[0]

    def getFramesWithRange(self, frameArray):
        frames = []
        for f in frameArray:
            img = Image.fromarray(self.__videoObject.frameAtFrameNumber(f))
            frames.append(img)

        return frames

    def saveFrame(self, dest_path = './image.png', frame=1):
        img = Image.fromarray(self.__videoObject.frameAtFrameNumber(frame))
        img.save(dest_path)
        if self.verbose_mode:
            print("Saved frame {0} to {1}".format(frame, dest_path))

    def showDetails(self):
        print ( "\nVideo")
        print ("\tFilename         :", self.filename)
        print ("\tDuration         :", self.video_duration)
        print ("\tWidth            :", self.target_resolution[1])
        print ("\tHeight           :", self.target_resolution[0])
        print ("\tFrame Rate       :", self.fps)
        print ("\tFrame Count      :", self.frameCount)

    def close(self):
        self.__videoObject.close()
