"""Generic class for all image sources"""
import cv2
from PIL import Image


class VKCamera:
    def __init__(self, verbose_mode=False):
        """Constructor for generic image source class.
        Use subclassed instances of this class

        Args:
            verbose_mode (bool): Additional class detail logging.
        """

        self.verbose_mode = verbose_mode
        self.video_object = None

        if verbose_mode:
            print(self)

    def get_frame(self):
        """Takes the next available frame from the relevant camera object.
         This method MUST be overridden by child classes
        """
        raise NotImplementedError

    def frame_count(self):
        """The number of frames in the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_COUNT property - zero if a live camera.
        """
        return int(self.video_object.get(cv2.CAP_PROP_FRAME_COUNT))

    def width(self):
        """The pixel width of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_WIDTH property.
        """
        return int(self.video_object.get(cv2.CAP_PROP_FRAME_WIDTH))

    def height(self):
        """The pixel height of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_HEIGHT property.
        """
        return int(self.video_object.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def fps(self):
        """The frames per second of the video resource.

        Returns:
            (int): The CAP_PROP_FPS property.
        """
        return int(self.video_object.get(cv2.CAP_PROP_FPS))

    def save_frame(self, dest_path='./image.png'):
        """Grabs a frame and saves it to file.

        Args:
            dest_path (str): destination for saved image.
        """
        frame = 0

        if hasattr(self, "set_position"):
            self.set_position(frame_number=100)

        _frame = self.get_frame()

        cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB, _frame)
        img = Image.fromarray(_frame)
        img.save(dest_path)

        if self.verbose_mode:
            print("Saved frame {0} to {1}".format(frame, dest_path))

    def close(self):
        if self.video_object is not None:
            print(self.__class__, "Closing video object...")
            self.video_object.release()

    def __repr__(self):
        """Overriding repr

        Returns:
            (str): A string representation of the object
        """
        return '{} | {}'.format(self.__class__, self.__hash__())

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        return "\nCamera Source:" \
               "\n\tFilename         : {0}" \
               "\n\tWidth            : {1}" \
               "\n\tHeight           : {2}" \
               "\n\tFrame Rate       : {3}" \
               "\n\tFrame Count      : {4}".format(self.filepath,
                                                   self.width(),
                                                   self.height(),
                                                   self.fps(),
                                                   self.frame_count())
