"""
Standard video handing and frame grabbing using ffmpeg.
Most of this code is ripped from the MoviePy codebase at:
https://github.com/Zulko/moviepy
"""

import subprocess as sp
import warnings
from subprocess import DEVNULL
import re
import numpy as np
import os


class FFMPEG_VideoReader:

    def __init__(self, filename, print_infos=False, bufsize = None,
                 pix_fmt="rgb24", check_duration=True,
                 target_resolution=None, resize_algo='bicubic',
                 fps_source='tbr'):

        self.filename = filename
        self.proc = None
        infos = ffmpeg_parse_infos(filename, print_infos, check_duration, fps_source)

        self.fps = infos['video_fps']
        self.size = infos['video_size']
        self.rotation = infos['video_rotation']
        self.duration = infos['video_duration']
        self.ffmpeg_duration = infos['duration']
        self.nframes = infos['video_nframes']

        if print_infos:
            # print the whole info text returned by FFMPEG
            print("FPS:             ", self.fps)
            print("Size:            ", self.size)
            print("Duration:        ", self.duration)
            print("ffmpeg_duration: ", self.ffmpeg_duration)
            print("Frames:          ", self.nframes)

        if target_resolution:
            # revert the order, as ffmpeg used (width, height)
            target_resolution = target_resolution[1], target_resolution[0]

            if None in target_resolution:
                ratio = 1
                for idx, target in enumerate(target_resolution):
                    if target:
                        ratio = target / self.size[idx]
                self.size = (int(self.size[0] * ratio), int(self.size[1] * ratio))
            else:
                self.size = target_resolution

        self.resize_algo = resize_algo
        self.infos = infos
        self.pix_fmt = pix_fmt
        if pix_fmt == 'rgba':
            self.depth = 4
        else:
            self.depth = 3

        if bufsize is None:
            w, h = self.size
            bufsize = self.depth * w * h + 100

        self.bufsize= bufsize
        self.initialize()
        self.pos = 1
        while not hasattr(self, 'lastread'):
            self.lastread = self.read_frame()


    def initialize(self, starttime=0):
        """Opens the file, creates the pipe. """

        self.close() # if any

        if starttime != 0 :
            offset = min(1, starttime)
            i_arg = ['-ss', "%.06f" % (starttime - offset),
                     '-i', self.filename,
                     '-ss', "%.06f" % offset]
        else:
            i_arg = [ '-i', self.filename]

        cmd = (["ffmpeg"] + i_arg +
               ['-loglevel', 'error',
                '-f', 'image2pipe',
                '-vf', 'scale=%d:%d' % tuple(self.size),
                '-sws_flags', self.resize_algo,
                "-pix_fmt", self.pix_fmt,
                '-vcodec', 'rawvideo', '-'])
        popen_params = {"bufsize": self.bufsize,
                        "stdout": sp.PIPE,
                        "stderr": sp.PIPE,
                        "stdin": DEVNULL}

        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000

        self.proc = sp.Popen(cmd, **popen_params)

    def seekToTime(self, t):
        """
        Seek directly to time t (doesn't retrieve a frame).
        """
        pos = int(self.fps*t + 0.00001)+1

        if not self.proc:
            self.initialize(t)
            self.pos = pos

        else:
            if (pos < self.pos) or (pos > self.pos + 100):
                self.initialize(t)
                self.pos = pos
            else:
                self.skip_frames(pos-self.pos-1)

    def seekToFrameNumber(self, f):
        """
        ffmpeg prefers time rather than frame.
        """
        seekTimeInSeconds = (f-1)/self.fps
        self.seekToTime(seekTimeInSeconds)

    def skip_frames(self, n=1):
        """Reads and throws away n frames """
        w, h = self.size
        for i in range(n):
            self.proc.stdout.read(self.depth*w*h)
            #self.proc.stdout.flush()
        self.pos += n


    def read_frame(self):
        w, h = self.size
        nbytes= self.depth*w*h

        s = self.proc.stdout.read(nbytes)
        if len(s) != nbytes:

            warnings.warn("Warning: in file %s, "%(self.filename)+
                   "%d bytes wanted but %d bytes read,"%(nbytes, len(s))+
                   "at frame %d/%d, at time %.02f/%.02f sec. "%(
                    self.pos,self.nframes,
                    1.0*self.pos/self.fps,
                    self.duration)+
                   "Using the last valid frame instead.",
                   UserWarning)

            if not hasattr(self, 'lastread'):
                raise IOError(("Failed to read the first frame of video file %s.")%(self.filename))

            result = self.lastread

        else:

            result = np.fromstring(s, dtype='uint8')
            result.shape =(h, w, len(s)//(w*h))
            self.lastread = result

        return result

    def frameAtTime(self, t):
        """
        Read a file video frame at time t.
        """

        pos = int(self.fps*t + 0.00001)+1

        # Initialize proc if it is not open
        if not self.proc:
            self.initialize(t)
            self.pos = pos
            self.lastread = self.read_frame()

        if pos == self.pos:
            return self.lastread
        else:
            if (pos < self.pos) or (pos > self.pos + 100):
                self.initialize(t)
                self.pos = pos
            else:
                self.skip_frames(pos-self.pos-1)
            result = self.read_frame()
            self.pos = pos
            return result

    def frameAtFrameNumber(self, f):
        """
        ffmpeg prefers time rather than frame.
        """
        seekTimeInSeconds = (f-1)/self.fps
        return self.frameAtTime(seekTimeInSeconds)

    def close(self):
        if self.proc:
            self.proc.terminate()
            self.proc.stdout.close()
            self.proc.stderr.close()
            self.proc.wait()
            self.proc = None
        if hasattr(self, 'lastread'):
            del self.lastread



def is_string(obj):
    """ Returns true if s is string or string-like object,
    compatible with Python 2 and Python 3."""
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)

def cvsecs(time):
    """ Will convert any time into seconds.
    Here are the accepted formats:

    >>> cvsecs(15.4) -> 15.4 # seconds
    >>> cvsecs( (1,21.5) ) -> 81.5 # (min,sec)
    >>> cvsecs( (1,1,2) ) -> 3662 # (hr, min, sec)
    >>> cvsecs('01:01:33.5') -> 3693.5  #(hr,min,sec)
    >>> cvsecs('01:01:33.045') -> 3693.045
    >>> cvsecs('01:01:33,5') #coma works too
    """

    if is_string(time):
        if (',' not in time) and ('.' not in time):
            time = time + '.0'
        expr = r"(\d+):(\d+):(\d+)[,|.](\d+)"
        finds = re.findall(expr, time)[0]
        nums = list( map(float, finds) )
        return ( 3600*int(finds[0])
                + 60*int(finds[1])
                + int(finds[2])
                + nums[3]/(10**len(finds[3])))

    elif isinstance(time, tuple):
        if len(time)== 3:
            hr, mn, sec = time
        elif len(time)== 2:
            hr, mn, sec = 0, time[0], time[1]
        return 3600*hr + 60*mn + sec

    else:
        return time


def ffmpeg_parse_infos(filename, print_infos=False, check_duration=True,
                       fps_source='tbr'):
    """Get file infos using ffmpeg.

    Returns a dictionnary with the fields:
    "video_found", "video_fps", "duration", "video_nframes",
    "video_duration", "audio_found", "audio_fps"

    "video_duration" is slightly smaller than "duration" to avoid
    fetching the uncomplete frames at the end, which raises an error.
    """

    # open the file in a pipe, provoke an error, read output
    is_GIF = filename.endswith('.gif')
    cmd = ["ffmpeg", "-i", filename]
    if is_GIF:
        cmd += ["-f", "null", "/dev/null"]

    popen_params = {"bufsize": 10**5,
                    "stdout": sp.PIPE,
                    "stderr": sp.PIPE,
                    "stdin": DEVNULL}

    if os.name == "nt":
        popen_params["creationflags"] = 0x08000000

    proc = sp.Popen(cmd, **popen_params)

    proc.stdout.readline()
    proc.terminate()
    infos = proc.stderr.read().decode('utf8')
    del proc

    lines = infos.splitlines()
    if "No such file or directory" in lines[-1]:
        raise IOError(("WTF! %s could not be found!")%filename)

    result = dict()

    # get duration (in seconds)
    result['duration'] = None

    if check_duration:
        try:
            keyword = ('frame=' if is_GIF else 'Duration: ')
            # for large GIFS the "full" duration is presented as the last element in the list.
            index = -1 if is_GIF else 0
            line = [l for l in lines if keyword in l][index]
            match = re.findall("([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])", line)[0]
            result['duration'] = cvsecs(match)
        except:
            raise IOError(("MoviePy error: failed to read the duration of file %s.\n"
                           "Here are the file infos returned by ffmpeg:\n\n%s")%(
                              filename, infos))

    # get the output line that speaks about video
    lines_video = [l for l in lines if ' Video: ' in l and re.search('\d+x\d+', l)]

    result['video_found'] = ( lines_video != [] )

    if result['video_found']:
        try:
            line = lines_video[0]

            # get the size, of the form 460x320 (w x h)
            match = re.search(" [0-9]*x[0-9]*(,| )", line)
            s = list(map(int, line[match.start():match.end()-1].split('x')))
            result['video_size'] = s
        except:
            raise IOError(("MoviePy error: failed to read video dimensions in file %s.\n"
                           "Here are the file infos returned by ffmpeg:\n\n%s")%(
                              filename, infos))

        # Get the frame rate. Sometimes it's 'tbr', sometimes 'fps', sometimes
        # tbc, and sometimes tbc/2...
        # Current policy: Trust tbr first, then fps unless fps_source is
        # specified as 'fps' in which case try fps then tbr

        # If result is near from x*1000/1001 where x is 23,24,25,50,
        # replace by x*1000/1001 (very common case for the fps).

        def get_tbr():
            match = re.search("( [0-9]*.| )[0-9]* tbr", line)

            # Sometimes comes as e.g. 12k. We need to replace that with 12000.
            s_tbr = line[match.start():match.end()].split(' ')[1]
            if "k" in s_tbr:
                tbr = float(s_tbr.replace("k", "")) * 1000
            else:
                tbr = float(s_tbr)
            return tbr

        def get_fps():
            match = re.search("( [0-9]*.| )[0-9]* fps", line)
            fps = float(line[match.start():match.end()].split(' ')[1])
            return fps

        if fps_source == 'tbr':
            try:
                result['video_fps'] = get_tbr()
            except:
                result['video_fps'] = get_fps()

        elif fps_source == 'fps':
            try:
                result['video_fps'] = get_fps()
            except:
                result['video_fps'] = get_tbr()

        # It is known that a fps of 24 is often written as 24000/1001
        # but then ffmpeg nicely rounds it to 23.98, which we hate.
        coef = 1000.0/1001.0
        fps = result['video_fps']
        for x in [23,24,25,30,50]:
            if (fps!=x) and abs(fps - x*coef) < .01:
                result['video_fps'] = x*coef

        if check_duration:
            result['video_nframes'] = int(result['duration']*result['video_fps'])+1
            result['video_duration'] = result['duration']
        else:
            result['video_nframes'] = 1
            result['video_duration'] = None
        # We could have also recomputed the duration from the number
        # of frames, as follows:
        # >>> result['video_duration'] = result['video_nframes'] / result['video_fps']

        # get the video rotation info.
        try:
            rotation_lines = [l for l in lines if 'rotate          :' in l and re.search('\d+$', l)]
            if len(rotation_lines):
                rotation_line = rotation_lines[0]
                match = re.search('\d+$', rotation_line)
                result['video_rotation'] = int(rotation_line[match.start() : match.end()])
            else:
                result['video_rotation'] = 0
        except:
            raise IOError(("MoviePy error: failed to read video rotation in file %s.\n"
                           "Here are the file infos returned by ffmpeg:\n\n%s")%(
                              filename, infos))


    lines_audio = [l for l in lines if ' Audio: ' in l]

    result['audio_found'] = lines_audio != []

    if result['audio_found']:
        line = lines_audio[0]
        try:
            match = re.search(" [0-9]* Hz", line)
            result['audio_fps'] = int(line[match.start()+1:match.end()])
        except:
            result['audio_fps'] = 'unknown'

    return result
