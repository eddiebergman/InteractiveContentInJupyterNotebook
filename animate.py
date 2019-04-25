import time
from threading import Thread
from bokeh.io import push_notebook

class Pause:

    def __init__(self, duration):
        self._duration = duration

    def length(self):
        return 1

    def init(self):
        return

    def view_frame(self, frame_index):
        time.sleep(self._duration)
        return

class UpdateAnimation:

    def __init__(self, source, data):
        self._source = source
        self._data = data

    def length(self):
        return len(self._data)

    def init(self):
        return

    def view_frame(self, frame_index):
        frame_data = self._data[frame_index]
        self._source.patch(frame_data)

class StreamAnimation:

    def __init__(self, source, data):
        self._source = source
        self._data = data

    def length(self):
        return len(self._data)

    def init(self):
        d = {}
        for key in self._data[0].keys():
            d[key] = []
        self._source.data.update(d)

    def view_frame(self, frame_index):
        self._source.stream(self._data[frame_index])

class FrameAnimation:

    def __init__(self, source, frame_gen, frame_values):
        self._source = source
        self._frame_gen = frame_gen
        self._frame_values = frame_values
        self._frame_data = []
        for frame in frame_values:
            data = self._frame_gen(frame)
            self._frame_data.append(data)

    def length(self):
        return self._frame_values.size

    def init(self):
        return

    def view_frame(self, frame_index):
        self._source.data.update(self._frame_data[frame_index])

class AnimationSet:

    def __init__(self, animations):
        self._animations = animations
        self._max_length = 0
        for animation in self._animations:
            self._max_length = animation.length() if animation.length() > self._max_length else self._max_length

    def length(self):
        return self._max_length

    def view_frame(self, frame_index):
        divisor = frame_index / self._max_length
        for animation in self._animations:
            f_idx = int(animation.length() * divisor)
            animation.view_frame(f_idx)

    def init(self):
        for animation in self._animations:
            animation.init()

class Animate(Thread):
    def __init__(self, animation, handle, fps=30, hook=None):
        Thread.__init__(self)
        self._animation = animation
        self._fps = fps
        self._interval = 1 / self._fps
        self._handle = handle
        self._hook = hook
        self.run()

    def run(self):
        self._animation.init()
        for frame in range(0, self._animation.length()):
            self._animation.view_frame(frame)
            if self._hook:
                self._hook(set_idx, frame)
            time.sleep(self._interval)
            push_notebook(handle=self._handle)


class AnimateSets(Thread):

    def __init__(self, animation_sets, handle, fps=30, hook=None):
        Thread.__init__(self)
        self._asets = animation_sets
        self._fps = fps
        self._interval = 1 / self._fps
        self._handle = handle
        self._hook = hook
        for animation_set in animation_sets:
            animation_set.init()

    def run(self):
        for aset in self._asets:
            aset.init()

        for set_idx, aset in enumerate(self._asets):
            for frame in range(0, aset.length()):
                aset.view_frame(frame)
                if self._hook:
                    self._hook(set_idx, frame)
                time.sleep(self._interval)
                push_notebook(handle=self._handle)
