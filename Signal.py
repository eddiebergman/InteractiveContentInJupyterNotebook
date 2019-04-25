# Bokeh
from bokeh.io import show
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models.tickers import FixedTicker, SingleIntervalTicker
from bokeh.models.markers import Circle
from bokeh.models import ColumnDataSource, BoxAnnotation, Arrow, OpenHead, Label, Span
from bokeh.models.formatters import FuncTickFormatter
from bokeh.models.mappers import LinearColorMapper

from animate import StreamAnimation, Animate, AnimationSet, AnimateSets

import signal_plots as plots
import numpy as np
import math_util as m
import bokehedit as bk


line_defaults = {
    'legend':None,
    'line_width':3, 'line_color':'blue', 'line_alpha':0.4
}
span_defaults = {
    'dimension':'height', 'line_color':'purple', 'line_dash':'dashed'
}
stem_defaults = {
    'line_color':'red', 'line_alpha':0.6, 'line_width':3
}
noise_line_defaults = {
    'line_color' : 'purple', 'line_alpha': 0.8, 'line_width':2
}
noise_stem_defaults = {
    'line_color' : 'purple', 'line_alpha': 0.2, 'line_width':1, 'line_dash':'dashed'
}

class Signal:

    def __init__(self, x, y, frequency=None):
        self._ds = ColumnDataSource({'x': x, 'y': y})
        self._stem_ds = None
        self._noise = 0
        self._noise_ds = None
        # line, spans, stems, noise
        self._views = ['line', 'spans']
        self._interacts = []
        self._handle = None
        self._frequency = frequency
        self._line_opts = line_defaults.copy()
        self._span_opts = span_defaults.copy()
        self._stem_opts = stem_defaults.copy()
        self._noise_line_opts = noise_line_defaults.copy()
        self._noise_stem_opts = noise_stem_defaults.copy()

    def sample(self, sample_amount):
        indicies = np.linspace(0, self.samples()-1, sample_amount).astype(int)
        return {'x' : self.x()[indicies], 'y': self.y()[indicies]}

    def copy(self):
        return Signal(np.copy(self.x()), np.copy(self.y()))

    def duration(self):
        return self.x()[-1] - self.x()[0]

    def samples(self):
        return self.x().size

    def frequency(self):
        return self._frequency

    def sample_rate(self):
        return self.samples() / self.duration()

    def add_view(self, view):
        self._views.append(view)

    def add_noise(self, strength=1, std_dev=1):
        self._noise = self.y() + strength * np.random.normal(scale=std_dev, size=self.samples())
        self.add_view('noise')

    def remove_noise(self):
        self._noise = 0
        self.remove_view('noise')

    def add_random_spike(self, strength=1, size=1):
        range_end = self.samples() - size
        start = np.random.randint(0, range_end+1)
        sl = slice(start, start+size)
        spike = self.y()[sl] + np.ones(size) * strength
        spike_patch = {
            'y': [ (sl, spike) ]
        }
        self.patch(spike_patch)

    def remove_view(self, view):
        self._views.remove(view)

    def data_source(self):
        return self._ds

    def data(self):
        return self._ds.data

    def x(self):
        return self._ds.data['x']

    def y(self):
        return self._ds.data['y']

    def add(self, ys):
        if ys.size == self.y().size:
            self._ds.data['y'] += ys
        else:
            print('ys {} Must be the same size y {}'.format(ys.size, self.y().size))

    def update(self, new_data):
        self._ds.data.update(new_data)
        if self._stem_ds:
            self._stem_ds.data.update({**new_data,'y0': np.zeros(new_data['x'].size) })

    def patch(self, patch_data):
        self._ds.patch(patch_data)

    def first(self):
        return self.x()[0]

    def last(self):
        return self.x()[-1]

    def max(self):
        return np.max(self.y())

    def min(self):
        return np.min(self.y())

    def noise_max(self):
        return np.max(self.y() + self._noise) if self._noise is not 0 else self.max()

    def noise_min(self):
        return np.min(self.y() + self._noise) if self._noise is not 0 else self.min()

    def update_line_opts(self, opts):
        self._line_opts.update(opts)

    def update_span_opts(self, opts):
        self._span_opts.update(opts)

    def update_stem_opts(self, opts):
        self._stem_opts.update(opts)

    def update_noise_line_opts(self, opts):
        self._noise_line_opts.update(opts)

    def update_noise_stem_opts(self, opts):
        self._noise_stem_opts.update(opts)

    def create_fig(self, title=None, size=(400, 400), scale='2pi', fig_opts={}, **kwargs):
        if scale == 'time' and fig_opts == {}:
            fig_opts['xaxis'] = ['central', 'major_only']
        return plots.signal_basic(self, title, size, fig_opts, **kwargs)

    def create_renderers(self, fig, views=[], scale='time'):
        views = self._views + views
        if 'line' in views:
            self.create_line_renderer(fig)
        if 'spans' in views:
            self.create_span_renderers(fig, scale)
        if 'stems' in views:
            self.create_stem_renderer(fig)
        if 'noise' in views:
            self.create_noise_renderer(fig)

    def get_fig(self, title=None, size=(400,400), scale='2pi', views=[], fig_opts={}):
        fig = self.create_fig(title, size, scale, fig_opts=fig_opts)
        self.create_renderers(fig, views=views, scale=scale)
        return fig


    def show(self, title=None, size=(400,400), scale='2pi',
            animated=False, frames=64, views=[]):
        fig = self.get_fig(title, size, scale,  views)
        self._handle = show(fig, notebook_handle=True)
        if animated:
            self.handle_animation(frames)
        return fig, self._handle

    def create_line_renderer(self, fig):
        fig.line(x='x', y='y', source=self.data_source(), **self._line_opts)

    def create_stem_renderer(self, fig):
        if not self._stem_ds:
            self._stem_ds = ColumnDataSource({
                'x': self.x(), 'y': self.y(), 'y0': np.zeros(self.y().size)
            })
        fig.segment(x0='x', x1='x', y0='y0', y1='y', source=self._stem_ds, **self._stem_opts)

    def create_noise_renderer(self, fig):
        if not self._noise_ds:
            self._noise_ds = ColumnDataSource({
                'x': self.x(), 'y': self.y(), 'y_noise': self.y() + self._noise
            })
            fig.line(x='x', y='y_noise', source=self._noise_ds, **self._noise_line_opts)
            fig.segment(x0='x', x1='x', y0='y', y1='y_noise', source=self._noise_ds, **self._noise_stem_opts)

    def create_span_renderers(self, fig, scale):
        interval = 2*np.pi if scale == '2pi' else 1
        start = m.floor(self.first(), base=interval)
        cur = start
        while cur <= self.last():
            fig.add_layout(Span(location=cur, **self._span_opts))
            cur += interval

    def create_line_animation(self, frames=64):
        xs = np.array_split(self.x(), frames)
        ys = np.array_split(self.y(), frames)
        data = [{ 'x': xs[frame], 'y': ys[frame] } for frame in range(0, frames)]
        return StreamAnimation(self._ds, data)

    def create_stem_animation(self, frames=64):
        xs = np.array_split(self._stem_ds.data['x'], frames)
        ys = np.array_split(self._stem_ds.data['y'], frames)
        y0 = np.array_split(self._stem_ds.data['y0'], frames)
        data = [{ 'x': xs[frame], 'y': ys[frame], 'y0': y0[frame]} for frame in range(0, frames)]
        return StreamAnimation(self._stem_ds, data)

    def create_noise_animation(self, frames=64):
        xs = np.array_split(self._noise_ds.data['x'], frames)
        ys = np.array_split(self._noise_ds.data['y'], frames)
        y_noises = np.array_split(self._noise_ds.data['y_noise'], frames)
        data = [{ 'x': xs[frame], 'y': ys[frame], 'y_noise': y_noises[frame]}
                for frame in range(0, frames)]
        return StreamAnimation(self._noise_ds, data)

    def handle_animation(self, frames):
        anims = []
        if 'line' in self._views:
            line_anim = self.create_line_animation(frames)
            anims.append(line_anim)
        if 'stems' in self._views:
            stem_anim = self.create_stem_animation(frames)
            anims.append(stem_anims)
        if 'noise' in self._views:
            noise_anim = self.create_noise_animation(frames)
            anims.append(noise_anim)

        aset = AnimationSet(anims)
        AnimateSets([aset], self._handle).run()
