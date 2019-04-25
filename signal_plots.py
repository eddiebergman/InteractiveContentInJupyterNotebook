# Notebook
from IPython.display import Audio, display
from ipywidgets import interact, Layout, Box, Button

# Bokeh
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models.tickers import FixedTicker, SingleIntervalTicker
from bokeh.models.markers import Circle
from bokeh.models import ColumnDataSource, BoxAnnotation, Arrow, OpenHead, Label, Span
from bokeh.models.formatters import FuncTickFormatter
from bokeh.models.mappers import LinearColorMapper

from bokeh.palettes import RdYlGn

# Others
import numpy as np
from scipy.fftpack import fft
from scipy.signal import square
from figs import sig_fig

def signal_basic(signal, title=None, size=(400,400), fig_opts=None):
    ymin, ymax = signal.noise_min(), signal.noise_max()
    xmin, xmax = signal.first(), signal.last()
    opts = {
        'xaxis': ['major_only', '2pi_ticks', 'central'],
        'yaxis': ['major_only'],
        'ylabels': ([ymin, ymax], [str(ymin), str(ymax)])
    }
    if fig_opts:
        opts.update(fig_opts)

    fig = sig_fig(
        size=size, title=title, options=opts,
        y_range=[ymin*1.1, ymax*1.1], x_range=[xmin*1.1, xmax*1.1]
    )
    return fig
