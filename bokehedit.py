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

def maprange(x, a, b, c, d):
    return (x-a) * (d-c)/(b-a) + c

def myround(x, base=1):
    return base * np.round(x/base)

def pi_formatter_func():
    return '{:.2f}\u03C0'.format(tick / 3.14)

def pi_formatter():
    return FuncTickFormatter.from_py_func(pi_formatter_func)

def pistr(x):
    return '{:.2f}\u03C0'.format(x / np.pi)

def map_to_colours(numbers, low, high, palette):
    i = maprange(numbers, low, high, 0, len(palette)-1)
    if type(i) is np.ndarray:
        i = i.astype(int)
    return np.asarray(palette)[i]

def hide_grid(fig, axis='both'):
    if axis == 'x':
        fig.xgrid.visible = False
    elif axis == 'y':
        fig.ygrid.visible = False
    elif axis == 'both':
        fig.grid.visible = False

def fixed_ticks(fig, ticks, axis='x'):
    if axis == 'x':
        fig.xaxis.ticker = FixedTicker(ticks=ticks)
    elif axis == 'y':
        fig.yaxis.ticker = FixedTicker(ticks=ticks)
    elif axis == 'both':
        fig.xaxis.ticker = FixedTicker(ticks=ticks)
        fig.yaxis.ticker = FixedTicker(ticks=ticks)

def hide_axis(fig, axis='both'):
    if axis == 'x':
        fig.xaxis.visible = False
    elif axis == 'y':
        fig.yaxis.visible = False
    elif axis == 'both':
        fig.axis.visible = False

def hide_border(fig):
    fig.outline_line_color = None

def no_ticks(fig, axis='both'):
    if axis == 'x':
        fig.xaxis.major_label_text_font_size = '0pt'
        fig.xaxis.minor_tick_line_color = None
        fig.xaxis.major_tick_line_color = None

    if axis == 'y':
        fig.yaxis.major_label_text_font_size = '0pt'
        fig.yaxis.minor_tick_line_color = None
        fig.yaxis.major_tick_line_color = None

    if axis =='both':
        fig.xaxis.major_label_text_font_size = '0pt'
        fig.yaxis.major_label_text_font_size = '0pt'
        fig.xaxis.minor_tick_line_color = None
        fig.yaxis.minor_tick_line_color = None
        fig.xaxis.major_tick_line_color = None
        fig.yaxis.major_tick_line_color = None

def major_ticks(fig, axis='both'):
    if axis == 'x':
        fig.xaxis.minor_tick_line_color = None
    if axis == 'y':
        fig.yaxis.minor_tick_line_color = None
    if axis =='both':
        fig.xaxis.minor_tick_line_color = None
        fig.yaxis.minor_tick_line_color = None

def pi_ticks(fig, interval, axis='x'):
    ticker = SingleIntervalTicker(interval=interval)
    if axis == 'x' or axis =='both':
        fig.xaxis.ticker = ticker
        fig.xaxis.formatter = pi_formatter()
    if axis == 'y' or axis =='both':
        fig.yaxis.ticker = ticker
        fig.yaxis.formatter = pi_formatter()


def labelled_fixed(fig, ticks, labels, axis='y'):
    d = {}
    for i, tick in enumerate(ticks):
        d.update({tick: labels[i]})

    if axis=='x' or axis=='both':
        fig.xaxis.ticker = ticks
        fig.xaxis.major_label_overrides = d
    if axis=='y' or axis=='both':
        fig.yaxis.ticker = ticks
        fig.yaxis.major_label_overrides = d

def hide_all(fig):
    bokeh_hide_axis(fig, axis='both')
    bokeh_hide_grid(fig, axis='both')
    bokeh_hide_border(fig)

def central_axis(fig, axis='both'):
        if axis == 'x':
            fig.xaxis.fixed_location = 0
        elif axis == 'y':
            fig.yaxis.fixed_location = 0
        elif axis == 'both':
            fig.xaxis.fixed_location = 0
            fig.yaxis.fixed_location = 0
