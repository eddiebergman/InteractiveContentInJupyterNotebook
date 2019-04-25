# Bokeh
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models.tickers import FixedTicker, SingleIntervalTicker
from bokeh.models.markers import Circle
from bokeh.models import ColumnDataSource, BoxAnnotation, Arrow, OpenHead, Label, Span
from bokeh.models.formatters import FuncTickFormatter
from bokeh.models.mappers import LinearColorMapper
import bokehedit as bk
import numpy as np

"""
opts = {
    'yaxis ||''xaxis':
        [hide,
         pi_ticks,
         2pi_ticks,
         labelled,
         central,
         major_only],

    'ylabels' || 'xlabels': ( [ticks], [labels] )
}
"""
def axis_configure(fig, axis, opts, labels):
        if 'hide' in opts:
            bk.hide_axis(fig, axis)

        if 'pi_ticks' in opts:
            bk.pi_ticks(fig, np.pi, axis=axis)

        if '2pi_ticks' in opts:
            bk.pi_ticks(fig, 2*np.pi, axis=axis)

        if 'labelled' in opts:
            bk.labelled_fixed(fig, labels[0], labels[1], axis=axis)

        if 'central' in opts:
            bk.central_axis(fig, axis=axis)

        if 'major_only' in opts:
            bk.major_ticks(fig)


def sig_fig(size=(300,300),
            title=None,
            grid=False,
            border=False,
            options={},
            **kwargs):

    fig = figure(toolbar_location=None,
                title=title,
                plot_height=size[1],
                plot_width=size[0],
                **kwargs)

    if not grid:
        bk.hide_grid(fig)

    if not border:
        bk.hide_border(fig)

    if 'xaxis' in options:
        axis = 'x'
        axis_configure(fig, 'x', options['xaxis'], options.get('xlabels', None))

    if 'yaxis' in options:
        axis = 'y'
        axis_configure(fig, 'y', options['yaxis'], options.get('ylabels', None))

    return fig

def add_spans(fig, locations, dimension='height', **kwargs):
    for loc in locations:
        fig.add_layout(Span(
            location=loc, dimension=dimension, **kwargs
        ))
