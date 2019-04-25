# Seaborn + Matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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

def bokeh_hide_grid(fig, axis='both'):
    if axis == 'x':
        fig.xgrid.visible = False
    elif axis == 'y':
        fig.ygrid.visible = False
    elif axis == 'both':
        fig.grid.visible = False

def bokeh_fixed_ticks(fig, ticks, axis='x'):
    if axis == 'x':
        fig.xaxis.ticker = FixedTicker(ticks=ticks)
    elif axis == 'y':
        fig.yaxis.ticker = FixedTicker(ticks=ticks)
    elif axis == 'both':
        fig.xaxis.ticker = FixedTicker(ticks=ticks)
        fig.yaxis.ticker = FixedTicker(ticks=ticks)

def bokeh_hide_axis(fig, axis='both'):
    if axis == 'x':
        fig.xaxis.visible = False
    elif axis == 'y':
        fig.yaxis.visible = False
    elif axis == 'both':
        fig.axis.visible = False

def bokeh_hide_border(fig):
    fig.outline_line_color = None

def bokeh_no_ticks(fig, axis='both'):
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

def bokeh_major_ticks(fig, axis='both'):
    if axis == 'x':
        fig.xaxis.minor_tick_line_color = None
    if axis == 'y':
        fig.yaxis.minor_tick_line_color = None
    if axis =='both':
        fig.xaxis.minor_tick_line_color = None
        fig.yaxis.minor_tick_line_color = None

def bokeh_pi_ticks(fig, interval, axis='x'):
    if axis == 'x':
        fig.xaxis.ticker = SingleIntervalTicker(interval=interval)
        fig.xaxis.formatter = pi_formatter()

def bokeh_labelled_fixed(fig, ticks, labels, axis='y'):
    d = {}
    for i, tick in enumerate(ticks):
        d.update({tick: labels[i]})

    if axis=='x' or axis=='both':
        fig.xaxis.ticker = ticks
        fig.xaxis.major_label_overrides = d
    if axis=='y' or axis=='both':
        fig.yaxis.ticker = ticks
        fig.yaxis.major_label_overrides = d

def bokeh_hide_all(fig):
    bokeh_hide_axis(fig, axis='both')
    bokeh_hide_grid(fig, axis='both')
    bokeh_hide_border(fig)

def bokeh_central_axis(fig, axis='both'):
        if axis == 'x':
            fig.xaxis.fixed_location = 0
        elif axis == 'y':
            fig.yaxis.fixed_location = 0
        elif axis == 'both':
            fig.xaxis.fixed_location = 0
            fig.yaxis.fixed_location = 0

def sin_and_cos_plot(f, A):
    t = np.linspace(0, 4*np.pi, 256)
    fig = figure(title="Sin wave at f = {}".format(f),
                  plot_height=300,
                  plot_width=800,
                  x_range=[t[0], t[-1]+1],
                  y_range=[-A-1, A+1],
                  toolbar_location=None)

    sin_source = ColumnDataSource({'x': t, 'y': A * np.sin(f*t)})
    cos_source = ColumnDataSource({'x': t, 'y': A * np.cos(f*t)})
    fig.line(
        x='x', y='y', source=sin_source, legend='sin(t)',
        line_width=4, line_color='blue', line_alpha=0.7
    )
    fig.line(
        x='x', y='y', source=cos_source, legend='cos(t)',
        line_width=4, line_color='red', line_alpha=0.7

    )
    fig.add_layout(Span(
        location=2*np.pi, dimension='height', line_color='purple', line_dash='dashed'
    ))
    fig.add_layout(Span(
        location=4*np.pi, dimension='height', line_color='purple', line_dash='dashed'
    ))

    bokeh_hide_grid(fig)
    bokeh_central_axis(fig, axis='x')
    bokeh_pi_ticks(fig, np.pi)
    bokeh_hide_border(fig)
    bokeh_labelled_fixed(fig, [-A, A], ['-A', 'A'], axis='y')
    bokeh_major_ticks(fig)

    show(fig)

def wave_animated(t, wave, title=None, line_width=4, scale='pi'):
    A = np.max(np.round(wave))
    fig = figure(title=title,
                  plot_height=300,
                  plot_width=800,
                  x_range=[t[0], t[-1]],
                  y_range=[-A-1, A+1],
                  toolbar_location=None)
    data = {'x': t, 'y': wave}
    source = ColumnDataSource({'x':[], 'y':[]})
    fig.line(
        x='x', y='y', source=source,
        line_width=line_width,
    )
    fig.add_layout(Span(
        location=t[-1]/2, dimension='height', line_color='purple', line_dash='dashed'
    ))
    fig.add_layout(Span(
        location=t[-1], dimension='height', line_color='purple', line_dash='dashed'
    ))

    bokeh_hide_grid(fig)
    bokeh_central_axis(fig, axis='x')

    if scale == 'pi':
        bokeh_pi_ticks(fig, np.pi)
    else:
        bokeh_fixed_ticks(fig, np.linspace(t[0], t[-1], 5))

    bokeh_hide_border(fig)
    bokeh_labelled_fixed(fig, [-A, A], [str(-A), str(A)], axis='y')
    bokeh_major_ticks(fig)

    handle = show(fig, notebook_handle=True)
    LoopAnimation(
        data=data, source=source, frames=t.size, handle=handle, fps=30,
        condition=lambda frame, elapsed, args: frame < args['frames'],
    ).start()

def bare_bones_sin_wave(t, A, f, wave):
    fig = figure(title="Sin wave at f = {}".format(f),
                  plot_height=400,
                  plot_width=800,
                  toolbar_location=None)
    fig.line(
        x=t, y=wave,
        line_width=5,
    )
    fig.add_layout(Span(
        location=2*np.pi, dimension='height', line_color='purple', line_dash='dashed'
    ))

    bokeh_hide_grid(fig)
    bokeh_central_axis(fig, axis='x')
    bokeh_pi_ticks(fig, np.pi)
    bokeh_hide_border(fig)
    bokeh_labelled_fixed(fig, [-A, A], [str(-A), str(A)], axis='y')
    bokeh_major_ticks(fig)
    show(fig)

def note_plot(data, fs):
    sns.set_context("notebook")
    duration = data.size / fs
    fig = plt.figure(figsize=(35, 20))
    with sns.axes_style(style="dark"):
        ax = plt.subplot(1, 1, 1)
        ax.set_xticks([0, data.size-1])
        ax.set_xticklabels(['0', '{:.2f}'.format(duration)])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    return sns.lineplot(data=data)

def sample_rate_interactive(f=4):
    res = 512
    t = np.linspace(0, 1, res, endpoint=True)
    signal = np.sin(f * 2 * np.pi * t)

    fig1 = figure(title="Sampling a signal x(t)",
                  plot_height=300,
                  plot_width=800,
                  toolbar_location=None)
    bokeh_hide_axis(fig1, axis='y')
    bokeh_hide_grid(fig1, axis='both')
    bokeh_hide_border(fig1)
    bokeh_central_axis(fig1, axis='x')
    fig1.xaxis.axis_label = "Time (s)"

    # clamp anim_s between [0, fs]
    line_source = ColumnDataSource({'x': t, 'y': signal})
    samples_source = ColumnDataSource({
        'x': np.empty(res), 'y0': np.zeros(res), 'y1': np.empty(res)
    })

    line1 = fig1.line(
        x='x', y='y', source=line_source, line_width=3
    )
    line2 = fig1.line(
        x='x', y='y1', source=samples_source, line_width=1, line_color='red'
    )
    sample_lines = fig1.segment(
        x0='x', y0='y0', x1='x', y1='y1', source=samples_source,
        line_width=1, line_color='red'
    )
    sample_circles = fig1.circle(
        x='x', y='y1', source=samples_source, line_color='red'
    )


    handle = show(fig1, notebook_handle=True)
    def update(Fs=7):
        i = np.linspace(0, res, Fs, endpoint=False).astype(int)
        samples_source.data.update({
            'x': t[i], 'y1': signal[i]
        })
        push_notebook(handle=handle)
    return update

def multiply_signals(random_signal, f1=4):
    t = np.linspace(0, 1, random_signal.size)
    y_range = (-2, 2)
    y_ticks = [-1, 0, 1]
    fig1 = figure(title="x1", plot_width=700, plot_height=200,
                    toolbar_location=None, y_range=y_range)
    bokeh_hide_all(fig1)
    fig1.yaxis.visible = True
    fig1.yaxis.ticker = y_ticks

    fig2 = figure(title="x2", plot_width=700, plot_height=200,
                    toolbar_location=None, y_range=y_range)
    bokeh_hide_all(fig2)
    fig2.yaxis.visible = True
    fig2.yaxis.ticker = y_ticks

    fig3 = figure(title="x3 = x1 * x2", plot_width=700, plot_height=200,
                    toolbar_location=None, y_range=y_range)
    bokeh_hide_all(fig3)
    fig3.yaxis.visible = True
    fig3.yaxis.ticker = y_ticks

    detector_signal = np.sin(f1 * np.pi * 2 * t)
    combined = random_signal * detector_signal

    w1_source = ColumnDataSource({'x': t, 'y': random_signal})
    w2_source = ColumnDataSource({'x': t, 'y': detector_signal})
    w3_source = ColumnDataSource({'x': t, 'y': combined})
    fig1.line(x='x', y='y', source=w1_source, line_color='blue', line_width=2)
    fig2.line(x='x', y='y', source=w2_source, line_color='red', line_width=2)
    fig3.line(x='x', y='y', source=w3_source, line_color='purple', line_width=2)

    fig3.add_layout(BoxAnnotation(top=3, bottom=0, fill_alpha=0.1, fill_color='green'))
    fig3.add_layout(BoxAnnotation(top=0, bottom=-3, fill_alpha=0.1, fill_color='red'))
    handle = show(column(fig1, fig2, fig3), notebook_handle=True)

    def update(f1=5):
        new_signal = np.sin(f1 * np.pi * 2 * t)
        w2_source.data.update({'y': new_signal})
        w3_source.data.update({'y': random_signal * new_signal})
        push_notebook(handle)


    return update


def dot_product_vectors_plot(x1, x2, x3, x4):

    r1 = max( np.max(np.abs(x1)), np.max(np.abs(x2)) ) + 1
    r2 = max( np.max(np.abs(x3)), np.max(np.abs(x4)) ) + 1

    fig1 = figure(
        title="x1 , x2", plot_width = 400, plot_height=400,
        toolbar_location=None, x_range=(-r1, r1), y_range=(-r1, r1)
    )
    fig2 = figure(
        title="x3 , x4", plot_width = 400, plot_height=400,
        toolbar_location=None, x_range=(-r2, r2), y_range=(-r2, r2)
    )

    c1 = 'purple'
    c2 = 'orange'
    bokeh_hide_border(fig1)
    bokeh_hide_grid(fig1)
    bokeh_central_axis(fig1)
    fig1.xaxis.minor_tick_line_color = None
    fig1.yaxis.minor_tick_line_color = None
    fig1.xaxis.ticker = [x1[0], x2[0]]
    fig1.yaxis.ticker = [x1[1], x2[1]]

    bokeh_hide_border(fig2)
    bokeh_hide_grid(fig2)
    bokeh_central_axis(fig2)
    fig2.xaxis.minor_tick_line_color = None
    fig2.yaxis.minor_tick_line_color = None
    fig2.xaxis.ticker = [x3[0], x4[0]]
    fig2.yaxis.ticker = [x3[1], x4[1]]

    fig1.line(x=[x1[0]], y=[x1[1]])

    fig2.line(x=[x1[0]], y=[x1[1]])

    x1_vec = Arrow(x_start=0, y_start=0, x_end=x1[0], y_end=x1[1],
        end=OpenHead(size=9, line_color=c1), line_color=c1
    )
    x2_vec = Arrow(x_start=0, y_start=0, x_end=x2[0], y_end=x2[1],
        end=OpenHead(size=9, line_color=c2), line_color=c2
    )
    x3_vec = Arrow(x_start=0, y_start=0, x_end=x3[0], y_end=x3[1],
        end=OpenHead(size=9, line_color=c1), line_color=c1
    )
    x4_vec = Arrow(x_start=0, y_start=0, x_end=x4[0], y_end=x4[1],
        end=OpenHead(size=9, line_color=c2), line_color=c2
    )


    dp1 = np.dot(x1, x2)
    m1 = np.linalg.norm(x1)
    p1 = [x1[0] * dp1 / (m1*m1) , x1[1] * dp1 / (m1*m1)]

    dp2 = np.dot(x3, x4)
    m2 = np.linalg.norm(x3)
    p2 = [x3[0] * dp2 / (m2*m2) , x3[1] * dp2 / (m2*m2)]

    proj1_vec = Arrow(x_start=0, y_start=0, x_end=p1[0], y_end=p1[1],
        end=OpenHead(size=9, line_color='grey'), line_color='grey'
    )
    proj2_vec = Arrow(x_start=0, y_start=0, x_end=p2[0], y_end=p2[1],
        end=OpenHead(size=9, line_color='grey'), line_color='grey'
    )

    fig1.line(x=[x2[0], p1[0]], y=[x2[1], p1[1]],
        line_color='grey', line_dash='dashed')
    fig1.line(x=[0, p1[0]], y=[0, p1[1]], line_width=5, line_color='grey'
        ,line_alpha=0.5)

    fig2.line(x=[x4[0], p2[0]], y=[x4[1], p2[1]],
        line_color='grey', line_dash='dashed')
    fig2.line(x=[0, p2[0]], y=[0, p2[1]], line_width=5, line_color='grey'
        ,line_alpha=0.5)

    fig1.add_layout(x1_vec)
    fig1.add_layout(x2_vec)
    fig1.add_layout(proj1_vec)

    fig2.add_layout(x3_vec)
    fig2.add_layout(x4_vec)
    fig2.add_layout(proj2_vec)


    d_range = max(dp1, dp2) + 2
    fig3 = figure(
        title='<x1, x2> = {}'.format(dp1), plot_width=400, plot_height=100,
        toolbar_location=None, x_range = (-d_range, d_range)
    )
    bokeh_hide_all(fig3)
    fig3.xaxis.visible = True
    fig4 = figure(
        title='<x3, x4> = {}'.format(dp2), plot_width=400, plot_height=100,
        toolbar_location=None, x_range = (-d_range, d_range)
    )
    bokeh_hide_all(fig4)

    fig4.xaxis.visible = True
    fig3.line(x=[0, dp1], y=[0.5, 0.5], line_width=5, line_color='blue'
        ,line_alpha=0.5, line_cap='round')

    fig4.line(x=[0, dp2], y=[0.5, 0.5], line_width=5, line_color='blue'
        ,line_alpha=0.5, line_cap='round')

    show(column(row(fig1, fig2), row(fig3, fig4)))

def detector(fs, random_signal, f_range, dot_product=False, with_cos=False):
    if with_cos:
        dot_product = True

    freqs = np.arange(f_range[0], f_range[1]+1)
    t = np.linspace(0, 1, fs)
    sin_signals = [np.sin(freq * 2 * np.pi * t) for freq in freqs]
    if with_cos:
        cos_signals = [np.cos(freq * 2 * np.pi * t) for freq in freqs]

    sigs_source = ColumnDataSource({'x': t})

    sin_figs = []
    sin_dps = []
    sin_fig_colours = []

    cos_figs = []
    cos_dps = []
    cos_fig_colours = []

    if dot_product:
        sin_dps = np.array([np.dot(random_signal, sig) for sig in sin_signals])
        dp_max = np.max(np.abs(sin_dps))

        if with_cos:
            cos_dps = np.array([np.dot(random_signal, sig) for sig in cos_signals])
            cos_dp_max = np.max(np.abs(cos_dps))
            dp_max = max(cos_dp_max, cos_dp_max)
            cos_fig_colours = map_to_colours(cos_dps, 0, dp_max, RdYlGn[5][::-1])

        sin_fig_colours = map_to_colours(sin_dps, 0, dp_max, RdYlGn[5][::-1])

    for i, sin_sig in enumerate(sin_signals):
            sin_title = None

            title = 'f:{}'.format(freqs[i])
            if dot_product:
                dp = int(sin_dps[i])
                sin_title = title + ' | {}'.format(dp)

            if with_cos:
                dp = int(sin_dps[i])
                if np.abs(cos_dps[i]) > np.abs(sin_dps[i]):
                    dp = int(cos_dps[i])
                cos_title = title + ' | {}'.format(dp)



            sin_source_key = 'sin_y{}'.format(i)
            sigs_source.data.update({ sin_source_key: sin_sig })

            if with_cos:
                cos_source_key = 'cos_y{}'.format(i)
                sigs_source.data.update({ cos_source_key: cos_signals[i]})

            title = sin_title if sin_title else title
            sin_fig = figure(
                title=title,
                plot_width=90,
                plot_height=60,
                toolbar_location=None,
            )
            bokeh_hide_all(sin_fig)

            if with_cos:
                cos_fig = figure(
                    title=cos_title,
                    plot_width=90,
                    plot_height=60,
                    toolbar_location=None,
                )
                bokeh_hide_all(cos_fig)

            if dot_product:
                sin_fig.background_fill_color = sin_fig_colours[i]
                sin_fig.background_fill_alpha = 0.4

            if with_cos:
                cos_fig.background_fill_color = cos_fig_colours[i]
                cos_fig.background_fill_alpha = 0.4


            sin_fig.line(x='x', y=sin_source_key, source=sigs_source)
            sin_figs.append(sin_fig)

            if with_cos:
                cos_fig.line(x='x', y=cos_source_key, source=sigs_source)
                cos_figs.append(cos_fig)

    fig = figure(
        title='Random f Signal',
        plot_width=800,
        plot_height=200,
        toolbar_location=None,
    )
    bokeh_hide_all(fig)
    fig.line(x=np.arange(0,fs), y=random_signal)

    if with_cos:
        show(column(row(sin_figs), row(cos_figs), fig))
    else:
        show(column(row(sin_figs), fig))

def complex_detector(fs, random_signal, f_range, phase=False):
    freqs = np.arange(f_range[0], f_range[1]+1)
    t = np.linspace(0, 1, fs)

    i = np.complex(0, 1)
    complex_signals = [np.exp(i * freq * 2 * np.pi * t) for freq in freqs]
    dps = np.array([np.dot(random_signal, sig) for sig in complex_signals])
    dp_max = max(
        np.max(np.abs(np.real(dps))),
        np.max(np.abs(np.imag(dps)))
    )
    m = dp_max * 1.2

    fig_colours = map_to_colours(dps, 0, dp_max, RdYlGn[5][::-1])
    figs = []
    for i, sig in enumerate(complex_signals):
            dp = dps[i]
            a = np.real(dp)
            b = np.imag(dp)
            r = np.abs(dp)
            theta = np.angle(dp)

            title = 'f:{} | {:,.0f}'.format(freqs[i], np.floor(r))
            if phase:
                title = title + ' | {:.1f}'.format(np.degrees(theta))

            fig = figure(
                title=title,
                plot_width=200,
                plot_height=200,
                x_range=[-m, m],
                y_range=[-m, m],
                toolbar_location=None,
            )
            bokeh_hide_border(fig)
            bokeh_hide_grid(fig)
            bokeh_central_axis(fig)
            fig.background_fill_color = fig_colours[i]
            fig.background_fill_alpha = 0.2

            if np.abs(a) > 1 or np.abs(b) > 1:
                fig.xaxis.ticker = [np.around(a)]
                fig.yaxis.ticker = [np.around(b)]
            else:
                fig.xaxis.ticker = []
                fig.yaxis.ticker = []

            if phase:
                c = 'purple'
                arrow = Arrow(x_start=0, y_start=0, x_end=a, y_end=b,
                    end=OpenHead(size=9, line_color=c),
                    line_color=c, line_width=4, line_alpha=0.8
                )
                fig.add_layout(arrow)
            else:
                c1 = 'green'
                c2 = 'blue'
                real_arrow = Arrow(x_start=0, y_start=0, x_end=a, y_end=0,
                    end=OpenHead(size=9, line_color=c1),
                    line_color=c1, line_width=4, line_alpha=0.8
                )
                imag_arrow = Arrow(x_start=0, y_start=0, x_end=0, y_end=b,
                    end=OpenHead(size=9, line_color=c2),
                    line_color=c2, line_width=4, line_alpha=0.8
                )
                fig.add_layout(real_arrow)
                fig.add_layout(imag_arrow)

            if i == 1:
                fig.xaxis.axis_label = "Real (z)"
                fig.yaxis.axis_label = "Imag (z)"

            figs.append(fig)

    fig = figure(
        title='Random f Signal',
        plot_width=800,
        plot_height=200,
        toolbar_location=None,
    )
    bokeh_hide_all(fig)
    fig.line(x=np.arange(0,fs), y=random_signal)

    show(column(fig, row(figs)))

def detector_plot(t, signal, detector_f):
    fig1 = figure(title="x1", plot_width=700, plot_height=200,
                    toolbar_location=None, y_range=(-1, 1))
    bokeh_hide_all(fig1)
    fig1.yaxis.visible = True
    fig1.yaxis.ticker = [-1, 0, 1]

    fig2 = figure(title="x2", plot_width=700, plot_height=200,
                    toolbar_location=None, y_range=(-1, 1))
    bokeh_hide_all(fig2)
    fig2.yaxis.visible = True
    fig2.yaxis.ticker = [-1, 0, 1]

    fig3 = figure(title="x3 = x1 * x2", plot_width=700, plot_height=200,
                    toolbar_location=None, y_range=(-1, 1))
    bokeh_hide_all(fig3)
    fig3.yaxis.visible = True
    fig3.yaxis.ticker = [-1, 0, 1]

    detector_signal = np.sin(detector_f * np.pi * 2 * t)
    combined = signal * detector_signal

    w1_source = ColumnDataSource({'x': t, 'y': signal})
    w2_source = ColumnDataSource({'x': t, 'y': detector_signal})
    w3_source = ColumnDataSource({'x': t, 'y': combined})

    segment_source = ColumnDataSource({
        'x': [], 'y1': [], 'y2': [], 'y3': [], 'zeros': []
    })
    segment_data = {
        'x':t, 'y1': signal, 'y2': detector_signal,
        'y3': combined, 'zeros': np.zeros(t.size)
    }


    w1_line = fig1.line(x='x', y='y', line_cap='round', source=w1_source)
    w2_line = fig2.line(x='x', y='y', line_cap='round', source=w2_source)
    w3_line = fig3.line(x='x', y='y', line_cap='round', source=w3_source)

    seg1 = fig1.segment(x0='x', x1='x', y0='zeros', y1='y1',
        line_color='green', source=segment_source)
    seg2 = fig2.segment(x0='x', x1='x', y0='zeros', y1='y2',
        line_color='orange', source=segment_source)
    seg3 = fig3.segment(x0='x', x1='x', y0='zeros', y1='y3',
        line_color='red', source=segment_source)


    dp_data = [np.sum(combined[0:i]) for i in np.arange(0, t.size)]
    dp_max = max(np.abs(dp_data))

    fig4 = figure(title="<x1, x2>", plot_height=600, plot_width=100,
                    toolbar_location=None, y_range=(-dp_max, dp_max))
    bokeh_hide_all(fig4)
    fig4.yaxis.visible = True

    dp_line = fig4.line(x=[1, 1], y=[], line_cap='round', line_color='red')

    handle = show(row(column(fig1, fig2, fig3), fig4), notebook_handle=True)

    def update(frame, elapsed, args):
        y = args['dp_data'][frame % args['frames']]
        args['dp_source'].data['y'] = [0, y]

    LoopAnimation(
        data=segment_data, source=segment_source, frames=t.size, handle=handle,
        condition=lambda frame, elapsed, args: frame < args['frames'],
        hook=update, reset=False,
        args={
            'dp_source': dp_line.data_source, 'dp_data': dp_data,
        },
        fps=20
    ).start()

def a0_term_plot(a0_sin, a0_square):
    fs = 256
    f = 3

    t = np.linspace(0, 2 * np.pi, fs)
    l = t + np.linspace(-2, 2, fs)
    s1 = np.sin(f * t)
    s2 = np.sin(f * t) + a0_sin
    s3 = a0_square * square(f * t) + a0_square


    fig = figure(plot_height=300,
                 plot_width=800,
                 toolbar_location=None)
    bokeh_hide_grid(fig, axis='both')
    bokeh_hide_border(fig)
    bokeh_central_axis(fig, axis='x')


    pi_ticks = np.linspace(0, 2 * np.pi, 9)
    fig.yaxis.ticker = [0, a0_sin, a0_square]
    fig.xaxis.ticker = pi_ticks
    fig.xaxis.formatter = pi_formatter()

    fig.line(x=t, y=s1, line_color='blue')
    fig.line(x=t, y=s2, line_color='green')
    fig.line(x=t, y=s3, line_color='red')
    span_1 = Span(location=0, line_color='blue',
                    line_alpha=0.5, line_dash='dashed',
                    line_width=2, dimension='width')
    span_2 = Span(location=a0_sin, line_color='green',
                    line_alpha=0.5, line_dash='dashed',
                    line_width=2, dimension='width')
    span_3 = Span(location=a0_square, line_color='red',
                    line_alpha=0.5, line_dash='dashed',
                    line_width=2, dimension='width')
    fig.add_layout(span_1)
    fig.add_layout(span_2)
    fig.add_layout(span_3)
    show(fig)

def complex_notations_plot(*args):
    zs = np.array(args)
    m = max(np.max(np.real(np.abs(zs))), np.max(np.imag(np.abs(zs)))) + 1

    fig1 = figure(title="Complex Plane (Rectangular)",
                 plot_height=400,
                 plot_width=400,
                 x_range=[-m, m],
                 y_range=[-m, m],
                 toolbar_location=None)

    bokeh_central_axis(fig1, axis='both')
    bokeh_major_ticks(fig1)
    bokeh_hide_border(fig1)

    fig2 = figure(title="Complex Plane (Polar)",
                 plot_height=400,
                 plot_width=400,
                 x_range=[-m, m],
                 y_range=[-m, m],
                 toolbar_location=None)

    bokeh_central_axis(fig2, axis='both')
    bokeh_no_ticks(fig2)
    bokeh_hide_border(fig2)
    bokeh_hide_grid(fig2)


    for z in zs:
        c = 'green'
        z1_arrow = Arrow(x_start=0, y_start=0, x_end=np.real(z), y_end=np.imag(z),
            end=OpenHead(size=9, line_color=c), line_color=c
        )
        z2_arrow = Arrow(x_start=0, y_start=0, x_end=np.real(z), y_end=np.imag(z),
            end=OpenHead(size=9, line_color=c), line_color=c
        )
        fig1.add_layout(z1_arrow)
        fig2.add_layout(z2_arrow)


    show(row(fig1, fig2))

def audio_signal(fs, signal):
    duration = signal.size / fs
    t = np.linspace(0, duration, signal.size)

    fig = figure(
                 plot_height=300,
                 plot_width=800,
                 x_range=(0, t[-1]))
    bokeh_hide_all(fig)

    source = ColumnDataSource({'x': t, 'y': signal})
    fig.xaxis.visible = True
    fig.xaxis.ticker = [0, t[t.size//2], t[-1]]
    fig.xaxis.axis_label = "Time (s)"


    line = fig.line(x='x', y='y', source=source, line_width=1)
    handle = show(fig, notebook_handle=True)

def signal_animated(t, signal):
    y_max = np.max(signal)
    fig = figure(title="Signal x[n]",
                 plot_height=300,
                 plot_width=800,
                 x_range=(0, t[-1]),
                 y_range=(-y_max, y_max),
                 toolbar_location=None)
    bokeh_hide_all(fig)

    data = {'x': t, 'y': signal}
    source = ColumnDataSource({'x': [], 'y': []})

    line = fig.line(x='x', y='y', source=source, line_width=1)
    handle = show(fig, notebook_handle=True)
    LoopAnimation(
        data=data, source=source, frames=signal.size, handle=handle,
        condition=lambda frame, elapsed, args: frame < args['frames'],
    ).start()

def multiply_signals_with_phase(signal, f):
        t = np.linspace(-2*np.pi, 2*np.pi, signal.size)
        y_range = (-2, 2)
        y_ticks = [-1, 0, 1]
        fig1 = figure(title="x1", plot_width=700, plot_height=200,
                        toolbar_location=None, y_range=y_range)
        bokeh_hide_all(fig1)
        fig1.yaxis.visible = True
        fig1.yaxis.ticker = y_ticks

        fig2 = figure(title="x2", plot_width=700, plot_height=200,
                        toolbar_location=None, y_range=y_range)
        bokeh_hide_all(fig2)
        fig2.yaxis.visible = True
        fig2.yaxis.ticker = y_ticks

        fig3 = figure(title="x3 = x1 * x2", plot_width=700, plot_height=200,
                        toolbar_location=None, y_range=y_range)
        bokeh_hide_all(fig3)
        fig3.yaxis.visible = True
        fig3.yaxis.ticker = y_ticks

        detector_signal = np.sin(f * t)
        combined = signal * detector_signal

        w1_source = ColumnDataSource({'x': t, 'y': signal})
        w2_source = ColumnDataSource({'x': t, 'y': detector_signal})
        w3_source = ColumnDataSource({'x': t, 'y': combined})
        fig1.line(x='x', y='y', source=w1_source, line_color='blue', line_width=2)
        fig2.line(x='x', y='y', source=w2_source, line_color='red', line_width=2)
        fig3.line(x='x', y='y', source=w3_source, line_color='purple', line_width=2)

        fig3.add_layout(BoxAnnotation(top=3, bottom=0, fill_alpha=0.1, fill_color='green'))
        fig3.add_layout(BoxAnnotation(top=0, bottom=-3, fill_alpha=0.1, fill_color='red'))

        fig1.add_layout(Span(location=0, line_color='grey', dimension='width'))
        fig2.add_layout(Span(location=0, line_color='grey', dimension='width'))
        fig3.add_layout(Span(location=0, line_color='grey', dimension='width'))

        for i in np.linspace(-2*np.pi, 2*np.pi, 9):
            fig1.add_layout(Span(location=i, line_color='grey', dimension='height'))
            fig2.add_layout(Span(location=i, line_color='grey', dimension='height'))
            fig3.add_layout(Span(location=i, line_color='grey', dimension='height'))

        handle = show(column(fig1, fig2, fig3), notebook_handle=True)

        def update(phase=0):
            new_signal = np.sin(f * t + phase)
            w2_source.data.update({'y': new_signal})
            w3_source.data.update({'y': signal * new_signal})
            push_notebook(handle)


        return update

def phase_plot():
    pi2 = 2 * np.pi
    fs = 256
    t = np.linspace(-pi2, pi2, fs)
    fig = figure(plot_height=300,
                 plot_width=800,
                 x_range=(-pi2, pi2),
                 toolbar_location=None,
                 y_range=(-1, 1.3))
    bokeh_hide_all(fig)

    cos_source = ColumnDataSource({'x': t, 'y': np.cos(t)})
    sin_source = ColumnDataSource({'x': t, 'y': np.empty(t.size)})
    sin_orig_source = ColumnDataSource({'x': t, 'y': np.sin(t)})
    phase_source =ColumnDataSource({'x': [0, np.pi/2], 'y': [1, 1]})

    cos_line = fig.line(x='x', y='y', source=cos_source, line_color='red',
                        legend='Cos(x)')
    sin_line = fig.line(x='x', y='y', source=sin_source, legend='Sin(x + 𝜑)')
    sin_orig_line = fig.line(x='x', y='y', source=sin_orig_source,
            line_alpha=0.5, line_dash='dashed', legend='Sin(x)')

    phase_diff_line = fig.line(x='x', y='y', source=phase_source, line_color='grey',
                        legend='Phase Difference')

    phase_diff_box = BoxAnnotation(left=0, right=0, fill_color='orange',
                        fill_alpha=0.1, line_color='black')

    for i in np.linspace(-pi2, pi2, 5):
        fig.add_layout(Span(
            location=i, dimension='height', line_color='purple', line_dash='dashed'
        ))

    phase_diff_label = Label(x=0, y=1.1, text="")

    fig.add_layout(phase_diff_box)
    fig.add_layout(phase_diff_label)
    fig.legend.location = "bottom_right"
    fig.xaxis.ticker = np.linspace(-pi2, pi2, 9)
    fig.xaxis.formatter = pi_formatter()
    fig.xaxis.visible = True

    handle = show(fig, notebook_handle=True)

    def update(phase=0):
        phase_rads = phase * (np.pi/180)
        phase_diff = np.pi/2 - phase_rads
        sin_source.data.update({
            'y' : np.sin(t + phase_rads)
        })
        phase_source.data.update({
            'x' : [0, phase_diff]
        })
        phase_diff_box.right = phase_diff
        phase_diff_label.text = pistr(np.abs(phase_diff))
        phase_diff_label.x = phase_diff / 2
        push_notebook(handle=handle)


    interact(update, phase=(-360, 360))

def basic_fourier_plot(f, y_fft):
    fig1 = figure(title="Sampled Signal x[n]",
                  plot_height=300,
                  plot_width=800,
                  toolbar_location=None)
    fig1.line(t, signal, line_width=1)
    bokeh_hide_all(fig1)
    show(fig1)

def sampling_animated(t, signal, fs):
    y_max = np.max(signal)
    fig1 = figure(title="Sampling a signal x(t)",
                  plot_height=300,
                  plot_width=800,
                  toolbar_location=None)
    bokeh_hide_axis(fig1, axis='y')
    bokeh_hide_grid(fig1, axis='both')
    bokeh_hide_border(fig1)
    fig1.xaxis.axis_label = "Time (s)"

    fig2 = figure(title="The sampled signal x[n]",
                  plot_height=300,
                  plot_width=800,
                  toolbar_location=None,
                  y_range=(-y_max, y_max),
                  x_range=(0, t[-1]))
    bokeh_hide_all(fig2)

    fs = min(signal.size, max(fs, 0)) * t[-1]

    line_source = ColumnDataSource({'x': t, 'y': signal})
    anim_i = np.linspace(0, signal.size-1, fs).astype(int)

    segment_data = {
        'x': t[anim_i], 'y0': np.zeros(anim_i.size), 'y1': signal[anim_i]
    }
    segments_source = ColumnDataSource({'x': [], 'y0': [], 'y1': []})

    line1 = fig1.line(
        x='x', y='y', source=line_source, line_width=1
    )
    line2 = fig2.line(
        x='x', y='y1', source=segments_source, line_width=1, line_color='red'
    )
    segments = fig1.segment(
        x0='x', y0='y0', x1='x', y1='y1', source=segments_source,
        line_width=1, line_color='red'
    )

    handle = show(column(fig1, fig2), notebook_handle=True)

    LoopAnimation(
        data=segment_data, source=segments_source, frames=anim_i.size,
        handle=handle, condition=lambda frame, elapsed, args: frame < args['frames'],
        fps=20
    ).start()

def sound_bit_detail_plot(sound, bits=1):

    fig1 = figure(title="Sound to digital",
                  plot_height=300,
                  plot_width=800,
                  toolbar_location=None)
    levels = bits ** 2
    if levels == 1:
        base = max(sound) / (levels)
    else:
        base = max(sound) / (levels-1)
    digital_signal = myround(sound, base=base)

    t = np.linspace(0, 1, sound.size)
    sound_source = ColumnDataSource({
        'x':t, 'y':sound
    })
    digital_signal_source = ColumnDataSource({
        'x':t, 'y': digital_signal
    })
    fig1.line(x='x', y='y', source=sound_source, legend='Sound')
    fig1.line(x='x', y='y', source=digital_signal_source,
        legend='Digital signal', line_color='orange')

    fig1.xaxis.axis_label = "Time"
    fig1.legend.location = "bottom_right"

    bokeh_hide_grid(fig1, axis='both')
    bokeh_fixed_ticks(fig1, [0], axis='x')
    bokeh_hide_axis(fig1, axis='y')
    bokeh_hide_border(fig1)

    handle = show(fig1, notebook_handle=True)

def sound_bit_detail_interactive():
    t = np.linspace(0, 1, 512)
    sound = np.sin(2 * 2 * np.pi * t) + 1
    fig1 = figure(title="Sound to digital",
                  plot_height=300,
                  plot_width=800,
                  toolbar_location=None)

    t = np.linspace(0, 1, sound.size)
    sound_source = ColumnDataSource({
        'x':t, 'y':sound
    })
    digital_signal_source = ColumnDataSource({
        'x':t, 'y': []
    })
    fig1.line(x='x', y='y', source=sound_source, legend='Sound Pressure')
    fig1.line(x='x', y='y', source=digital_signal_source,
        legend='Digital signal', line_color='orange')

    fig1.xaxis.axis_label = "Time"
    fig1.yaxis.ticker = [0, 2]
    fig1.yaxis.major_label_overrides = {0:'-A', 2:'A'}
    fig1.legend.location = "bottom_right"

    bokeh_hide_grid(fig1, axis='both')
    bokeh_hide_axis(fig1, axis='x')
    bokeh_hide_border(fig1)

    handle = show(fig1, notebook_handle=True)
    def update(bits=1):
        levels = bits ** 2
        if levels == 1:
            base = max(sound) / (levels)
        else:
            base = max(sound) / (levels-1)
        rounded = myround(sound, base=base)
        digital_signal_source.data.update({'y': rounded})

        push_notebook(handle=handle)
    interact(update, bits=(1,16))

def aliasing_plot_animated(sample_freq, f0):
    time = 1
    resolution = 512
    t = np.linspace(0, time, resolution)
    sigf0 = np.sin(f0 * 2 * np.pi * t)

    alias_count = 5
    aliases_i = np.arange(0, alias_count)
    aliases = aliases_i * sample_freq + f0
    sig_aliases = [np.sin(alias * 2 * np.pi * t ) for alias in aliases]

    sample_t = np.linspace(0, time, int(sample_freq*time)+1)
    sample_y = np.sin(f0 * 2 * np.pi * sample_t)

    samples_source = ColumnDataSource({
        'x': sample_t,
        'y0': np.zeros(sample_t.size), 'y1': sample_y
    })
    f0_source = ColumnDataSource({'x': t, 'y': sigf0})
    fk_source = ColumnDataSource({'x': [], 'y': []})

    fig1 = figure(title="Aliases overlayed with f0",
                  plot_height=300,
                  plot_width=800,
                  toolbar_location=None)

    bokeh_hide_axis(fig1, axis='y')
    bokeh_hide_grid(fig1, axis='both')
    bokeh_fixed_ticks(fig1, [0, 0.5, 1], axis='x')
    bokeh_hide_border(fig1)
    fig1.xaxis.axis_label = 'Time (s)'
    sample_circles = fig1.circle(
        x='x', y='y1', source=samples_source, fill_color='red', line_color='red'
    )
    sample_segments = fig1.segment(
        x0='x', x1='x', y0='y0', y1='y1', source=samples_source,
        line_width=1, line_color='red'
    )
    f0_line = fig1.line(
        x='x', y='y', source=f0_source, line_width=1, line_alpha=0.4,
        line_color='grey'
    )
    fk_line = fig1.line(
        x='x', y='y', source=fk_source, line_width=1
    )

    handle = show(fig1, notebook_handle=True)
    data_frames = [{'x': t, 'y': y} for y in sig_aliases]

    FrameAnimation(
        data_frames=data_frames, source=fk_source, handle=handle, fps=0.5,
        args={'aliases':aliases},
        condition=lambda frame, elapsed, args: elapsed < 20 * 60,
    ).start()

def time_and_frequency_plot():
    fmax = 100
    time = 1
    ticks = fmax * time
    fig1 = figure(title="Time Domain",
                  plot_height=100,
                  plot_width=800,
                  x_range=(0, time),
                  toolbar_location=None)
    bokeh_hide_axis(fig1, axis='y')
    bokeh_hide_grid(fig1, axis='both')
    bokeh_hide_border(fig1)

    fig2 = figure(title="Frequency Domain (Hz)",
                  plot_height=100,
                  plot_width=800,
                  x_range=(0, fmax+5),
                  toolbar_location=None)
    bokeh_hide_axis(fig2, axis='y')
    bokeh_hide_grid(fig2, axis='both')
    bokeh_hide_border(fig2)

    # bokeh_hide_all(fig2)

    t_source = ColumnDataSource({
        'x': np.empty(ticks), 'zeros': np.zeros(ticks), 'ones': np.ones(ticks)}
    )
    f_source = ColumnDataSource({'x':[1, 1], 'y':[0, 1]})

    time_markers = fig1.segment(
        x0='x', y0='zeros', x1='x', y1='ones', source=t_source, line_width=2
    )
    f_marker = fig2.line(
        x='x', y='y', source=f_source, line_width=2
    )
    handle = show(column(fig1, fig2), notebook_handle=True)

    def animate(frame, elapsed, args):
        f = (frame % fmax) + 1
        args['f_source'].data.update({'x': [f, f]})
        args['t_source'].data.update({'x': args['ticks'] * (1.0/f)})
        args['axis'].ticker = [f]
        push_notebook(handle=args['handle'])

    Animation(
        animate=animate,
        args={
            'ticks': np.arange(0, ticks), 'axis': fig2.axis,
            'f_source': f_source, 't_source':t_source, 'handle':handle},
        condition=lambda frame, elapsed, args: elapsed < 10,
        fps=10
    ).start()

def amplitude_spectrum_plot_interact():
    time = 1
    fmax = 10
    amax = 10
    resolution = 128

    fig1 = figure(title="Signal x(t)",
                  plot_height=300,
                  plot_width=400,
                  y_range=(-(amax+2), (amax+2)),
                  toolbar_location=None)
    bokeh_hide_all(fig1)

    fig2 = figure(title="Spectrum of x(t)",
                  plot_height=300,
                  plot_width=400,
                  x_range=(0, fmax+2),
                  y_range=(0, (amax+2)),
                  toolbar_location=None)
    bokeh_hide_grid(fig2, axis='both')
    bokeh_hide_border(fig2)
    bokeh_fixed_ticks(fig2, [3, 0], axis='x')
    bokeh_fixed_ticks(fig2, [3, 0], axis='x')
    fig2.xaxis.axis_label = "Frequency (Hz)"
    fig2.yaxis.axis_label = "Amplitude"


    t = np.linspace(0, time, resolution)
    wave_source = ColumnDataSource({'x':t, 'y':[]})
    spec_source = ColumnDataSource({'x':[], 'y':[], 'y0': [0, 0]})

    wave_line = fig1.line(x='x', y='y', source=wave_source, line_width=3)
    spec_segs = fig2.segment(
        x0='x', x1='x', y0='y0' ,y1='y',
        source=spec_source
    )
    handle = show(row(fig1, fig2), notebook_handle=True)

    def update(f1=3, f2=0, a1=3, a2=0):
        wave_source.data['y'] = a1*np.sin(f1*2*np.pi*t) + a2*np.sin(f2*2*np.pi*t)
        spec_source.data['y'] = [a1, a2]
        spec_source.data['x'] = [f1, f2]
        fig2.xaxis.ticker = [f1, f2]
        fig2.yaxis.ticker = [a1, a2]
        push_notebook(handle=handle)
    interact(update, f1=(0, fmax), f2=(0, fmax), a1=(0, amax), a2=(0, amax))

def transform_plot(signal, t, samples):
    amp_max = 10
    freq_max = 20
    f1 = 5
    f2 = 20

    fig1 = figure(title="Signal x(t)",
                  plot_height=300,
                  plot_width=400,
                  y_range=(-(amp_max+2), (amp_max+2)),
                  toolbar_location=None)

    fig2 = figure(title="Transform Signal X(f)",
                  plot_height=300,
                  plot_width=400,
                  x_range=(0, freq_max+10),
                  y_range=(0, (amp_max+2)),
                  toolbar_location=None)

    bokeh_hide_grids(fig1, fig2)
    bokeh_hide_axis(fig1, axis='both')
    bokeh_fixed_ticks(fig1, [-amp_max, amp_max], axis='y')
    bokeh_fixed_ticks(fig2, [f1, f2])

    def transform(sig):
        return (np.abs(fft(sig)[0:t.size//2]) * 2) / t.size

    y_fft = transform(signal)
    x_fft = np.linspace(0, samples / 2, t.size / 2)

    signal_line = fig1.line(t, signal, line_width=1)

    signal_fft_segments = fig2.segment(
        x0=x_fft, y0=np.zeros(y_fft.size),
        x1=x_fft, y1=y_fft, line_width=1)

    fig_handle = show(row(fig1, fig2), notebook_handle=True)

    def update_function(freq1=5, freq2=20, amp1=10, amp2=2):
        comp1 = amp1 * np.sin(freq1 * 2 * np.pi * t)
        comp2 = amp2 * np.sin(freq2 * 2 * np.pi * t)
        new_signal = comp1 + comp2
        new_signal_fft = transform(new_signal)

        signal_line.data_source.data['y'] = new_signal
        signal_fft_segments.data_source.data['y1'] = new_signal_fft

        bokeh_fixed_ticks(fig2, [freq1, freq2])
        push_notebook(handle=fig_handle)

    interact(update_function, freq1=(0, freq_max), freq2=(0, freq_max),
             amp1=(0, amp_max), amp2=(0, amp_max))
