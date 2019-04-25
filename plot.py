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
from animate import StreamAnimation, FrameAnimation, UpdateAnimation, Pause, Animate, AnimationSet, AnimateSets
from Signal import Signal, stem_defaults

class InvisibleAudio(Audio):
    def _repr_html_(self):
        audio = super()._repr_html_()
        audio = audio.replace('<audio', f'<audio onended="this.parentNode.removeChild(this)"')
        return f'<div style="display:none; height:0px">{audio}</div>'

def scale_time(title='Time scaling', size=(400, 400), animated=False):
    frames = 50
    scale = 2*np.pi
    m = max(1, scale) + 0.5
    t = np.linspace(-m*2, m*2, 256)
    y = np.sin(2 * np.pi * t)

    sig1 = Signal(t, y)
    sig1.remove_view('spans')

    fig = sig1.get_fig(title=title, size=size, scale='2pi')

    marker_source = ColumnDataSource({'x':[1, 1], 'y': [0, 1]})
    marker_ref = fig.line(x=[1,1], y=[0,1], line_width=2,
                    line_color='purple', line_dash='dashed')
    marker = fig.line(x='x', y='y', source=marker_source,
                    line_width=2, line_color='purple')
    label = Label(x=1, y=1, text='1 second marker', render_mode='css')

    fig.add_layout(label)
    handle = show(fig, notebook_handle=True)

    if animated:
        def gen_marker_frame(frame_value):
            return {'x': [frame_value, frame_value]}

        def gen_sig_frame(frame_value):
            return {'x': t * frame_value}


        frame_values = np.linspace(1, scale, frames)
        a1 = FrameAnimation(marker_source, gen_marker_frame, frame_values)
        a2 = FrameAnimation(sig1.data_source(), gen_sig_frame, frame_values)
        aset = AnimationSet([a1, a2])
        AnimateSets([aset], handle, fps=25).run()

    else:
        ones = np.asarray([1, 1])
        def update(scale=1):
            marker_update = { 'x' : ones * scale }
            signal_update = { 'x' : t * scale }
            marker_source.data.update(marker_update)
            sig1.update(signal_update)
            push_notebook(handle=handle)

        interact(update, scale=(-1*np.pi, 2*np.pi))

def add_functions(t, ft, gt, xt, animated=False):
    sig_ft = Signal(t, ft)
    sig_gt = Signal(t, gt)
    sig_xt = Signal(t, xt)
    y_max = max(sig_ft.max(), sig_gt.max(), sig_xt.max())
    y_min = min(sig_ft.min(), sig_gt.min(), sig_xt.min())

    def update_axis_range(fig):
        fig.x_range.start = t[0]
        fig.x_range.end = t[-1]
        fig.y_range.start = y_min
        fig.y_range.end = y_max

    fig_ft = sig_ft.create_fig(size=(900,300), title='f(t)', scale='time')
    update_axis_range(fig_ft)
    fig_ft.xaxis.axis_label = 't'
    fig_ft.yaxis.axis_label = 'f(t)'
    sig_ft.update_line_opts({'line_color':'red'})
    sig_ft.create_renderers(fig_ft, ['line', 'stems'])

    fig_gt = sig_gt.create_fig(size=(900,300), title='g(t)', scale='time')
    update_axis_range(fig_gt)
    fig_gt.xaxis.axis_label = 't'
    fig_gt.yaxis.axis_label = 'g(t)'
    sig_gt.update_line_opts({'line_color':'green'})
    sig_gt.update_stem_opts({'line_color':'green'})
    sig_gt.create_renderers(fig_gt, ['line', 'stems'])

    fig_xt = sig_xt.create_fig(size=(1800,300), title='x(t)', scale='time')
    update_axis_range(fig_xt)
    fig_xt.xaxis.axis_label = 't'
    fig_xt.yaxis.axis_label = 'x(t)'
    sig_xt.update_line_opts({'line_color':'purple', 'line_width': 5, 'line_alpha':1})
    sig_xt.create_renderers(fig_xt, ['line'])

    # sig_gt.update_line_opts({'line_width':1, 'line_alpha':0.6})
    sig_ft.create_renderers(fig_xt, ['stems'])
    # sig_ft.update_line_opts({'line_width':1, 'line_alpha':0.6})
    # sig_ft.create_renderers(fig_xt, ['line'])


    extra_ds = ColumnDataSource({'x':t, 'y0': np.zeros(t.size), 'y1': gt})
    opts = stem_defaults.copy()
    opts.update({'line_color':'green'})
    fig_xt.segment(x0='x', x1='x', y0='y0', y1='y1', source=extra_ds, **opts)

    handle = show(column(row(fig_ft, fig_gt), fig_xt), notebook_handle=True)

    if animated:
        ft_anim = sig_ft.create_stem_animation(frames=t.size)
        gt_anim = sig_gt.create_stem_animation(frames=t.size)

        xs = np.array_split(extra_ds.data['x'], t.size)
        y1 = np.array_split(extra_ds.data['y1'], t.size)
        y0 = np.array_split(extra_ds.data['y0'], t.size)

        data = [{ 'x': xs[frame], 'y1': y1[frame], 'y0': y0[frame]} for frame in range(0, t.size)]
        gt_on_figxt = StreamAnimation(extra_ds, data)


        slide_to_data = [{
            'y0': [ (frame, ft[frame]) ],
            'y1': [ (frame, xt[frame]) ]
        } for frame in range(0, t.size)]

        slide_to = UpdateAnimation(extra_ds, slide_to_data)

        a1set = AnimationSet([ft_anim])
        a2set = AnimationSet([gt_anim, gt_on_figxt])
        a3set = AnimationSet([slide_to])
        AnimateSets([a1set, a2set, a3set], handle, fps=45).run()

def sampling_a_signal(original_signal, hide_original=False):
    line_options = {'line_color': 'purple', 'legend': 'continuous signal'}
    original_signal.update_line_opts(line_options)


    sample_options = {'line_color': 'orange', 'legend': 'Samples'}
    sample_line_options = {'line_color': 'red', 'legend': 'sampled signal'}
    sampled_signal = original_signal.copy()
    sampled_signal.update_line_opts(sample_line_options)
    sampled_signal.update_stem_opts(sample_options)

    fig = original_signal.create_fig(size=(1920, 300), title="Sampling a Signal")
    original_signal.create_line_renderer(fig)
    sampled_signal.create_stem_renderer(fig)

    fig2 = sampled_signal.create_fig(size=(1920, 300), title="Sampled Result")

    sampled_signal.create_line_renderer(fig2)
    if hide_original:
        sampled_signal.create_stem_renderer(fig2)

        handle = show(fig2, notebook_handle=True)
    else:
        handle = show(column(fig,fig2), notebook_handle=True)

    def update(samples=4):
        samples_data = original_signal.sample(samples)
        sampled_signal.update(samples_data)
        push_notebook(handle)

    interact(update, samples=(0, original_signal.samples()))


def digitize_sound(sig, max_bits=12):
    sig.update({
        'y': sig.y() - sig.min()
    })  # Move lower bound to y=0 to make rounding easier
    sig.update_line_opts({'legend': 'Audio Signal'})

    digital_signal = sig.copy()
    digital_signal.update_line_opts({
        'line_color':'orange', 'line_alpha':1, 'legend':'Digital Signal'
    })

    fig = sig.create_fig(
        title="Sound to Digital", size=(1920, 300), scale='time',
        fig_opts = {
            'yaxis': ['labelled'],
            'xaxis': ['major_only'],
            'ylabels': ([0, sig.max()], ['-A', 'A']),
        }
    )
    fig.x_range.end = sig.duration() / 64
    fig.yaxis.axis_label = 'Amplutide'
    fig.xaxis.axis_label = 'Time (t)'
    digital_signal.create_line_renderer(fig)

    def on_click_button(_):
        audio_element = InvisibleAudio(digital_signal.y(), rate=digital_signal.sample_rate(), autoplay=True)
        display(audio_element)

    button = Button(description="Play Digital Audio")
    button.on_click(on_click_button)

    handle = show(fig, notebook_handle=True)
    display(button)
    def update(bits=3):
        # Round to the nearest level based on bits used
        levels = 2 ** bits
        base = sig.max() / (levels-1)
        rounded = base * np.round(sig.y()/base)
        digital_signal.update({'y': rounded})
        push_notebook(handle=handle)


    interact(update, bits=(1,max_bits))

def aliases_figure(f0, fs):
    t = np.linspace(0, 1, 50*fs)
    y = np.sin(f0 * 2*np.pi * t)
    sig = Signal(t, y)
    alias_sig = Signal(t, y)

    fig = sig.get_fig(size=(1080,300),title="Aliases of {} Hz signal".format(f0))
    alias_sig.update_line_opts({'line_color' : 'orange', 'line_alpha': 1})
    alias_sig.create_line_renderer(fig)

    alias_x = np.linspace(0, 1, fs, endpoint=False)
    alias_y = np.sin(f0 * 2*np.pi * alias_x)
    alias_points = ColumnDataSource({
        'x' : alias_x,
        'y' : alias_y,
        'y0': np.zeros(alias_x.size)
    })
    fig.circle(x='x', y='y', source=alias_points, fill_color='red', line_width=3, line_color='red')
    fig.segment(x0='x', x1='x', y0='y0', y1='y', source=alias_points,line_width=3, line_color='red')
    handle = show(fig, notebook_handle=True)

    def frame_gen(fv):
        return {'y': np.sin(fv*2*np.pi*t)}

    anim_sets = []
    for k in range(1, 6):
        f_previous = f0 + (k-1)*fs
        f_alias = f0 + k*fs
        frame_values = np.linspace(f_previous, f_alias, 60)
        anim = FrameAnimation(alias_sig.data_source(), frame_gen, frame_values)
        anim_sets.append(AnimationSet([anim]))
        anim_sets.append(AnimationSet([Pause(1)]))

    AnimateSets(anim_sets, handle).run()

def signal_grid(signals, opts={'view': 'default'}):
    fig_opts = {}
    if 'view' in opts:
        if opts['view'] == 'default':
            fig_opts.update({'yaxis': ['hide']})

    figs = []
    for signal in signals:
        fig = signal.get_fig(
            size=(200, 120), title='{} Hz'.format(signal.frequency()),
            fig_opts=fig_opts
        )
        figs.append(fig)


    fig_rows = np.array_split(figs, len(signals)/10)
    rows = [row(list(fig_row)) for fig_row in fig_rows]
    show(column(rows))
