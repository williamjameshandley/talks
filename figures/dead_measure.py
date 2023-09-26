import lecture_style
from anesthetic.examples.perfect_ns import planck_gaussian
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
from matplotlib import patheffects, rcParams
rcParams['path.effects'] = [patheffects.withStroke(linewidth=1, foreground='white')]

nlive = 100
samples = planck_gaussian(nlive)

x = 'omegabh2'
y = 'omegach2'

fig, axes = plt.subplots(1,4, figsize=(7*4*0.5,7*0.5*1.5))

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.plot(samples[x], samples[y], 'C0.', ms=0.5)

axes[0].get_children()[-1].get_zorder()
axes[0].set_zorder(10)
axes[0].get_children()[-1].get_zorder()

def get_box(i):
    live = samples.live_points(i)
    xmin = live[x].min()
    xmax = live[x].max()
    ymin = live[y].min()
    ymax = live[y].max()
    return xmin, xmax, ymin, ymax

def inter_axis_line(fig, ax0, ax1, x, y, *args, **kwargs):
    coord0 = fig.transFigure.inverted().transform(ax0.transData.transform([x, y]))
    coord1 = fig.transFigure.inverted().transform(ax1.transData.transform([x, y]))
    return Line2D([coord0[0], coord1[0]], [coord0[1], coord1[1]], transform=fig.transFigure,*args, **kwargs)

def draw_zoom_lines(ax0, ax1, xmin, xmax, ymin, ymax):
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    zorder = max([child.get_zorder() for child in ax0.get_children()] +
                 [ax0.get_zorder()])
    ax1.set_zorder(zorder+2)
    lines = list(fig.lines)
    for x_ in [xmin, xmax]:
        for y_ in [ymin, ymax]:
            lines.append(inter_axis_line(fig, ax0, ax1, x_, y_, color='k',
                                         zorder=zorder+1,
                                         lw=rcParams['axes.linewidth']))
    rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='k',
                     zorder=zorder+1, lw=rcParams['axes.linewidth'])
    ax0.add_patch(rect)
    fig.lines = lines


fig.tight_layout()
fig.canvas.draw()
fig.canvas.flush_events()
shift = 0.065

pos = axes[1].get_position()
pos.x0 -= shift
pos.x1 -= shift
pos.y0 -= shift
pos.y1 -= shift
axes[1].set_position(pos)

pos = axes[2].get_position()
pos.x0 -= shift*2
pos.x1 -= shift*2
axes[2].set_position(pos)

pos = axes[3].get_position()
pos.x0 -= shift*3
pos.x1 -= shift*3
pos.y0 -= shift
pos.y1 -= shift
axes[3].set_position(pos)

ax = fig.add_subplot(1,4,4)
ax.set_box_aspect(1)
ax.set_xticks([])
ax.set_yticks([])
ax.plot(samples[x], samples[y], 'C0.', ms=0.5)
axes = np.concatenate([axes, [ax]])

fig.canvas.draw()
fig.canvas.flush_events()
plot_live_points = True

for k, (i, ax0, ax1) in enumerate(zip(range(nlive*20, nlive*100, nlive*20), axes[:-1], axes[1:])):
    if plot_live_points:
        live = samples.live_points(i)
        ax0.plot(live[x], live[y], f'C{k+1}o')
        ax1.plot(live[x], live[y], f'C{k+1}o')
    draw_zoom_lines(ax0, ax1, *get_box(i))

fig.canvas.draw()
fig.canvas.flush_events()

if plot_live_points:
    filename = 'dead_measure_live.pdf'
else:
    filename = 'dead_measure.pdf'

fig.savefig(filename, transparent=False, bbox_inches='tight')

