#!/usr/bin/env python

import os
import sys
import json
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.patches
import matplotlib.collections
import matplotlib.path
import matplotlib.gridspec
import seaborn as sns
import numpy as np
import scipy.misc
import glob

plt.rc("font", family="sans-serif")
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
]
sns.set_style("white")
sns.set_context("talk")

from lineage_lib import track


class Faked(object):
    def __init__(self, interval):
        self.interval = interval

    def __getitem__(self, x):
        return (x - 1) * self.interval


class DeadCells:
    def __init__(self):
        self.d = {}

    def set_death_frame(self, frame, generation, cell_ref, length):
        if frame not in self.d:
            self.d[frame] = []
        self.d[frame].append((generation, cell_ref, length, False))

    def __getitem__(self, f):
        # return any dead cells which died <= f
        r = []
        for death_f in self.d.keys():
            if death_f <= f:
                r.extend(self.d[death_f])
        return r


def get_px():
    if os.path.exists(".WF1_100"):
        return 0.062893  # calibration for 1px in um for 100x objective (WF1)
    elif os.path.exists(".WF2_63"):
        return 0.12254  # calibration of 1px in um for 63x objective (WF2)
    elif os.path.exists("timings.json"):
        return json.loads(open("timings.json").read())["px"]
    else:
        return 0.1031746  # calibration of 1px in um for 63x objective (WF3)


def __(x):
    return datetime.datetime.strptime("{0} {1}".format(*x), "%d.%m.%y %H:%M")


def get_frame_timings():
    """Convert frames to times (in minutes)"""
    if os.path.exists("timings.json"):
        f = json.loads(open("timings.json").read())
        i = f["pass_delay"]
        t = f["timings"]
        t0 = __(t[0])
        T = []
        for t_ in t:
            t1 = __(t_) - t0
            for x in range(t_[2]):
                T.append(int(t1.seconds / 60))
                t1 += datetime.timedelta(minutes=i)
        return T
    else:
        return Faked(15)
    return


def frames_to_times(x, T, in_hours=False):
    if type(x) is not list:
        return in_hours and T[x - 1] / 60 or T[x - 1]
    else:
        return in_hours and [T[z - 1] / 60 for z in x] or [T[z - 1] for z in x]


def create_path(cell_data, x0, colour):
    vertices = []
    codes = []

    arc1 = matplotlib.path.Path.arc(90, 270)
    add = np.array([x0 + 1, 0])
    arc1_vertices = arc1._vertices + add
    vertices.extend([
        tuple(x) for x in list(arc1_vertices)
    ])
    codes.extend(list(arc1._codes))

    vertices.append((x0 + cell_data[3] - 1, -1))
    codes.append(matplotlib.path.Path.LINETO)

    arc2 = matplotlib.path.Path.arc(270, 90)
    add = np.array([x0 + cell_data[3] - 1, 0])
    arc2_vertices = arc2._vertices + add
    vertices.extend([
        tuple(x) for x in list(arc2_vertices)
    ])
    codes.extend(list(arc2._codes))

    vertices.append((x0 + 1, 1))
    codes.append(matplotlib.path.Path.LINETO)

    path = matplotlib.path.Path(vertices, codes)
    if colour == "none":
        edgecolour = "0.8"
    else:
        edgecolour = "k"
    return matplotlib.patches.PathPatch(
        path,
        facecolor=colour,
        edgecolor=edgecolour,
        lw=1
    )


def create_frame_patches(frame, data, dead_data, colours):
    data.extend(dead_data)
    max_gen = max([x[0] for x in data])
    cell_data = sorted(
        [
            (
                c[0],
                int("{0:b}".format(c[1]).zfill(max_gen + 1)[::-1], 2),
                c[1],
                c[2],
                c[3],
            ) for c in data
        ],
        key=lambda x: x[1]
    )
    # input("F{0}, gen {1}: {2}".format(frame, max_gen, cell_data))
    xstart = 0
    cells = []
    for c in cell_data:
        if c[4]:
            colour = colours.pop(0)
        else:
            colour = "none"
        cell_patch = create_path(c, xstart, colour)
        cells.append(cell_patch)
        xstart += c[3]
    collection = matplotlib.collections.PatchCollection(
        cells, match_original=True,
    )
    return collection, xstart


def get_completed(frame, data):
    fs = np.array(list(data.keys()))
    comp_frames = np.where(fs < frame)[0]
    out = []
    for x in comp_frames:
        f = fs[int(x)]
        out.extend(data[f])
    return out


def create_lineage_animation(PX, T, lineage, lin_num):
    cell_ref = 0b000000000000
    queue = [(0, cell_ref, lineage)]

    frames_lineage_data = {}
    frames_mesh_data = {}
    frames_length_data = {}
    completed_length_data = {}
    dead_cells = DeadCells()
    while queue:
        generation, cell_ref, lin = queue.pop()
        lengths = lin.lengths(PX)
        frames = lin.frames()
        cells = lin.cells
        length_colour = json.dumps([float(_) for _ in np.random.rand(3, 1)])
        length_buffer = []
        for f, l, c in zip(frames, lengths, cells):
            length_buffer.append((f, l, length_colour))
            if f not in frames_lineage_data:
                frames_lineage_data[f] = []
            frames_lineage_data[f].append((generation, cell_ref, l, True))

            if f not in frames_mesh_data:
                frames_mesh_data[f] = []
            frames_mesh_data[f].append(c.mesh)

            if f not in frames_length_data:
                frames_length_data[f] = []
            frames_length_data[f].append(np.array(length_buffer))

        if frames[-1] not in completed_length_data:
            completed_length_data[frames[-1]] = []
        completed_length_data[frames[-1]].append(np.array(length_buffer))

        if lin.children:
            # determine pole assignment
            generation += 1
            queue.insert(0, (generation, cell_ref, lin.children[0]))
            cell_ref += (1 << generation)
            queue.insert(0, (generation, cell_ref, lin.children[1]))
        else:
            dead_cells.set_death_frame(f + 1, generation, cell_ref, lengths[-1])

    fig1 = plt.figure(figsize=(10, 7.5))
    gspec = matplotlib.gridspec.GridSpec(
        3, 2,
        width_ratios=[1, 1],
        height_ratios=[2, 0.2, 1]
    )
    mask_ax = fig1.add_subplot(gspec[0, 1], aspect="equal")
    mask_ax.xaxis.set_ticks_position("none")
    mask_ax.set_xticklabels([])
    mask_ax.yaxis.set_ticks_position("none")
    mask_ax.set_yticklabels([])

    image_ax = fig1.add_subplot(gspec[0, 0], aspect="equal")
    image_ax.axis("off")

    length_ax = fig1.add_subplot(gspec[2, :])
    length_ax.set_xlabel("Time (h)")
    length_ax.set_ylabel("Length (\si{\micro\metre})")
    sns.despine(ax=length_ax)

    lineage_ax = fig1.add_subplot(gspec[1, :])
    lineage_ax.axis("off")

    file_names = sorted(
        glob.glob("B/*.tif"),
        key=lambda x: int(x.split(".tif")[0].split("B/")[1])
    )
    artists_fig1 = []
    maxx = 0
    for frame in sorted(frames_lineage_data.keys()):
        frame_artists_fig1 = []
        cells = frames_mesh_data[frame]
        fn = file_names[frame - 1]
        image = scipy.misc.imread(fn)
        frame_artists_fig1.append(
            image_ax.imshow(image, cmap=plt.cm.gray)
        )
        del image
        lengths = frames_length_data[frame]
        colours = [json.loads(_[0, 2]) for _ in lengths]
        for mesh, colour in zip(cells, colours):
            poly = np.array(
                list(mesh[:, 0:2]) +
                list(mesh[:, 2:4][::-1])
            )
            l, = image_ax.plot(
                poly[:, 0], poly[:, 1], ls="-", color="y", alpha=0.7, lw=1
            )
            frame_artists_fig1.append(l)
            p = matplotlib.patches.Polygon(
                poly,
                facecolor=colour,
                edgecolor="k",
                lw=0.1,
            )
            cell_mask = mask_ax.add_patch(p)
            frame_artists_fig1.append(cell_mask)

        lengths.extend(get_completed(frame, completed_length_data))
        for l in lengths:
            t = frames_to_times(
                [int(_) for _ in l[:, 0]], T, in_hours=True
            )
            l, = length_ax.plot(t, l[:, 1], color=json.loads(l[0, 2]))
            frame_artists_fig1.append(l)

        frame_data = frames_lineage_data[frame]
        patch, xend = create_frame_patches(frame, frame_data, dead_cells[frame], colours=colours)
        frame_artists_fig1.append(
            lineage_ax.add_collection(patch),
        )

        artists_fig1.append(frame_artists_fig1)
        if xend > maxx:
            maxx = xend

    Writer = matplotlib.animation.writers["ffmpeg"]
    writer = Writer(fps=3, bitrate=10000)

    mask_ax.autoscale_view(True, True, True)
    # increase limits by 10 pixels and make square
    mx = list(mask_ax.get_xlim())
    my = list(mask_ax.get_ylim())
    if np.abs(mx[1] - mx[0]) > np.abs(my[1] - my[0]):
        # increase y range
        d = np.abs(np.abs(mx[1] - my[0]) - np.abs(my[1] - my[0]))
        my[0] = my[0] - (d / 2)
        my[1] = my[1] + (d / 2)
    elif np.abs(mx[1] - mx[0]) < np.abs(my[1] - my[0]):
        # increase x range
        d = np.abs(np.abs(mx[1] - my[0]) - np.abs(my[1] - my[0]))
        mx[0] = mx[0] - (d / 2)
        mx[1] = mx[1] + (d / 2)

    mx[0] = mx[0] - 10
    mx[1] = mx[1] + 10
    my[0] = my[0] - 10
    my[1] = my[1] + 10
    mask_ax.set_xlim(mx)
    mask_ax.set_ylim(my)
    image_ax.set_xlim(mx)
    image_ax.set_ylim(my)
    lineage_ax.set_xlim([-1, maxx + 1])
    lineage_ax.set_ylim([-2, 2])
    fig1.tight_layout()
    anim1 = matplotlib.animation.ArtistAnimation(
        fig1,
        artists_fig1,
        interval=20,
        repeat_delay=3000,
        blit=True,
    )
    if not os.path.exists("animations"):
        os.mkdir("animations")
    anim1.save("animations/animation-{0:03d}.mp4".format(lin_num), writer=writer)
    print("Saved animation-{0:03d}.mp4".format(lin_num))
    plt.close("all")


def main():
    L = track.Lineage()
    PX = get_px()
    T = get_frame_timings()
    init_cells = L.frames[0].cells
    lineages = []
    for init_cell in init_cells:
        cell_lineage = track.SingleCellLineage(init_cell.id, L)
        if "-s" in sys.argv:
            # skip first division
            if cell_lineage.children:
                lineages.append(cell_lineage.children[0])
                lineages.append(cell_lineage.children[1])
        else:
            lineages.append(cell_lineage)

    lin_num = 1
    for lineage in lineages:
        create_lineage_animation(PX, T, lineage, lin_num)
        lin_num += 1

if __name__ == "__main__":
    main()
