#!/usr/bin/env python

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches
# matplotlib.rcParams["text.usetex"] = True
import glob
import scipy.misc

from lineage_lib import track


class AxBase(object):
    def __init__(self, ax):
        self.ax = ax
        self._postinit()

    def _postinit(self):
        pass

    def __getattr__(self, attr):
        return getattr(self.ax, attr)


class NoAx(AxBase):
    def _postinit(self):
        self.ax.axis("off")


class ManualAssign(object):
    def __init__(self, c0, c1):
        self.assignment = None
        self.phase_files = sorted(glob.glob("B/*.tif"))
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect(
            "pick_event", self.pick_event
        )
        self.fig.canvas.mpl_connect(
            "key_press_event", self.key_press
        )

        self.ax0 = NoAx(self.fig.add_subplot(121))
        self.ax1 = NoAx(self.fig.add_subplot(
            122, sharex=self.ax0, sharey=self.ax0
        ))
        self.ax1.set_title(r"Select NEW pole")

        self.plot_c1(c1)
        self.plot_c0(c0, c1)

    def get_assignment(self):
        plt.tight_layout()
        plt.show()
        return self.assignment

    def plot_mesh(self, c, ax):
        ax.plot(c.mesh[:, 0], c.mesh[:, 1], "y")
        ax.plot(c.mesh[:, 2], c.mesh[:, 3], "y")

    def pick_event(self, e):
        if self.assignment is not None:
            # reset
            self.handle0.set_color("b")
            self.handle1.set_color("b")
        e.artist.set_color("r")
        self.assignment = e.artist.pole_assignment
        plt.draw()

    def key_press(self, e):
        if e.key == "enter":
            plt.close()

    def plot_c0(self, c0, c1):
        if c0:
            i0 = scipy.misc.imread(self.phase_files[c0.frame - 1])
            self.plot_mesh(c0, self.ax0)
        else:
            i1 = scipy.misc.imread(self.phase_files[c1.frame - 1])
            i0 = np.zeros(i1.shape)
        self.ax0.imshow(i0, cmap=plt.cm.gray)

    def plot_c1(self, c1):
        i1 = scipy.misc.imread(self.phase_files[c1.frame - 1])
        self.ax1.imshow(i1, cmap=plt.cm.gray)

        p0 = c1.mesh[0, 0:2]
        p1 = c1.mesh[-1, 0:2]
        self.handle0 = matplotlib.patches.Circle(p0, picker=True)
        self.handle0.pole_assignment = 0
        self.handle1 = matplotlib.patches.Circle(p1, picker=True)
        self.handle1.pole_assignment = -1
        self.ax1.add_artist(self.handle0)
        self.ax1.add_artist(self.handle1)


class PoleAssign(object):
    def __init__(self, frames=None):
        if not frames:
            l = track.Lineage()
            frames = l.frames
        self.frames = frames

    def get_poles(self, cell):
        pole1 = cell.mesh[0, 0:2]
        pole2 = cell.mesh[-1, 0:2]

        midpoint = len(cell.mesh) / 2
        mtest = int(midpoint)
        if mtest == midpoint:
            # get value directly
            midcell = cell.mesh[mtest - 1, 0:2]
        else:
            midcell1 = cell.mesh[int(midpoint - 1), 0:2]
            midcell2 = cell.mesh[int(midpoint), 0:2]
            midcells = np.vstack([midcell1, midcell2])
            midcell = midcells.mean(axis=0)
        return pole1, midcell, pole2

    def get_distances(self, master, other):
        out = []
        for c in other:
            dist = np.sqrt(
                (master[0] - c[0]) ** 2 +
                (master[1] - c[1]) ** 2
            )
            out.append(dist)
        return np.array(out)

    def assign_new_pole(self, mother, daughter):
        # returns 1, 0, or None
        # -1 = new_pole is last mesh point
        # 0 = new_pole is first mesh point
        # None = unknown / unclear
        if not mother:
            return None

        mother_cell = self.frames.cell(mother)
        daughter_cell = self.frames.cell(daughter)
        m1, m2, m3 = self.get_poles(mother_cell)
        p1, _, p2 = self.get_poles(daughter_cell)
#        print(np.vstack([m1, m2, m3]))
#        print(np.vstack([p1, p2]))

        p1dist = self.get_distances(p1, [m1, m2, m3]).argmin()
        p2dist = self.get_distances(p2, [m1, m2, m3]).argmin()
        if p1dist == 1:
            return 0
        elif p2dist == 1:
            return -1
        else:
            return None

    def manual_assignment(self, c0, c1):
        M = ManualAssign(c0, c1)
        assignment = M.get_assignment()
        return assignment

    def assign_poles(self, force=False):
        if os.path.exists("poles.json") and not force:
            return json.loads(open("poles.json").read())

        assignments = {}
        # iterate through cell lineages
        progenitors = list(self.frames[0].cells)
        for progenitor in progenitors:
            # assign a pole to the first member of the lineage only
            # orientation parameter will cover rest in spot_analysis routines
            pole_assignment = self.assign_new_pole(progenitor.parent, progenitor.id)
            if not pole_assignment:
                pole_assignment = self.manual_assignment(
                    self.frames.cell(progenitor.parent),  # parent      (n - 1)
                    progenitor,  # this cell                            (n)
                )

            assignments[progenitor.id] = pole_assignment
            while type(progenitor.children) is str:
                progenitor = self.frames.cell(progenitor.children)
            if progenitor.children:
                progenitors.append(self.frames.cell(progenitor.children[0]))
                progenitors.append(self.frames.cell(progenitor.children[1]))

        open("poles.json", "w").write(json.dumps(assignments))
        return assignments
