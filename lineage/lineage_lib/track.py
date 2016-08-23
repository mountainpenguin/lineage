#!/usr/bin/env python

"""This module is designed to provide tools that allow assignment of cell
   lineages from MicrobeTracker results.
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import scipy.io
import scipy.misc
import skimage.draw
import uuid
import os
import json
import logging
import glob
import time

from lineage_lib import poles

__version__ = "0.1"
__author__ = "Miles Priestman <priestman.miles@gmail.com>"


class PoleAge(object):
    def __init__(self, age_known=False, age_start=0):
        self.age_known = age_known
        self.age = age_start

    def __iadd__(self, x):
        self.age += x

    def __add__(self, x):
        new_age = self.age + 1
        return PoleAge(self.age_known, new_age)

    def __repr__(self):
        if self.age_known:
            return "{0}".format(self.age)
        else:
            return "+{0}".format(self.age)


class SingleCellLineage(object):
    """Class for defining and determining a single cell lineage"""
    def __init__(self, init_id, L, pole1_age=None, pole2_age=None):
        c = L.frames.cell(init_id)
        self.cells = [c]
        self.pole1_age = pole1_age or PoleAge()
        self.pole2_age = pole2_age or PoleAge()

        i = 0
        while type(c.children) is str:
            c = L.frames.cell(c.children)
            prev_c = self.cells[i]
            c = self._orient_cell(prev_c, c)
            self.cells.append(c)
            i += 1

        # last cell
        if type(c.children) is list:
            pole_assignments = self.get_poles(
                c,
                c.children[0],
                c.children[1],
                L
            )
            child1_pole1 = PoleAge(True, 0)
            child1_pole2 = PoleAge(True, 0)
            child2_pole1 = PoleAge(True, 0)
            child2_pole2 = PoleAge(True, 0)

            old1 = pole_assignments["old1"]
            old2 = pole_assignments["old2"]

            if old1[0] == 0 and old1[1] == 0:
                child1_pole1 = self.pole1_age + 1
            elif old1[0] == 0 and old1[1] == 1:
                child1_pole2 = self.pole1_age + 1
            elif old1[0] == 1 and old1[1] == 0:
                child2_pole1 = self.pole1_age + 1
            elif old1[0] == 1 and old1[1] == 1:
                child2_pole2 = self.pole1_age + 1

            if old2[0] == 0 and old2[1] == 0:
                child1_pole1 = self.pole2_age + 1
            elif old2[0] == 0 and old2[1] == 1:
                child1_pole2 = self.pole2_age + 1
            elif old2[0] == 1 and old2[1] == 0:
                child2_pole1 = self.pole2_age + 1
            elif old2[0] == 1 and old2[1] == 1:
                child2_pole2 = self.pole2_age + 1

            self.children = [
                SingleCellLineage(c.children[0], L, child1_pole1, child1_pole2),
                SingleCellLineage(c.children[1], L, child2_pole1, child2_pole2)
            ]
        else:
            self.children = None

    def _calc_distance(self, p1, p2):
        return (
            np.sqrt(
                (p2[0] - p1[0]) ** 2 +
                (p2[1] - p1[1]) ** 2
            )
        )

    def _manual_keypress(self, event):
        if event.key not in ["enter", "escape"]:
            return

        if event.key == "escape":
            # restart
            self.fig._status = 1
            for p in self.ax_work.patches:
                p.set_picker(True)
                p.set_facecolor((0.8, 0.8, 0.8, 0.5))
                p.set_radius(3)
            self.ax_work.set_title("Select NEW pole (child1)")
            plt.draw()

        elif event.key == "enter":
            if self.fig._status < 4:
                return
            plt.close()

    def _manual_pick(self, event):
        if self.fig._status == 1 and event.artist._pole_id["child"] == 2:
            return
        elif self.fig._status == 2 and event.artist._pole_id["child"] == 1:
            return

        if (time.time() - self.fig._last_click) < 0.2:
            return

        if self.fig._status == 1:
            self.fig._new_pole1 = event.artist._pole_id
            self.ax_work.set_title("Select NEW pole (child2)")
            event.artist.set_facecolor("none")
            event.artist.set_radius(1)
            plt.draw()
        elif self.fig._status == 2:
            self.fig._new_pole2 = event.artist._pole_id
            self.ax_work.set_title("Select YELLOW pole")
            event.artist.set_facecolor("none")
            event.artist.set_radius(1)
            plt.draw()
        elif self.fig._status == 3:
            self.fig._yellow_pole = event.artist._pole_id
            self.ax_work.set_title("Select GREEN pole")
            event.artist.set_facecolor("y")
            plt.draw()
        else:
            self.fig._green_pole = event.artist._pole_id
            self.ax_work.set_title("Press ENTER to accept")
            event.artist.set_facecolor("g")
            plt.draw()
            return

        event.artist.set_picker(None)
        self.fig._status += 1
        self.fig._last_click = time.time()

    def _manual_fix(self, mother, child1, child2, mother_poles, child1_poles, child2_poles):
        # manual fix
        self.fig = plt.figure()
        ax_mother = self.fig.add_subplot(121, aspect="equal")
        ax_mother.plot(mother.mesh[:, 0], mother.mesh[:, 1], "k-")
        ax_mother.plot(mother.mesh[:, 2], mother.mesh[:, 3], "k-")
        mother_p1 = matplotlib.patches.Circle(mother_poles[0], radius=3, facecolor="y", lw=2)
        mother_p2 = matplotlib.patches.Circle(mother_poles[1], radius=3, facecolor="g", lw=2)
        ax_mother.add_patch(mother_p1)
        ax_mother.add_patch(mother_p2)
        ax_mother.axis("off")

        self.ax_work = self.fig.add_subplot(122, aspect="equal")
        self.ax_work.plot(child1.mesh[:, 0], child1.mesh[:, 1], "k-")
        self.ax_work.plot(child1.mesh[:, 2], child1.mesh[:, 3], "k-")
        self.ax_work.plot(child2.mesh[:, 0], child2.mesh[:, 1], "k-")
        self.ax_work.plot(child2.mesh[:, 2], child2.mesh[:, 3], "k-")
        kwargs = {
            "facecolor": (0.8, 0.8, 0.8, 0.5),
            "radius": 3,
            "lw": 2,
            "picker": True,
        }

        c11 = matplotlib.patches.Circle(child1_poles[0], **kwargs)
        c11._pole_id = {"child": 1, "pole_num": 1}
        c12 = matplotlib.patches.Circle(child1_poles[1], **kwargs)
        c12._pole_id = {"child": 1, "pole_num": 2}
        c21 = matplotlib.patches.Circle(child2_poles[0], **kwargs)
        c21._pole_id = {"child": 2, "pole_num": 1}
        c22 = matplotlib.patches.Circle(child2_poles[1], **kwargs)
        c22._pole_id = {"child": 2, "pole_num": 2}
        self.ax_work.add_patch(c11)
        self.ax_work.add_patch(c12)
        self.ax_work.add_patch(c21)
        self.ax_work.add_patch(c22)
        self.ax_work.axis("off")
        self.ax_work.set_title("Select NEW pole (child1)")

        self.fig.canvas.mpl_connect("pick_event", self._manual_pick)
        self.fig.canvas.mpl_connect("key_press_event", self._manual_keypress)
        self.fig._status = 1
        self.fig._last_click = time.time()
        plt.show()

        new_pole1 = (0, self.fig._new_pole1["pole_num"] - 1)
        new_pole2 = (1, self.fig._new_pole2["pole_num"] - 1)

        ref = self.fig._yellow_pole
        old_pole1 = (ref["child"] - 1, ref["pole_num"] - 1)

        ref = self.fig._green_pole
        old_pole2 = (ref["child"] - 1, ref["pole_num"] - 1)
        return new_pole1, new_pole2, old_pole1, old_pole2

    def get_poles(self, mother, child1_id, child2_id, L):
        if os.path.exists(".pole_data/{0}.json".format(mother.id)):
            return json.loads(open(".pole_data/{0}.json".format(mother.id)).read())

        mother_poles = (
            np.array([mother.mesh[0, 0], mother.mesh[0, 1]]),
            np.array([mother.mesh[-1, 0], mother.mesh[-1, 1]])
        )
        child1 = L.frames.cell(child1_id)
        child1_poles = (
            np.array([child1.mesh[0, 0], child1.mesh[0, 1]]),
            np.array([child1.mesh[-1, 0], child1.mesh[-1, 1]])
        )
        child2 = L.frames.cell(child2_id)
        child2_poles = (
            np.array([child2.mesh[0, 0], child2.mesh[0, 1]]),
            np.array([child2.mesh[-1, 0], child2.mesh[-1, 1]])
        )

        # nearest child poles
        pairs = [
            (0, 0), (0, 1), (1, 0), (1, 1)
        ]
        d = [
            self._calc_distance(child1_poles[a], child2_poles[b])
            for a, b in pairs
        ]
        d_ref = pairs[d.index(min(d))]

        new_pole1_idx = d_ref[0]
        new_pole2_idx = d_ref[1]

        avail = [child1_poles[not d_ref[0]], child2_poles[not d_ref[1]]]
        avail_idxs = [int(not d_ref[0]), int(not d_ref[1])]

        d1 = [
            self._calc_distance(mother_poles[0], avail[0]),
            self._calc_distance(mother_poles[0], avail[1])
        ]
        d2 = [
            self._calc_distance(mother_poles[1], avail[0]),
            self._calc_distance(mother_poles[1], avail[1])
        ]
        child_order = [d1.index(min(d1)), d2.index(min(d2))]
        a1_idx = d1.index(min(d1))
        old_pole1 = avail[a1_idx]
        old_pole1_idx = avail_idxs[a1_idx]
        a2_idx = d2.index(min(d2))
        old_pole2 = avail[a2_idx]
        old_pole2_idx = avail_idxs[a2_idx]

        if old_pole1.sum() == old_pole2.sum() or min(d) > 20 or min(d1) > 20 or min(d2) > 20:
            new_pole1, new_pole2, old_pole1, old_pole2 = self._manual_fix(
                mother, child1, child2, mother_poles, child1_poles, child2_poles
            )
            child_order = [old_pole1[0], old_pole2[0]]
            new_pole1_idx = new_pole1[1]
            new_pole2_idx = new_pole2[1]
            old_pole1_idx = old_pole1[1]
            old_pole2_idx = old_pole2[1]

        out_data = {
            "old1": (child_order[0], old_pole1_idx),
            "new1": (0, new_pole1_idx),
            "new2": (1, new_pole2_idx),
            "old2": (child_order[1], old_pole2_idx),
        }
        if not os.path.exists(".pole_data"):
            os.mkdir(".pole_data")
        with open(".pole_data/{0}.json".format(mother.id), "w") as fn:
            fn.write(json.dumps(out_data))
        return out_data

    def _do_reversal(self, c):
        c.mesh = c.mesh[::-1]
        c.steparea = c.steparea[::-1]
        c.steplength = c.steplength[::-1]
        c.stepvolume = c.stepvolume[::-1]
        return c

    def _orient_cell(self, prev_cell, new_cell):
        fn = os.path.join(".orientation_data/{0}-{1}".format(prev_cell.id, new_cell.id))
        if os.path.exists(fn):
            reversal = bool(open(fn).read())
            if reversal:
                return self._do_reversal(new_cell)
            else:
                return new_cell

        reversal = False
        prev1 = prev_cell.mesh[0, 0:2]
        prev2 = prev_cell.mesh[-1, 0:2]

        prev_centre = np.array(prev_cell.centre)
        new_centre = np.array(new_cell.centre)
        translation = new_centre - prev_centre

        new_l = new_cell.mesh[:, 0:2] - translation

        new1 = new_l[0]
        new2 = new_l[-1]

        d1 = [
            self._calc_distance(prev1, new1),
            self._calc_distance(prev1, new2)
        ]
        d2 = [
            self._calc_distance(prev2, new1),
            self._calc_distance(prev2, new2)
        ]
        # nearest new_pole to prev_pole1
        d1_idx = d1.index(min(d1))
        # nearest new_pole to prev_pole2
        d2_idx = d2.index(min(d2))

        if max([min(d1), min(d2)]) < 15 and d1_idx is not d2_idx:
            if d1_idx != 0:
                reversal = True
        else:
            # needs manual assignment
            self._fig = plt.figure()
            ax = self._fig.add_subplot(111, aspect="equal")
            ax.axis("off")
            ax.set_title("Select indicated pole")
            ax.plot(prev_cell.mesh[:, 0], prev_cell.mesh[:, 1], "k-")
            ax.plot(prev_cell.mesh[:, 2], prev_cell.mesh[:, 3], "k-")
            ax.plot(prev_cell.mesh[0, 0], prev_cell.mesh[0, 1], "ro", ms=10)
            new_r = new_cell.mesh[:, 2:4] - translation
            ax.plot(new_l[:, 0], new_l[:, 1], "r-", zorder=-1)
            ax.plot(new_r[:, 0], new_r[:, 1], "r-", zorder=-1)

            kwargs = {
                "radius": 2,
                "edgecolor": "k",
                "lw": 2,
                "facecolor": (0.4, 0.4, 0.4, 0.5),
                "picker": True,
            }
            c1 = matplotlib.patches.Circle(new_l[0], **kwargs)
            c1._pole_num = 0
            c2 = matplotlib.patches.Circle(new_l[-1], **kwargs)
            c2._pole_num = 1
            ax.add_patch(c1)
            ax.add_patch(c2)

            self._fig.canvas.mpl_connect("pick_event", self._pickevent)

            plt.show()
            plt.close()

            if self._picked_pole == 1:
                reversal = True

        if not os.path.exists(".orientation_data"):
            os.mkdir(".orientation_data")
        with open(fn, "w") as x:
            x.write(str(int(reversal)))

        if reversal:
            return self._do_reversal(new_cell)
        else:
            return new_cell

    def _pickevent(self, event):
        self._picked_pole = event.artist._pole_num
        plt.close()

    def lengths(self, px=None):
        if px:
            return [x.length[0][0] * px for x in self.cells]
        else:
            return [x.length[0][0] for x in self.cells]

    def frames(self):
        return [x.frame for x in self.cells]


class Cell(object):
    """Simple class that holds arbritrary information about a cell.

    Args:
      cell (numpy.ndarray): input data from MATLAB file.

    Note:
        No attributes are guaranteed, they depend only on the inputted `numpy`
        array structure (via `dtype.names`). However, attributes that are
        automatically included by other methods in this module are listed
        below.

    Attributes:
        id (str): unique UUID that identifies the cell.
        frame (int): frame number of the cell
        mt_idx (int): cell number that MicrobeTracker assigns
        py_idx (int): cell number that Lineage assigns
        parent (str, None):
            `id` of the parent cell from the previous frame (i.e. does not
            imply a division event).

            *None* if no linked parent.

        children (str, tuple, None):
            if *str*: `id` of the same cell in the next frame.

            if *tuple*:
                `id` of both daughter cells in the next frame.

            *None* if no linked child (may imply that the cell has died, moved
            out of the field of view, or that the frame is the final frame).
        centre (tuple):
            tuple of XY cell centre coordinates, both as `float`.
    """
    def __init__(self, cell, dead=False):
        for d in range(len(cell.dtype)):
            self.__dict__[cell.dtype.names[d]] = cell[d]


class Frames(object):
    """Very basic container class for all frames in an image sequence

    Args:
        frames (list): list of :class:`Frame` objects.

    Attributes:
        frames (list): list of :class:`Frame` objects.
    """
    def __init__(self, frames):
        self.frames = frames
        self._idx = {}
        for f in self.frames:
            for id in f._idx.keys():
                self._idx[id] = f.frame

    def cell(self, id):
        """Return a :class:`Cell` object with corresponding `id`, searching
        all frames within the stack.

        Args:
            id (str): id (uuid) of cell

        Returns:
            :class:`Cell`: if `id` is found, None otherwise.
        """
        if id in self._idx:
            return self.frames[self._idx[id]].cell(id)

    def __getitem__(self, idx):
        return self.frames[idx]

    def __len__(self):
        return len(self.frames)


class Frame(object):
    """Container class for all cells in a single image frame.

    Args:
        frame (int): frame number.
        cells (list): list of :class:`Cell` objects.

    Attributes:
      frame (int): frame number (should be 0-indexed).
      cells (list): list of :class:`Cell` objects.
    """
    def __init__(self, frame, cells):
        self.frame = frame
        self.cells = cells
        # make cell id index
        self._idx = {}
        for c in self.cells:
            self._idx[c.id] = c.py_idx

    def cell(self, id):
        """Return a :class:`Cell` object with corresponding `id`, searching
        within the cells in the frame.

        Searches for cells within the current frame only, use
        :func:`Frames.cell` for searching all frames simultaneously.

        Args:
            id (str): id (uuid) of cell

        Returns:
            :class:`Cell`: if `id` is found, None otherwise.
        """
        if id in self._idx:
            return self.cells[self._idx[id]]


class EmptyFrame(object):
    def __init__(self):
        self.frame = None
        self.cells = []

    def cell(self, id):
        return None


class Lineage(object):
    """Contains methods for analysing and assigning lineages

    Args:
        cellmat (str): path to MicrobeTracker meshes file, defaults to
            `mt/mt.mat`
        uuidfile (str): path to uuid file created by Lineage, defaults to
            `uuids.json`
        lineagefile (str): path to lineages file created by Lineage, defaults
            to `lineages.json`
        alignmat (str): path to MicrobeTracker alignment file, defaults to
            `mt/alignment.mat`

    Note:
        Attributes are set by the methods :func:`load_alignment`,
        :func:`load_uuids`, :func:`load_lineages`, and :func:`load_cells` which
        are called on init. See below for descriptions of these attributes
    """
    def __init__(self, cellmat="mt/mt.mat", uuidfile="uuids.json",
                 lineagefile="lineages.json", alignmat="mt/alignment.mat"):
        self.CELLMAT = cellmat
        self.ALIGNMAT = alignmat
        self.UUIDFILE = uuidfile
        self.LINEAGEFILE = lineagefile

        self.load_alignment()
        self.load_uuids()
        self.load_lineages()
        self.load_cells()

    def load_alignment(self):
        """Loads alignment files as created by MicrobeTracker

        Note:
            This will be improved in the future to refine these
            alignments.

        Sets the following attributes:

        Attributes:
            alignment (numpy.ndarray): 2 x N array containing X and Y offsets
                for each of N frames.

                Currently, these values are pulled directly from
                MicrobeTracker, but will be improved in the future.

        Returns:
            None

        Raises:
            IOError: if alignment file is not found
        """
        logging.info("Loading alignment data...")
        if not os.path.exists(self.ALIGNMAT):
            raise IOError("Alignment file not found ({0})".format(self.ALIGNMAT))

        alignment = scipy.io.loadmat(self.ALIGNMAT)
        logging.info(">>> loaded")
        alignment = alignment["shiftframes"]
        alignment_x = alignment[0][0][0][0]
        alignment_y = alignment[0][0][1][0]
        alignment = np.array([alignment_x, alignment_y]).T
        logging.info(">>> parsed")
        self.alignment = alignment

    def load_uuids(self):
        """Load any UUIDs set by a prior run of Lineage

        Sets the following attributes:

        Attributes:
            uuids (dict): References all cell UUIDs according to
                MicrobeTracker's cell numbering system

                Cells are referenced with the following key structure:
                    `frame`:`mt_idx`
                Where `frame` is the frame number of the cell, and `mt_idx`
                is the cell number assigned by MicrobeTracker

                Warning:
                    If cell numbers are changed by subsequent runs of
                    MicrobeTracker, these UUIDs will no longer be valid.
        """
        logging.info("Loading UUIDs...")
        if os.path.exists(self.UUIDFILE):
            self.uuids = json.loads(open(self.UUIDFILE).read())
        else:
            self.uuids = {}
            # "frame:mt_idx": uuid

    def load_lineages(self):
        """Load any previously determined lineages by Lineage

        Sets the following attributes:

        Attributes:
            lineages (dict): Links each cell `id` with its parent from the
                previous frame, and any children in the subsequent frame

                The key is the cell `id` and the value is a tuple with
                structure:
                    (`parent_id`, `child_id`)
                `parent_id` (str, None) is the id of the parent cell or None,
                and `child_id` (str, tuple, None) is the id of the child (str),
                both child ids (tuple), or None

                Example:
                    If no division has occured:
                        (`parent`, `child`)
                    If a division has occured:
                        (`parent`, (`child1`, `child2`))
                    If a death has occured:
                        (`parent`, None)
        """
        logging.info("Loading lineages...")
        if os.path.exists(self.LINEAGEFILE):
            self.lineages = json.loads(open(self.LINEAGEFILE).read())
        else:
            self.lineages = {}
            # uuid: parent, child

    def load_cells(self):
        """Load meshes created by MicrobeTracker

        Sets the following attributes:

        Attributes:
            frames (:class:`Frames`): Contains all data from MicrobeTracker

        Returns:
            None

        Raises:
            IOError: if meshes file is not found
            SysError: if there is a problem with the cell data
        """
        if not os.path.exists(self.CELLMAT):
            raise IOError(
                "Meshes file couldn't be found ({0})".format(self.CELLMAT)
            )

        logging.info("Loading cell data...")
        cells = scipy.io.loadmat(self.CELLMAT)
        logging.info(">>> loaded")
        cellList = cells["cellList"][0]

        cell_total = 0
        pos_idx = 0
        frames = []

        for pos in cellList:
            if pos.shape[0] == 0:
                pos_idx += 1
                continue
            cell_idx = 0
            py_cell_idx = 0
            frame_cells = []
            for c in pos[0]:
                if c.shape[0] != 0 and c.shape[1] != 0:
                    cell = Cell(c[0][0])
                    if "length" not in cell.__dict__:
                        raise SystemError("Please reselect Cell {0} in Frame {1}".format(
                            cell_idx + 1, pos_idx + 1
                        ))
                    elif cell.mesh.shape == (1, 1):
                        pass
                    else:
                        cell.frame = pos_idx + 1
                        cell.mt_idx = cell_idx + 1
                        cell.py_idx = py_cell_idx
                        uuid_chk = "frame:{0}:{1}".format(
                            cell.frame, cell.mt_idx
                        )
                        if uuid_chk in self.uuids:
                            cell.id = self.uuids[uuid_chk]
                        else:
                            cell.id = uuid.uuid4().urn[9:]
                            self.uuids[uuid_chk] = cell.id

                        # check for lineages
                        if cell.id in self.lineages:
                            cell.parent = self.lineages[cell.id][0]
                            cell.children = self.lineages[cell.id][1]
                        else:
                            cell.parent = None
                            cell.children = None

                        cell.box_centre = self.get_box_centre(cell)
                        cell.centre = self.get_mesh_centre(cell)

                        py_cell_idx += 1
                        cell_total += 1
                        frame_cells.append(cell)
                cell_idx += 1
            frame = Frame(pos_idx, frame_cells)
            frames.append(frame)
            pos_idx += 1
        logging.info(">>> parsed")
        logging.info(">>> {0} frames, {1} cells".format(
            pos_idx, cell_total
        ))

        logging.info("Saving UUIDs...")
        open(self.UUIDFILE, "w").write(json.dumps(self.uuids))
        logging.info(">>> saved")

        self.frames = Frames(frames)

    def get_mesh_centre(self, cell):
        """Return the coordinates for cell centre using the mesh of the given
        cell.

        Uses the midpoint of the middle mesh coordinates.

        Args:
            cell (:class:`Cell`): Cell in question

        Returns:
            (numpy.float64, numpy.float64): X and Y coordinates for cell centre
        """
        xl = cell.mesh[:, 0]
        if len(xl) % 2 == 1:
            mid = len(xl) // 2
            middle_x1 = xl[mid]
            middle_y1 = cell.mesh[:, 1][mid]
            middle_x2 = cell.mesh[:, 2][mid]
            middle_y2 = cell.mesh[:, 3][mid]
        else:
            # average middle xs
            # 10 elements
            mid2 = len(xl) // 2
            mid1 = mid2 - 1
            middle_x1 = np.mean([xl[mid1], xl[mid2]])
            middle_y1 = np.mean([
                cell.mesh[:, 1][mid1], cell.mesh[:, 1][mid2]
            ])
            middle_x2 = np.mean([
                cell.mesh[:, 2][mid1], cell.mesh[:, 2][mid2]
            ])
            middle_y2 = np.mean([
                cell.mesh[:, 3][mid1], cell.mesh[:, 3][mid2]
            ])

        return np.mean([middle_x1, middle_x2]), np.mean([middle_y1, middle_y2])

    def get_box_centre(self, cell):
        """Return the coordinates for cell centre of the given cell.

        Averages all the outline coordinates.

        Args:
            cell (:class:`Cell`): Cell in question.

        Returns:
            (numpy.float64, numpy.float64): X and Y coordinates for cell centre
        """
        xl = cell.mesh[:, 0]
        yl = cell.mesh[:, 1]
        xr = cell.mesh[:, 2]
        yr = cell.mesh[:, 3]
        c_x = np.mean([xl.mean(), xr.mean()])
        c_y = np.mean([yl.mean(), yr.mean()])
        return c_x, c_y

    def get_overlap(self, c1, c2):
        """Return area of overlap between two cells

        Constructs two circles centred on each cell, with the same diameter as
        the cell length, and calculates the area of overlap between the two
        cells.

        If the distance between the two cell centres is greater than 50px,
        the match is rejected without constructing any circles.

        The area of overlap is multipled by an angle factor, which biases
        overlaps against rotation, i.e. if a potential child cell is
        significantly rotated, its returned overlap area will be reduced.
        The angle of the cell axis is determined by constructing a line between
        the cell poles, and determining the rotation from the X-axis. They are
        normalised to ensure that the greatest possible angle of rotation is
        90 degrees.

        Angle factors:
            90deg rotation => 0.5
            45deg rotation => 0.75
            0deg rotation => 1

        Args:
            c1 (:class:`Cell`): Reference cell
            c2 (:class:`Cell`): Candidate cell

        Returns:
            float: area of overlap (pixels^2) between c1 and c2
        """
        if not hasattr(c1, "radius"):
            c1.radius = c1.length[0][0] / 2

        if not hasattr(c2, "radius"):
            c2.radius = c2.length[0][0] / 2

        # calculate distance between cell centres
        dist = np.sqrt(
            ((c1.centre[0] - c2.centre[0]) ** 2) +
            ((c1.centre[1] - c2.centre[1]) ** 2)
        )

        if dist > 50:
            return 0

        ang1 = np.arccos(
            ((c2.radius ** 2) +
             (dist ** 2) -
             (c1.radius ** 2)) /
            (2 * c2.radius * dist)
        )

        ang2 = np.arccos(
            ((c1.radius ** 2) +
             (dist ** 2) -
             (c2.radius ** 2)) /
            (2 * c1.radius * dist)
        )

        # area of intersection
        area = (ang1 * (c2.radius ** 2) -
                0.5 * (c2.radius ** 2) * np.sin(2 * ang1) +
                ang2 * (c1.radius ** 2) -
                0.5 * (c1.radius ** 2) * np.sin(2 * ang2))

        # determine angle of rotation of cell 1
        p1_x1 = c1.mesh[0, 0]
        p1_y1 = c1.mesh[0, 1]
        p1_x2 = c1.mesh[-1, 0]
        p1_y2 = c1.mesh[-1, 1]

        a1 = np.arctan(
            (p1_y2 - p1_y1) /
            (p1_x2 - p1_x1)
        )
        if a1 < 0:
            a1 += np.pi

        p2_x1 = c2.mesh[0, 0]
        p2_y1 = c2.mesh[0, 1]
        p2_x2 = c2.mesh[-1, 0]
        p2_y2 = c2.mesh[-1, 1]

        a2 = np.arctan(
            (p2_y2 - p2_y1) /
            (p2_x2 - p2_x1)
        )
        if a2 < 0:
            a2 += np.pi

        a_delta = np.abs(a2 - a1)

        # determine angle factor
        # pi / 2 => 0.5
        # pi / 4 => 0.75
        # 0      => 1
        angle_factor = (np.pi - a_delta) / np.pi

        area *= angle_factor

        return area

    def get_distance(self, c1, c2, max_dist=50):
        """Return modified distance between two cells

        Determines distance by calculating the average distance between three
        points on each cell: the cell centre, and each pole.

        The distance is multiplied by an angle factor, which biases distances
        against cells which are rotated.
        The angle of the cell axis is determined by constructing a line between
        the cell poles, and determining the difference in rotation from the
        X-axis.

        Angle factors:
            - 90deg rotation => 2
            - 45deg rotation => 1.5
            - 0deg rotation => 1

        Poles are arranged to ensure pole reversal doesn't occur.

        Args:
            c1 (:class:`Cell`): Reference cell
            c2 (:class:`Cell`): Candidate cell
            max_dist: Maximum distance permitted between two cells to be
                considered, if over this limit, function returns -1

        Returns:
            float: Average distance between `c1` and `c2`
        """
        p1_1, p1_2 = self._get_poles(c1)
        p2_1, p2_2 = self._get_poles(c2)

        pole1_dist, pole2_dist, p2_1, p2_2 = self.get_pole_distances(
            p1_1, p1_2, p2_1, p2_2
        )

        centre_dist = np.sqrt(
            ((c1.centre[0] - c2.centre[0]) ** 2) +
            ((c1.centre[1] - c2.centre[1]) ** 2)
        )

        dist = np.mean([pole1_dist, pole2_dist, centre_dist])

        if dist > max_dist:
            return -1

        # calculate angle factor
        a1 = np.arctan(
            (p1_2[1] - p1_1[0]) /
            (p1_2[0] - p1_1[0])
        )
        if a1 < 0:
            a1 += np.pi

        a2 = np.arctan(
            (p2_2[1] - p2_1[0]) /
            (p2_2[0] - p2_1[0])
        )
        if a2 < 0:
            a2 += np.pi

        a_delta = np.abs(a2 - a1)
        angle_factor = 1 + (a_delta / (np.pi / 2))

        dist *= angle_factor

        return dist

    def _get_poles(self, c):
        return (
            (c.mesh[0, 0], c.mesh[0, 1]),
            (c.mesh[-1, 0], c.mesh[-1, 1])
        )

    def get_pole_distances(self, p1_1, p1_2, p2_1, p2_2):
        pole1_dist = np.sqrt(
            ((p2_1[0] - p1_1[0]) ** 2) +
            ((p2_1[1] - p1_1[1]) ** 2)
        )
        pole2_dist = np.sqrt(
            ((p2_2[0] - p1_2[0]) ** 2) +
            ((p2_2[1] - p1_2[1]) ** 2)
        )

        # if p2_2 is closer to p1_1 than p1_2, swap p2_1 and p2_2
        alt_dist = np.sqrt(
            ((p2_2[0] - p1_1[0]) ** 2) +
            ((p2_2[1] - p1_1[1]) ** 2)
        )
        if alt_dist < pole2_dist:
            # swap p2_1 and p2_2
            p_tmp = p2_2
            p2_2 = p2_1
            p2_1 = p_tmp
            del p_tmp

            pole1_dist = alt_dist
            pole2_dist = np.sqrt(
                ((p2_2[0] - p1_2[0]) ** 2) +
                ((p2_2[1] - p1_2[1]) ** 2)
            )
        return pole1_dist, pole2_dist, p2_1, p2_2

    def search_daughters(self, c1, n1_frame):
        # top_candidates
        # - cells with one pole (first pole) close to c1 centre

        p1_1, p1_2 = self._get_poles(c1)

        scores = []
        for cell in n1_frame.cells:
            p2_1, p2_2 = self._get_poles(cell)
            pole1_dist, pole2_dist, p2_1, p2_2 = self.get_pole_distances(
                p1_1, p1_2, p2_1, p2_2
            )

            # calculate distance between p2_1 and c1 centre
            c1_dist = np.sqrt(
                ((p2_1[0] - c1.centre[0]) ** 2) +
                ((p2_1[1] - c1.centre[1]) ** 2)
            )

            # calculate distance between p2_2, and p1_1 and p1_2, use
            # lowest value for scoring
            p1_dist = np.sqrt(
                ((p2_2[0] - p1_1[0]) ** 2) +
                ((p2_2[1] - p1_1[1]) ** 2)
            )
            p2_dist = pole2_dist

            # aggregate by simple addition
            aggregate = c1_dist + p1_dist + p2_dist

            # perform in opposite direction
            # - distance between p2_2 and c1 centre
            # - distance between p2_1, and  p1_1 and p1_2, using lowest value
            #   for scoring
            c1_dist_ = np.sqrt(
                ((p2_2[0] - c1.centre[0]) ** 2) +
                ((p2_2[1] - c1.centre[1]) ** 2)
            )

            p1_dist_ = pole1_dist
            p2_dist_ = np.sqrt(
                ((p2_1[0] - p1_2[0]) ** 2) +
                ((p2_1[1] - p1_2[1]) ** 2)
            )

            aggregate_ = c1_dist_ + p1_dist_ + p2_dist_

            # use lowest aggregate score

            scores.append({
                "score": min([aggregate, aggregate_]),
                "id": cell.id,
                "c1_dist": c1_dist,
                "p1_dist": p1_dist,
                "p2_dist": p2_dist,
            })

        s = sorted(scores, key=lambda x: x["score"])
        return s[0]["id"], s[1]["id"]

    def guess_lineage(self):
        """Naive guess at lineages using XY positions and cell length only.

        Updates the following attributes:

        Attributes:
            lineages (dict): As per :func:`load_lineages`

        Returns:
            None

        Similarly to :func:`interactive_track`, start from all "progenitor"
        cells in frame 1. For each of these cells, select a cell in the next
        frame which is spatially closest, if the closest cell is significantly
        smaller in length, then choose the two closest cells as daughter
        cells.

        See :func:`get_distance` for how spatial distance is calculated.

        If no cell can be found, initiate a cell death.

        If the closest cell is significantly shorter than the potential parent
        cell, look for a division event.

        Note:
            Improvements planned:

            - ensuring daughters have similar cell
              length (e.g. within 20% tolerance?)
            - using cell morphology in some clever way to determine
              division events
        """
        logging.info("Guessing lineages based on positions...")
        temp_lineage = {}
        progenitors = list(self.frames[0].cells)
        while len(progenitors) > 0:
            progenitor = progenitors.pop()
#        for progenitor in progenitors:
            logging.debug(">>> Parent cell: %s", progenitor.id)
            lineage = [progenitor.id]
#            # radius
#            radius = progenitor.length[0][0] / 2
#            progenitor.radius = radius

            cell = progenitor
            while True:
                try:
                    n1_frame = self.frames[cell.frame]
                except IndexError:
                    ending = "pseudo-death (out of frames)"
                    break
                distances = []
                for pot in n1_frame.cells:
    #                area = self.get_overlap(
    #                    cell, pot
    #                )
                    dist = self.get_distance(
                        cell, pot
                    )
                    if dist >= 0:
                        distances.append((
                            dist, pot.py_idx, pot.id
                        ))

                if not distances:
                    ending = "death"
                    break

                minimum = min(distances, key=lambda x: x[0])
                child = n1_frame.cells[minimum[1]]

                # check for cell length change
                length_change = child.length[0][0] - cell.length[0][0]
                length_perc = length_change / cell.length[0][0]

                if length_perc < -0.25:
                    # if more than 25% change (negative) in length,
                    # look for daughters
                    daughters = self.search_daughters(
                        cell, n1_frame
                    )
                    lineage.append(daughters)
                    ending = "division"
                    break
                else:
                    lineage.append(child.id)
                cell = child
            logging.debug(
                (">>> Lineage completed, %i members ending "
                 "in %s (%i - %i)"),
                len(lineage),
                ending,
                progenitor.frame,
                (type(lineage[-1]) is tuple and
                 self.frames.cell(lineage[-1][0]).frame or
                 self.frames.cell(lineage[-1]).frame)
            )

            # add lineage to self.lineages
            for i in range(len(lineage)):
                lin_ = lineage[i]
                if type(lin_) is tuple:
                    for lin in lin_:
                        if lin in temp_lineage and i < (len(lineage) - 1):
                            parent = temp_lineage[lin][0]
                            temp_lineage[lin] = [parent, lineage[i + 1]]
                        elif lin in temp_lineage and i >= (len(lineage) - 1):
                            parent = temp_lineage[lin][0]
                            temp_lineage[lin] = [parent, None]
                        else:
                            if i == 0 and i < (len(lineage) - 1):
                                temp_lineage[lin] = [None, lineage[i + 1]]
                            elif i == 0 and i >= (len(lineage) - 1):
                                temp_lineage[lin] = [None, None]
                            elif i > 0 and i < (len(lineage) - 1):
                                temp_lineage[lin] = [
                                    lineage[i - 1], lineage[i + 1]
                                ]
                            elif i > 0 and i >= (len(lineage) - 1):
                                temp_lineage[lin] = [lineage[i - 1], None]
                else:
                    lin = lin_
                    if lin in temp_lineage and i < (len(lineage) - 1):
                        parent = temp_lineage[lin][0]
                        temp_lineage[lin] = [parent, lineage[i + 1]]
                    elif lin in temp_lineage and i >= (len(lineage) - 1):
                        parent = temp_lineage[lin][0]
                        temp_lineage[lin] = [parent, None]
                    else:
                        if i == 0 and i < (len(lineage) - 1):
                            temp_lineage[lin] = [None, lineage[i + 1]]
                        elif i == 0 and i >= (len(lineage) - 1):
                            temp_lineage[lin] = [None, None]
                        elif i > 0 and i < (len(lineage) - 1):
                            temp_lineage[lin] = [
                                lineage[i - 1],
                                lineage[i + 1]
                            ]
                        elif i > 0 and i >= (len(lineage) - 1):
                            temp_lineage[lin] = [lineage[i - 1], None]

            if len(lineage[-1]) == 2:
                d1 = self.frames.cell(lineage[-1][0])
                d2 = self.frames.cell(lineage[-1][1])
                progenitors.append(d1)
                progenitors.append(d2)

        self.lineages = temp_lineage
        logging.info("Finished guessing")
        self.write_lineage("lineages.guess.json")
        self.write_lineage(self.LINEAGEFILE)
        logging.info("Reloading cells...")
        self.load_cells()

    def interactive_track(self, p_ref=None):
        """Launch manual lineage tracker.

        Initiates a :class:`LineageMaker` instance, and proceeds through
        the assignment process.

        Updates the following attributes:

        Attributes:
            lineages (dict): As per :func:`load_lineages`

        Once launced, a matplotlib figure will open showing three
        images, N - 1, N, and N + 1 where N is the current frame.

        The parent of the current cell will be outlined in blue
        on the left-hand image, the current cell (the "mother")
        will be displayed on the centre image, and any daughter
        cells will be highlighted in green on the right-hand image.
        All cells defined by MicrobeTracker will be highlighted in
        yellow on the right-hand image if they are not set as
        daughter cells.

        Daughter cells can be selected by clicking on them, selected
        cells will be highlighted in green, and can be toggled by
        clicking again. The next frame can be progressed to by
        pressing `Enter`.

        If no cells are selected, a death event will be initiated,
        which must be confirmed by pressing the `y` key.

        If one cell is selected, the next frame will be progressed
        to without further input.

        If two cells are selected, a division event will be registered
        and the figure will close, and the next "progenitor" cell
        will be displayed in the new figure.

        The window can be maximised by pressing the `m` key, in which
        case the entire field will be displayed. The matplotlib
        zoom and pan tools can be used to narrow down the field.

        Returns:
            None
        """
        final_lineage = {}
        phase = sorted(glob.glob("B/*.tif"))
        if os.path.exists("F"):
            fluor = sorted(glob.glob("F/*.tif"))
        else:
            fluor = None
        progenitors = list(self.frames[0].cells)
        p_idx = 0
        self.accept_all_previews = False
        while len(progenitors) > 0:
#        for progenitor in progenitors:
            progenitor = progenitors.pop()
            p_idx += 1
#            logging.info(
#                "Lineage %i with %i progenitors (%i remaining)",
#                p_idx, len(progenitors), len(progenitors) - p_idx + 1
#            )
            logging.info(
                "Lineage %i (%i remaining)",
                p_idx, len(progenitors) + 1,
            )
            l = LineageMaker(
                progenitor, phase, self.alignment, self.frames, fluor=fluor
            )
            if self.accept_all_previews:
                p_ref = 100000

            self.accept_all_previews = l.start(bool(fluor), p_idx, p_ref)
            lineage = l.lineage
            logging.info("Lineage completed, %i members", len(lineage))
            logging.debug("Lineage members: %r", lineage)
            for i in range(len(lineage)):
                lin_ = lineage[i]
                if type(lin_) is tuple:
                    for lin in lin_:
                        if lin in final_lineage and i < (len(lineage) - 1):
                            parent = final_lineage[lin][0]
                            final_lineage[lin] = [parent, lineage[i + 1]]
                        elif lin in final_lineage and i >= (len(lineage) - 1):
                            parent = final_lineage[lin][0]
                            final_lineage[lin] = [parent, None]
                        else:
                            if i == 0 and i < (len(lineage) - 1):
                                final_lineage[lin] = [None, lineage[i + 1]]
                            elif i == 0 and i >= (len(lineage) - 1):
                                final_lineage[lin] = [None, None]
                            elif i > 0 and i < (len(lineage) - 1):
                                final_lineage[lin] = [
                                    lineage[i - 1], lineage[i + 1]
                                ]
                            elif i > 0 and i >= (len(lineage) - 1):
                                final_lineage[lin] = [lineage[i - 1], None]
                else:
                    lin = lin_
                    if lin in final_lineage and i < (len(lineage) - 1):
                        parent = final_lineage[lin][0]
                        final_lineage[lin] = [parent, lineage[i + 1]]
                    elif lin in final_lineage and i >= (len(lineage) - 1):
                        parent = final_lineage[lin][0]
                        final_lineage[lin] = [parent, None]
                    else:
                        if i == 0 and i < (len(lineage) - 1):
                            final_lineage[lin] = [None, lineage[i + 1]]
                        elif i == 0 and i >= (len(lineage) - 1):
                            final_lineage[lin] = [None, None]
                        elif i > 0 and i < (len(lineage) - 1):
                            final_lineage[lin] = [
                                lineage[i - 1],
                                lineage[i + 1]
                            ]
                        elif i > 0 and i >= (len(lineage) - 1):
                            final_lineage[lin] = [lineage[i - 1], None]

            logging.info(
                "Backing up temporary lineage file to <%s>",
                "lineages.json.tmp"
            )
            open("lineages.json.tmp", "w").write(json.dumps(final_lineage))
            if len(lineage[-1]) == 2:
                daughter1 = self.frames.cell(lineage[-1][0])
                daughter2 = self.frames.cell(lineage[-1][1])
                progenitors.append(daughter1)
                progenitors.append(daughter2)
        self.lineages = final_lineage
#        logging.info("Assigning poles")
#        P = poles.PoleAssign(self.frames)
#        P.assign_poles()
#        logging.info("Writing poles to <poles.json>")
        self.write_lineage()

    def write_lineage(self, fn="lineages.json"):
        """Write lineages to JSON file
        """
        logging.info("Writing lineage file to <%s>", fn)
        open(fn, "w").write(json.dumps(self.lineages))


class TempLineage(object):
    def __init__(self, progenitor_id, progenitor_frame, grandparent):
        self.lineage = [progenitor_id]
        self.start_frame = progenitor_frame
        self.grandparent_frame = progenitor_frame - 1
        self.grandparent = grandparent
        self.end = None
        self.end_frame = None

    def __getitem__(self, idx):
        return self.lineage[idx]

    def append(self, item):
        if not self.end:
            self.lineage += [item]
        else:
            raise ValueError("Lineage has already ended")

    def __iadd__(self, item):
        self.append(item)

    def __len__(self):
        return len(self.lineage)

    def finish(self, end_type, end_frame):
        self.end = end_type
        self.end_frame = end_frame

    def __repr__(self):
        if self.end:
            if self.end == "division":
                final = "[{0}..{1}, {2}..{3}]".format(
                    self.lineage[-1][0][:5], self.lineage[-1][0][-5:],
                    self.lineage[-1][1][:5], self.lineage[-1][1][-5:]
                )
            else:
                final = "{0}..{1}".format(
                    self.lineage[-1][:5], self.lineage[-1][-5:]
                )
            return ("TempLineage({0}..{1} -> {2}; {3} frames ({4}-{5}); "
                    "ending in {6})").format(
                self.lineage[0][:5], self.lineage[0][-5:],
                final,
                len(self.lineage),
                self.start_frame, self.end_frame,
                self.end
            )
        else:
            return ("TempLineage({0}..{1} -> ?; {2} frames so far; "
                    "still in progress)").format(
                self.lineage[0][:5], self.lineage[0][-5:],
                len(self.lineage)
            )


class LineageMaker(object):
    """Assignment of lineages via manual graphical selection.

    Extensively uses :mod:`matplotlib` and user input to
    determine cell lineages.

    Args:
        progenitor (:class:`Cell`): Initial cell to track
        files (list): List of image files sorted by frame number, each
            frame must be in a separate file (for now)
        alignment (numpy.ndarray): As per :func:`Lineage.load_alignment`
            above
        frames (:class:`Frames`): Data for each frame of the image sequence
        fluor (list or None): List of fluorescent image files sorted by frame
            number (optional)

    """
    def __init__(self, progenitor, phase, alignment, frames, fluor=None):
        self.progenitor = progenitor
        self.files_phase = phase
        self.files_fluor = fluor
        self.align = alignment
        if (self.progenitor.frame - 2) >= 0:
            self.n0_frame = frames[self.progenitor.frame - 2]
        else:
            self.n0_frame = EmptyFrame()
        self.n1_frame = frames[self.progenitor.frame - 1]
        try:
            self.n2_frame = frames[self.progenitor.frame]
        except IndexError:
            self.n2_frame = EmptyFrame()
        self.frames = frames
        self.daughters = []
        self.mother = self.progenitor.id
        self.parent = self.progenitor.parent
        self.lineage = [self.mother]
        self.death = False
        self.growth_confirm = False
        if os.path.exists("fluorescence.json"):
            self.fluor_data = json.loads(
                open("fluorescence.json").read()
            )
        else:
            self.fluor_data = {}

    def get_fluor(self, lineage):
        """ Calculates the mean fluorescence within a lineage of cells

        Args:
            lineage (:class:`TempLineage`): lineage of cells

        Return:
            NoneType

        Note:
            Saves the fluorescence data using the :func:`save_fluor` method.
        """
        for l in lineage:
            if type(l) is str:
                cell = self.frames.cell(l)
                fluor_img = scipy.misc.imread(
                    self.files_fluor[
                        cell.frame - 1
                    ]
                )
                offset = self.align[cell.frame - 1]
                c_x, c_y = cell.centre
                x_lower = c_x - 100
                y_lower = c_y - 100

                if y_lower - offset[0] < 0:
                    y_lower = offset[0]
                elif x_lower - offset[1] < 0:
                    x_lower = offset[1]

                width = 200
                y0 = y_lower - offset[0]
#                if y0 < 0:
#                    y0 = 0
#                elif y0 + width >= fluor_img.shape[1]:
#                    y0 = fluor_img.shape[1] - width - 1
                y1 = y0 + width

                x0 = x_lower - offset[1]
#                if x0 < 0:
#                    x0 = 0
#                elif x0 + width >= fluor_img.shape[0]:
#                    x0 = fluor_img.shape[0] - width - 1
                x1 = x0 + width

                fluor_crop = fluor_img[
                    y0:y1, x0:x1
                ]
                xs = np.array([
                    cell.mesh[:, 0] - x_lower,
                    (cell.mesh[:, 2] - x_lower)[::-1]
                ]).flatten()
                ys = np.array([
                    cell.mesh[:, 1] - y_lower,
                    (cell.mesh[:, 3] - y_lower)[::-1]
                ]).flatten()
                rr, cc = skimage.draw.polygon(ys, xs)
                fluor = np.mean(fluor_crop[rr, cc])
                self.fluor_data[l] = fluor
        self.save_fluor()

    def fluor(self, bounds, mesh):
        """ Fluorescent value for a given cell boundary

        Args:
            bounds (tuple): 4-element tuple of XY shift (y0, y1, x0, x1)
            mesh (tuple): 4-element tuple of mesh parameters (xl, yl, xr, yr)

        Note:
            Saves fluorescence values using the :func:`save_fluor` method.
        """
        fluor_img = scipy.misc.imread(
            self.files_fluor[self.frame_idx]
        )
        y0, y1, x0, x1 = bounds
        xl, yl, xr, yr = mesh
        fluor_crop = fluor_img[
            y0:y1, x0:x1
        ]
        xs = np.array([xl, xr[::-1]]).flatten()
        ys = np.array([yl, yr[::-1]]).flatten()
        rr, cc = skimage.draw.polygon(ys, xs)
        fluor = np.mean(fluor_crop[rr, cc])
        self.fluor_data[self.mother] = fluor
        self.save_fluor()

    def save_fluor(self):
        """ Saves fluorescence values from the :attr:`fluor_data` attribute
        as JSON data in the file `fluorescence.json`."""
        J = json.dumps(self.fluor_data)
        open("fluorescence.json", "w").write(J)

    def get_offset(self, cell, offset=None, cell2=None):
        """Return offset, bounds, and shifts for a given cell

        Args:
            cell (:class:`Cell`): Target cell
            offset (sequence): 2-element list of XY offset (optional)
            cell2 (:class:`Cell`): Sibling cell (optional)

        Returns:
            (bounds, shifts, offset)
        Return:
            tuple

        Note:
            bounds (tuple):
                - y_lower (float): Lower Y bound
                - x_lower (float): Lower X bound
            shifts (tuple):
                - xshift (float): X shift
                - yshift (float): Y shift
            offset (sequence):
                - x_offset (float): X offset
                - y_offset (float): Y offset
        """
        width = 200

        if cell2:
            c_x = np.mean([cell.centre[0], cell2.centre[0]])
            c_y = np.mean([cell.centre[1], cell2.centre[1]])
            x_lower = c_x - (width / 2)
            y_lower = c_y - (width / 2)
        else:
            x_lower = cell.centre[0] - (width / 2)
            y_lower = cell.centre[1] - (width / 2)

        xshift = x_lower
        yshift = y_lower

        if not offset:
            offset = self.align[cell.frame - 1]

        if y_lower - offset[0] < 0:
            y_lower = offset[0]
            yshift = offset[0]
        if x_lower - offset[1] < 0:
            x_lower = offset[1]
            xshift = offset[1]

        return (y_lower, x_lower), (xshift, yshift), offset

    def maximise_frame(self, ax, frame, fn_idx, frame_offset):
        """ Maximises a frame

        Args:
            ax (axis): Axes object for frame in question
            frame (:class:`Frame`): The data for the frame
            fn_idx (int): File index for self.files_phase
                (e.g. frame_idx + 1 for n2)
            frame_offset (int): offset for title (e.g. 2 for n2)
        """
        # maximise n2_frame
        if frame.frame is None:
            img = np.zeros((1000, 1000))
            offset = [0, 0]
        else:
            img = scipy.misc.imread(
                self.files_phase[fn_idx]
            )
            offset = self.align[fn_idx]

        ax.clear()
        ax.axis("off")
        ax.imshow(img, cmap=plt.cm.gray)
        ax.set_title("Frame {0}".format(self.frame_idx + frame_offset))

        if frame_offset == 0:
            # draw grandmother (parent)
            parent = frame.cell(self.parent)
            if parent:
                xs_l = parent.mesh[:, 0] - offset[1]
                ys_l = parent.mesh[:, 1] - offset[0]
                xs_r = parent.mesh[:, 2] - offset[1]
                ys_r = parent.mesh[:, 3] - offset[0]
                ax.plot(xs_l, ys_l, "y")
                ax.plot(xs_r, ys_r, "y")

        elif frame_offset == 1:
            # draw mother (mother)
            mother = frame.cell(self.mother)
            xs_l = mother.mesh[:, 0] - offset[1]
            ys_l = mother.mesh[:, 1] - offset[0]
            xs_r = mother.mesh[:, 2] - offset[1]
            ys_r = mother.mesh[:, 3] - offset[0]
            ax.plot(xs_l, ys_l, "r")
            ax.plot(xs_r, ys_r, "r")

        elif frame_offset == 2:
            cells = frame.cells
            for cell in cells:
                xs_l = cell.mesh[:, 0] - offset[1]
                ys_l = cell.mesh[:, 1] - offset[0]
                xs_r = cell.mesh[:, 2] - offset[1]
                ys_r = cell.mesh[:, 3] - offset[0]

                left = np.array([xs_l, ys_l]).T
                right = np.array([xs_r, ys_r]).T
                poly_xy = np.array(
                    list(left) +
                    list(reversed(list(right)))
                )
                poly = matplotlib.patches.Polygon(
                    poly_xy,
                    color="y",
                    picker=True,
                    alpha=0.8,
                )
                poly.set_picker(True)
                poly.gid = cell.id
                poly.selected = False
                if cell.parent == self.mother:
                    poly.set_color("b")
                    poly.selected = True
                ax.add_patch(poly)

    def next_frame(self):
        self.growth_confirm = False
        self.parent = self.mother
        self.mother = self.daughters[0]

        self.daughters = []

        self.frame_idx += 1

        self.n0_frame = self.frames[self.frame_idx - 1]
        self.n1_frame = self.frames[self.frame_idx]
        if self.frame_idx + 1 >= len(self.frames):
            self.n2_frame = EmptyFrame()
        else:
            self.n2_frame = self.frames[self.frame_idx + 1]

        logging.debug("n0_frame: %r", self.n0_frame.frame)
        logging.debug("n1_frame: %r", self.n1_frame.frame)
        logging.debug("n2_frame: %r", self.n2_frame.frame)

        logging.debug("parent: %s", self.parent)
        logging.debug("n0: %r", self.n0_frame.cell(self.parent))
        logging.debug("n1: %r", self.n1_frame.cell(self.parent))
        logging.debug("n2: %r", self.n2_frame.cell(self.parent))

        logging.debug("mother: %s", self.mother)
        logging.debug("n0: %r", self.n0_frame.cell(self.mother))
        logging.debug("n1: %r", self.n1_frame.cell(self.mother))
        logging.debug("n2: %r", self.n2_frame.cell(self.mother))

        width = 200

        grand_mother = self.n0_frame.cell(self.parent)
        bounds, shifts, offset = self.get_offset(grand_mother)
        n0_img = scipy.misc.imread(self.files_phase[self.frame_idx - 1])
        x0, x1 = bounds[0] - offset[0], bounds[0] - offset[0] + width
        y0, y1 = bounds[1] - offset[1], bounds[1] - offset[1] + width
        n0_crop = n0_img[
            x0:x1,
            y0:y1
        ]
        self.n0.clear()
        self.n0.axis("off")
        self.n0.imshow(n0_crop, cmap=plt.cm.gray)
        self.n0.set_title("Frame {0}".format(self.frame_idx))

        self.n0.plot(
            grand_mother.mesh[:, 0] - shifts[0],
            grand_mother.mesh[:, 1] - shifts[1],
            "b"
        )
        self.n0.plot(
            grand_mother.mesh[:, 2] - shifts[0],
            grand_mother.mesh[:, 3] - shifts[1],
            "b"
        )

        mother = self.n1_frame.cell(self.mother)
        bounds, shifts, offset = self.get_offset(mother)
        n1_img = scipy.misc.imread(self.files_phase[self.frame_idx])
        x0, x1 = bounds[0] - offset[0], bounds[0] - offset[0] + width
        y0, y1 = bounds[1] - offset[1], bounds[1] - offset[1] + width
        n1_crop = n1_img[
            x0:x1,
            y0:y1
        ]
        self.n1.clear()
        self.n1.axis("off")
        self.n1.imshow(n1_crop, cmap=plt.cm.gray)
        self.n1.set_title("Frame {0}".format(self.frame_idx + 1))

        xs_l = mother.mesh[:, 0] - shifts[0]
        ys_l = mother.mesh[:, 1] - shifts[1]
        xs_r = mother.mesh[:, 2] - shifts[0]
        ys_r = mother.mesh[:, 3] - shifts[1]

        self.n1.plot(xs_l, ys_l, "r")
        self.n1.plot(xs_r, ys_r, "r")
        if self.files_fluor:
            self.fluor(
                (x0, x1, y0, y1),
                (xs_l, ys_l, xs_r, ys_r)
            )

        if self.n2_frame.frame:
            n2_img = scipy.misc.imread(self.files_phase[self.frame_idx + 1])
            offset = self.align[self.frame_idx + 1]
        else:
            n2_img = np.zeros((1000, 1000))
            offset = [0, 0]

        x0, x1 = bounds[0] - offset[0], bounds[0] - offset[0] + width
        if x0 < 0:
            x1 -= x0
            offset[0] -= x0
            x0 = 0
        y0, y1 = bounds[1] - offset[1], bounds[1] - offset[1] + width
        if y0 < 0:
            y1 -= y0
            offset[1] -= y0
            y0 = 0

        n2_crop = n2_img[
            x0:x1,
            y0:y1
        ]
        self.n2.clear()
        self.n2.axis("off")
        self.n2.imshow(n2_crop, cmap=plt.cm.gray)
        self.n2.set_title("Frame {0}".format(self.frame_idx + 2))

        n2_cells = self.n2_frame.cells
        for n2_cell in n2_cells:
            xs_l = n2_cell.mesh[:, 0] - shifts[0]
            ys_l = n2_cell.mesh[:, 1] - shifts[1]
            xs_r = n2_cell.mesh[:, 2] - shifts[0]
            ys_r = n2_cell.mesh[:, 3] - shifts[1]

            if ((xs_l < 0).sum() > 0 or
                    (xs_l > width).sum() > 0 or
                    (xs_r < 0).sum() > 0 or
                    (xs_r > width).sum() > 0 or
                    (ys_l < 0).sum() > 0 or
                    (ys_l > width).sum() > 0 or
                    (ys_r < 0).sum() > 0 or
                    (ys_r > width).sum() > 0):
                pass
            else:
                left = np.array([xs_l, ys_l]).T
                right = np.array([xs_r, ys_r]).T
                poly_xy = np.array(
                    list(left) +
                    list(reversed(list(right)))
                )
                poly = matplotlib.patches.Polygon(
                    poly_xy,
                    color="y",
                    picker=True,
                    alpha=0.8,
                )
                poly.set_picker(True)
                poly.gid = n2_cell.id
                poly.selected = False
                if n2_cell.parent == self.mother:
                    if len(self.daughters) == 0:
                        poly.set_color("b")
                    else:
                        poly.set_color("c")
                    poly.selected = True
                    self.daughters.append(n2_cell.id)
                self.n2.add_patch(poly)

        info_msg1 = str(self.parent).ljust(36)
        info_msg2 = str(self.mother).ljust(36)
        info_msg3 = (self.daughters and
                     "\n".ljust(49).join(self.daughters)
                     or "None".ljust(36))
        logging.info(
            "Grand-mother: %s (frame: % 2i)",
            info_msg1, self.frame_idx
        )
        logging.info(
            "Mother      : %s (frame: % 2i)",
            info_msg2, self.frame_idx + 1
        )
        logging.info(
            "Daughters   : %s (frame: % 2i)",
            info_msg3, self.frame_idx + 2
        )

        plt.draw()

    def _previewkey(self, e):
        if e.key == "enter":
            # accept lineage and leave
            self.preview_conclusion = "skip"
            plt.close()
        elif e.key == "control":
            self.preview_conclusion = "review"
            plt.close()
        elif e.key == "escape":
            self.preview_conclusion = "accept"
            plt.close()

    def preview(self, p_idx=None, p_ref=None):
        self.preview_conclusion = None
        # run through lineage
        cell = self.progenitor
        logging.info("Previewing lineage from cell %s", cell.id)
        lin = TempLineage(cell.id, cell.frame - 1, cell.parent)
        while True:
            child = cell.children
            if not child:
                if cell.frame == len(self.files_phase):
                    lin.finish("frames", cell.frame - 1)
                else:
                    lin.finish("death", cell.frame - 1)
                break
            elif type(child) is not str:
                # division event
                lin.append(tuple(child))
                lin.finish("division", cell.frame)
                break
            lin.append(child)
            cell = self.frames.cell(child)

        if p_idx and p_ref and p_idx < p_ref:
            if p_ref == 100000:
                return (lin, "accept")
            else:
                return (lin, "skip")

        num_frames = len(lin)
        if lin.end == "death":
            num_frames += 1
        if lin.grandparent:
            num_frames += 1

        if num_frames <= 3:
            cols = num_frames
            rows = 1
        else:
            sqrt = np.sqrt(num_frames)
            if sqrt.is_integer():
                cols = int(sqrt)
                rows = int(sqrt)
            else:
                cols = int(sqrt) + 1
                rows = int(sqrt)

            if cols * rows < num_frames:
                rows += 1

        fig = plt.figure()
        plt.suptitle("Lineage preview. Press ENTER to accept or "
                     "CTRL to re-assign.")

        fig.canvas.mpl_connect(
            "key_press_event", self._previewkey
        )
        i_mod = 0
        if lin.grandparent:
            i_mod += 1

            plt.subplot(rows, cols, 1)

            frame_idx = lin.grandparent_frame
            cell = self.frames.cell(lin.grandparent)

            f = scipy.misc.imread(self.files_phase[frame_idx])

            bounds, shifts, offset = self.get_offset(cell)
            x_lower, y_lower = bounds

            f_crop = f[
                bounds[0] - offset[0]:bounds[0] - offset[0] + 200,
                bounds[1] - offset[1]:bounds[1] - offset[1] + 200
            ]

            plt.imshow(f_crop, cmap=plt.cm.gray)
            plt.axis("off")

            plt.plot(
                cell.mesh[:, 0] - shifts[0],
                cell.mesh[:, 1] - shifts[1],
                "y"
            )
            plt.plot(
                cell.mesh[:, 2] - shifts[0],
                cell.mesh[:, 3] - shifts[1],
                "y"
            )

        for i in range(num_frames):
            try:
                cell_id = lin[i]
            except IndexError:
                break

            plt.subplot(rows, cols, i + 1 + i_mod)
            second = False
            if type(cell_id) is tuple:
                cell_id = lin[i][0]
                second = True
                cell2 = self.frames.cell(lin[i][1])

            cell = self.frames.cell(cell_id)
            plt.title("Frame {0}".format(cell.frame))
            frame_idx = cell.frame - 1
            f = scipy.misc.imread(self.files_phase[frame_idx])

            if second:
                bounds, shifts, offset = self.get_offset(cell, cell2=cell2)
            else:
                bounds, shifts, offset = self.get_offset(cell)

            offset = self.align[frame_idx]
            xshift = shifts[0]
            yshift = shifts[1]

            f_crop = f[
                bounds[0] - offset[0]:bounds[0] - offset[0] + 200,
                bounds[1] - offset[1]:bounds[1] - offset[1] + 200,
            ]

            plt.imshow(f_crop, cmap=plt.cm.gray)
            plt.axis("off")

            if cell_id:
                plt.plot(
                    cell.mesh[:, 0] - xshift,
                    cell.mesh[:, 1] - yshift,
                    "r"
                )
                plt.plot(
                    cell.mesh[:, 2] - xshift,
                    cell.mesh[:, 3] - yshift,
                    "r"
                )
                if second:
                    plt.plot(
                        cell2.mesh[:, 0] - xshift,
                        cell2.mesh[:, 1] - yshift,
                        "c"
                    )
                    plt.plot(
                        cell2.mesh[:, 2] - xshift,
                        cell2.mesh[:, 3] - yshift,
                        "c"
                    )

        if lin.end == "death":
            # add death frame
            death_idx = i + 1 + i_mod
            f = scipy.misc.imread(self.files_phase[lin.end_frame + 1])
            # use last used offset
            f_crop = f[
                bounds[0] - offset[0]:bounds[0] - offset[0] + 200,
                bounds[1] - offset[1]:bounds[1] - offset[1] + 200
            ]
            plt.subplot(rows, cols, death_idx)
            plt.imshow(f_crop, cmap=plt.cm.gray)
            plt.axis("off")

        if plt.get_backend() == "Qt4Agg":
            figm = plt.get_current_fig_manager()
            figm.window.showMaximized()
            plt.show()
        else:
            plt.show()

        if self.preview_conclusion == "skip":
            # assign lineage and move on
            logging.info("Lineage confirmed, skipping to next progenitor cell")
            return (lin, "skip")
        elif self.preview_conclusion == "accept":
            logging.info("All lineages confirmed, skipping to end")
            return (lin, "accept")
        else:
            logging.info("Lineage rejected, moving to review")

    def end(self):
        # display lineage after assignment for confirmation
        # allow restarting if not satifactory

        # input("Write ending code!")
        return

    def start(self, fluor=None, p_idx=None, p_ref=None):
        # preview lineage if exists
        response = self.preview(p_idx, p_ref)
        if response:
            # get fluorescence for members of lineage
            if fluor:
                self.get_fluor(response[0])
            # assign this lineage
            self.lineage = response[0]
            if response[1] == "skip":
                return False
            elif response[1] == "accept":
                return True

        fig = plt.figure()
        self.frame_idx = self.progenitor.frame - 1
        logging.debug("Initial frame_idx: %i", self.frame_idx)

        width = 200

        self.n0 = plt.subplot(131)  # previous frame, i.e. blank
        plt.axis("off")
        if self.frame_idx == 0:
            n0_img = np.zeros((200, 200))
        else:
            n0_img = scipy.misc.imread(self.files_phase[self.frame_idx - 1])

        if self.progenitor.parent:
            grand_mother = self.frames.cell(self.progenitor.parent)
            bounds, shifts, offset = self.get_offset(grand_mother)
            n0_crop = n0_img[
                bounds[0] - offset[0]:bounds[0] - offset[0] + width,
                bounds[1] - offset[1]:bounds[1] - offset[1] + width
            ]
            plt.imshow(n0_crop, cmap=plt.cm.gray)
            plt.plot(
                grand_mother.mesh[:, 0] - shifts[0],
                grand_mother.mesh[:, 1] - shifts[1],
                "y"
            )
            plt.plot(
                grand_mother.mesh[:, 2] - shifts[0],
                grand_mother.mesh[:, 3] - shifts[1],
                "y"
            )
        else:
            plt.imshow(n0_img, cmap=plt.cm.gray)
            plt.title("Frame {0}".format(self.frame_idx))

        self.n1 = plt.subplot(132, sharex=self.n0, sharey=self.n0)  # the frame
        n1_img = scipy.misc.imread(self.files_phase[self.frame_idx])

        bounds, shifts, offset = self.get_offset(self.progenitor)

        x0, x1 = bounds[0] - offset[0], bounds[0] - offset[0] + width
        y0, y1 = bounds[1] - offset[1], bounds[1] - offset[1] + width
        n1_crop = n1_img[
            x0: x1,
            y0: y1
        ]
        plt.imshow(n1_crop, cmap=plt.cm.gray)
        plt.title("Frame {0}".format(self.frame_idx + 1))
        plt.axis("off")

        cell_xs_l = self.progenitor.mesh[:, 0] - shifts[0]
        cell_ys_l = self.progenitor.mesh[:, 1] - shifts[1]
        cell_xs_r = self.progenitor.mesh[:, 2] - shifts[0]
        cell_ys_r = self.progenitor.mesh[:, 3] - shifts[1]

        plt.plot(cell_xs_l, cell_ys_l, "r")
        plt.plot(cell_xs_r, cell_ys_r, "r")

#        self.fluor(
#            (y0, y1, x0, x1),
#            (cell_xs_l, cell_ys_l, cell_xs_r, cell_ys_r)
#        )

        plt.title("Frame {0}".format(self.frame_idx + 1))
        plt.axis("off")

        self.n2 = plt.subplot(133, sharex=self.n0, sharey=self.n0)

        # next frame
        try:
            n2_img = scipy.misc.imread(self.files_phase[self.frame_idx + 1])
            # use alignment to offset
            offset = self.align[self.frame_idx + 1]
        except IndexError:
            n2_img = np.zeros((1000, 1000))
            offset = [0, 0]

        x0, x1 = bounds[0] - offset[0], bounds[0] - offset[0] + width
        if x0 < 0:
            x1 -= x0
            # offset[0] -= x0
            x0 = 0
        y0, y1 = bounds[1] - offset[1], bounds[1] - offset[1] + width
        if y0 < 0:
            y1 -= y0
            # offset[1] -= y0
            y0 = 0

        n2_crop = n2_img[
            x0:x1,
            y0:y1
        ]

        plt.imshow(n2_crop, cmap=plt.cm.gray)
        plt.title("Frame {0}".format(self.frame_idx + 2))
        plt.axis("off")

        # display all cells on n2
        n2_cells = self.n2_frame.cells
        for n2_cell in n2_cells:
            xs_l = n2_cell.mesh[:, 0] - offset[1] - y0
            xs_r = n2_cell.mesh[:, 2] - offset[1] - y0
            ys_l = n2_cell.mesh[:, 1] - offset[0] - x0
            ys_r = n2_cell.mesh[:, 3] - offset[0] - x0

            if ((xs_l < 0).sum() > 0 or
                    (xs_l > width).sum() > 0 or
                    (xs_r < 0).sum() > 0 or
                    (xs_r > width).sum() > 0 or
                    (ys_l < 0).sum() > 0 or
                    (ys_l > width).sum() > 0 or
                    (ys_r < 0).sum() > 0 or
                    (ys_r > width).sum() > 0):
                logging.debug("Pass {0}".format(n2_cell.id))
                pass
            else:
                left = np.array([xs_l, ys_l]).T
                right = np.array([xs_r, ys_r]).T
                poly_xy = np.array(
                    list(left) +
                    list(reversed(list(right)))
                )
                poly = matplotlib.patches.Polygon(
                    poly_xy,
                    color="y",
                    picker=True,
                    alpha=0.8,
                )
                poly.set_picker(True)
                poly.gid = n2_cell.id
                poly.selected = False

                if n2_cell.parent == self.mother:
                    if len(self.daughters) == 0:
                        poly.set_color("b")
                    else:
                        poly.set_color("c")
                    poly.selected = True
                    self.daughters.append(n2_cell.id)

                self.n2.add_patch(poly)

        fig.canvas.mpl_connect(
            "key_press_event", self.plotkey
        )
        fig.canvas.mpl_connect(
            "pick_event", self.plotpick
        )

        info_msg1 = str(self.parent).ljust(36)
        info_msg2 = str(self.mother).ljust(36)
        info_msg3 = (self.daughters and
                     "\n".ljust(49).join(self.daughters)
                     or "None".ljust(36))
        logging.info(
            "Grand-mother: %s (frame: % 2i)",
            info_msg1, self.frame_idx
        )
        logging.info(
            "Mother      : %s (frame: % 2i)",
            info_msg2, self.frame_idx + 1
        )
        logging.info(
            "Daughters   : %s (frame: % 2i)",
            info_msg3, self.frame_idx + 2
        )

        plt.draw()
        if plt.get_backend() == "Qt4Agg":
            figm = plt.get_current_fig_manager()
            figm.window.showMaximized()
            plt.show()
        else:
            plt.show()
        self.end()

    def plotkey(self, e):
        if e.key == "enter":
            # submit lineage change and move frame
            if len(self.daughters) == 0:
                if not self.death:
                    logging.debug("Death event awaiting confirmation")
                    self.death = True
                    self.n2.set_title("Press y to confirm cell death")
                    plt.draw()

            elif len(self.daughters) == 1:
                self.death = False
                test_cell = self.n2_frame.cell(self.daughters[0])
                mother_cell = self.n1_frame.cell(self.mother)
                if mother_cell.length[0][0] > test_cell.length[0][0] * 1.1:
                    self.growth_confirm = True
                    self.n2.set_title("Press y to confirm growth")
                    plt.draw()
                else:
                    logging.info("Growth event")
                    self.lineage.append(self.daughters[0])
                    # move to next frame
                    self.next_frame()

            elif len(self.daughters) == 2:
                self.death = False
                logging.info("Division event")
                self.lineage.append(
                    (self.daughters[0],
                     self.daughters[1])
                )
                plt.close()

            else:
                self.death = False
                logging.warning("Too many daughters")

        elif e.key == "escape" and self.death:
            self.death = False
            self.n2.set_title("Death averted")
            plt.draw()

        elif e.key == "escape" and self.growth_confirm:
            self.growth_confirm = False
            self.n2.set_title("Growth averted")
            plt.draw()

        elif e.key == "y" and self.death and len(self.daughters) == 0:
            logging.info("Death event confirmed")
            plt.close()

        elif e.key == "y" and self.growth_confirm and len(self.daughters) == 1:
            logging.info("Growth event confirmed")
            self.lineage.append(self.daughters[0])
            self.next_frame()

        elif e.key == "m":
            logging.info("Maximising frames")
            self.maximise_frame(
                self.n0, self.n0_frame,
                (self.frame_idx - 1), 0,
            )
            self.maximise_frame(
                self.n1, self.n1_frame,
                self.frame_idx, 1,
            )
            self.maximise_frame(
                self.n2, self.n2_frame,
                (self.frame_idx + 1), 2
            )
            plt.draw()

    def plotpick(self, e):
        if not e.artist.selected:
            self.daughters.append(e.artist.gid)
            e.artist.selected = True
            if len(self.daughters) == 1:
                e.artist.set_color("b")
            elif len(self.daughters) == 2:
                e.artist.set_color("c")
            else:
                e.artist.set_color("r")
        else:
            try:
                self.daughters.remove(e.artist.gid)
            except ValueError:
                # daughter not in list, do nothing
                pass
            e.artist.selected = False
            e.artist.set_color("y")
        plt.draw()


if __name__ == "__main__":
#    logging.basicConfig(
#        format="[%(asctime)s] %(levelname) 10s: %(message)s",
#        datefmt="%Y-%m-%d %H:%M:%S",
#        level=logging.DEBUG
#    )
#    l = Lineage()
    print("You should probably import this as a module")
