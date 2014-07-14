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
import uuid
# import glob
import os
import json
import logging
import glob

__version__ = "0.1"
__author__ = "Miles Priestman <priestman.miles@gmail.com>"


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
    def __init__(self, cell):
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
        Attributes are set by the methods :func:`load_alignment` and
        :func:`load_cells` which are called on init. See below for
        descriptions of these attributes
    """
    def __init__(self, cellmat="mt/mt.mat", uuidfile="uuids.json",
                 lineagefile="lineages.json", alignmat="mt/alignment.mat"):
        self.load_alignment(alignmat)
        self.load_cells(cellmat, uuidfile, lineagefile)

    def load_alignment(self, alignmat="mt/alignment.mat"):
        """Loads alignment files as created by MicrobeTracker

        Note:
            This will be improved in the future to refine these
            alignments.

        Args:
            alignmat (str): path to MicrobeTracker alignment file,
                defaults to `mt/alignment.mat`

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
        if not os.path.exists(alignmat):
            raise IOError("Alignment file not found ({0})".format(alignmat))

        alignment = scipy.io.loadmat(alignmat)
        logging.info(">>> loaded")
        alignment = alignment["shiftframes"]
        alignment_x = alignment[0][0][0][0]
        alignment_y = alignment[0][0][1][0]
        alignment = np.array([alignment_x, alignment_y]).T
        logging.info(">>> parsed")
        self.alignment = alignment

    def load_cells(self, cellmat="mt/mt.mat", uuidfile="uuids.json",
                   lineagefile="lineages.json"):
        """Loads meshes created my MicrobeTracker, and any lineages previously
        created by Lineage

        Args:
            cellmat (str): path to MicrobeTracker meshes file, defaults to
                `mt/mt.mat`
            uuidfile (str): path to uuid file created by Lineage, defaults to
                `uuids.json`
            lineagefile (str): path to lineages file created by Lineage,
                defaults to `lineages.json`

        Sets the following attributes:

        Attributes:
            frames (:class:`Frames`): Contains all data from MicrobeTracker
            uuids (dict): References all cell UUIDs according to
                MicrobeTracker's cell numbering system

                Cells are referenced with the following key structure:
                    `frame`:`mt_idx`
                Where `frame` is the frame number of the cell, and `mt_idx`
                is the cell number assigned by MicrobeTracker

                Warning:
                    If cell numbers are changed by subsequent runs of
                    MicrobeTracker, these UUIDs will no longer be valid.

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

        Returns:
            None

        Raises:
            IOError: if meshes file is not found
            SysError: if there is a problem with the cell data
        """
        if not os.path.exists(cellmat):
            raise IOError(
                "Meshes file couldn't be found ({0})".format(cellmat)
            )

        logging.info("Loading cell data...")
        cells = scipy.io.loadmat(cellmat)
        logging.info(">>> loaded")
        cellList = cells["cellList"][0]

        cell_total = 0
        pos_idx = 0
        frames = []

        logging.info("Loading UUIDs...")
        if os.path.exists(uuidfile):
            self.uuids = json.loads(open(uuidfile).read())
        else:
            self.uuids = {}
            # "frame:mt_idx": uuid

        if os.path.exists(lineagefile):
            self.lineages = json.loads(open(lineagefile).read())
        else:
            self.lineages = {}
            # uuid: parent, child

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

                        cell.centre = self._centre(cell)

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
        open("uuids.json", "w").write(json.dumps(self.uuids))
        logging.info(">>> saved")

        self.frames = Frames(frames)

    def _centre(self, cell):
        """Return the coordinates for cell centre of the given cell.

        Args:
            cell (:class:`Cell`): Cell in question.

        Returns:
            X, Y (tuple, float): X and Y coordinates for cell centre
        """
        xl = cell.mesh[:, 0]
        yl = cell.mesh[:, 1]
        xr = cell.mesh[:, 2]
        yr = cell.mesh[:, 3]
        c_x = np.mean([xl.mean(), xr.mean()])
        c_y = np.mean([yl.mean(), yr.mean()])
        return c_x, c_y

    def guess_lineage(self):
        """Naive guess at lineages using XY positions and cell length only.

        Updates the following attributes:

        Attributes:
            lineages (dict): As per :func:`load_cells`

        Returns:
            None

        Similarly to :func:`interactive_track`, start from all "progenitor"
        cells in frame 1. For each of these cells, select a cell in the next
        frame which is spatially closest, if the closest cell is significantly
        smaller in length, then choose the two closest cells as daughter
        cells.

        Note:
            Improvements planned:

            - ensuring daughters have similar cell
              length (e.g. within 20% tolerance?)
            - using cell morphology in some clever way to determine
              division events
        """
        temp_lineage = {}
        progenitors = self.frames[0].cells
        for progenitor in progenitors:
            # use mesh to determine bounding circle
            # this is a circle centred on the cell centre, with a diameter
            # of the cell length

            # cell centre
            c_x, c_y = progenitor.centre
            # radius
            radius = progenitor.length[0][0] / 2

            n1_frame = self.frames[progenitor.frame + 1]
            # search n1 frame for cells > 60% within bounding circle
            for pot in n1_frame.cells:
                pot_radius = pot.length[0][0] / 2
                pot_x, pot_y = pot.centre

                print(
                    "c_x:", c_x,
                    "c_y:", c_y,
                    "r:", radius,
                )
                print(
                    "p_x:", pot_x,
                    "p_y:", pot_y,
                    "p_r:", pot_radius
                )

                # distance between cell centres
                dist = np.sqrt(
                    ((c_x - pot_x) ** 2) +
                    ((c_y - pot_y) ** 2)
                )
                print("dist:", dist)

                if dist > 50:
                    continue

                # half angle of the top intersection
                ang1 = np.arccos(
                    ((pot_radius ** 2) +
                     (dist ** 2) -
                     (radius ** 2)) /
                    (2 * pot_radius * dist)
                )
                print("ang1:", ang1)

                ang2 = np.arccos(
                    ((radius ** 2) +
                     (dist ** 2) -
                     (pot_radius ** 2)) /
                    (2 * radius * dist)
                )
                print("ang2:", ang2)

                # intersection area
                a = (ang1 * (pot_radius ** 2) -
                     0.5 * (pot_radius ** 2) * np.sin(2 * ang1) +
                     ang2 * (radius ** 2) -
                     0.5 * (radius ** 2) * np.sin(2 * ang2))

                print(a)
                break

            break

    def interactive_track(self):
        """Launch manual lineage tracker.

        Initiates a :class:`LineageMaker` instance, and proceeds through
        the assignment process.

        Updates the following attributes:

        Attributes:
            lineages (dict): As per :func:`load_cells`

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
        files = sorted(glob.glob("*.tif"))
        progenitors = self.frames[0].cells
        for progenitor in progenitors:
            l = LineageMaker(
                progenitor, files, self.alignment, self.frames
            )
            l.start()
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
        logging.info("Writing lineage file to <lineages.json>")
        open("lineages.json", "w").write(json.dumps(final_lineage))


class LineageMaker(object):
    """Assignment of lineages via manual graphical selection.

    Extensively uses :mod:`matplotlib` and user input to
    determine cell lineages.

    Args:
        progenitor (:class:`Cell`): Initial cell to track
        files (list): list of image files sorted by frame number, each
            frame must be in a separate file (for now)
        alignment (numpy.ndarray): As per :func:`Lineage.load_alignment`
            above

    """
    def __init__(self, progenitor, files, alignment, frames):
        self.progenitor = progenitor
        self.files = files
        self.align = alignment
        if (self.progenitor.frame - 2) >= 0:
            self.n0_frame = frames[self.progenitor.frame - 2]
        else:
            self.n0_frame = EmptyFrame()
        self.n1_frame = frames[self.progenitor.frame - 1]
        self.n2_frame = frames[self.progenitor.frame]
        self.frames = frames
        self.daughters = []
        self.mother = self.progenitor.id
        self.parent = self.progenitor.parent
        self.lineage = [self.mother]
        self.death = False

    def maximise_frame(self, ax, frame, fn_idx, frame_offset):
        """

        Args:
            ax (axis): Axes object for frame in question
            frame (:class:`Frame`): The data for the frame
            fn_idx (int): File index for self.files
                (e.g. frame_idx + 1 for n2)
            frame_offset (int): offset for title (e.g. 2 for n2)
        """
        # maximise n2_frame
        if frame.frame is None:
            img = np.zeros((1024, 1344))
        else:
            img = scipy.misc.imread(
                self.files[fn_idx]
            )

        ax.clear()
        ax.axis("off")
        ax.imshow(img, cmap=plt.cm.gray)
        ax.set_title("Frame {0}".format(self.frame_idx + frame_offset))

        offset = self.align[fn_idx]
        if frame_offset == 0:
            # draw grandmother (parent)
            parent = frame.cell(self.parent)
            if parent:
                xs_l = parent.mesh[:, 0] - offset[1]
                ys_l = parent.mesh[:, 1] - offset[0]
                xs_r = parent.mesh[:, 2] - offset[1]
                ys_r = parent.mesh[:, 3] - offset[0]
                ax.plot(xs_l, ys_l, "b")
                ax.plot(xs_r, ys_r, "b")

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
                    poly.set_color("g")
                    poly.selected = True
                ax.add_patch(poly)

    def next_frame(self):
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

        c_x, c_y = self.get_centre(
            self.n1_frame.cell(self.mother).mesh
        )
        x_lower = c_x - 100
        y_lower = c_y - 100
        width = 200

#        x_upper, y_upper, width, height = self.n1_frame.cell(
#            self.mother
#        ).box[0]
#        x_lower = int(x_upper - width)
#        y_lower = int(y_upper - height)
#        max_dim = max(width, height)
#        dim = max_dim * 3

        n0_img = scipy.misc.imread(self.files[self.frame_idx - 1])
        offset = self.align[self.frame_idx - 1]
        n0_crop = n0_img[
            y_lower - offset[0]:y_lower - offset[0] + width,
            x_lower - offset[1]:x_lower - offset[1] + width
        ]

        self.n0.clear()
        self.n0.axis("off")
        self.n0.imshow(n0_crop, cmap=plt.cm.gray)
        self.n0.set_title("Frame {0}".format(self.frame_idx))

        parent = self.n0_frame.cell(self.parent)
        xs_l = parent.mesh[:, 0] - x_lower
        ys_l = parent.mesh[:, 1] - y_lower
        xs_r = parent.mesh[:, 2] - x_lower
        ys_r = parent.mesh[:, 3] - y_lower
        self.n0.plot(xs_l, ys_l, "b")
        self.n0.plot(xs_r, ys_r, "b")

        n1_img = scipy.misc.imread(self.files[self.frame_idx])
        offset = self.align[self.frame_idx]
        n1_crop = n1_img[
            y_lower - offset[0]:y_lower - offset[0] + width,
            x_lower - offset[1]:x_lower - offset[1] + width
        ]
        self.n1.clear()
        self.n1.axis("off")
        self.n1.imshow(n1_crop, cmap=plt.cm.gray)
        self.n1.set_title("Frame {0}".format(self.frame_idx + 1))

        mother = self.n1_frame.cell(self.mother)
        xs_l = mother.mesh[:, 0] - x_lower
        ys_l = mother.mesh[:, 1] - y_lower
        xs_r = mother.mesh[:, 2] - x_lower
        ys_r = mother.mesh[:, 3] - y_lower
        self.n1.plot(xs_l, ys_l, "r")
        self.n1.plot(xs_r, ys_r, "r")

        if self.n2_frame.frame:
            n2_img = scipy.misc.imread(self.files[self.frame_idx + 1])
            offset = self.align[self.frame_idx + 1]
        else:
            n2_img = np.zeros((1024, 1344))
            offset = [0, 0]

        if y_lower - offset[0] < 0:
            offset = [0, 0]
        elif x_lower - offset[1] < 0:
            offset = [0, 0]

        n2_crop = n2_img[
            y_lower - offset[0]:y_lower - offset[0] + width,
            x_lower - offset[1]:x_lower - offset[1] + width
        ]

        self.n2.clear()
        self.n2.axis("off")
        self.n2.imshow(n2_crop, cmap=plt.cm.gray)
        self.n2.set_title("Frame {0}".format(self.frame_idx + 2))

        n2_cells = self.n2_frame.cells
        for n2_cell in n2_cells:
            xs_l = n2_cell.mesh[:, 0] - x_lower
            ys_l = n2_cell.mesh[:, 1] - y_lower
            xs_r = n2_cell.mesh[:, 2] - x_lower
            ys_r = n2_cell.mesh[:, 3] - y_lower

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
                    poly.set_color("g")
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

    def get_centre(self, mesh):
        xl = mesh[:, 0]
        yl = mesh[:, 1]
        xr = mesh[:, 2]
        yr = mesh[:, 3]

        c_x = np.mean([xl.mean(), xr.mean()])
        c_y = np.mean([yl.mean(), yr.mean()])

        return c_x, c_y

    def preview(self):
        # run through lineage
        pass

    def start(self):
        fig = plt.figure()
        self.frame_idx = self.progenitor.frame - 1
        logging.debug("Initial frame_idx: %i", self.frame_idx)

        c_x, c_y = self.get_centre(self.progenitor.mesh)
        x_lower = c_x - 100
        y_lower = c_y - 100
        width = 200

        self.n0 = plt.subplot(131)  # previous frame, i.e. blank
        plt.axis("off")
        if self.frame_idx == 0:
            n0_img = np.zeros((1024, 1344))
            offset = [0, 0]
        else:
            n0_img = scipy.misc.imread(self.files[self.frame_idx - 1])
            offset = self.align[self.frame_idx - 1]
        n0_crop = n0_img[
            y_lower - offset[0]:y_lower - offset[0] + width,
            x_lower - offset[1]:x_lower - offset[1] + width
        ]
        plt.imshow(n0_crop, cmap=plt.cm.gray)
        plt.title("Frame {0}".format(self.frame_idx))

        self.n1 = plt.subplot(132, sharex=self.n0, sharey=self.n0)  # the frame
        n1_img = scipy.misc.imread(self.files[self.frame_idx])
        offset = self.align[self.frame_idx]
        n1_crop = n1_img[
            y_lower - offset[0]:y_lower - offset[0] + width,
            x_lower - offset[1]:x_lower - offset[1] + width
        ]
        plt.imshow(n1_crop, cmap=plt.cm.gray)
        plt.title("Frame {0}".format(self.frame_idx + 1))
        plt.axis("off")

        cell_xs_l = self.progenitor.mesh[:, 0] - x_lower
        cell_ys_l = self.progenitor.mesh[:, 1] - y_lower
        cell_xs_r = self.progenitor.mesh[:, 2] - x_lower
        cell_ys_r = self.progenitor.mesh[:, 3] - y_lower
        plt.plot(cell_xs_l, cell_ys_l, "r")
        plt.plot(cell_xs_r, cell_ys_r, "r")

        plt.title("Frame {0}".format(self.frame_idx + 1))
        plt.axis("off")

        self.n2 = plt.subplot(133, sharex=self.n0, sharey=self.n0)
        # next frame
        n2_img = scipy.misc.imread(self.files[self.frame_idx + 1])
        # use alignment to offset
        offset = self.align[self.frame_idx + 1]
        n2_crop = n2_img[
            y_lower - offset[0]:y_lower - offset[0] + width,
            x_lower - offset[1]:x_lower - offset[1] + width
        ]
        plt.imshow(n2_crop, cmap=plt.cm.gray)
        plt.title("Frame {0}".format(self.frame_idx + 2))
        plt.axis("off")

        # display all cells on n2
        n2_cells = self.n2_frame.cells
        for n2_cell in n2_cells:
            xs_l = n2_cell.mesh[:, 0] - x_lower
            xs_r = n2_cell.mesh[:, 2] - x_lower

            ys_l = n2_cell.mesh[:, 1] - y_lower
            ys_r = n2_cell.mesh[:, 3] - y_lower

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
                    poly.set_color("g")
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
        plt.show()

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

        elif e.key == "escape":
            self.death = False
            self.n2.set_title("Death averted")
            plt.draw()

        elif e.key == "y" and self.death and len(self.daughters) == 0:
            logging.info("Death event confirmed")
            plt.close()

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
            e.artist.set_color("g")
        else:
            self.daughters.remove(e.artist.gid)
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
