#!/usr/bin/env python

import numpy as np
from lineage_lib import track
import json
import os


class PoleAssign(object):
    def __init__(self, lineage):
        self.lineage = lineage

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

        mother_cell = self.lineage.frames.cell(mother)
        daughter_cell = self.lineage.frames.cell(daughter)
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

    def assign_poles(self, force=False):
        if os.path.exists("poles.json") and not force:
            return json.loads(open("poles.json").read())

        assignments = {}
        # iterate through cell lineages
        progenitors = self.lineage.frames[0].cells
        for progenitor in progenitors:
            # assign a pole to the first member of the lineage only
            # orientation parameter will cover rest in spot_analysis routines
            pole_assignment = self.assign_new_pole(progenitor.parent, progenitor.id)
            assignments[progenitor.id] = pole_assignment
            while type(progenitor.children) is str:
                progenitor = self.lineage.frames.cell(progenitor.children)
            if progenitor.children:
                progenitors.append(self.lineage.frames.cell(progenitor.children[0]))
                progenitors.append(self.lineage.frames.cell(progenitor.children[1]))

        open("poles.json", "w").write(json.dumps(assignments))


def main():
    lineage = track.Lineage()
    p = PoleAssign(lineage)
    p.assign_poles()


if __name__ == "__main__":
    main()
