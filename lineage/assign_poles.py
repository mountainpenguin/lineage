#!/usr/bin/env python

from lineage_lib import track
from lineage_lib import poles


def main():
    lineage = track.Lineage()
    P = poles.PoleAssign(lineage.frames)
    P.assign_poles()

if __name__ == "__main__":
    main()
