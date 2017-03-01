#!/usr/bin/env python

import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np
import textwrap
import sys
import argparse

#import ete3
# fix for ete3 Qt4/Qt5 stuff
#import matplotlib
#matplotlib.use("Qt4Agg")

from lineage_lib import track
from lineage_lib import misc

class CustomNode(object):
    def __init__(self, lineage):
        self.lineage_id = lineage.lineage_id
        self.frames = [x.frame for x in lineage.cells]
        self.times = [x.t for x in lineage.cells]
        self.lengths = [x.length for x in lineage.cells]
        self.initial_length = lineage.cells[0].length
        self.final_length = lineage.cells[-1].length
        if not lineage.children:
            self.loss = True
            self.interdivision_time = None
            self.elongation_rate = None
            self.growth_rate = None
        else:
            self.loss = False
            self.interdivision_time = (lineage.cells[-1].t - lineage.cells[0].t) / 60
            self.elongation_rate = get_elongation_rate(self.times, self.lengths)
            self.growth_rate = get_growth_rate(self.times, self.lengths)

    def __repr__(self):
        return "<{0}>".format(self.lineage_id)

    def describe(self):
        print(textwrap.dedent(
        """
            Lineage ID: {lineage_id}
            Number of frames: {num_frames} ({first_frame} - {last_frame})
            Lengths (um): {lengths}
            Interdivision time (h): {interdivision_time}
            Elongation rate (um/h): {elongation_rate}
            Growth rate (h^-1): {growth_rate}
            Lineage ends in {end_type}
        """.format(
            num_frames=len(self.lengths),
            first_frame=self.frames[0],
            last_frame=self.frames[-1],
            end_type=self.loss and "loss" or "division",
            **self.__dict__
        )).strip())


NODES = {}

def get_custom_node(lin):
    if lin.lineage_id in NODES:
        return NODES[lin.lineage_id]
    else:
        n = CustomNode(lin)
        NODES[lin.lineage_id] = n
        return n

def get_node_dict(G):
    nodes = []
    for x in G.nodes():
        nodes.append((x.lineage_id, x))
    return dict(nodes)

def get_growth_rate(times, lengths):
    logL = np.log(lengths)
    lamb, logL0 = np.polyfit(np.array(times) - times[0], logL, 1)
    return lamb * 60  # hr^{-1}

def get_elongation_rate(times, lengths):
    gradient = np.polyfit(times, lengths, 1)[0]  # um / min
    return gradient * 60  # um / hr

def add_children(node, cell):
    if cell.children:
        n0 = get_custom_node(cell)
        n1 = get_custom_node(cell.children[0])
        n2 = get_custom_node(cell.children[1])

        node.add_node(n1)
        node.add_edge(n0, n1)
        add_children(node, cell.children[0])

        node.add_node(n2)
        node.add_edge(n0, n2)
        add_children(node, cell.children[1])

def _check_dir(path):
    if not os.path.exists(os.path.join(path, "timings.json")):
        return False
    return True

def intercolate(prev_L, new_L):
    if not prev_L:
        return new_L

    frame_idx = 0
    for new_frame in new_L.frames:
        try:
            prev_frame = prev_L.frames[frame_idx]
        except IndexError:
            prev_L.frames.frames.append(new_frame)
            for c in new_frame.cells:
                prev_L.frames._idx[c.id] = c.frame - 1
        else:
            for c in new_frame.cells:
                # add cell from new_frame into prev_frame
                prev_frame.cells.append(c)
                # update prev_frame index
                prev_frame._idx[c.id] = c
                # update prev_frames index
                prev_L.frames._idx[c.id] = c.frame - 1

        frame_idx += 1

    return prev_L


def process_path(path):
    orig_dir = os.getcwd()
    os.chdir(path)
    timings, rif_add, px = misc.get_timings()
    L = track.Lineage()
    initial_cells = L.frames[0].cells
    i = 1
    lineages = []
    graphs = []
    for init_cell in initial_cells:
        print("Handling cell lineage {0} ({1} of {2})".format(
            init_cell.id, i, len(initial_cells),
        ))
        tree = nx.DiGraph()
        cell_lineage = track.SingleCellLineage(
            init_cell.id,
            L,
            px_conversion=px,
            timings=timings,
            rif_cut=rif_add,
        )
        lineages.append(cell_lineage)
        tree.add_node(get_custom_node(cell_lineage))
        add_children(tree, cell_lineage)

        if not os.path.exists("networks"):
            os.mkdir("networks")

        nx.drawing.nx_agraph.write_dot(
            tree, "networks/network-{0}.dot".format(init_cell.id)
        )
        # nx.write_gpickle(tree, "networks/network-{0}.pickle".format(init_cell.id))
        pos = nx.drawing.nx_agraph.graphviz_layout(tree, prog="dot")
        fig = plt.figure()
        nx.draw(tree, pos, arrows=False, with_labels=False)
        fig.savefig("networks/network-{0}.pdf".format(init_cell.id))
        fig.clear()

        graphs.append(tree)

        i += 1
    os.chdir(orig_dir)
    return L, lineages, graphs


def main():
    parser = argparse.ArgumentParser(
        description="Script to handle single cell data hopefully usefully."
    )
    parser.add_argument(
        "process_list", metavar="directory", type=str, nargs="*",
        help="""
            Specify directories to process.
            Set to "*" to process all directories in the current working directory.
            Omit to process only the current directory.
        """
    )
    args = parser.parse_args()
    if not args.process_list:
        dirlist = ["."]
    elif args.process_list == ["*"]:
        dirlist = os.listdir(".")
    else:
        dirlist = []
        for d in args.process_list:
            if os.path.exists("{0}/timings.json".format(d)):
                dirlist.append(d)
            else:
                print("Directory <{0}> does not exist or isn't correctly arranged, skipping".format(d))

    dirlist = list(filter(_check_dir, dirlist))

    L = None
    lineages = []
    graphs = []
    i = 1
    for path in dirlist:
        print("Processing directory <{0}> ({1} of {2})".format(
            path, i, len(dirlist),
        ))
        this_L, this_lineages, this_graphs = process_path(path)
        L = intercolate(L, this_L)
        lineages.extend(this_lineages)
        graphs.extend(this_graphs)
        i += 1

    launch_confirm = input("Start interactive console? (y/N): ")
    if launch_confirm == "y":
        import code
        variables = {
            "plt": plt,
            "nx": nx,
            "np": np,
            "L": L,
            "lineages": lineages,
            "graphs": graphs,
            "get_node_dict": get_node_dict,
        }
        code.interact(local=variables)

if __name__ == "__main__":
    main()
