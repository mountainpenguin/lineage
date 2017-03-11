#!/usr/bin/env python


import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
plt.rc("font", family="sans-serif")
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
    r"\setlength\parindent{0pt}",
]
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")
sns.set_palette("colorblind")

# lineage library imports
from lineage_lib import track
from lineage_lib import misc


class StatsObject(object):
    def __init__(self, r=None, rp=None, m=None, c=None, n=None):
        self.pearson_r = r
        self.pearson_r2 = r ** 2
        self.pearson_rsq = r ** 2
        self.pearson_pvalue = rp
        self.slope = m
        self.intercept = c
        self.n = n

    def fmt_pearson(self):
        return r"$r=$ {0:.3f}, $r^2=$ {1:.3f} ($n=$ {2})".format(
            self.pearson_r,
            self.pearson_r2,
            self.n,
        )

    def fmt_lin_reg(self):
        return r"$m=$ {0:.3f}, $c=$ {1:.3f} ($n=$ {2})".format(
            self.slope,
            self.intercept,
            self.n,
        )

def get_cell_stats(parent, child1, child2, timings, pixel):
    p1 = parent.cells[0]
    p2 = parent.cells[-1]
    n1 = child1.cells[0]
    n2 = child1.cells[-1]
    o1 = child2.cells[0]
    o2 = child2.cells[-1]

    p_data = np.array([(timings[x.frame - 1] / 60, x.length[0][0] * pixel)
                       for x in parent.cells])
    n_data = np.array([(timings[x.frame - 1] / 60, x.length[0][0] * pixel)
                       for x in child1.cells])
    o_data = np.array([(timings[x.frame - 1] / 60, x.length[0][0] * pixel)
                       for x in child2.cells])

    if min([len(p_data), len(n_data), len(o_data)]) < 3:
        return

#    if max([p_data[-1, 0] - p_data[0, 0], n_data[-1, 0] - n_data[0, 0], o_data[-1, 0] - o_data[0, 0]]) > 6:
#        return

    returnable = pd.Series({
        "parent_id": parent.lineage_id,
        "new_id": child1.lineage_id,
        "old_id": child2.lineage_id,
        "source_dir": os.getcwd(),

        "p_li": p_data[0, 1],
        "p_lf": p_data[-1, 1],
        "p_al": p_data[-1, 1] - p_data[0, 1],
        "p_it": p_data[-1, 0] - p_data[0, 0],
        "p_er": np.polyfit(p_data[:, 0], p_data[:, 1], 1)[0],
        "p_gr": np.polyfit(p_data[:, 0] - p_data[0, 0], np.log(p_data[:, 1]), 1)[0],

        "cn_li": n_data[0, 1],
        "cn_lf": n_data[-1, 1],
        "cn_al": n_data[-1, 1] - n_data[0, 1],
        "cn_it": n_data[-1, 0] - n_data[0, 0],
        "cn_er": np.polyfit(n_data[:, 0], n_data[:, 1], 1)[0],
        "cn_gr": np.polyfit(n_data[:, 0] - n_data[0, 0], np.log(n_data[:, 1]), 1)[0],

        "co_li": o_data[0, 1],
        "co_lf": o_data[-1, 1],
        "co_al": o_data[-1, 1] - o_data[0, 1],
        "co_it": o_data[-1, 0] - o_data[0, 0],
        "co_er": np.polyfit(o_data[:, 0], o_data[:, 1], 1)[0],
        "co_gr": np.polyfit(o_data[:, 0] - o_data[0, 0], np.log(o_data[:, 1]), 1)[0],
    })

    if returnable["cn_er"] < 0.0 or returnable["co_er"] < 0.0:
        return

    return returnable


def order_children(child1, child2):
    c1m = max(child1.pole1_age, child1.pole2_age)
    c2m = max(child2.pole1_age, child2.pole2_age)
    if c1m > c2m:
        return child2, child1
    else:
        return child1, child2

def process_lineage(cell_lineage, timings, rif_add, pixel):
    lineage_id = cell_lineage.lineage_id
    data = pd.DataFrame(columns=[
        "parent_id", "new_id", "old_id", "source_dir",
        "p_li", "cn_li", "co_li",  # initial length: parent, child new, child old, both
        "p_lf", "cn_lf", "co_lf",  # final length
        "p_al", "cn_al", "co_al",  # added length
        "p_it", "cn_it", "co_it",  # interdivision time
        "p_er", "cn_er", "co_er",  # elongation rate
        "p_gr", "cn_gr", "co_gr",  # growth rate
    ])

    if not cell_lineage.children:
        return data

    elif not cell_lineage.parent_lineage:
        data = data.append(process_lineage(
            cell_lineage.children[0],
            timings,
            rif_add,
            pixel,
        ), ignore_index=True)
        data = data.append(process_lineage(
            cell_lineage.children[1],
            timings,
            rif_add,
            pixel,
        ), ignore_index=True)
    elif cell_lineage.parent_lineage and cell_lineage.children:
        d = get_cell_stats(
            cell_lineage,
            *order_children(*cell_lineage.children),
            timings,
            pixel,
        )
        if d is not None:
            data = data.append(d, ignore_index=True)

    return data

def process_dir():
    L = track.Lineage()
    timings, rif_add, pixel = misc.get_timings()
    dir_data = None
    for cell_lineage in L.frames[0].cells:
        cell_lineage = track.SingleCellLineage(cell_lineage.id, L)
        lineage_data = process_lineage(cell_lineage, timings, rif_add, pixel)
        if dir_data is not None:
            dir_data  = dir_data.append(lineage_data, ignore_index=True)
        else:
            dir_data = lineage_data
    return dir_data


def get_stats(data1, data2):
    r, p = scipy.stats.pearsonr(data1, data2)
    return StatsObject(r=r, rp=p, n=len(data1))


def process_sources(dir_sources, dirs=None, force=False):
    if not dirs:
        dirs = "."

    if (os.path.exists("correlation_data.pandas") and force) or not os.path.exists("correlation_data.pandas"):
        orig_dir = os.getcwd()
        i = 0
        data = None
        for d in dirs:
            os.chdir(d)
            print("Processing {0}".format(d))
            if os.path.exists("mt/alignment.mat"):
                dir_data = process_dir()
                if data is not None:
                    data = data.append(dir_data, ignore_index=True)
                else:
                    data = dir_data
                print("Got {0} cells".format(len(dir_data)))
            else:
                print("Skipping, missing critical files")
            i += 1
            os.chdir(orig_dir)
        data.to_pickle("correlation_data.pandas")
    else:
        data = pd.read_pickle("correlation_data.pandas")

    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig2 = plt.figure(figsize=(8.27, 11.69))
    random_ax = fig2.add_subplot(111)
    random_ax.axis("off")

    i = 1
    parameters = [
        ("li", r"initial length (\si{\micro\metre})"),
        ("lf", r"final length (\si{\micro\metre})"),
        ("al", r"added length (\si{\micro\metre})"),
        ("it", r"interdivision time (\si{\hour})"),
        ("er", r"linear elongation rate (\si{\micro\metre\per\hour})"),
        ("gr", r"exponential elongation rate (\si{\per\hour})"),
    ]

    for param, label in parameters:
        p_data = data["p_{0}".format(param)]
        cn_data = data["cn_{0}".format(param)]
        co_data = data["co_{0}".format(param)]
        p2_data = p_data.append(p_data, ignore_index=True)
        c2_data = cn_data.append(co_data, ignore_index=True)

        ax = fig.add_subplot(3, 2, i)
        ax.axis("equal")
        lims = min([p_data.min(), c2_data.min()]), max([p_data.max(), c2_data.max()])
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        sns.despine(ax=ax)
        sns.regplot(p_data, cn_data, ax=ax, label="New")
        sns.regplot(p_data, co_data, ax=ax, label="Old")
        sns.regplot(p2_data, c2_data, ax=ax, scatter=False)
        ax.set_xlabel("Parent {0}".format(label))
        ax.set_ylabel("Child {0}".format(label))
        ax.legend()
        all_stats = get_stats(p2_data, c2_data)
        new_stats = get_stats(p_data, cn_data)
        old_stats = get_stats(p_data, co_data)
        x, y = 0.03, 0.97
        ax.text(x, y, "All: {0}".format(all_stats.fmt_pearson()), transform=ax.transAxes)
        ax.text(x, y - 0.05, "New: {0}".format(new_stats.fmt_pearson()), transform=ax.transAxes)
        ax.text(x, y - 0.10, "Old: {0}".format(old_stats.fmt_pearson()), transform=ax.transAxes)


        child_stats = get_stats(cn_data, co_data)
        ax2 = fig2.add_subplot(3, 2, i)
        ax2.axis("equal")
        lims = c2_data.min(), c2_data.max()
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        sns.despine(ax=ax2)
        sns.regplot(cn_data, co_data, ax=ax2)
        ax2.set_xlabel("New-pole child {0}".format(label))
        ax2.set_ylabel("Old-pole child {0}".format(label))
        ax2.text(x, y, child_stats.fmt_pearson(), transform=ax2.transAxes)

        i += 1

    fig.tight_layout()
    fig.savefig("parent-daughter-correlation.pdf")

    fig2.tight_layout()
    fig2.savefig("daughter-daughter-correlation.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mother/daughter correlation"
    )
    parser.add_argument(
        "-f", "--force", default=False, action="store_true",
        help="""
            force reacquisition of data
        """
    )
    parser.add_argument(
        "process_list", metavar="process", type=str, nargs="*",
        help="""
            specify which process_list to handle.
        """
    )
    args = parser.parse_args()
    if args.process_list:
        dirlist = []
        sources = []
        a = json.loads(open(args.process_list[0]).read())
        for x, y in a.items():
            dirlist.extend([os.path.join(x, _) for _ in y])
            sources.extend([os.path.basename(x) for _ in y])
    else:
        sources, dirlist = None, None

    process_sources(sources, dirlist, force=args.force)
