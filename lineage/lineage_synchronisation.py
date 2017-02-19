#!/usr/bin/env python

""" Test synchronisation within microcolonies """

# stdlib imports
import argparse
import json
import os

# scientific imports
import numpy as np
import scipy.fftpack
import scipy.signal
import scipy.interpolate
import pandas as pd
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

# lineage library imports
from lineage_lib import track
from lineage_lib import misc


def process_lineage(cell_lineage, timings, rif_add, pixel, generation=0, lineage_id=None):
    # time vs birth_length
    if not lineage_id:
        lineage_id = cell_lineage.lineage_id
    data = pd.DataFrame(columns=[
        "lineage_id", "cell_id", "birth_hour", "birth_length", "generation",
        "doubling_time",
    ])
    birth_hour = timings[cell_lineage.cells[0].frame - 1] / 60
    if birth_hour > 0 and cell_lineage.children:
        division_hour = timings[cell_lineage.cells[-1].frame - 1] / 60
        data = data.append(
            pd.Series({
                "lineage_id": lineage_id,
                "cell_id": cell_lineage.lineage_id,
                "birth_hour": timings[cell_lineage.cells[0].frame - 1] / 60,
                "birth_length": cell_lineage.cells[0].length[0][0] * pixel,
                "generation": generation + 1,
                "doubling_time": division_hour - birth_hour,
            }),
            ignore_index=True
        )

    if cell_lineage.children:
        for child in cell_lineage.children:
            child_data = process_lineage(
                child, timings, rif_add, pixel,
                generation=generation + 1,
                lineage_id=lineage_id,
            )
            data = data.append(child_data, ignore_index=True)
    return data


def process_dir():
    L = track.Lineage()
    initial_cells = L.frames[0].cells
    timings, rif_add, pixel = misc.get_timings()
    dir_data = None
    for cell_lineage in initial_cells:
        cell_lineage = track.SingleCellLineage(cell_lineage.id, L)
        lineage_data = process_lineage(cell_lineage, timings, rif_add, pixel)
        if dir_data is not None:
            dir_data = dir_data.append(lineage_data, ignore_index=True)
        else:
            dir_data = lineage_data
    return dir_data


def process_sources(dir_sources, dirs=None, force=False):
    if not dirs:
        dirs = "."
        dir_sources = dirs

    if (os.path.exists("synchro_data.pandas") and force) or not os.path.exists("synchro_data.pandas"):
        orig_dir = os.getcwd()
        i = 0
        data = None
        for d in dirs:
            os.chdir(d)
            source = dir_sources[i]
            print("Processing {0}".format(d))
            if os.path.exists("mt/alignment.mat"):
                dir_data = process_dir()
                if data is not None:
                    data = data.append(dir_data, ignore_index=True)
                else:
                    data = dir_data
                print("Got {0} cells".format(len(dir_data)))
            else:
                print("Skipping, data not analysed")
            i += 1
            os.chdir(orig_dir)
        data.to_pickle("synchro_data.pandas")
    else:
        data = pd.read_pickle("synchro_data.pandas")

    threshold_generations = 4
    long_ids = []
    for lineage_id in data.lineage_id.unique():
        subset = data[data.lineage_id == lineage_id]
        if max(subset.generation) >= threshold_generations:
            long_ids.append(lineage_id)

    long_data = data[data.lineage_id.isin(long_ids)]

    mc_wanted = [
        "3f04e754-f30c-43d8-bcee-612c8edc3447",
        "4c2356fe-5486-46d9-ade4-29fd7bcb22e0",
    ]

    u = list(long_data.lineage_id.unique())
    u.pop(u.index(mc_wanted[0]))
    u.pop(u.index(mc_wanted[1]))
    unique = mc_wanted + u
    microcolony_num = 1
    general_stats = []
    for lineage_id in unique:
        dataset = long_data[long_data.lineage_id == lineage_id]
        mean = dataset.doubling_time.mean()
        std = dataset.doubling_time.std()
        ci = get_ci(dataset.doubling_time)
        n = len(dataset)
        cv = 100 * std / mean
        long_data.ix[long_data.lineage_id == lineage_id, "microcolony"] = microcolony_num
        general_stats.append((
            microcolony_num,
            lineage_id,
            mean,
            std,
            ci,
            cv,
            n
        ))
        microcolony_num += 1

    general_stats = pd.DataFrame(general_stats, columns=[
        "microcolony",
        "lineage_id",
        "_mean",
        "_std",
        "_ci",
        "_cv",
        "_n",
    ])

    # perform stats
    # Kruskal-Wallis H-test
    d = [np.array(long_data[long_data.microcolony == x].doubling_time) for x in long_data.microcolony.unique()]
    print(d)
    kw = scipy.stats.mstats.kruskalwallis(*d)
    print(kw)

    fig = plt.figure(figsize=(6.4, 6.4))

    ax1a = fig.add_subplot(411)
    ax1a.spines["top"].set_visible(False)

    dataset = long_data[long_data.lineage_id == mc_wanted[0]]
    ax1a.plot(
        dataset.birth_hour,
        dataset.birth_length,
        marker=".",
        ls="none",
        ms=8,
    )
    ax1a.set_xlabel("Birth Time (\si{\hour})")
    ax1a.set_ylabel("Initial length (\si{\micro\metre})")
    ax1a.set_title("Birth event timing (microcolony \#1)")

    ax1b = ax1a.twinx()
    ax1b.spines["top"].set_visible(False)

    T = 0.75
    bins = np.arange(
        min(dataset.birth_hour),
        max(dataset.birth_hour) + T,
        T
    )
    h = np.histogram(dataset.birth_hour, bins=bins)
    ax1b.plot(
        bins[:-1] + (T / 2), h[0],
        "r-",
        lw=2
    )
    ax1b.set_ylabel("\# cells", color="r")
    for ticklabel in ax1b.get_yticklabels():
        ticklabel.set_color("r")

    ax2a = fig.add_subplot(412, sharex=ax1a)
    ax2a.spines["top"].set_visible(False)
    dataset = long_data[long_data.lineage_id == mc_wanted[1]]
    ax2a.plot(
        dataset.birth_hour,
        dataset.birth_length,
        marker=".",
        ls="none",
        ms=8,
    )
    ax2a.set_xlabel("Birth Time (\si{\hour})")
    ax2a.set_ylabel("Initial length (\si{\micro\metre})")
    ax2a.set_title("Birth event timing (microcolony \#2)")

    ax2b = ax2a.twinx()
    ax2b.spines["top"].set_visible(False)

    T = 0.75
    bins = np.arange(
        min(dataset.birth_hour),
        max(dataset.birth_hour) + T,
        T
    )
    h = np.histogram(dataset.birth_hour, bins=bins)
    ax2b.plot(
        bins[:-1] + (T / 2), h[0],
        "r-",
        lw=2
    )
    ax2b.set_ylabel("\# cells", color="r")
    for ticklabel in ax2b.get_yticklabels():
        ticklabel.set_color("r")


    ax3 = fig.add_subplot(413)
    sns.despine(ax=ax3)
    sns.barplot(
        x="microcolony",
        y="doubling_time",
        data=long_data,
        color="0.5",
        ax=ax3
    )
#
#    sns.swarmplot(
#        x="microcolony",
#        y="doubling_time",
#        data=long_data,
#        color="0.3",
#        alpha=0.7,
#        ax=ax3
#    )
#    ax3.errorbar(
#        general_stats.microcolony,
#        general_stats._mean,
#        general_stats._std,
#        marker=".",
#        ms=12,
#        linestyle="none",
#        color="b",
#    )
    ax3.set_xlabel("Microcolony")
    ax3.set_ylabel("Interdivision time (\si{\hour})")
    ax3.set_title("Mean interdivision time")

    ax4 = fig.add_subplot(414, sharex=ax3)
    sns.despine(ax=ax4)

    sns.barplot(
        x="microcolony",
        y="_cv",
        data=general_stats,
        color="0.8",
        ax=ax4
    )
    ax4.set_xlabel("Microcolony")
    ax4.set_ylabel("CV (\si{\percent})")
    ax4.set_title("Interdivision time coefficient of variation")

    fig.tight_layout()
    fig.savefig("synchronisation.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchro Script"
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
            if multiple arguments and -c is False:
                combines datasets into a single plot
            elif multiple arguments and -c is True:
                handles each dataset individually
            elif no arguments:
                handles the current directory
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
