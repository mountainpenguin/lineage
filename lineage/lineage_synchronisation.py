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

    # Attempt at copying Aldridge 2012 Fig. 4B (Fourier analysis)
    for lineage in data.lineage_id.unique():
        # lineage = "ce6b27b4-73e8-4508-b4b4-4cf5a71f41ff"
        dataset = data[data.lineage_id == lineage]
        if dataset.generation.max() < 6:
            continue

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = ax1.twinx()
        ax3 = fig.add_subplot(212)

        ax1.plot(
            dataset.birth_hour,
            dataset.birth_length,
            ls="none",
            marker=".",
            ms=8,
        )
        ax1.set_ylabel("Initial length (\si{\micro\metre})")
        ax1.set_xlabel("Time (\si{\hour})")

        T = 0.75
        # 0.75 as per Aldridge et al. 2012 (from visual inspection of Fig. 4B)
        bins = np.arange(
            min(dataset.birth_hour),
            max(dataset.birth_hour) + T,
            T
        )
        h = np.histogram(dataset.birth_hour, bins=bins)
        ax2.plot(
            bins[:-1] + (T / 2), h[0],
            "r-",
            lw=2,
        )
        ax2.set_ylabel("\# cells", color="r")
        for ticklabel in ax2.get_yticklabels():
            ticklabel.set_color("r")

#        ax3.set_ylabel("PSD ($V^2$ \si{\per{\hertz}})")
        ax3.set_xlabel(r"Periodicity (\si{\hour})")
        a = np.abs(scipy.fftpack.rfft(h[0]))[1:]
        freqs = 1 / scipy.fftpack.rfftfreq(len(h[0]), d=T)[1:]
        max_freq = freqs[np.argmax(a)]
        print("max_freq:", max_freq)
        # f, Pxx = scipy.signal.periodogram(h[0], fs=T)
#        ax3.bar(freqs, a, width=0.01, lw=2)
        ax3.plot(freqs, a, "b--", alpha=.75)
        # ax3.plot(xnew, ynew, "k-")

        freq_data = pd.DataFrame({
            "freq": freqs,
            "a": a,
        })
        sns.tsplot(freq_data, time="freq", value="a")

#        histo_data = np.vstack([
#            bins[:-1] + (T / 2),
#            h[0]
#        ]).T
#        print(histo_data)
#        print(scipy.signal.correlate(histo_data, histo_data))

        fig.tight_layout()
#        plt.show()
        plt.savefig("synchronisation.pdf")
        plt.close()


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
