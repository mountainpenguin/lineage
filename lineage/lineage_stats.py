#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rc("font", family="sans-serif")
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
]
sns.set_style("white")
sns.set_context("paper")

"""
def main():
    L = track.Lineage()
    init = L.frames[0].cells
    lineages = []
    for x in init:
        lin = track.SingleCellLineage(x.id, L, assign_poles=False, orient=False)
        lineages.append(lin)
    print(lineages)
"""


def main():
    stats = [
        ("elongation_rate", r"Elongation Rate (\si{\micro\metre{\per\hour}})"),
        ("doubling_time", r"Doubling Time (\si{\hour})"),
        ("birth_length", r"Birth Length (\si{\micro\metre})"),
        ("division_length", r"Division Length (\si{\micro\metre})"),
        ("septum_placement", r"Septum Placment (\si{\percent})"),
    ]
    for stat, label in stats:
        data = np.load("data/{0}.npy".format(stat))
        # prune out negatives
        data = data[data >= 0]
        # plot histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
#        ax.set_title(stat.replace("_", " ").title())
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        if stat == "doubling_time":
            sns.distplot(data, ax=ax, kde=False, bins=np.arange(min(data), max(data) + 0.25, 0.25))
        else:
            sns.distplot(data, ax=ax, kde=False)

        sns.despine()
        fig.tight_layout()
        fig.savefig("data/{0}.pdf".format(stat))


if __name__ == "__main__":
    main()
