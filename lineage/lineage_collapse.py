#!/usr/bin/env python

import pandas as pd
import seaborn as sns
sns.set_context("paper")
sns.set_style("white")
sns.set_palette("colorblind")
import matplotlib.pyplot as plt
plt.rc("font", family="sans-serif")
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
    r"\setlength\parindent{0pt}",
]
import numpy as np


def distplot(data, label, ax, **kws):
    counts, bins = np.histogram(data, density=True)
    bin_width = bins[1] - bins[0]
    x_centres = (bins + bin_width)[:-1]
    ax.plot(x_centres, counts, label=label)
#    freq = counts / counts.sum()
#    ax.plot(x_centres, freq, label=label)
#    sns.distplot(data, label=label, ax=ax, **kws)


def main():
    glycerol = pd.read_pickle("glycerol/data-tree.pandas")
    acetate = pd.read_pickle("acetate/data-tree.pandas")
    pyruvate = pd.read_pickle("pyruvate/data-tree.pandas")

    i = 1
    parameters = [
        ("initial_length", r"Initial length (\si{\micro\metre})"),
        ("final_length", r"Final length (\si{\micro\metre})"),
        ("added_length", r"Added length (\si{\micro\metre})"),
        ("doubling_time", r"Interdivision time (\si{\hour})"),
        ("elong_rate", r"Linear elongation rate (\si{\micro\metre\per\hour})"),
        ("growth_rate", r"Exponential elongation rate (\si{\per\hour})"),
    ]

    dkw = {
        "hist_kws": {
        },
        "kde": True,
        "norm_hist": True,
        "hist": False,
    }

#    fig1 = plt.figure(figsize=(11.69, 8.27))
#    fig2 = plt.figure(figsize=(11.69, 8.27))

    for data_type, label in parameters:
        fig1 = plt.figure(figsize=(3.5, 2.4))
        ax1 = fig1.add_subplot(111)
#        ax1 = fig1.add_subplot(2, 3, i)
        sns.despine(ax=ax1)
        distplot(glycerol[data_type], label="Glycerol", ax=ax1, **dkw)
        distplot(acetate[data_type], label="Acetate", ax=ax1, **dkw)
        distplot(pyruvate[data_type], label="Pyruvate", ax=ax1, **dkw)
        ax1.set_xlabel(label)
        ax1.set_ylabel("PDF")
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig("collapsed/{0}-raw.pdf".format(data_type))

        fig2 = plt.figure(figsize=(3.5, 2.4))
        ax2 = fig2.add_subplot(111)
#        ax2 = fig2.add_subplot(2, 3, i)
        sns.despine(ax=ax2)
        distplot(glycerol[data_type] / glycerol[data_type].mean(), label="Glycerol", ax=ax2, **dkw)
        distplot(acetate[data_type] / acetate[data_type].mean(), label="Acetate", ax=ax2, **dkw)
        distplot(pyruvate[data_type] / pyruvate[data_type].mean(), label="Pyruvate", ax=ax2, **dkw)
        ax2.set_xlabel("Normalised {0}".format(label.lower()))
        ax2.set_ylabel("PDF")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig("collapsed/{0}-scaled.pdf".format(data_type))

        i += 1

#    fig1.savefig("not-collapsed.pdf")
#    fig2.savefig("collapsed.pdf")


if __name__ == "__main__":
    main()
