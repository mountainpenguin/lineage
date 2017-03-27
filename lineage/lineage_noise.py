#!/usr/bin/env python

""" Script for plotting noise for various carbon sources with bootstrapping"""
import warnings
import os

import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
sns.set_context("paper")

plt.rc("font", family="sans-serif")
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
    r"\setlength\parindent{0pt}",
]

warnings.filterwarnings("ignore")


class Main(object):
    def __init__(self):
        self.import_data()
        self.plot_data()

    def import_data(self):
        self.glycerol = pd.read_pickle("glycerol/data-tree.pandas")
        self.acetate = pd.read_pickle("acetate/data-tree.pandas")
        self.pyruvate = pd.read_pickle("pyruvate/data-tree.pandas")
        self.datasets = [self.glycerol, self.acetate, self.pyruvate]
        self.datalabels = ["Glycerol", "Acetate", "Pyruvate"]
        self.datacolour = [
            (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
            (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
            (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
        ]

    def get_pole(self, x, isold):
        if isold:
            return x[(x.pole_age > 1)]
        else:
            return x[(x.pole_age == 1) & (x.age_known)]

    def bootstrap(self, data, statistic=np.mean, alpha=0.05, num_samples=100):
        n = len(data)
        samples = np.random.choice(data, (num_samples, n))
        stat = np.sort(statistic(samples, 1))
        return (
            stat[int((alpha / 2.0) * num_samples)],
            stat[int((1 - alpha / 2.0) * num_samples)]
        )

    def plot_data(self):
        wanted_vars = [
            ("initial_length", "Initial length", r"\si{\micro\metre}"),
            # ("final_length", "Final length", r"\si{\micro\metre}"),
            # ("added_length", "Added length", r"\si{\micro\metre}"),
            # ("elong_rate", "Elongation rate", r"\si{\micro\metre\per\hour}"),
            ("growth_rate", "Growth rate", r"\si{\per\hour}"),
            # ("doubling_time", "Interdivision time", r"\si{\hour}"),
            ("asymmetry", "Asymmetry", None),
        ]
        fig = plt.figure(figsize=(3.5, 2.4 * len(wanted_vars)))
        rows, cols, ax_num = len(wanted_vars), 2, 1
        for var, label, unit in wanted_vars:
            ax = fig.add_subplot(rows, cols, ax_num)
            sns.despine(ax=ax)
            self.plot_variable(var, ax)
            if unit:
                ylabel = "{0} ({1})".format(label, unit)
            else:
                ylabel = label
            ax.set_ylabel(ylabel)
            if ax_num == 1:
                self.add_legend(ax)
            elif ax_num == rows * cols - 1:
                ax.set_xticklabels(self.datalabels, rotation=90)
            ax_num += 1

            ax = fig.add_subplot(rows, cols, ax_num)
            sns.despine(ax=ax)
            self.plot_variable(var, ax, cv=True)
            ax.set_ylabel(r"CV {0} (\si{{\percent}})".format(label.lower()))
            if ax_num == 2:
                self.add_legend(ax)
            elif ax_num == rows * cols:
                ax.set_xticklabels(self.datalabels, rotation=90)
            ax_num += 1

        if not os.path.exists("noise"):
            os.mkdir("noise")

        fig.tight_layout()
        fig.savefig(
            "noise/temp_noise.pdf",
            bbox_inches="tight",
            transparent=True
        )

    def cv(self, x, axis=None):
        return 100 * x.std(axis=axis) / x.mean(axis=axis)

    def plot_variable(self, var, ax, cv=False):
        i = 1
        print("=" * 54)
        print(var)
        for dataset in self.datasets:
            newpole = self.get_pole(dataset, False)
            oldpole = self.get_pole(dataset, True)
            dataset = dataset[(dataset.pole_age > 1) | (dataset.age_known)]
            if cv:
                cv_both = self.cv(dataset[var])
                cv_ci_both = self.bootstrap(
                    dataset[var], self.cv
                )
                cv_new = self.cv(newpole[var])
                # bootstrap to get the CI of the CV
                cv_ci_new = self.bootstrap(
                    newpole[var],
                    self.cv,
                )
                cv_old = self.cv(oldpole[var])
                cv_ci_old = self.bootstrap(
                    oldpole[var],
                    self.cv,
                )
            else:
                cv_both = np.mean(dataset[var])
                cv_ci_both = self.bootstrap(dataset[var])
                cv_new = np.mean(newpole[var])
                cv_ci_new = self.bootstrap(newpole[var])
                cv_old = np.mean(oldpole[var])
                cv_ci_old = self.bootstrap(oldpole[var])

            print("{4:8s} [{0}] Both: {1:.5f} <-> {2:.5f} <-> {3:.5f}".format(
                cv and "cv" or "  ",
                cv_ci_both[0], cv_both, cv_ci_both[1],
                self.datalabels[i - 1],
            ))
            print("{4:8s} [{0}]  New: {1:.5f} <-> {2:.5f} <-> {3:.5f}".format(
                cv and "cv" or "  ",
                cv_ci_new[0], cv_new, cv_ci_new[1],
                self.datalabels[i - 1],
            ))
            print("{4:8s} [{0}]  Old: {1:.5f} <-> {2:.5f} <-> {3:.5f}".format(
                cv and "cv" or "  ",
                cv_ci_old[0], cv_old, cv_ci_old[1],
                self.datalabels[i - 1],
            ))

            jitter = 0.15
            ax_args = [
                i - jitter,
                cv_new,
                [[cv_new - cv_ci_new[0], cv_ci_new[1] - cv_new]],
                # np.diff(cv_ci_new),
            ]
            ax_kwargs = {
                "color": self.datacolour[i - 1],
                "fmt": "o",
                "lw": 2,
                "mew": 1,
            }
            ax.errorbar(*ax_args, **ax_kwargs)

#            ax_args = [
#                i,
#                cv_both,
#                [[cv_both - cv_ci_both[0], cv_ci_both[1] - cv_both]],
#            ]
#            ax_kwargs = {
#                "color": self.datacolour[i - 1],
#                "fmt": "x",
#                "lw": 2,
#                "mew": 1
#            }
#            ax.errorbar(*ax_args, **ax_kwargs)

            ax_args = [
                i + jitter,
                cv_old,
                [[cv_old - cv_ci_old[0], cv_ci_old[1] - cv_old]],
                #np.diff(cv_ci_old)
            ]
            ax_kwargs = {
                "color": self.datacolour[i - 1],
                "fmt": "o",
                "lw": 2,
                "mew": 1,
                "mfc": "none",
                "mec": self.datacolour[i - 1],
            }
            ax.errorbar(*ax_args, **ax_kwargs)

            i += 1

        ax.set_xticks([1, 2, 3])
#        ax.set_xticklabels([x[1] for x in self.datasets], rotation=90)
        ax.set_xticklabels([])
        ax.set_xlim([0.5, 3.5])

    def add_legend(self, ax):
        black_filled = plt.Line2D(
            [], [],
            color="black",
            marker="o",
            mew=1,
            linestyle="none",
            label="New",
        )
        black_unfilled = plt.Line2D(
            [], [],
            color="black",
            marker="o",
            mew=1,
            mfc="none",
            linestyle="none",
            label="Old",
        )
        ax.legend(handles=[black_filled, black_unfilled])


if __name__ == "__main__":
    M = Main()







