#!/usr/bin/env python

""" Script for plotting noise for various carbon sources with bootstrapping"""
import warnings
import os

import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
            (202 / 256, 51 / 256, 0),
            (49 / 256, 99 / 256, 206 / 256),
            (255 / 256, 155 / 256, 0)
        ]

    def get_pole(self, x, isold):
        if isold:
            return x[(x.pole_age > 1)]
        else:
            return x[(x.pole_age == 1) & (x.age_known)]

    def bootstrap_slope(self, data, alpha=0.05, num_samples=100):
        n = len(data)
        sample_ids = np.random.choice(data.id, (num_samples, n))
        stat = []
        for sample_id in sample_ids:
            sample = data[data.id.isin(sample_id)]
            cv = self.slope(sample.initial_length, sample.final_length, cv=True)
            stat.append(cv)
        stat = np.sort(stat)
        return (
            stat[int((alpha / 2.0) * num_samples)],
            stat[int((1 - alpha / 2.0) * num_samples)]
        )

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
            ("initial_length", "Birth length", r"\si{\micro\metre}"),
            # ("final_length", "Division length", r"\si{\micro\metre}"),
            # ("added_length", "Added length", r"\si{\micro\metre}"),
            # ("elong_rate", "Linear elongation rate", r"\si{\micro\metre\per\hour}"),
            ("growth_rate", "Exponential growth rate", r"\si{\per\hour}"),
            # ("doubling_time", "Interdivision time", r"\si{\hour}"),
            ("asymmetry", "Division asymmetry", None),
            ("slope", r"Slope, $a$", None),
        ]

        if not os.path.exists("noise"):
            os.mkdir("noise")

#        fig = plt.figure(figsize=(3.5, 2.4 * len(wanted_vars)))
#        rows, cols, ax_num = len(wanted_vars), 2, 1
        for var, label, unit in wanted_vars:
#            ax = fig.add_subplot(rows, cols, ax_num)
            fig = plt.figure(figsize=(3.34, 2.4))
            ax = fig.add_subplot(121)
            sns.despine(ax=ax)
            if var == "slope":
                ax.plot([0.5, 3.5], [1, 1], "k--")

            self.plot_variable(var, ax)
            if unit:
                ylabel = "{0} ({1})".format(label, unit)
            else:
                ylabel = label
            ax.set_ylabel(ylabel)
#            if ax_num == 1:
#                self.add_legend(ax)
#            elif ax_num == rows * cols - 1:
#                ax.set_xticklabels(self.datalabels, rotation=90)
#            ax_num += 1
            if var == "slope":
                self.add_legend(
                    ax,
                    bbox_to_anchor=[0.6, 1],
                    bbox_transform=ax.transAxes,
                )
            else:
                self.add_legend(ax)
            ax.set_xticklabels(self.datalabels, rotation=90)

#            ax = fig.add_subplot(rows, cols, ax_num)
            ax = fig.add_subplot(122)
            sns.despine(ax=ax)
            self.plot_variable(var, ax, cv=True)
            ax.set_ylabel(r"CV {0} (\si{{\percent}})".format(label.lower()))
#            if ax_num == 2:
#                self.add_legend(ax)
#            elif ax_num == rows * cols:
#                ax.set_xticklabels(self.datalabels, rotation=90)
#            ax_num += 1
            self.add_legend(ax)
            ax.set_xticklabels(self.datalabels, rotation=90)

            fig.tight_layout()
            fig.savefig(
                "noise/{0}.pdf".format(var),
                transparent=True
            )

#        fig.tight_layout()
#        fig.savefig(
#            "noise/temp_noise.pdf",
#            transparent=True
#        )

    def cv(self, x, axis=None):
        return 100 * x.std(axis=axis) / x.mean(axis=axis)

    def slope(self, Lb, Ld, cv=False):
        twotail = 0.975
        tstatistic = scipy.stats.t.ppf(twotail, df=(len(Lb) - 2))
        A = np.vstack([Lb, np.ones(len(Lb))]).T
        linalg = scipy.linalg.lstsq(A, Ld)
        m, c = linalg[0]
        sum_y_res = np.sum((Ld - Ld.mean()) ** 2)
        Syx = np.sqrt(sum_y_res / (len(Lb) - 2))
        sum_x_res = np.sum((Lb - Lb.mean()) ** 2)
        Sb = Syx / np.sqrt(sum_x_res)
        merror = tstatistic * Sb

        if cv:
            return 100 * merror / m
        else:
            return m, merror

    def plot_variable(self, var, ax, cv=False):
        i = 1
        print("=" * 54)
        print(var)
        for dataset in self.datasets:
            if var != "slope":
                dataset = dataset[~np.isnan(dataset[var])]
            newpole = self.get_pole(dataset, False)
            oldpole = self.get_pole(dataset, True)
            # dataset = dataset[(dataset.pole_age > 1) | (dataset.age_known)]
            if var == "slope" and cv:
                cv_both = self.slope(dataset.initial_length, dataset.final_length, cv=True)
                cv_ci_both = self.bootstrap_slope(dataset)
                cv_new = self.slope(newpole.initial_length, newpole.final_length, cv=True)
                cv_ci_new = self.bootstrap_slope(newpole)
                cv_old = self.slope(oldpole.initial_length, oldpole.final_length, cv=True)
                cv_ci_old = self.bootstrap_slope(oldpole)
            elif var == "slope" and not cv:
                cv_both, cv_ci_both = self.slope(dataset.initial_length, dataset.final_length)
                print("{2:8s} [  ; n={3:03d}] Both: {0:.5f} +/i {1:.5f}".format(
                    cv_both, cv_ci_both, self.datalabels[i - 1], len(dataset)
                ))
                cv_new, cv_ci_new = self.slope(newpole.initial_length, newpole.final_length)
                print("{2:8s} [  ; n={3:03d}] New: {0:.5f} +/i {1:.5f}".format(
                    cv_new, cv_ci_new, self.datalabels[i - 1], len(newpole)
                ))
                cv_old, cv_ci_old = self.slope(oldpole.initial_length, oldpole.final_length)
                print("{2:8s} [  ; n={3:03d}] Old: {0:.5f} +/i {1:.5f}".format(
                    cv_old, cv_ci_old, self.datalabels[i - 1], len(oldpole)
                ))
            elif cv:
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

            if var != "slope":
                print("{4:8s} [{0}; n={5:03d}] Both: {1:.5f} <-> {2:.5f} <-> {3:.5f}".format(
                    cv and "cv" or "  ",
                    cv_ci_both[0], cv_both, cv_ci_both[1],
                    self.datalabels[i - 1],
                    len(dataset[var]),
                ))
                print("{4:8s} [{0}; n={5:03d}]  New: {1:.5f} <-> {2:.5f} <-> {3:.5f}".format(
                    cv and "cv" or "  ",
                    cv_ci_new[0], cv_new, cv_ci_new[1],
                    self.datalabels[i - 1],
                    len(newpole[var]),
                ))
                print("{4:8s} [{0}; n={5:03d}]  Old: {1:.5f} <-> {2:.5f} <-> {3:.5f}".format(
                    cv and "cv" or "  ",
                    cv_ci_old[0], cv_old, cv_ci_old[1],
                    self.datalabels[i - 1],
                    len(oldpole[var]),
                ))

            jitter = 0.3
            ax_args = [
                i - jitter,
                cv_new,
            ]
            if type(cv_ci_new) is not tuple:
                ax_args.append(cv_ci_new)
            else:
                ax_args.append(
                    [[cv_new - cv_ci_new[0], cv_ci_new[1] - cv_new]]
                )

            ax_kwargs = {
                "color": self.datacolour[i - 1],
                "fmt": "o",
                "lw": 2,
                "mew": 1,
            }

            ax.errorbar(*ax_args, **ax_kwargs)

            ax_args = [
                i,
                cv_both,
            ]
            if type(cv_ci_both) is not tuple:
                ax_args.append(cv_ci_both)
            else:
                ax_args.append(
                    [[cv_both - cv_ci_both[0], cv_ci_both[1] - cv_both]]
                )

            ax_kwargs = {
                "color": self.datacolour[i - 1],
                "fmt": "s",
                "lw": 2,
                "mew": 1,
            }
            ax.errorbar(*ax_args, **ax_kwargs)

            ax_args = [
                i + jitter,
                cv_old,
            ]
            if type(cv_ci_old) is not tuple:
                ax_args.append(cv_ci_old)
            else:
                ax_args.append(
                    [[cv_old - cv_ci_old[0], cv_ci_old[1] - cv_old]]
                )

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
        ax.set_xlim([0.5, 3.5])

    def add_legend(self, ax, **legend_kws):
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
        black_square = plt.Line2D(
            [], [],
            color="black",
            marker="s",
            mew=1,
            linestyle="none",
            label="Both",
        )
        ax.legend(
            handles=[black_filled, black_square, black_unfilled],
            **legend_kws
        )


if __name__ == "__main__":
    M = Main()
