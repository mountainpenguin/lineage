#!/usr/bin/env python


""" Model cell growth using a noisy linear map

    final_size = (a * initial_size) + b + noise
    L_F(n) = aL_I(n) + b + \eta

    where:
        n = generation number
        L_F(n) = cell size before division for generation n
        L_I(n) = cell size at birth for generation n
        a = gradient of regression line
        b = intercept of regression line
        eta = noise
"""


import argparse
from lineage_lib import track
from lineage_lib import misc
import numpy as np
import scipy.stats
import scipy.special
import statsmodels.formula.api as sm
import json
import datetime
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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


def process_old():
#    lineage_file = np.load("lineages.npz")
#    lineage_data = lineage_file["lineages"]
    lineage_info = json.loads(open("lineages.json").read())
    T, rif_add, pixel = misc.get_timings()

    data = pd.DataFrame(columns=[
        "initial_id", "final_id",
        "initial_length", "final_length",
        "initial_area", "final_area",
        "length_ratio", "area_ratio",
        "doubling_time", "growth_rate", "elong_rate",
    ])
    for lin in lineage_info:
        if lin["parent"] and lin["children"] and len(lin["lineage"]) > 5:
            lin_data = {}
            lineage = lin["lineage"]
            initial_length = lineage[0][2] * pixel
            final_length = lineage[-1][2] * pixel
            length_ratio = final_length / initial_length

            # get doubling time
            initial_frame = lineage[0][1]
            final_frame = lineage[-1][1]
            doubling_time = (T[final_frame] - T[initial_frame]) / 60

            # get growth rate
            times = np.array([T[x[1]] / 60 for x in lineage])
            lengths = [x[2] * pixel for x in lineage]
            logL = np.log(lengths)
            growth_rate, _ = np.polyfit(times - times[0], logL, 1)

            # get elongation rate
            elong_rate, _ = np.polyfit(times, lengths, 1)

            lin_data["initial_id"] = lineage[0][0]
            lin_data["final_id"] = lineage[-1][0]
            lin_data["initial_length"] = initial_length
            lin_data["final_length"] = final_length
            lin_data["initial_area"] = np.NaN
            lin_data["final_area"] = np.NaN
            lin_data["doubling_time"] = doubling_time
            lin_data["growth_rate"] = growth_rate
            lin_data["elong_rate"] = elong_rate
            lin_data["length_ratio"] = length_ratio
            lin_data["area_ratio"] = np.NaN
            lin_series = pd.Series(lin_data)
            data = data.append(lin_series, ignore_index=True)

    return data

def process_process(process_queue, L, T, rif_add, pixel):
    out_data = pd.DataFrame(columns=[
        "initial_id", "final_id",
        "initial_length", "final_length",
        "initial_area", "final_area",
        "length_ratio", "area_ratio",
        "doubling_time", "growth_rate", "elong_rate",
    ])

    for cell in process_queue:
        num_frames = 1
        # get daughters
        initial_cell = L.frames.cell(cell.id)
        lengths = []
        times = []
        while type(cell.children) is str:
            lengths.append(cell.length[0][0] * pixel)
            times.append(T[cell.frame - 1] / 60)

            cell = L.frames.cell(cell.children)
            num_frames += 1
            if T[cell.frame - 1] > rif_add:
                cell.children = None

        if type(cell.children) is list and num_frames > 5:
            lin_data = {}
            process_queue.append(L.frames.cell(cell.children[0]))
            process_queue.append(L.frames.cell(cell.children[1]))

            initial_length = initial_cell.length[0][0] * pixel
            final_length = cell.length[0][0] * pixel

            initial_area = initial_cell.area[0][0] * pixel * pixel
            final_area = cell.area[0][0] * pixel * pixel

            lin_data["initial_id"] = initial_cell.id
            lin_data["initial_length"] = initial_length
            lin_data["initial_area"] = initial_area
            lin_data["final_id"] = cell.id
            lin_data["final_length"] = final_length
            lin_data["final_area"] = final_area

            lin_data["length_ratio"] = final_length / initial_length
            lin_data["area_ratio"] = final_area / initial_area

            lin_data["doubling_time"] = (
                T[cell.frame - 1] - T[initial_cell.frame - 1]
            ) / 60

            times = np.array(times)
            logL = np.log(lengths)
            growth_rate, logL0 = np.polyfit(times - times[0], logL, 1)
            lin_data["growth_rate"] = growth_rate

            elong_rate, _ = np.polyfit(times, lengths, 1)
            lin_data["elong_rate"] = elong_rate

            lin_series = pd.Series(lin_data)
            out_data = out_data.append(lin_series, ignore_index=True)

    return out_data

def process_process_with_poles(process_queue, T, rif_add, pixel):
    out_data = pd.DataFrame(columns=[
        "initial_id", "final_id",
        "initial_length", "final_length",
        "initial_area", "final_area",
        "length_ratio", "area_ratio",
        "doubling_time", "growth_rate", "elong_rate",
        "old_pole", "pole_age", "age_known",
    ])

    for lin in process_queue:
        if type(lin.children) is list:
            lin_data = {}
            initial_cell = lin.cells[0]
            final_cell = lin.cells[-1]

            if T[final_cell.frame - 1] > rif_add:
                continue

            if len(lin.cells) < 5:
                continue

            m = max(lin.pole1_age, lin.pole2_age).age
            if lin.pole1_age.age_known and lin.pole2_age.age_known:
                lin_data["age_known"] = True
            else:
                lin_data["age_known"] = False

            lin_data["pole_age"] = m
            if m > 1:
                lin_data["old_pole"] = True
            else:
                lin_data["old_pole"] = False

            lin_data["initial_id"] = initial_cell.id
            lin_data["initial_length"] = initial_cell.length[0][0] * pixel
            lin_data["initial_area"] = initial_cell.area[0][0] * pixel * pixel

            lin_data["final_id"] = final_cell.id
            lin_data["final_length"] = final_cell.length[0][0] * pixel
            lin_data["final_area"] = final_cell.area[0][0] * pixel * pixel

            lin_data["length_ratio"] = final_cell.length[0][0] / initial_cell.length[0][0]
            lin_data["area_ratio"] = final_cell.area[0][0] / initial_cell.area[0][0]
            lin_data["doubling_time"] = (
                T[final_cell.frame - 1] - T[initial_cell.frame - 1]
            ) / 60

            cell_frames = lin.frames()
            cell_lengths = lin.lengths(pixel)

            times = np.array([T[x - 1] for x in cell_frames]) / 60
            logL = np.log(cell_lengths)
            growth_rate, _ = np.polyfit(times - times[0], logL, 1)
            lin_data["growth_rate"] = growth_rate

            elong_rate, _ = np.polyfit(times, cell_lengths, 1)
            lin_data["elong_rate"] = elong_rate

            lin_series = pd.Series(lin_data)
            out_data = out_data.append(lin_series, ignore_index=True)

            process_queue.append(lin.children[0])
            process_queue.append(lin.children[1])

    return out_data

def process_dir(with_poles, with_age, debug=False):
    try:
        L = track.Lineage()
    except:
        print("Error getting lineage information")
        if os.path.exists("lineages.npz"):
            print("But lineages.npz exists")
            return process_old()
        return None
    initial_cells = L.frames[0].cells
    # only follow cells after first division
    process_queue = []
    for cell_lineage in initial_cells:
        if with_poles or with_age:
            cell_lineage = track.SingleCellLineage(cell_lineage.id, L, debug=debug)
            if type(cell_lineage.children) is list:
                process_queue.append(cell_lineage.children[0])
                process_queue.append(cell_lineage.children[1])
        else:
            while type(cell_lineage.children) is str:
                cell_lineage = L.frames.cell(cell_lineage.children)

            if type(cell_lineage.children) is list:
                process_queue.append(L.frames.cell(cell_lineage.children[0]))
                process_queue.append(L.frames.cell(cell_lineage.children[1]))

    T, rif_add, pixel = misc.get_timings()
    if with_poles or with_age:
        result = process_process_with_poles(process_queue, T, rif_add, pixel)
    else:
        result = process_process(process_queue, L, T, rif_add, pixel)

    return result

def get_stats(xdata, ydata, fit="linear", ci=95):
    """ Return data statistics

    Input arguments:
        `xdata`
        `ydata`
        `fit`
        `ci`

    Fit Types:
        linear:
            Fits a linear regression line to the data

            Returns:
                (gradient, \pm 95%),
                (y-intercept, \pm 95%),
                pearson r,

    """
    if fit == "linear":
        twotail = 1 - (1 - ci / 100) / 2
        tstatistic = scipy.stats.t.ppf(twotail, df=(len(xdata) - 2))

        A = np.vstack([xdata, np.ones(len(xdata))]).T
#        results = sm.OLS(
#            ydata, A
#        ).fit()
#        print(results.summary())
#        input("...")

        linalg = scipy.linalg.lstsq(A, ydata)
        m, c = linalg[0]
        sum_y_residuals = np.sum((ydata - ydata.mean()) ** 2)
        Syx = np.sqrt(sum_y_residuals / (len(xdata) - 2))
        sum_x_residuals = np.sum((xdata - xdata.mean()) ** 2)
        Sb = Syx / np.sqrt(sum_x_residuals)
        merror = tstatistic * Sb

        Sa = Syx * np.sqrt(
            np.sum(xdata ** 2) / (len(xdata) * sum_x_residuals)
        )
        cerror = tstatistic * Sa

        r, rp = scipy.stats.pearsonr(xdata, ydata)

        return [
            (m, merror),
            (c, cerror),
            (r, rp),
        ]
    else:
        raise NotImplementedError

#def get_mstd(data):
#    xdata = data.initial_length
#    ydata = data.final_length
#    tstatistic = scipy.stats.t.ppf(0.975, df=(len(xdata) - 2))
#    A = np.vstack([xdata, np.ones(len(xdata))]).T
#    linalg = scipy.linalg.lstsq(A, ydata)
#    m, c = linalg[0]
#    sum_y_residuals = np.sum((ydata - ydata.mean()) ** 2)
#    Syx = np.sqrt(sum_y_residuals / (len(xdata) - 2))
#    sum_x_residuals = np.sum((xdata - xdata.mean()) ** 2)
#    Sb = Syx / np.sqrt(sum_x_residuals)
#
#    return (m, Sb, len(xdata))

def plot_fake(ax, label):
    x = np.mean(ax.get_xlim())
    y = np.mean(ax.get_ylim())
    ax.plot(x, y, color="none", alpha=1, label=label)

def fmt_dec(var, places):
    distance = -int(np.log10(np.abs(var)))
    if distance < places:
        fmtter = "{{0:.{0}f}}".format(places)
    elif distance > places:
        fmtter = "{0:.3g}"
    else:
        fmtter = "{{0:.{0}f}}".format(distance)
    return fmtter.format(var)


def add_stats(ax, xdata, ydata, msymbol="m", csymbol="c"):
    (stats_m, stats_merr), (stats_c, stats_cerr), (r, rpval) = get_stats(
        xdata, ydata
    )
    plot_fake(
        ax,
        "{msymbol} = {0} $\pm$ {1}".format(
            fmt_dec(stats_m, 5),
            fmt_dec(stats_merr, 3),
            msymbol=msymbol,
        )
    )
    plot_fake(
        ax,
        "{csymbol} = {0} $\pm$ {1}".format(
            fmt_dec(stats_c, 5),
            fmt_dec(stats_cerr, 3),
            csymbol=csymbol,
        )
    )
    # plot_fake(ax, r"r$^2$ = {rsq} (p = {rpval})".format(
    plot_fake(ax, r"r = {r:.3g}, r$^2$ = {rsq:.3g}".format(
        r=r,
        rsq=r ** 2,
        rpval=fmt_dec(rpval, 3),
    ))
    plot_fake(ax, "n = {0}".format(len(xdata)))


def plot_joint(xdata, ydata, xlab, ylab, fn="noisy_linear_map", suffix=""):
    fig = plt.figure()
    kws = dict(
        x=xdata,
        y=ydata,
        kind="reg",
        joint_kws={
            "ci": 95,
            "scatter_kws": {
                "s": 40,
                "color": "darkred",
                "alpha": 0.5,
            },
            "marker": "x",
        },
    )
    if fn == "initial-doubling":
        kws["marginal_kws"] = {
            "bins": np.arange(
                min(ydata),
                max(ydata) + 0.25,
                0.25
            )
        }
    if fn in ["noisy_linear_map", "noisy-linear-map-new-pole", "noisy-linear-map-old-pole"]:
        kws["xlim"] = [1, 9]
        kws["ylim"] = [2, 16]

    g = sns.jointplot(**kws)
    ((stats_m, stats_merror),
     (stats_c, stats_cerror),
     (stats_r, stats_rp)) = get_stats(xdata, ydata)
    annotation = r"""
a = {m} $\pm$ {me}
b = {c} $\pm$ {ce}
r = {r}, r$^2$ = {rsq}
n = {n}"""
    if fn == "noisy_linear_map" or "noisy-linear-map" in fn:
        annotation += r"""
$\langle L_I \rangle$ = {im}
$\langle L_F \rangle$ = {fm}"""
        x_ci = float(np.diff(scipy.stats.t.interval(0.95, len(xdata) - 1, loc=xdata.mean(), scale=xdata.sem()))[0])
        y_ci = float(np.diff(scipy.stats.t.interval(0.95, len(ydata) - 1, loc=ydata.mean(), scale=ydata.sem()))[0])
        print("<L_I>=", xdata.mean(), "sd=", xdata.std(), "sem=", xdata.sem(), "ci=", x_ci)
        print("<L_F>=", ydata.mean(), "sd=", ydata.std(), "sem=", ydata.sem(), "ci=", y_ci)
    annotation = annotation.format(
        m=fmt_dec(stats_m, 5),
        me=fmt_dec(stats_merror, 3),
        c=fmt_dec(stats_c, 5),
        ce=fmt_dec(stats_cerror, 3),
        r=fmt_dec(stats_r, 3),
        rsq=fmt_dec(stats_r ** 2, 3),
        n=len(xdata),
        im=fmt_dec(xdata.mean(), 4),
        fm=fmt_dec(ydata.mean(), 4),
    )
    annotation += "\n"
    g.annotate(
        lambda x,y: (x,y),
        template="\n".join(annotation.split("\n")[1:-1]),
        loc="upper left",
        fontsize=12,
    )
#    xlab = "Initial cell length (\si{\micro\metre})"
#    ylab = "Final cell length (\si{\micro\metre})"
    g.set_axis_labels(xlab, ylab, fontsize=12)

    plt.savefig("{0}{1}.pdf".format(fn, suffix))
    plt.close()


def plot_error(ax, xdata, ydata):
    fig = plt.figure()
    ax_resid = fig.add_subplot(1, 2, 1)
    ax_resid.set_title("Residuals")
    ax_resid.set_xlabel("Residual")
    ax_resid.set_ylabel("Frequency")
    ax_qq = fig.add_subplot(1, 2, 2)

#    o = np.ones((len(xdata),))
#    A = np.vstack([xdata, np.ones(len(xdata))]).T
#    result = sm.OLS(
#        ydata, A
#    ).fit()
#    m, c = result.params
    (m, _), (c, _), _ = get_stats(xdata, ydata)
    expected = m * xdata + c
    residuals = ydata - expected
    sns.distplot(residuals, ax=ax_resid, kde=False, norm_hist=True)

    # fit Gaussian
    hist, bin_edges = np.histogram(residuals, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    gauss = lambda x, A, mu, sigma: A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    p0 = [1.0, 0.0, 1.0]
    coeff, var_matrix = scipy.optimize.curve_fit(gauss, bin_centres, hist, p0=p0)
    limits = ax_resid.get_xlim()
    xspace = np.linspace(limits[0], limits[1], 200)
    hist_fit = gauss(xspace, *coeff)
    ax_resid.plot(xspace, hist_fit, label="Fitted Gaussian")
    plot_fake(ax_resid, "$\mu =$ {0:.5f}".format(coeff[1]))
    plot_fake(ax_resid, "$\sigma =$ {0:.5f}".format(coeff[2]))

    ax_resid.legend(loc=1)

    # plot Q-Q probability plot
    scipy.stats.probplot(residuals, plot=ax_qq)
    lines = ax_qq.lines
    for l in lines:
        if l.get_marker() == "o" and l.get_color() == "b":
            l.set_color(sns.color_palette()[0])
            l.set_alpha(0.5)
        else:
            l.set_color(sns.color_palette()[1])
    # test normality
    normal_kscore, normal_pvalue = scipy.stats.mstats.normaltest(residuals)
    plot_fake(ax_qq, "isnormal $p=$ {0:.5g}".format(normal_pvalue))
    ax_qq.legend(loc=2)

    sns.despine()

    fig.savefig("noisy-noise.pdf")
    plt.close()


def plot_regplot(ax, xdata, ydata, xlabel, ylabel, mlabel="m", clabel="c"):
    common_kws = {
        "ci": 95,
        "scatter_kws": {
            "s": 10,
            "color": "darkred",
            "alpha": 0.5,
        },
        "marker": "x",
    }
    sns.regplot(
        x=xdata,
        y=ydata,
        **common_kws
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    add_stats(ax, xdata, ydata, mlabel, clabel)
    plt.legend(loc=2)
    sns.despine()


def _iqr(a):
    """From seaborn/utils.py"""
    q1 = scipy.stats.scoreatpercentile(a, 25)
    q3 = scipy.stats.scoreatpercentile(a, 75)
    return q3 - q1


def _freedman_diaconis_bins(a):
    """From seaborn/distributions.py"""
    a = np.asarray(a)
    h = 2 * _iqr(a) / (len(a) ** (1 / 3))
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        return int(np.ceil((a.max() - a.min()) / h))


def process_root(dir_sources, dirs=None, with_poles=False, with_age=False, force=False, debug=False):
    if not dirs:
        dirs = "."
#        dirs = list(filter(lambda x: os.path.isdir(x), sorted(os.listdir())))
        dir_sources = dirs

    if (os.path.exists("data.pandas") and force) or not os.path.exists("data.pandas"):
        columns = [
            "source", "sub_source",
            "initial_id", "final_id",
            "initial_length", "final_length",
            "initial_area", "final_area",
            "added_length",
            "length_ratio", "area_ratio",
            "doubling_time", "growth_rate", "elong_rate",
        ]
        if with_poles:
            columns.extend(["old_pole", "pole_age", "age_known"])

        data = pd.DataFrame(columns=columns)

        orig_dir = os.getcwd()
        i = 0
        for d in dirs:
            os.chdir(d)
            source = dir_sources[i]
            print("Processing {0}".format(d))
            if os.path.exists("mt/alignment.mat") or os.path.exists("lineages.npz"):
                out_data  = process_dir(with_poles, with_age, debug)
                if out_data is not None:
                    out_data["source"] = [source] * len(out_data)
                    out_data["sub_source"] = [os.path.basename(d)] * len(out_data)
                    out_data["added_length"] = out_data.final_length - out_data.initial_length
                    data = pd.concat([data, out_data], ignore_index=True)
                    print("Got {0} cells".format(len(out_data)))
                else:
                    print("No cells returned")
            else:
                print("Skipping, no cells")
            i += 1
            os.chdir(orig_dir)

        print("Got {0} cells that divide twice during observation period".format(len(data)))
        if len(data) == 0:
            return

        data.to_pickle("data.pandas")
    else:
        data = pd.read_pickle("data.pandas")

    xlab = "Initial cell length (\si{\micro\metre})"
    plot_joint(
        data.initial_length, data.final_length,
        xlab, "Final cell length (\si{\micro\metre})"
    )
    plot_error(None, data.initial_length, data.final_length)

    plot_joint(
        data.initial_length, data.added_length,
        xlab, "Added length (\si{\micro\metre})",
        "initial-added"
    )
    plot_joint(
        data.initial_length, data.doubling_time,
        xlab, "Interdivision time (\si{\hour})",
        "initial-doubling"
    )

    plot_joint(
        data.initial_length, data.elong_rate,
        xlab, "Elongation rate (\si{\micro\metre\per\hour})",
        "initial-elongation"
    )

    plot_joint(
        data.initial_length, data.growth_rate,
        xlab, "Growth rate (\si{\per\hour})",
        "initial-growth"
    )

    if with_age:
#        for x in range(int(data.pole_age.max())):
#            data_subset = data[data.pole_age == (x + 1)]
#            try:
#                plot_joint(
#                    data_subset.initial_length, data_subset.final_length,
#                    xlab, "Final cell length (\si{\micro\metre})",
#                    "noisy-linear-map-gen-{0}".format(x + 1)
#                )
#            except ValueError:
#                pass
#
        # plot pole data swarms
        fig, ax = plt.subplots(3, 2, figsize=(8, 12))
        sns.despine()
        ax = ax.flatten()
        for i, y in zip(range(6), [
            "initial_length", "final_length", "added_length",
            "doubling_time", "elong_rate", "growth_rate"
        ]):
#            sns.swarmplot(
#                x="pole_age",
#                y=y,
#                data=data[data.age_known == True],
#                color="0.25",
#                alpha=0.75,
#                ax=ax[i],
#            )
            sns.boxplot(
                x="pole_age",
                y=y,
                data=data[data.age_known == True],
                ax=ax[i],
            )
            # plot overall data mean
            ax[i].axhline(
                data[data.age_known == True][y].mean(),
                color="r",
                lw=1,
                alpha=.5,
                ls="--",
            )
            ax[i].set_xlabel("Pole age (generations)")
        ax[0].set_ylabel("Initial length (\si{\micro\metre})")
        ax[1].set_ylabel("Final length (\si{\micro\metre})")
        ax[2].set_ylabel("Added length (\si{\micro\metre})")
        ax[3].set_ylabel("Interdivision time (\si{\hour})")
        ax[4].set_ylabel("Elongation rate (\si{\micro\metre\per\hour})")
        ax[5].set_ylabel("Growth rate (\si{\per\hour})")
        fig.tight_layout()
        sns.despine()
        fig.savefig("pole_age_boxplots.pdf")
        plt.close()

    elif with_poles:
        data_new = data[(data.pole_age == 1 & data.age_known)]
        data_old = data[data.pole_age > 1]
        plot_joint(
            data_new.initial_length, data_new.final_length,
            xlab, "Final cell length (\si{\micro\metre})",
            "noisy-linear-map-new-pole"
        )
        plot_joint(
            data_old.initial_length, data_old.final_length,
            xlab, "Final cell length (\si{\micro\metre})",
            "noisy-linear-map-old-pole"
        )

        # plot pole data histograms
        plot_distplot_comparisons(
            data[(data.pole_age == 1 & data.age_known)],
            data[data.pole_age > 1],
            labels=[
                "New pole",
                "Old pole"
            ]
        )

        fig, ax = plt.subplots(3, 2, figsize=(5, 12))
        sns.despine()
        ax = ax.flatten()
        for i, y in zip(range(6), [
            "initial_length", "final_length", "added_length",
            "doubling_time", "elong_rate", "growth_rate"
        ]):
#            sns.swarmplot(
#                x="old_pole",
#                y=y,
#                data=data,
#                color="0.25",
#                alpha=0.75,
#                ax=ax[i],
#            )
            sns.boxplot(
                x="old_pole",
                y=y,
                data=data,
                ax=ax[i],
            )

            for tf in [True, False]:
                ax[i].plot(
                    [-0.4 + (1 * tf), 0.4 + (1 * tf)],
                    [data[data.old_pole == tf][y].mean(),
                    data[data.old_pole == tf][y].mean()],
                    color="r",
                    lw=2,
                    alpha=.5,
                    ls="--",
                )

            ylims = ax[i].get_ylim()
            stat_bar_spacing = (ylims[1] - ylims[0]) / 20
            stat_bar_y = data[y].max() + stat_bar_spacing
            ax[i].plot(
                [0, 0, 1, 1],
                [
                    stat_bar_y,
                    stat_bar_y + stat_bar_spacing / 5,
                    stat_bar_y + stat_bar_spacing / 5,
                    stat_bar_y
                ],
                lw=1.5,
                color="k",
            )
            ttest = scipy.stats.ttest_ind(
                data[data.pole_age == 1][y],
                data[data.pole_age > 1][y],
                equal_var=False
            )
            ax[i].text(
                0.5,
                stat_bar_y + stat_bar_spacing / 2,
                r"$p=$ \num{{{0:.4g}}}".format(ttest.pvalue),
                ha="center",
                va="bottom",
                color="k",
                fontsize=12,
            )
            ax[i].set_xlabel("Pole inherited")
            ax[i].xaxis.set_ticklabels(["New pole", "Old pole"])
        ax[0].set_ylabel("Initial length (\si{\micro\metre})")
        ax[1].set_ylabel("Final length (\si{\micro\metre})")
        ax[2].set_ylabel("Added length (\si{\micro\metre})")
        ax[3].set_ylabel("Interdivision time (\si{\hour})")
        ax[4].set_ylabel("Elongation rate (\si{\micro\metre\per\hour})")
        ax[5].set_ylabel("Growth rate (\si{\per\hour})")
        fig.tight_layout()
        fig.savefig("pole_boxplots.pdf")
        plt.close()


def plot_distplot_comparisons(*datasets, labels=None, filename="pole_histograms"):
    fig, axes = plt.subplots(3, 2)
    ax = axes.flatten()
    sns.despine()

    # plot dists
    variables = [
        "initial_length", "final_length", "added_length",
        "doubling_time", "elong_rate", "growth_rate",
    ]

    var_idx = 0
    for var in variables:
        # determine significance
        bins = None
        for dataset, label in zip(datasets, labels):
            if var == "doubling_time":
                bins = np.arange(
                    min(dataset[var]),
                    max(dataset[var]) + 0.25,
                    0.25
                )
                sns.distplot(
                    dataset[var], kde=False, ax=ax[var_idx],
                    bins=np.arange(
                        min(dataset[var]),
                        max(dataset[var]) + 0.25,
                        0.25
                    ),
                    norm_hist=True,
                )
            elif var_idx == 5:
                sns.distplot(
                    dataset[var],
                    kde=True,
                    ax=ax[var_idx],
                    label=label,
                    norm_hist=True,
                )
            else:
                sns.distplot(
                    dataset[var],
                    kde=True,
                    ax=ax[var_idx],
                    norm_hist=True,
                )

        ttest = scipy.stats.ttest_ind(
            datasets[0][var], datasets[1][var],
            equal_var=False
        )
        if ttest.pvalue < 0.0001:
            plabel = "****"
        elif ttest.pvalue < 0.001:
            plabel = "***"
        elif ttest.pvalue < 0.01:
            plabel = "**"
        elif ttest.pvalue < 0.05:
            plabel = "*"
        else:
            plabel = "ns"
        plabel = "$p=${0:.3g}".format(ttest.pvalue)
        if bins is None:
            bins0 = _freedman_diaconis_bins(datasets[0][var])
            bins1 = _freedman_diaconis_bins(datasets[1][var])
        else:
            bins0 = bins
            bins1 = bins
        hist0 = np.histogram(datasets[0][var], bins0, density=True)
        max0_idxs = np.argwhere(hist0[0] == np.max(hist0[0])).flatten()
        if len(max0_idxs) == 1:
            max0_x = hist0[1][max0_idxs[0]] + (hist0[1][2] - hist0[1][1]) / 2
        elif len(max0_idxs) == 2:
            max0_x = np.mean([
                hist0[1][max0_idxs[0]] + (hist0[1][2] - hist0[1][1]) / 2,
                hist0[1][max0_idxs[1]] + (hist0[1][2] - hist0[1][1]) / 2
            ])
        max0_y = np.max(hist0[0])

        hist1 = np.histogram(datasets[1][var], bins1, density=True)
        max1_idxs = np.argwhere(hist1[0] == np.max(hist1[0])).flatten()
        if len(max1_idxs) == 1:
            max1_x = hist1[1][max1_idxs[0]] + (hist1[1][2] - hist1[1][1]) / 2
        elif len(max1_idxs) == 2:
            max1_x = np.mean([
                hist1[1][max1_idxs[0]] + (hist1[1][2] - hist1[1][1]) / 2,
                hist1[1][max1_idxs[1]] + (hist1[1][2] - hist1[1][1]) / 2
            ])
        max1_y = np.max(hist1[0])

        p_x = np.mean([max0_x, max1_x])
        p_y = np.max([max0_y, max1_y]) + np.diff(ax[var_idx].get_ylim()) / 10

        ax[var_idx].text(
            p_x, p_y, plabel, ha="center", va="center", color="k", fontsize=12
        )

        var_idx += 1

    ax[3].set_xlim([0, ax[3].get_xlim()[1]])
    ax[5].legend()

    # set labels
    ax[0].set_xlabel("Initial length (\si{\micro\metre})")
    ax[1].set_xlabel("Final length (\si{\micro\metre})")
    ax[2].set_xlabel("Added length (\si{\micro\metre})")
    ax[3].set_xlabel("Interdivision time (\si{\hour})")
    ax[4].set_xlabel("Elongation rate (\si{\micro\metre\per\hour})")
    ax[5].set_xlabel("Growth rate (\si{\per\hour})")

    fig.tight_layout()
    fig.savefig("{0}.pdf".format(filename))
    plt.close()



def main():
    parser = argparse.ArgumentParser(
        description="Noisy Linear Mapper"
    )

    excl_group = parser.add_mutually_exclusive_group()
    excl_group.add_argument(
        "-p", "--poles", default=False, action="store_true",
        help="""
            split cells by pole inheritance (new/old inheritors)
        """
    )
    excl_group.add_argument(
        "-a", "--age", default=False, action="store_true",
        help="""
            split cells by age of the oldest pole
        """
    )
    parser.add_argument(
        "-c", "--comparison", default=False, action="store_true",
        help="""
            compare maps for different conditions, expects at least one
            `process_list` argument
        """
    )
    parser.add_argument(
        "-f", "--force", default=False, action="store_true",
        help="""
            force reacquisition of data
        """
    )
    parser.add_argument(
        "-d", "--debug", default=False, action="store_true",
        help="""
            debug pole assignments on division
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
    if args.comparison and len(args.process_list) < 2:
        parser.error("Comparisons require at least two inputs")
    elif args.comparison:
        raise NotImplementedError("Arbitrary comparisons haven't been written")

    if len(args.process_list) > 1:
        raise NotImplementedError("Combining datasets hasn't been implemented")
    elif args.process_list:
        dirlist = []
        sources = []
        a = json.loads(open(args.process_list[0]).read())
        for x, y in a.items():
            dirlist.extend([os.path.join(x, _) for _ in y])
            sources.extend([os.path.basename(x) for _ in y])
    else:
        sources, dirlist = None, None

    process_root(sources, dirlist, with_poles=args.poles, with_age=args.age, force=args.force, debug=args.debug)

if __name__ == "__main__":
    main()
