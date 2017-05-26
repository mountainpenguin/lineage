#!/usr/bin/env python


""" Model cell growth using a noisy linear map

    final_size = (a * initial_size) + b + noise
    L_d(n) = aL_b(n) + b + \eta

    where:
        n = generation number
        L_d(n) = cell size before division for generation n
        L_b(n) = cell size at birth for generation n
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
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns
import warnings
import textwrap
import networkx as nx
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

settings = {}


class CustomNode(object):
    def __init__(self, lineage):
        self.lineage_id = lineage.lineage_id
        self.frames = [x.frame for x in lineage.cells]
        self.times = [x.t for x in lineage.cells]
        self.lengths = [x.length for x in lineage.cells]
        self.initial_length = lineage.cells[0].length
        self.final_length = lineage.cells[-1].length
        self.initial_time = self.times[0]
        self.final_time = self.times[-1]
        self.source = os.path.basename(os.getcwd())
        self.pole1_age = lineage.pole1_age
        self.pole2_age = lineage.pole2_age
        if self.pole1_age.age_known and self.pole2_age.age_known:
            self.age_known = True
        else:
            self.age_known = False
        self.pole_age = max(self.pole1_age, self.pole2_age).age
        self.old_pole = self.pole_age > 1

        if not lineage.children:
            self.loss = True
            self.interdivision_time = None
            self.elongation_rate = None
            self.growth_rate = None
        else:
            self.loss = False
            self.interdivision_time = (
                lineage.cells[-1].t -
                lineage.cells[0].t
            ) / 60
            self.elongation_rate = self._get_elongation_rate()
            self.growth_rate = self._get_growth_rate()

    def _get_elongation_rate(self):
        # self.times, self.lengths
        gradient = np.polyfit(self.times, self.lengths, 1)[0]
        return gradient * 60  # um / hr

    def _get_growth_rate(self):
        # self.times, self.lengths
        logL = np.log(self.lengths)
        lamb, logL0 = np.polyfit(
            np.array(self.times) - self.times[0],
            logL,
            1
        )
        return lamb * 60  # um / hr

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
            )
        ).strip())


def bootstrap(data, statistic=np.mean, alpha=0.05, num_samples=100):
    """ Copied from lineage_noise.py """
    n = len(data)
    samples = np.random.choice(data, (num_samples, n))
    stat = np.sort(statistic(samples, 1))
    return (
        stat[int((alpha / 2.0) * num_samples)],
        stat[int((1 - alpha / 2.0) * num_samples)]
    )


def process_old():
    # lineage_file = np.load("lineages.npz")
    # lineage_data = lineage_file["lineages"]
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

            lin_data["length_ratio"] = (
                final_cell.length[0][0] /
                initial_cell.length[0][0]
            )
            lin_data["area_ratio"] = (
                final_cell.area[0][0] /
                initial_cell.area[0][0]
            )
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


def process_dir():
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
        if settings["with_poles"] or settings["with_age"]:
            cell_lineage = track.SingleCellLineage(
                cell_lineage.id,
                L,
                debug=settings["debug"]
            )
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
    if settings["with_poles"] or settings["with_age"]:
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
        A = np.vstack([xdata, np.ones(len(xdata))]).T
        results = sm.OLS(
            ydata, A
        ).fit()

        m, c = results.params
        Sb, Sa = results.bse
        conf = np.array(results.conf_int())
        merror = float(np.diff(conf[0]) / 2)
        cerror = float(np.diff(conf[1]) / 2)
        print("m:", m, "c:", c, "mse:", Sb, "cse:", Sa, "merror:", merror, "cerror:", cerror, "n:", len(xdata))
        r, rp = scipy.stats.pearsonr(xdata, ydata)

        return [
            (m, merror),
            (c, cerror),
            (r, rp),
        ]
    else:
        raise NotImplementedError

# def get_mstd(data):
#     xdata = data.initial_length
#     ydata = data.final_length
#     tstatistic = scipy.stats.t.ppf(0.975, df=(len(xdata) - 2))
#     A = np.vstack([xdata, np.ones(len(xdata))]).T
#     linalg = scipy.linalg.lstsq(A, ydata)
#     m, c = linalg[0]
#     sum_y_residuals = np.sum((ydata - ydata.mean()) ** 2)
#     Syx = np.sqrt(sum_y_residuals / (len(xdata) - 2))
#     sum_x_residuals = np.sum((xdata - xdata.mean()) ** 2)
#     Sb = Syx / np.sqrt(sum_x_residuals)
#
#     return (m, Sb, len(xdata))


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


def draw_annotation(g, xdata, ydata, fn):
    ((stats_m, stats_merror),
     (stats_c, stats_cerror),
     (stats_r, stats_rp)) = get_stats(xdata, ydata)
#    annotation = r"""
#a = {m} $\pm$ {me}
#b = {c} $\pm$ {ce}
#r = {r}, r$^2$ = {rsq}
#n = {n}"""
#    if fn == "noisy_linear_map" or "noisy-linear-map" in fn:
#        annotation += r"""
#$\langle L_I \rangle$ = {im}
#$\langle L_F \rangle$ = {fm}"""
#        x_ci = float(np.diff(scipy.stats.t.interval(
#            0.95,
#            len(xdata) - 1,
#            loc=xdata.mean(),
#            scale=xdata.sem()
#        ))[0])
#        y_ci = float(np.diff(scipy.stats.t.interval(
#            0.95,
#            len(ydata) - 1,
#            loc=ydata.mean(),
#            scale=ydata.sem()
#        ))[0])
#        print(
#            "<L_I>=", xdata.mean(),
#            "sd=", xdata.std(),
#            "sem=", xdata.sem(),
#            "ci=", x_ci
#        )
#        print(
#            "<L_F>=", ydata.mean(),
#            "sd=", ydata.std(),
#            "sem=", ydata.sem(),
#            "ci=", y_ci
#        )
#    annotation = annotation.format(
#        m=fmt_dec(stats_m, 5),
#        me=fmt_dec(stats_merror, 3),
#        c=fmt_dec(stats_c, 5),
#        ce=fmt_dec(stats_cerror, 3),
#        r=fmt_dec(stats_r, 3),
#        rsq=fmt_dec(stats_r ** 2, 3),
#        n=len(xdata),
#        im=fmt_dec(xdata.mean(), 4),
#        fm=fmt_dec(ydata.mean(), 4),
#    )
#    annotation += "\n"

    annotation = """
slope     = {m} $\\pm$ {me}
intercept = {c} $\\pm$ {ce}
n         = {n}
    """.format(
        m=fmt_dec(stats_m, 3),
        me=fmt_dec(stats_merror, 3),
        c=fmt_dec(stats_c, 3),
        ce=fmt_dec(stats_cerror, 3),
        n=len(xdata),
    )
    g.annotate(
        lambda x, y: (x, y),
        template="\n".join(annotation.split("\n")[1:-1]),
        loc="upper left",
        fontsize=12,
    )


def plot_joint(
    xdata, ydata,
    xlab, ylab,
    fn="noisy_linear_map",
    suffix="",
    xlim=None, ylim=None
):
    if settings["binned"]:
        # bin data by xdata
        counts, bins = np.histogram(xdata)
        bin_width = bins[1] - bins[0]

        unbinned_data = pd.DataFrame([xdata, ydata]).transpose()
        binned_data = pd.DataFrame()
        for bin_min in bins[:-1]:
            bin_max = bin_min + bin_width
            if bin_max == bins[-1]:
                upper_lim = (unbinned_data[xdata.name] <= bin_max)
            else:
                upper_lim = (unbinned_data[xdata.name] < bin_max)
            yvals = unbinned_data[
                (unbinned_data[xdata.name] >= bin_min) &
                (upper_lim)
            ][ydata.name]
            if len(yvals) >= settings["binthreshold"]:
                q1, q3 = yvals.quantile([0.25, 0.75])
                q_lower = yvals.mean() - q1
                q_upper = q3 - yvals.mean()

                # bootstrap for 95% confidence interval
                y_95 = bootstrap(yvals)

                binned_data = binned_data.append({
                    "x_centre": bin_min + (bin_width / 2),
                    "y_mean": yvals.mean(),
                    "y_std": yvals.std(),
                    "y_err": yvals.std() / np.sqrt(len(yvals)),
                    "q_25": q1,
                    "q_75": q3,
                    "q_lower": q_lower,
                    "q_upper": q_upper,
                    "y_95_lower": yvals.mean() - y_95[0],
                    "y_95_upper": y_95[1] - yvals.mean(),
                    "n": len(yvals)
                }, ignore_index=True)
        # suffix = "{0}-binned".format(suffix)
    else:
        binned_data, unbinned_data = None, None

    print("Plotting {0}{1} (xlab: {2}, ylab: {3}".format(
        fn, suffix, xlab, ylab
    ))

    # xlim_set, ylim_set = None, None
    g = sns.JointGrid(xdata, ydata, xlim=xlim, ylim=ylim)

    marginal_args = [
        sns.distplot
    ]
    marginal_kws = {
        "kde": True,
        "hist_kws": {
            "edgecolor": "k",
        },
    }

    if fn == "initial-doubling":
        marginal_kws["bins"] = np.arange(
            min(ydata),
            max(ydata) + 0.25,
            0.25
        )

    g = g.plot_marginals(*marginal_args, **marginal_kws)

    fit_kws = {
        "color": "0.1",
        "alpha": 0.7,
        "ls": "--",
    }
    scatter_kws = {
        "s": 40,
        "alpha": 0.5,
        "color": "darkred",
        "marker": ".",
    }

    if settings["binned"]:
        scatter_kws["alpha"] = 0.3
        g.ax_joint.errorbar(
            binned_data.x_centre, binned_data.y_mean, yerr=np.array(binned_data[["y_95_lower", "y_95_upper"]]).T,
            marker="o",
            ms=10,
            mec="0.1",
            mew=3,
            mfc="w",
            lw=3,
            color="0.1",
            capsize=5,
        )
    if settings["regression"]:
        m, c, r = get_stats(xdata, ydata)
        x_ = np.array(g.ax_joint.get_xlim())
        # draw line
        g.ax_joint.plot(
            x_, x_ * m[0] + c[0],
            **fit_kws
        )
        # draw CI
        ci_region = matplotlib.patches.Polygon(
            [[0, c[0] - c[1]],
             [0, c[0] + c[1]],
             [x_[-1], x_[-1] * m[0] + c[0] - c[1]],
             [x_[-1], x_[-1] * m[0] + c[0] + c[1]]],
            color="0.1",
            alpha=0.1,
        )
        g.ax_joint.add_patch(ci_region)
        g.ax_joint.set_xlim(x_)

    g.ax_joint.scatter(
        xdata,
        ydata,
        **scatter_kws
    )

    draw_annotation(g, xdata, ydata, fn)
    g.set_axis_labels(xlab, ylab, fontsize=16)
    g.ax_joint.tick_params(axis="both", which="major", labelsize=13)

    plt.tight_layout()
    plt.savefig("{0}{1}.pdf".format(fn, suffix), transparent=True)
    plt.close()


def _gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def plot_error(ax, xdata, ydata):
    fig = plt.figure()
    ax_resid = fig.add_subplot(1, 2, 1)
    ax_resid.set_title("Residuals")
    ax_resid.set_xlabel("Residual")
    ax_resid.set_ylabel("Frequency")
    ax_qq = fig.add_subplot(1, 2, 2)

    # o = np.ones((len(xdata),))
    # A = np.vstack([xdata, np.ones(len(xdata))]).T
    # result = sm.OLS(
    #     ydata, A
    # ).fit()
    # m, c = result.params
    (m, _), (c, _), _ = get_stats(xdata, ydata)
    expected = m * xdata + c
    residuals = ydata - expected
    sns.distplot(residuals, ax=ax_resid, kde=False, norm_hist=True)

    # fit Gaussian
    hist, bin_edges = np.histogram(residuals, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    p0 = [1.0, 0.0, 1.0]
    coeff, var_matrix = scipy.optimize.curve_fit(
        _gauss, bin_centres, hist, p0=p0
    )
    limits = ax_resid.get_xlim()
    xspace = np.linspace(limits[0], limits[1], 200)
    hist_fit = _gauss(xspace, *coeff)
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

    fig.savefig("noisy-noise.pdf", transparent=True)
    plt.close()


def plot_regplot(ax, xdata, ydata, xlabel, ylabel, mlabel="m", clabel="c"):
    common_kws = {
        "ci": 95,
        "scatter_kws": {
            "s": 10,
            "color": "darkred",
            "alpha": 0.5,
        },
        "marker": ".",
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


def get_custom_node(lin, nodes):
    if lin.lineage_id in nodes:
        return nodes[lin.lineage_id]
    else:
        n = CustomNode(lin)
        nodes[lin.lineage_id] = n
        return n


def cell_filter(x):
    if (
        x.timings[x.cells[-1].frame - 1] < x.rif_cut and
        len(x.cells) > 5 and
        x.children
    ):
        return True
    return False


def add_children(tree, cell, nodes):
    if cell.children:
        n0 = get_custom_node(cell, nodes)
        if cell_filter(cell.children[0]):
            n1 = get_custom_node(cell.children[0], nodes)
            n1.asymmetry_parent = n1.initial_length / n0.final_length
            n1.asymmetry = n1.initial_length / n0.final_length
            tree.add_node(n1)
            tree.add_edge(n0, n1)
            add_children(tree, cell.children[0], nodes)

        if cell_filter(cell.children[1]):
            n2 = get_custom_node(cell.children[1], nodes)
            n2.asymmetry_parent = n2.initial_length / n0.final_length
            n2.asymmetry = n2.initial_length / n0.final_length
            tree.add_node(n2)
            tree.add_edge(n0, n2)
            add_children(tree, cell.children[1], nodes)

        if cell_filter(cell.children[0]) and cell_filter(cell.children[1]):
            n1.asymmetry = n1.initial_length / (n1.initial_length + n2.initial_length)
            n2.asymmetry = n2.initial_length / (n1.initial_length + n2.initial_length)


def process_tree(dirs):
    orig_dir = os.getcwd()
    if not dirs:
        dirs = ["."]

    if os.path.exists("data-tree.pandas") and not settings["force"]:
        data = pd.read_pickle("data-tree.pandas")
    else:
        graphs = []
        for d in dirs:
            os.chdir(d)
            print("Processing {0}".format(d))
            if not os.path.exists("mt/mt.mat"):
                print("No mt file, skipping")
                continue
            timings, rif_add, px = misc.get_timings()
            L = track.Lineage()
            initial_cells = L.frames[0].cells
            # only follow cells after second division
            process_queue = []
            for init_cell in initial_cells:
                cell_lineage = track.SingleCellLineage(
                    init_cell.id,
                    L,
                    debug=settings["debug"],
                    px_conversion=px,
                    timings=timings,
                    rif_cut=rif_add
                )
                if type(cell_lineage.children) is list:
                    if type(cell_lineage.children[0].children) is list:
                        process_queue.append(cell_lineage.children[0].children[0])
                        process_queue.append(cell_lineage.children[0].children[1])
                    if type(cell_lineage.children[1].children) is list:
                        process_queue.append(cell_lineage.children[1].children[0])
                        process_queue.append(cell_lineage.children[1].children[1])
#                    process_queue.append(cell_lineage.children[0])
#                    process_queue.append(cell_lineage.children[1])

            j = 1
            nodes = {}
            for cell_lineage in process_queue:
                print("Handling cell lineage {0} ({1} of {2})".format(
                    cell_lineage.lineage_id, j, len(process_queue)
                ))
                tree = nx.DiGraph()
                root_node = get_custom_node(cell_lineage, nodes)
                root_node.asymmetry = np.NaN
                root_node.asymmetry_parent = np.NaN
                tree.add_node(root_node)
                add_children(tree, cell_lineage, nodes)
                graphs.append(tree)
                j += 1
            os.chdir(orig_dir)

        # update graphs
        for graph in graphs:
            roots = [
                node for node, degree in graph.degree().items() if degree == 2
            ]
            for node in roots:
                matchRootNode(node)
            for node in graph:
                matchDivision(node, graph.successors(node))

        # get data from graphs
        data = pd.DataFrame()
        for graph in graphs:
            for node in graph.nodes():
                if node.growth_rate and node.interdivision_time > 0.5:
                    row = pd.Series({
                        "id": node.lineage_id,
                        "initial_length": node.initial_length,
                        "final_length": node.final_length,
                        "added_length": node.final_length - node.initial_length,
                        "doubling_time": node.interdivision_time,
                        "growth_rate": node.growth_rate,
                        "elong_rate": node.elongation_rate,
                        "initial_time": node.initial_time,
                        "final_time": node.final_time,
                        "source": node.source,
                        "pole1_age": node.pole1_age,
                        "pole2_age": node.pole2_age,
                        "age_known": node.age_known,
                        "pole_age": node.pole_age,
                        "old_pole": node.old_pole,
                        "asymmetry": node.asymmetry,
                        "asymmetry_parent": node.asymmetry_parent,
                    })
                    data = data.append(row, ignore_index=True)
        data.to_pickle("data-tree.pandas")

    return data


def get_growth_curve(times, lengths):
    if len(times) < 2:
        return
    logL = np.log(lengths)
    lamb, logL0 = np.polyfit(np.array(times) - times[0], logL, 1)
    return lambda x: np.exp(logL0) * np.exp(lamb * (x - times[0]))

def matchRootNode(node):
    vol = get_growth_curve(node.times, node.lengths)
    if vol:
        node.initial_length = vol(node.times[0])

def matchDivision(mother, children):
    # written by Philipp
    if len(children) < 2:
        return None

    volM = get_growth_curve(mother.times, mother.lengths)
    vol1 = get_growth_curve(children[0].times, children[0].lengths)
    vol2 = get_growth_curve(children[1].times, children[1].lengths)
    if not (volM and vol1 and vol2):
        return

    res = scipy.optimize.minimize_scalar(
        lambda x: abs(vol1(x) + vol2(x) - volM(x)),
        bounds=(
            mother.times[-1],
            min(children[0].times[0], children[1].times[0])
        ),
        method="bounded"
    )
    tau = res.x

    vT = vol1(tau) + vol2(tau)
    children[0].initial_length = vol1(tau)
    children[1].initial_length = vol2(tau)
    mother.final_length = volM(tau)

    mother.final_time = tau
    children[0].initial_time = tau
    children[1].initial_time = tau
    mother.interdivision_time = (mother.final_time - mother.initial_time) / 60
    children[0].interdivision_time = (
        children[0].final_time - children[0].initial_time
    ) / 60
    children[1].interdivision_time = (
        children[1].final_time - children[1].initial_time
    ) / 60

    children[0].asymmetry_parent = children[0].initial_length / mother.final_length
    children[1].asymmetry_parent = children[1].initial_length / mother.final_length
    children[0].asymmetry = vol1(tau) / vT
    children[1].asymmetry = vol2(tau) / vT

def process_root(dir_sources, dirs=None):
    if not dirs:
        dirs = "."
        # dirs = list(filter(lambda x: os.path.isdir(x), sorted(os.listdir())))
        dir_sources = dirs

    if (
        (os.path.exists("data.pandas") and settings["force"]) or
        not os.path.exists("data.pandas")
    ):
        columns = [
            "source", "sub_source",
            "initial_id", "final_id",
            "initial_length", "final_length",
            "initial_area", "final_area",
            "added_length",
            "length_ratio", "area_ratio",
            "doubling_time", "growth_rate", "elong_rate",
        ]
        if settings["with_poles"]:
            columns.extend(["old_pole", "pole_age", "age_known"])

        data = pd.DataFrame(columns=columns)

        orig_dir = os.getcwd()
        i = 0
        for d in dirs:
            os.chdir(d)
            source = dir_sources[i]
            print("Processing {0}".format(d))
            if (
                os.path.exists("mt/mt.mat") or
                os.path.exists("lineages.npz")
            ):
                out_data = process_dir()
                if out_data is not None:
                    out_data["source"] = [source] * len(out_data)
                    out_data["sub_source"] = [
                        os.path.basename(d)
                    ] * len(out_data)
                    out_data["added_length"] = (
                        out_data.final_length - out_data.initial_length
                    )
                    data = pd.concat([data, out_data], ignore_index=True)
                    print("Got {0} cells".format(len(out_data)))
                else:
                    print("No cells returned")
            else:
                print("Skipping, no cells")
            i += 1
            os.chdir(orig_dir)

        if len(data) == 0:
            return

        data.to_pickle("data.pandas")
    else:
        data = pd.read_pickle("data.pandas")

    return data


def plot_data(data):
    xlab = "Birth length (\si{\micro\metre})"
    plot_joint(
        data.initial_length, data.final_length,
        xlab, "Division length (\si{\micro\metre})"
    )
    plot_error(None, data.initial_length, data.final_length)

    plot_joint(
        data.elong_rate, data.added_length,
        "Linear elongation rate (\si{\micro\metre\per\hour})",
        "Added length (\si{\micro\metre})",
        "elongation-added",
        ylim=(0, 10),
    )
    plot_joint(
        data.growth_rate, data.added_length,
        "Exponential growth rate (\si{\per\hour})",
        "Added length (\si{\micro\metre})",
        "growth-added",
        ylim=(0, 10),
    )

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
        xlab, "Linear elongation rate (\si{\micro\metre\per\hour})",
        "initial-elongation"
    )

    plot_joint(
        data.initial_length, data.growth_rate,
        xlab, "Exponential growth rate (\si{\per\hour})",
        "initial-growth"
    )

    if settings["with_age"]:
        dcolumns = ["age", "gradient", "ci", "n"]
        generation_gradient = pd.DataFrame(columns=dcolumns)
        for x in range(int(data.pole_age.max())):
            data_subset = data[(data.pole_age == (x + 1)) & (data.age_known)]
            try:
                if x <= 0 or len(data_subset) <= 1:
                    continue
                stats = get_stats(
                    data_subset.initial_length,
                    data_subset.final_length
                )
                gen_data = pd.Series({
                    "age": x + 1,
                    "gradient": stats[0][0],
                    "ci": stats[0][1],
                    "n": len(data_subset),
                    "initial_length_mean": data_subset.initial_length.mean(),
                    "initial_length_std": data_subset.initial_length.std(),
                    "initial_length_ci": float(
                        np.diff(scipy.stats.t.interval(
                            0.95,
                            len(data_subset.initial_length) - 1,
                            loc=data_subset.initial_length.mean(),
                            scale=data_subset.initial_length.sem()
                        ))[0]
                    ),
                    "growth_rate_mean": data_subset.growth_rate.mean(),
                    "growth_rate_std": data_subset.growth_rate.std(),
                    "growth_rate_ci": float(
                        np.diff(scipy.stats.t.interval(
                            0.95,
                            len(data_subset.growth_rate) - 1,
                            loc=data_subset.growth_rate.mean(),
                            scale=data_subset.growth_rate.sem()
                        ))[0]
                    ),
                })
                generation_gradient = generation_gradient.append(
                    gen_data,
                    ignore_index=True
                )
                plot_joint(
                    data_subset.initial_length, data_subset.final_length,
                    xlab, "Division length (\si{\micro\metre})",
                    "noisy-linear-map-gen-{0}".format(x + 1)
                )
            except ValueError:
                pass

        generation_gradient.to_pickle("data/generation_gradient.pandas")
        fig = plt.figure(figsize=(2.4, 2.4))
        ax = fig.add_subplot(1, 1, 1)
        sns.despine()
        err_style = {
            "fmt": "o",
            "lw": 2,
            "mew": 1
        }
        ax.errorbar(
            generation_gradient.age,
            generation_gradient.gradient,
            yerr=generation_gradient.ci,
            **err_style
        )

        # also plot "new" pole cells (i.e. pole_age = 1) and all old pole cells
        new_pole_cells = data[(data.pole_age == 1) & (data.age_known)]
        new_stats = get_stats(
            new_pole_cells.initial_length,
            new_pole_cells.final_length
        )
        ax.errorbar(
            1,
            new_stats[0][0],
            new_stats[0][1],
            **err_style
        )

        old_pole_cells = data[(data.pole_age > 1)]
        old_stats = get_stats(
            old_pole_cells.initial_length,
            old_pole_cells.final_length
        )
        ax.errorbar(
            generation_gradient.age.max() + 1,
            old_stats[0][0],
            old_stats[0][1],
            **err_style
        )

        ax.set_xlim([.5, generation_gradient.age.max() + 1.5])
        ax.set_xticks(range(1, int(generation_gradient.age.max()) + 2))
        ax.plot(ax.get_xlim(), [1, 1], linestyle="--", color="k")
        max_y = ax.get_ylim()[1]
        ax.set_ylim([0, max_y])

        ax.set_xlabel("Pole age")
        ax.set_ylabel("Slope, $a$")

        fig.canvas.draw()

        xticklabels = [x.get_text() for x in ax.get_xticklabels()]
        try:
            # new_idx = xticklabels.index("$1$")
            # xticklabels[new_idx] = "New"
            old_idx = xticklabels.index("${0}$".format(
                int(generation_gradient.age.max() + 1)
            ))
            xticklabels[old_idx] = "$>1$"
        except (IndexError, ValueError):
            print("'Manually' assigning xticklabels")
            xticklabels = ["", "$1$"]
            xticklabels.extend([
                str(x) for x in range(
                    1,
                    int(generation_gradient.age.max() + 1)
                )
            ])
            xticklabels.append("$>1$")
        ax.set_xticklabels(xticklabels)

        fig.tight_layout()
        fig.savefig(
            "noisy-linear-map-generation-gradient.pdf",
            transparent=True
        )

        # plot pole age vs initial length
        fig = plt.figure(figsize=(2.4, 2.4))
        ax = fig.add_subplot(1, 1, 1)
        sns.despine()
        ax.errorbar(
            generation_gradient.age,
            generation_gradient.initial_length_mean,
            yerr=generation_gradient.initial_length_ci,
            **err_style
        )

        ax.errorbar(
            1,
            new_pole_cells.initial_length.mean(),
            float(
                np.diff(scipy.stats.t.interval(
                    0.95,
                    len(new_pole_cells.initial_length) - 1,
                    loc=new_pole_cells.initial_length.mean(),
                    scale=new_pole_cells.initial_length.sem()
                ))[0]
            ),
            **err_style
        )
        ax.errorbar(
            generation_gradient.age.max() + 1,
            old_pole_cells.initial_length.mean(),
            float(
                np.diff(scipy.stats.t.interval(
                    0.95,
                    len(old_pole_cells.initial_length) - 1,
                    loc=old_pole_cells.initial_length.mean(),
                    scale=old_pole_cells.initial_length.sem()
                ))[0]
            ),
            **err_style
        )

        ax.set_xlim([.5, generation_gradient.age.max() + 1.5])
        ax.set_xticks(range(1, int(generation_gradient.age.max() + 2)))
        max_y = ax.get_ylim()[1]
        # ax.set_ylim([0, max_y])
        ax.set_xlabel("Pole age")
        ax.set_ylabel(r"Birth length (\si{\micro\metre})")
        ax.set_xticklabels(xticklabels)

        fig.tight_layout()
        fig.savefig(
            "noisy-linear-map-generation-initial-length.pdf",
            transparent=True
        )

        fig = plt.figure(figsize=(2.4, 2.4))
        ax = fig.add_subplot(1, 1, 1)
        sns.despine()
        ax.errorbar(
            generation_gradient.age,
            generation_gradient.growth_rate_mean,
            yerr=generation_gradient.growth_rate_ci,
            **err_style
        )

        ax.errorbar(
            1,
            new_pole_cells.growth_rate.mean(),
            float(
                np.diff(scipy.stats.t.interval(
                    0.95,
                    len(new_pole_cells.growth_rate) - 1,
                    loc=new_pole_cells.growth_rate.mean(),
                    scale=new_pole_cells.growth_rate.sem()
                ))[0]
            ),
            **err_style
        )
        ax.errorbar(
            generation_gradient.age.max() + 1,
            old_pole_cells.growth_rate.mean(),
            float(
                np.diff(scipy.stats.t.interval(
                    0.95,
                    len(old_pole_cells.growth_rate) - 1,
                    loc=old_pole_cells.growth_rate.mean(),
                    scale=old_pole_cells.growth_rate.sem()
                ))[0]
            ),
            **err_style
        )

        ax.set_xlim([.5, generation_gradient.age.max() + 1.5])
        ax.set_xticks(range(1, int(generation_gradient.age.max() + 2)))
        max_y = ax.get_ylim()[1]
        # ax.set_ylim([0, max_y])
        ax.set_xlabel("Pole age")
        ax.set_ylabel(r"Exponential growth rate (\si{\per\hour})")
        ax.set_xticklabels(xticklabels)

        fig.tight_layout()
        fig.savefig(
            "noisy-linear-map-generation-growth-rate.pdf",
            transparent=True
        )



        # plot pole data swarms
        fig, ax = plt.subplots(3, 2, figsize=(8, 12))
        sns.despine()
        ax = ax.flatten()

        age_known_data = data[data.age_known == True]
        plottable = []
        for age in range(1, int(age_known_data.pole_age.max()) + 1):
            num_records = len(age_known_data[age_known_data.pole_age == age])
            print("{0} cells with known age of {1}".format(num_records, age))
            if num_records > 2:
                plottable.append(age)

        restricted_data = age_known_data[
            age_known_data.pole_age.isin(plottable)
        ]
        # add old pole data
        old_pole_data = data[data.pole_age > 1]
        print("{0} cells with age > 1 (including unknown)".format(
            len(old_pole_data)
        ))
        old_pole_data.pole_age = restricted_data.pole_age.max() + 1
        restricted_data = restricted_data.append(old_pole_data)

        for i, y in zip(range(6), [
            "initial_length", "final_length", "added_length",
            "doubling_time", "elong_rate", "growth_rate"
        ]):
            # sns.swarmplot(
            #     x="pole_age",
            #     y=y,
            #     data=data[data.age_known == True],
            #     color="0.25",
            #     alpha=0.75,
            #     ax=ax[i],
            # )
            sns.boxplot(
                x="pole_age",
                y=y,
                data=restricted_data,
                ax=ax[i],
            )

            # plot overall data mean
            ax[i].axhline(
                restricted_data[
                    restricted_data.pole_age < restricted_data.pole_age.max()
                ][y].mean(),
                color="k",
                lw=1,
                alpha=.5,
                ls="--",
            )
            ax[i].set_xlabel("Pole age (generations)")
            xticklabels = [
                "${0}$".format(str(x))
                for x in range(1, int(restricted_data.pole_age.max()))
            ]
            xticklabels.append("$>1$")
            ax[i].set_xticklabels(xticklabels)

        ax[0].set_ylabel("Birth length (\si{\micro\metre})")
        ax[1].set_ylabel("Division length (\si{\micro\metre})")
        ax[2].set_ylabel("Added length (\si{\micro\metre})")
        ax[3].set_ylabel("Interdivision time (\si{\hour})")
        ax[4].set_ylabel("Linear elongation rate (\si{\micro\metre\per\hour})")
        ax[5].set_ylabel("Exponential growth rate (\si{\per\hour})")
        fig.tight_layout()
        sns.despine()

        fig.savefig("pole_age_boxplots.pdf", transparent=True)
        plt.close()

    elif settings["with_poles"]:
        data_new = data[((data.pole_age == 1) & (data.age_known))]
        data_old = data[data.pole_age > 1]
        plot_joint(
            data_new.initial_length, data_new.final_length,
            xlab, "Division length (\si{\micro\metre})",
            "noisy-linear-map-new-pole"
        )
        plot_joint(
            data_old.initial_length, data_old.final_length,
            xlab, "Division length (\si{\micro\metre})",
            "noisy-linear-map-old-pole"
        )

        plot_joint(
            data_new.initial_length, data_new.added_length,
            xlab, "Added length (\si{\micro\metre})",
            "initial-added-new-pole"
        )

        plot_joint(
            data_old.initial_length, data_old.added_length,
            xlab, "Added length (\si{\micro\metre})",
            "initial-added-old-pole"
        )

        # plot pole data histograms
        plot_distplot_comparisons(
            data[((data.pole_age == 1) & (data.age_known))],
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
            # sns.swarmplot(
            #     x="old_pole",
            #     y=y,
            #     data=data,
            #     color="0.25",
            #     alpha=0.75,
            #     ax=ax[i],
            # )
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
        ax[0].set_ylabel("Birth length (\si{\micro\metre})")
        ax[1].set_ylabel("Division length (\si{\micro\metre})")
        ax[2].set_ylabel("Added length (\si{\micro\metre})")
        ax[3].set_ylabel("Interdivision time (\si{\hour})")
        ax[4].set_ylabel("Linear elongation rate (\si{\micro\metre\per\hour})")
        ax[5].set_ylabel("Exponential growth rate (\si{\per\hour})")
        fig.tight_layout()
        fig.savefig("pole_boxplots.pdf", transparent=True)
        plt.close()


def plot_distplot_comparisons(
        *datasets,
        labels=None,
        filename="pole_histograms"
):
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
        shared_kws = {
            "norm_hist": True,
            "hist_kws": {
                "edgecolor": "k",
            },
        }
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
                    **shared_kws
                )
            elif var_idx == 5:
                sns.distplot(
                    dataset[var],
                    kde=True,
                    ax=ax[var_idx],
                    label=label,
                    **shared_kws
                )
            else:
                sns.distplot(
                    dataset[var],
                    kde=True,
                    ax=ax[var_idx],
                    **shared_kws
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
    ax[0].set_xlabel("Birth length (\si{\micro\metre})")
    ax[1].set_xlabel("Division length (\si{\micro\metre})")
    ax[2].set_xlabel("Added length (\si{\micro\metre})")
    ax[3].set_xlabel("Interdivision time (\si{\hour})")
    ax[4].set_xlabel("Linear elongation rate (\si{\micro\metre\per\hour})")
    ax[5].set_xlabel("Exponential growth rate (\si{\per\hour})")

    fig.tight_layout()
    fig.savefig("{0}.pdf".format(filename), transparent=True)
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
        "-b", "--binned", default=False, action="store_true",
        help="""
            bin (most) plots by initial length
        """
    )
    parser.add_argument(
        "-t", "--binthreshold", default=5, type=int,
        help="""
            add threshold number of values per bin for plotting, defaults to 5
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
        "-r", "--regression", default=False, action="store_true",
        help="""
            plot regression line on plot
        """
    )
    parser.add_argument(
        "-m", "--treemode", default=False, action="store_true",
        help="""
            use tree method for obtaining data (interpolates initial length,
            final length, and interdivision time)
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

    settings["dir_sources"] = sources
    settings["dirs"] = dirlist
    settings["with_poles"] = args.poles
    settings["with_age"] = args.age
    settings["force"] = args.force
    settings["binned"] = args.binned
    settings["binthreshold"] = args.binthreshold
    settings["debug"] = args.debug
    settings["regression"] = args.regression
    settings["treemode"] = args.treemode
    if args.treemode:
        data = process_tree(dirlist)
    else:
        data = process_root(sources, dirlist)

    print(
        "Got {0} cells".format(
            len(data)
        )
    )
    plot_data(data)



if __name__ == "__main__":
    main()
