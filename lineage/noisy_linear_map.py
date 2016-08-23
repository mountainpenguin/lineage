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
sns.set_style("white")
sns.set_context("paper")

plt.rc("font", family="sans-serif")
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
    r"\setlength\parindent{0pt}",
]


def _gettimestamp(day, time, *args):
    return datetime.datetime.strptime(
        "{0} {1}".format(day, time),
        "%d.%m.%y %H:%M"
    )


def _timediff(day, time, t0):
    t1 = _gettimestamp(day, time)
    td = t1 - t0
    s = td.days * 24 * 60 * 60
    s += td.seconds
    m = s // 60
    return m


def get_timings():
    timing_data = json.loads(open("timings.json").read())
    timings = timing_data["timings"]
    try:
        pass_delay = timing_data["pass_delay"]
    except KeyError:
        pass_delay = 15
    pixel = timing_data["px"]
    T = []
    t0 = _gettimestamp(*timings[0])
    if "add" in timing_data:
        rif_add = _timediff(
            *timing_data["add"], t0
        )
    else:
        rif_add = _timediff(
            timings[-1][0], timings[-1][1], t0
        ) + timings[-1][2] * pass_delay

    for d1, t1, frames in timings:
        frame_time = _timediff(d1, t1, t0)
        for _ in range(frames):
            T.append(frame_time)
            frame_time += pass_delay

    return T, rif_add, pixel


def process_old():
#    lineage_file = np.load("lineages.npz")
#    lineage_data = lineage_file["lineages"]
    lineage_info = json.loads(open("lineages.json").read())
    T, rif_add, pixel = get_timings()

    data = {
        "initial_ids": [],
        "final_ids": [],
        "initial_lengths": [],
        "final_lengths": [],
        "initial_areas": [],
        "final_areas": [],
        "doubling_times": [],
        "growth_rates": [],
        "elong_rates": [],
        "length_ratios": [],
        "area_ratios": [],
    }
    for lin in lineage_info:
        if lin["parent"] and lin["children"] and len(lin["lineage"]) > 5:
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

            data["initial_ids"].append(lineage[0][0])
            data["final_ids"].append(lineage[-1][0])
            data["initial_lengths"].append(initial_length)
            data["final_lengths"].append(final_length)
            data["initial_areas"].append(np.NaN)
            data["final_areas"].append(np.NaN)
            data["doubling_times"].append(doubling_time)
            data["growth_rates"].append(growth_rate)
            data["elong_rates"].append(elong_rate)
            data["length_ratios"].append(length_ratio)
            data["area_ratios"].append(np.NaN)

    return data


def process_dir():
    try:
        L = track.Lineage()
    except:
        print("Error getting lineage information")
        if os.path.exists("lineages.npz"):
            print("But lineages.npz exists")
            return process_old()
        return {}
    initial_cells = L.frames[0].cells
    # only follow cells after first division
    process_queue = []
    for cell in initial_cells:
        while type(cell.children) is str:
            cell = L.frames.cell(cell.children)

        if type(cell.children) is list:
            process_queue.append(L.frames.cell(cell.children[0]))
            process_queue.append(L.frames.cell(cell.children[1]))

    initial_ids = []
    final_ids = []
    initial_lengths = []
    final_lengths = []
    length_ratios = []
    initial_areas = []
    final_areas = []
    area_ratios = []
    doubling_times = []
    growth_rates = []
    elong_rates = []

    T, rif_add, pixel = get_timings()
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
            process_queue.append(L.frames.cell(cell.children[0]))
            process_queue.append(L.frames.cell(cell.children[1]))

            initial_length = initial_cell.length[0][0] * pixel
            final_length = cell.length[0][0] * pixel

            initial_area = initial_cell.area[0][0] * pixel * pixel
            final_area = cell.area[0][0] * pixel * pixel

            initial_ids.append(initial_cell.id)
            initial_lengths.append(initial_length)
            initial_areas.append(initial_area)
            final_ids.append(cell.id)
            final_lengths.append(final_length)
            final_areas.append(final_area)

            length_ratios.append(final_length / initial_length)
            area_ratios.append(final_area / initial_area)

            doubling_times.append((T[cell.frame - 1] - T[initial_cell.frame - 1]) / 60)

            times = np.array(times)
            logL = np.log(lengths)
            growth_rate, logL0 = np.polyfit(times - times[0], logL, 1)
            growth_rates.append(growth_rate)

            elong_rate, _ = np.polyfit(times, lengths, 1)
            elong_rates.append(elong_rate)

    data = {
        "initial_ids": initial_ids,
        "final_ids": final_ids,
        "initial_lengths": initial_lengths,
        "final_lengths": final_lengths,
        "initial_areas": initial_areas,
        "final_areas": final_areas,
        "doubling_times": doubling_times,
        "growth_rates": growth_rates,
        "elong_rates": elong_rates,
        "length_ratios": length_ratios,
        "area_ratios": area_ratios,
    }

    return data

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
                pearson r^2,

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
            (r ** 2, rp),
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
    (stats_m, stats_merr), (stats_c, stats_cerr), (rsq, rpval) = get_stats(
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
    plot_fake(ax, r"r$^2$ = {rsq:.3g}".format(
        rsq=rsq,
        rpval=fmt_dec(rpval, 3),
    ))
    plot_fake(ax, "n = {0}".format(len(xdata)))


def plot_joint(xdata, ydata, xlab, ylab, fn="noisy_linear_map", suffix=""):
    fig = plt.figure()
    g = sns.jointplot(
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
    ((stats_m, stats_merror),
     (stats_c, stats_cerror),
     (stats_r2, stats_rp)) = get_stats(xdata, ydata)
    annotation = r"""
a = {m} $\pm$ {me},\newline
b = {c} $\pm$ {ce},\newline
r$^2$ = {rsq}, p = {rp}\newline
n = {n}
    """.format(
        m=fmt_dec(stats_m, 5),
        me=fmt_dec(stats_merror, 3),
        c=fmt_dec(stats_c, 5),
        ce=fmt_dec(stats_cerror, 3),
        rsq=fmt_dec(stats_r2, 3),
        rp=fmt_dec(stats_rp, 3),
        n=len(xdata),
    )
    annotation = "\n".join(annotation.split("\n")[1:-1])
    g.annotate(
        lambda x,y: (x,y),
        template=annotation,
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


def process_root(dir_sources, dirs=None):
    if not dirs:
        dirs = "."
#        dirs = list(filter(lambda x: os.path.isdir(x), sorted(os.listdir())))
        dir_sources = dirs

    initial_ids = []
    final_ids = []
    initial_lengths = []
    final_lengths = []
    initial_areas = []
    final_areas = []
    doubling_times = []
    growth_rates = []
    elong_rates = []
    length_ratios = []
    area_ratios = []
    sources = []

    orig_dir = os.getcwd()
    i = 0
    for d in dirs:
        os.chdir(d)
        source = dir_sources[i]
        print("Processing {0}".format(d))
        if os.path.exists("mt/alignment.mat") or os.path.exists("lineages.npz"):
            out_data  = process_dir()
            if out_data:
                initial_ids.extend(out_data["initial_ids"])
                final_ids.extend(out_data["final_ids"])
                initial_lengths.extend(out_data["initial_lengths"])
                final_lengths.extend(out_data["final_lengths"])
                initial_areas.extend(out_data["initial_areas"])
                final_areas.extend(out_data["final_areas"])
                doubling_times.extend(out_data["doubling_times"])
                elong_rates.extend(out_data["elong_rates"])
                growth_rates.extend(out_data["growth_rates"])
                length_ratios.extend(out_data["length_ratios"])
                area_ratios.extend(out_data["area_ratios"])
                sources.extend([source] * len(out_data["initial_ids"]))
                print("Got {0} cells".format(len(out_data["initial_ids"])))
            else:
                print("No cells returned")
        else:
            print("Skipping, no cells")
        i += 1
        os.chdir(orig_dir)

    print("Got {0} cells that divide twice during observation period".format(len(initial_ids)))
    if len(initial_ids) == 0:
        return

    data = pd.DataFrame({
        "initial_id": initial_ids,
        "final_id": final_ids,
        "initial_length": initial_lengths,
        "final_length": final_lengths,
        "length_ratio": length_ratios,
        "initial_area": initial_areas,
        "final_area": final_areas,
        "area_ratio": area_ratios,
        "doubling_time": doubling_times,
        "growth_rate": growth_rates,
        "elong_rate": elong_rates,
        "source": sources,
    })

    xlab = "Initial cell length (\si{\micro\metre})"
    plot_joint(
        data.initial_length, data.final_length,
        xlab, "Final cell length (\si{\micro\metre})"
    )
    plot_error(None, data.initial_length, data.final_length)

    plot_joint(
        data.initial_length, data.final_length - data.initial_length,
        xlab, "Added Length (\si{\micro\metre})",
        "initial-added"
    )
    plot_joint(
        data.initial_length, data.doubling_time,
        xlab, "Doubling Time (\si{\hour})",
        "initial-doubling"
    )

    plot_joint(
        data.initial_length, data.elong_rate,
        xlab, "Elongation Rate (\si{\micro\metre\per\hour})",
        "initial-elongation"
    )

    plot_joint(
        data.initial_length, data.growth_rate,
        xlab, "Growth Rate (\si{\per\hour})",
        "initial-growth"
    )
    plt.close()
    data.to_pickle("data.pandas")


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
