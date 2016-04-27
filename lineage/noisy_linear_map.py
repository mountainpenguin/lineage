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


from lineage_lib import track
import numpy as np
import scipy.stats
import json
import datetime
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


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
    rif_add = _timediff(
        *timing_data["add"], t0
    )

    for d1, t1, frames in timings:
        sm = _timediff(d1, t1, t0)
        for _ in range(frames):
            T.append(sm)
            sm += pass_delay

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

            data["initial_ids"].append(lineage[0][0])
            data["final_ids"].append(lineage[-1][0])
            data["initial_lengths"].append(initial_length)
            data["final_lengths"].append(final_length)
            data["initial_areas"].append(np.NaN)
            data["final_areas"].append(np.NaN)
            data["doubling_times"].append(doubling_time)
            data["growth_rates"].append(growth_rate)
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

    data = {
        "initial_ids": initial_ids,
        "final_ids": final_ids,
        "initial_lengths": initial_lengths,
        "final_lengths": final_lengths,
        "initial_areas": initial_areas,
        "final_areas": final_areas,
        "doubling_times": doubling_times,
        "growth_rates": growth_rates,
        "length_ratios": length_ratios,
        "area_ratios": area_ratios,
    }
    return data


def process_root(dir_sources, dirs=None):
    if not dirs:
        dirs = filter(lambda x: os.path.isdir(x), sorted(os.listdir()))

    initial_ids = []
    final_ids = []
    initial_lengths = []
    final_lengths = []
    initial_areas = []
    final_areas = []
    doubling_times = []
    growth_rates = []
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
        "source": sources,
    })

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(3, 2, 1)
    sns.regplot(
        x="initial_length",
        y="final_length",
        data=data,
        ci=95,
    )
    ax.set_xlabel("Initial cell length (um)")
    ax.set_ylabel("Final cell length (um)")

    # get regression
    pf = np.polyfit(data.initial_length, data.final_length, 1)
    x = np.linspace(data.initial_length.min(), data.initial_length.max(), 50)
    y = pf[0] * x + pf[1]
    pearson_r, pearson_p = scipy.stats.pearsonr(data.initial_length, data.final_length)
    plt.plot(x, y, color="none", alpha=1, label="a = {0:.5f}".format(pf[0]))
    plt.plot(x, y, color="none", alpha=1, label="b = {0:.5f}".format(pf[1]))
    plt.plot(x, y, color="none", alpha=1, label="r = {0:.5f}".format(pearson_r))
    plt.plot(x, y, color="none", alpha=1, label="n = {0}".format(len(data)))
    plt.legend(loc=2)
    sns.despine()

    ax = fig.add_subplot(3, 2, 2)
    if not np.isnan(data.initial_area.values[0]):
        sns.regplot(
            x="initial_area",
            y="final_area",
            data=data,
            ci=95,
        )
        ax.set_xlabel("Initial cell area (um^2)")
        ax.set_ylabel("Final cell area (um^2)")
        # get regression
        pf = np.polyfit(data.initial_area.dropna(), data.final_area.dropna(), 1)
        x = np.linspace(data.initial_area.min(), data.initial_area.max(), 50)
        y = pf[0] * x + pf[1]
        pearson_r, pearson_p = scipy.stats.pearsonr(data.initial_area.dropna(), data.final_area.dropna())
        plt.plot(x, y, color="none", alpha=1, label="a = {0:.5f}".format(pf[0]))
        plt.plot(x, y, color="none", alpha=1, label="b = {0:.5f}".format(pf[1]))
        plt.plot(x, y, color="none", alpha=1, label="r = {0:.5f}".format(pearson_r))
        plt.plot(x, y, color="none", alpha=1, label="n = {0}".format(len(data.initial_area.dropna())))
        plt.legend(loc=2)
    ax.set_xlabel("Initial cell area (um^2)")
    ax.set_ylabel("Final cell area (um^2)")

    sns.despine()

    ax = fig.add_subplot(3, 2, 3)
    sns.distplot(
        data.doubling_time, kde=False
    )
    ax.set_xlabel("Doubling Time (h)")
    sns.despine()

    ax = fig.add_subplot(3, 2, 4)
    sns.distplot(
        data.growth_rate, kde=False
    )
    ax.set_xlabel("Growth Rate (h^{-1})")
    sns.despine()

    ax = fig.add_subplot(3, 2, 5)
    sns.distplot(
        data.length_ratio, kde=False
    )
    ax.set_xlabel("Length ratio (L_F / L_I)")
#    sns.regplot(
#        x="doubling_time",
#        y="length_ratio",
#        data=data,
#        ci=95
#    )
#    # get regression
#    pf = np.polyfit(data.doubling_time, data.length_ratio, 1)
#    x = np.linspace(data.doubling_time.min(), data.doubling_time.max(), 50)
#    y = pf[0] * x + pf[1]
#    pearson_r, pearson_p = scipy.stats.pearsonr(data.doubling_time, data.length_ratio)
#    plt.plot(x, y, color="none", alpha=1, label="y = mx + c")
#    plt.plot(x, y, color="none", alpha=1, label="m = {0:.5f}".format(pf[0]))
#    plt.plot(x, y, color="none", alpha=1, label="x = {0:.5f}".format(pf[1]))
#    plt.plot(x, y, color="none", alpha=1, label="r = {0:.5f}".format(pearson_r))
#    plt.plot(x, y, color="none", alpha=1, label="n = {0}".format(len(data)))
#    plt.legend(loc=2)
    sns.despine()

    ax = fig.add_subplot(3, 2, 6)

    if not np.isnan(data.initial_area.values[0]):
        sns.distplot(
            data.area_ratio.dropna(), kde=False
        )
    ax.set_xlabel("Area ratio (A_F / A_I)")
#    sns.regplot(
#        x="growth_rate",
#        y="length_ratio",
#        data=data,
#        ci=95
#    )
#    # get regression
#    pf = np.polyfit(data.growth_rate, data.length_ratio, 1)
#    x = np.linspace(data.growth_rate.min(), data.growth_rate.max(), 50)
#    y = pf[0] * x + pf[1]
#    pearson_r, pearson_p = scipy.stats.pearsonr(data.growth_rate, data.length_ratio)
#    plt.plot(x, y, color="none", alpha=1, label="y = mx + c")
#    plt.plot(x, y, color="none", alpha=1, label="m = {0:.5f}".format(pf[0]))
#    plt.plot(x, y, color="none", alpha=1, label="x = {0:.5f}".format(pf[1]))
#    plt.plot(x, y, color="none", alpha=1, label="r = {0:.5f}".format(pearson_r))
#    plt.plot(x, y, color="none", alpha=1, label="n = {0}".format(len(data)))
#    plt.legend(loc=2)
    sns.despine()

    plt.tight_layout()
    plt.savefig("noisy_linear_map.pdf")

    plt.close()
    for source in np.unique(dir_sources):
        vars = [
            "initial_length", "final_length", "length_ratio",
            "doubling_time", "growth_rate"
        ]
        if not np.isnan(data[(data.source == source)].initial_area.values[0]):
            vars = [
                "initial_length", "final_length", "length_ratio",
                "initial_area", "final_area", "area_ratio",
                "doubling_time", "growth_rate"
            ]

        fig = plt.figure()
        sns.pairplot(
            data[(data.source == source)],
            vars=vars,
        )
        plt.tight_layout()
        plt.savefig("{0}.pdf".format(source))
        plt.close()

    fig = plt.figure()
    sns.pairplot(
        data,
        vars=[
            "initial_length", "final_length", "length_ratio",
            "doubling_time", "growth_rate",
        ],
        hue="source"
    )
    plt.tight_layout()
    plt.savefig("all_data.pdf")

    data.to_pickle("data.pandas")


def main():
    try:
        process_list = sys.argv[1]
    except IndexError:
        sources = None
        dirlist = None
    else:
        a = json.loads(open(process_list).read())
        dirlist = []
        sources = []
        for x, y in a.items():
            dirlist.extend([os.path.join(x, _) for _ in y])
            sources.extend([os.path.basename(x) for _ in y])

    process_root(sources, dirlist)


if __name__ == "__main__":
    main()
