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

    return T, rif_add


def process_dir():
    try:
        L = track.Lineage()
    except:
        print("Error getting lineage information")
        return [], [], [], []
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

    T, rif_add = get_timings()
    for cell in process_queue:
        num_frames = 1
        # get daughters
        initial_cell = L.frames.cell(cell.id)
        lengths = []
        times = []
        while type(cell.children) is str:
            lengths.append(cell.length[0][0])
            times.append(T[cell.frame - 1] / 60)

            cell = L.frames.cell(cell.children)
            num_frames += 1
            if T[cell.frame - 1] > rif_add:
                cell.children = None

        if type(cell.children) is list and num_frames > 5:
            process_queue.append(L.frames.cell(cell.children[0]))
            process_queue.append(L.frames.cell(cell.children[1]))

            initial_ids.append(initial_cell.id)
            initial_lengths.append(initial_cell.length[0][0])
            initial_areas.append(initial_cell.area[0][0])
            final_ids.append(cell.id)
            final_lengths.append(cell.length[0][0])
            final_areas.append(cell.area[0][0])

            length_ratios.append(cell.length[0][0] / initial_cell.length[0][0])
            area_ratios.append(cell.area[0][0] / initial_cell.area[0][0])

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


def process_root(dirs=None):
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

    orig_dir = os.getcwd()
    for d in dirs:
        os.chdir(d)
        print("Processing {0}".format(d))
        if os.path.exists("mt/alignment.mat"):
            out_data  = process_dir()
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
            print("Got {0} cells".format(len(out_data["initial_ids"])))
        else:
            print("Skipping, no cells")
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
    })

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(3, 2, 1)
    sns.regplot(
        x="initial_length",
        y="final_length",
        data=data,
        ci=95,
    )
    ax.set_xlabel("Initial cell length (px)")
    ax.set_ylabel("Final cell length (px)")

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
    sns.regplot(
        x="initial_area",
        y="final_area",
        data=data,
        ci=95,
    )
    ax.set_xlabel("Initial cell area (px^2)")
    ax.set_ylabel("Final cell area (px^2)")
    # get regression
    pf = np.polyfit(data.initial_area, data.final_area, 1)
    x = np.linspace(data.initial_area.min(), data.initial_area.max(), 50)
    y = pf[0] * x + pf[1]
    pearson_r, pearson_p = scipy.stats.pearsonr(data.initial_area, data.final_area)
    plt.plot(x, y, color="none", alpha=1, label="a = {0:.5f}".format(pf[0]))
    plt.plot(x, y, color="none", alpha=1, label="b = {0:.5f}".format(pf[1]))
    plt.plot(x, y, color="none", alpha=1, label="r = {0:.5f}".format(pearson_r))
    plt.plot(x, y, color="none", alpha=1, label="n = {0}".format(len(data)))
    plt.legend(loc=2)
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
    sns.regplot(
        x="doubling_time",
        y="length_ratio",
        data=data,
        ci=95
    )
    # get regression
    pf = np.polyfit(data.doubling_time, data.length_ratio, 1)
    x = np.linspace(data.doubling_time.min(), data.doubling_time.max(), 50)
    y = pf[0] * x + pf[1]
    pearson_r, pearson_p = scipy.stats.pearsonr(data.doubling_time, data.length_ratio)
    plt.plot(x, y, color="none", alpha=1, label="y = mx + c")
    plt.plot(x, y, color="none", alpha=1, label="m = {0:.5f}".format(pf[0]))
    plt.plot(x, y, color="none", alpha=1, label="x = {0:.5f}".format(pf[1]))
    plt.plot(x, y, color="none", alpha=1, label="r = {0:.5f}".format(pearson_r))
    plt.plot(x, y, color="none", alpha=1, label="n = {0}".format(len(data)))
    plt.legend(loc=2)
    sns.despine()

    ax = fig.add_subplot(3, 2, 6)
    sns.regplot(
        x="growth_rate",
        y="length_ratio",
        data=data,
        ci=95
    )
    # get regression
    pf = np.polyfit(data.growth_rate, data.length_ratio, 1)
    x = np.linspace(data.growth_rate.min(), data.growth_rate.max(), 50)
    y = pf[0] * x + pf[1]
    pearson_r, pearson_p = scipy.stats.pearsonr(data.growth_rate, data.length_ratio)
    plt.plot(x, y, color="none", alpha=1, label="y = mx + c")
    plt.plot(x, y, color="none", alpha=1, label="m = {0:.5f}".format(pf[0]))
    plt.plot(x, y, color="none", alpha=1, label="x = {0:.5f}".format(pf[1]))
    plt.plot(x, y, color="none", alpha=1, label="r = {0:.5f}".format(pearson_r))
    plt.plot(x, y, color="none", alpha=1, label="n = {0}".format(len(data)))
    plt.legend(loc=2)

    sns.despine()

    plt.tight_layout()

    plt.savefig("noisy_linear_map.pdf")

    plt.close()

    fig = plt.figure()
    sns.pairplot(
        data,
        vars=[
            "initial_length", "final_length", "length_ratio",
            "initial_area", "final_area", "area_ratio",
            "doubling_time", "growth_rate",
        ],
    )
    plt.tight_layout()
    plt.savefig("all_data.pdf")


def main():
    try:
        process_list = sys.argv[1]
    except IndexError:
        dirlist = None
    else:
        a = json.loads(open(process_list).read())
        dirlist = []
        for x, y in a.items():
            dirlist.extend([os.path.join(x, _) for _ in y])

    process_root(dirlist)


if __name__ == "__main__":
    main()
