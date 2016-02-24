#!/usr/bin/env python

import os
import lineage_plot
import xlwt
import arial10
import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")
import matplotlib.pyplot as plt
import sys


class Sheet(object):
    def __init__(self, sheet):
        self.sheet = sheet
        self.widths = {}

    def write(self, r, c, label="", *args, **kwargs):
        self.sheet.write(r, c, label, *args, **kwargs)
        width = int(arial10.fitwidth(str(label)))
        if width > self.widths.get(c, 0):
            self.widths[c] = width
            self.sheet.col(c).width = width

    def __getattr__(self, attr):
        return getattr(self.sheet, attr)


def decorate_top_sheet(sheet):
    headerrows = [
        (0, "Doubling Time"),
        (8, "Elongation Rate"),
        (16, "Division Length"),
        (24, "Cell Length"),
        (32, "Mini-cells")
    ]
    for rownum, desc in headerrows:
        sheet.write(rownum, 0, desc)
        i = 1
        for x in ["n", "mean", "SD", "SEM", "95%", "unit"]:
            sheet.write(rownum + i, 0, x)
            i += 1


def add_data(sheet, colnum, heading, *args):
    rownums = [0, 8, 16, 24, 32]
    i = 0
    for n, mean, std, sem, ci, unit in args:
        rownum = rownums[i]
        sheet.write(
            rownum, colnum, heading
        )
        ii = 1
        for what in [n, mean, std, sem, ci, unit]:
            sheet.write(rownum + ii, colnum, what)
            ii += 1
        i += 1


def add_master_data(sheet, *args):
    sheet.write(0, 1, "n")
    sheet.write(0, 2, "mean")
    sheet.write(0, 3, "SD")
    sheet.write(0, 4, "SEM")
    sheet.write(0, 5, "95%")
    sheet.write(0, 6, "unit")

    sheet.write(1, 0, "Doubling Time")
    sheet.write(2, 0, "Elongation Rate")
    sheet.write(3, 0, "Division Length")
    sheet.write(4, 0, "Cell Length")
    sheet.write(5, 0, "Mini-cells")

    rownum = 1
    for n, mean, std, sem, ci, unit in args:
        colnum = 1
        for x in [n, mean, std, sem, ci, unit]:
            sheet.write(rownum, colnum, x)
            colnum += 1
        rownum += 1


def get_mini_data(end_length):
    cl = np.array(end_length)
    n = len(cl)
    m = cl.mean()
    std = cl.std()
    sem = scipy.stats.sem(end_length).flatten()[0]
    ci = sem * scipy.stats.t.ppf(1.95/2, n - 1)
    unit = "\u03BCm"

    nmini = len(cl[cl < 2.5])
    minival = (nmini / n) * 100
    miniunit = "%"
    return [
        (n, m, std, sem, ci, unit),
        (nmini, minival, "", "", "", miniunit)
    ]


def get_data(P):
    d = get_mini_data(P.end_length)
    return [
        P.doubling_time.get_all_data()[0][1:],
        P.growth_rate.get_all_data()[0][1:],
        P.div_length.get_all_data()[0][1:],
        d[0],
        d[1],
    ]


def get_master_data(P):
    labels = [
        "doubling_time",
        "growth_rate",
        "div_length",
        "end_length",
    ]
    full_data = []

    for label in labels:
        data = getattr(P, label)
        if type(data) is not list:
            data = np.array(data.all_data).flatten()
        else:
            data = np.array(data)

        full_data.append(
            (label, pd.Series(data))
        )

    cl = np.array(P.end_length)
    full_data.append(
        ("mini_cells", (len(cl[cl < 2.5]) / len(cl)) * 100)
    )
    return dict(full_data)


def plot_master_data(doubling, elong, div, end, mini, nvalues):
    labels = [
        "Doubling Time",
        "Elongation Rate",
        "Division Length",
        "Endpoint Length",
    ]
    full_data = [
        doubling,
        elong,
        div,
        end
    ]
    units = [
        "\si{\hour}",
        "\si{\micro\metre\per\hour}",
        "\si{\micro\metre}",
        "\si{\micro\metre}"
    ]
    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 10.5))
    # 203mm x 140mm
    # 8in x 5.5in
    i = 1
    for label, data, unit in zip(labels, full_data, units):
        plt.subplot(3, 2, i)
        ax = sns.boxplot(data=data)
#        ax = sns.stripplot(data=data, jitter=True, color="0.3", edgecolor="none")
#        ax = sns.swarmplot(data=data, color=".25", alpha=0.6)
        _, labels = plt.xticks()
        ax.set_xticklabels(labels, rotation=90)
        sns.despine()
        plt.ylabel("{0} ({1})".format(label, unit))
        # plt.title(label)
        i += 1

    labels = [
        "Mini-cells",
        "Cells Analysed",
    ]
    full_data = [
        mini,
        nvalues,
    ]
    units = [
        "\si{\percent}",
        None
    ]
    for label, data, unit in zip(labels, full_data, units):
        plt.subplot(3, 2, i)
        ax = sns.barplot(x=data.index, y=data.values)
        _, labels = plt.xticks()
        ax.set_xticklabels(labels, rotation=90)
        sns.despine()
        if not unit:
            plt.ylabel(label)
        else:
            plt.ylabel("{0} ({1})".format(label, unit))
        i += 1
    plt.tight_layout()
    plt.savefig(os.path.join("lineage_output", "data.pdf"))
    plt.close()


def run(indirs, outdir):
    default_kwargs = {
        "method": "gradient",
        "suffix": "",
        "phases": False,
        "print_data": False,
        "write_excel": False,
    }
    kwargs1 = dict(default_kwargs)
    kwargs1["write_pdf"] = True

    kwargs2 = dict(default_kwargs)
    kwargs2["write_pdf"] = False

    # generate lineage-plot for subdir
    # rename to subdir
    # if len(subdir) > 1:
    #   generate lineage-plot for topdir
    #   rename to topdir
    # generate lineage-plot for all data
    # rename to alldata

    master_wb = xlwt.Workbook()
    master_sheet = Sheet(master_wb.add_sheet("All"))
    if not os.path.exists("lineage_output"):
        os.mkdir("lineage_output")
    if not os.path.exists(os.path.join("lineage_output", outdir)):
        os.mkdir(os.path.join("lineage_output", outdir))

    masterpaths = []
    for topdir, subdirs in indirs:
        if type(topdir) is tuple:
            topdir, toplabel = topdir
        else:
            toplabel = topdir
        top_sheet = Sheet(master_wb.add_sheet(toplabel))
        decorate_top_sheet(top_sheet)

        toppaths = []
        colnum = 2
        for subdir in subdirs:
            path = [os.path.join(topdir, subdir)]
            toppaths.extend(path)
            masterpaths.extend(path)
            P = lineage_plot.Plotter(path, **kwargs1)
            P.start()

            os.rename("growth-traces.pdf", os.path.join(
                "lineage_output", outdir, "{0}-{1}.pdf".format(
                    topdir, subdir
                )
            ))

            add_data(top_sheet, colnum, subdir, *get_data(P))
            colnum += 1

        if len(toppaths) > 1:
            P = lineage_plot.Plotter(toppaths, **kwargs1)
            P.start()
            os.rename("growth-traces.pdf", os.path.join(
                "lineage_output", outdir, "{0}.pdf".format(topdir)
            ))

        elif len(toppaths) == 1:
            os.rename(
                os.path.join("lineage_output", outdir, "{0}-{1}.pdf".format(
                    topdir, subdir
                )),
                os.path.join("lineage_output", outdir, "{0}.pdf".format(
                    topdir
                ))
            )

        if len(toppaths) >= 1:
            # write 'all' data
            add_data(top_sheet, 1, "All", *get_data(P))

    if len(masterpaths) > 1:
        # generate 'all' data
        P = lineage_plot.Plotter(masterpaths, **kwargs1)
        P.start()
        os.rename("growth-traces.pdf", os.path.join(
            "lineage_output", outdir, "{0}.pdf".format(outdir)
        ))
    elif len(masterpaths) == 1:
        os.rename(
            os.path.join("lineage_output", outdir, "{0}.pdf".format(topdir)),
            os.path.join("lineage_output", outdir, "{0}.pdf".format(outdir))
        )

    if len(masterpaths) >= 1:
        add_master_data(master_sheet, *get_data(P))
        master_wb.save(
            os.path.join("lineage_output", outdir, "{0}.xls".format(outdir))
        )

    return get_master_data(P)


if __name__ == "__main__":
    dirlist = {
        "Other": [
            ["14-7-15", ["4-1", "9-1", "10-2", "12-1", "13-2", "16-1"]],
            ["120815", ["1-1", "9-2", "12-2"]],
            ["delParA pMENDAB_2", ["zoom 1", "zoom 3", "zoom 4", "zoom 5"]],
            ["delParA pMENDAB_3", ["zoom 1", "zoom 2"]],
            ["parAB 14-7-7 miles", ["09"]],
            ["ParAB 12", ["zoom 1"]],
        ],
        # non-random
#        "delParA pMENDAB": [
#            [("delParA pMENDAB_2", "2"), ["zoom 1", "zoom 3", "zoom 4", "zoom 5"]],
#            [("delParA pMENDAB_3", "3"), ["zoom 1", "zoom 2"]],
#        ],
#        "ParAB": [
#            [("parAB 14-7-7 miles", "14-7-7"), ["09"]],
#            [("ParAB 12", "12"), ["zoom 1"]],
#        ],
#        "other": [
#            ["14-7-15", ["4-1", "9-1", "10-2", "12-1", "13-2", "16-1"]],
#            ["120815", ["1-1", "9-2", "12-2"]],
#        ],
        # end
        "WT ParA int": [
            [("WT ParA int", "3"), [
                "ParA int 3-1", "ParA int 3-2",
                "ParA int 3-3", "ParA int 3-4",
                "ParA int 3-5",
            ]],
            [("WT ParA int", "4"), [
                "ParA int 4-1", "ParA int 4-2",
                "ParA int 4-3", "ParA int 4-4",
            ]],
        ],
        "WT ParB int": [
            [("WT ParB int", "6"), ["6-1", "6-2"]],
            [("WT ParB int", "7"), ["7-1", "7-2"]],
            [("WT ParB int", "8"), ["8-1"]],
            [("WT ParB int", "13"), ["13-1"]],
        ],
        "WT ParAB int": [
            [("WT ParAB int", "10"), [
                "pMENDAB_10-1", "pMENDAB_10-2", "pMENDAB_10-3", "pMENDAB_10-4",
                "pMENDAB_10-5", "pMENDAB_10-6", "pMENDAB_10-7",
            ]],
            [("WT ParAB int", "int 10"), [
                "WT ParAB int 10",
            ]],
        ],
        "WT episomal ParB": [
            [("WT episomal ParB", "WT pstB 5"), ["WT pstB 5"]],
        ]
    }

    backup_path = "lineage_output/.backup"
    if not os.path.exists("lineage_output"):
        os.mkdir("lineage_output")
    if not os.path.exists("lineage_output/.backup"):
        os.mkdir("lineage_output/.backup")

    if not os.path.exists("lineage_output/.backup/doubling.pkl") or "-s" in sys.argv:
        doubling = {}
        elong = {}
        div = {}
        end = {}
        mini = {}
        nvalues = {}
        for outdir, indirs in dirlist.items():
            data = run(indirs, outdir)

            doubling[outdir] = data["doubling_time"]
            elong[outdir] = data["growth_rate"]
            div[outdir] = data["div_length"]
            end[outdir] = data["end_length"]
            mini[outdir] = data["mini_cells"]
            nvalues[outdir] = len(elong[outdir])

        doubling = pd.DataFrame(doubling)
        doubling.to_pickle("lineage_output/.backup/doubling.pkl")
        elong = pd.DataFrame(elong)
        elong.to_pickle("lineage_output/.backup/elongation.pkl")
        div = pd.DataFrame(div)
        div.to_pickle("lineage_output/.backup/div.pkl")
        end = pd.DataFrame(end)
        end.to_pickle("lineage_output/.backup/end.pkl")
        mini = pd.Series(mini)
        mini.to_pickle("lineage_output/.backup/mini.pkl")
        nvalues = pd.Series(nvalues)
        nvalues.to_pickle("lineage_output/.backup/nvalues.pkl")
    else:
        doubling = pd.read_pickle("lineage_output/.backup/doubling.pkl")
        elong = pd.read_pickle("lineage_output/.backup/elongation.pkl")
        div = pd.read_pickle("lineage_output/.backup/div.pkl")
        end = pd.read_pickle("lineage_output/.backup/end.pkl")
        mini = pd.read_pickle("lineage_output/.backup/mini.pkl")
        nvalues = pd.read_pickle("lineage_output/.backup/nvalues.pkl")

    plot_master_data(doubling, elong, div, end, mini, nvalues)
