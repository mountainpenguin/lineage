#!/usr/bin/env python

from __future__ import print_function
from lineage_lib import track
import numpy as np
# import matplotlib.patches
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import datetime
import scipy.optimize
import scipy.stats
import argparse
import xlwt
plt.rc("font", family="sans-serif")
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
]
import seaborn as sns
sns.set_style("white")

if not Axes3D:
    print("Uh oh")


class Storage(object):
    def __init__(self, description, unit):
        self.description = description
        self.unit = unit
        self.all_data = []
        self.phase1 = []
        self.phase2 = []
        self.phase3 = []

    def set_phase_boundaries(self, p2, p3):
        self.p2 = p2
        self.p3 = p3

    def append(self, x):
        self.all_data.append(x)

    def append_p1(self, x):
        self.phase1.append(x)

    def append_p2(self, x):
        self.phase2.append(x)

    def append_p3(self, x):
        self.phase3.append(x)

    def append_p(self, cell, x):
        if not self.p2 or not self.p3:
            raise NotImplementedError
        if cell.frame <= self.p2:
            self.phase1.append(x)
        elif cell.frame > self.p2 and cell.frame <= self.p3:
            self.phase2.append(x)
        else:
            self.phase3.append(x)

    def mean_growth(self, phase=0, ci=False, threshold=False):
        x = np.array([self.all_data, self.phase1, self.phase2, self.phase3][phase])

        if threshold:
            x = x[x > threshold]

        if len(x) == 0:
            return None, None

        SEM = scipy.stats.sem(x).flatten()[0]
        if not ci:
            return (
                np.mean(x),
                np.std(x),
                SEM
            )
        else:
            ci = SEM * scipy.stats.t.ppf(1.95/2, len(x) - 1)
            return (
                np.mean(x),
                np.std(x),
                SEM,
                ci
            )

    def get_all_data(self):
        if len(self.all_data) < 1:
            return [(None, 0, 0, 0, 0, 0, self.unit)]
        m, std, sem, ci = self.mean_growth(0, ci=True)
        return [(None, len(self.all_data), m, std, sem, ci, self.unit)]

    def get_data(self):
        data = []  # all, phase1, phase2, phase3: (phase, n, mean, sem, unit)

        phase = 0
        for x in [self.all_data,
                  self.phase1,
                  self.phase2,
                  self.phase3]:
            if len(x) < 1:
                # no divisions
                data.append((
                    phase and "Phase {0}".format(phase) or "All Data",
                    0, 0, 0, 0, 0, self.unit
                ))
            else:
                m, std, sem, ci = self.mean_growth(phase, ci=True)
                data.append((
                    phase and "Phase {0}".format(phase) or "All Data",
                    len(x), m, std, sem, ci, self.unit
                ))
            phase += 1
        return data

    def print_data(self):
        phase = 0
        for x in [self.all_data,
                  self.phase1,
                  self.phase2,
                  self.phase3]:
            if len(x) < 1:
                print("{0} ({1}): No divisions".format(
                    self.description,
                    phase and "Phase {0}".format(phase) or "All Data"
                ))
            else:
                m, std, sem, ci = self.mean_growth(phase, ci=True)
                print("{0} ({1}): {2:.5f}{3} \u00B1 {4:.5f} (n = {5}) [S.D. {6:.5f}]".format(
                    self.description,
                    phase and "Phase {0}".format(phase) or "All Data",
                    m,
                    self.unit,
                    ci,
                    len(x),
                    std
                ))
            phase += 1

    def print_all_data(self):
        if len(self.all_data) < 1:
            print("{0}: No divisions".format(self.description))
        else:
            m, std, sem, ci = self.mean_growth(0, ci=True)
            print("{0}: {1:.5f}{2} \u00B1 {3:.5f} (n = {4}) [S.D. {5:.5f}]".format(
                self.description,
                m,
                self.unit,
                ci,
                len(self.all_data),
                std
            ))


class Plotter(object):
    def __init__(self, paths, method, suffix, phases, write_excel=True, write_pdf=True, print_data=True):
        self.PATHS = paths
        self.PASS_DELAY = 15  # pass delay in minutes
        self.PX = 0.12254  # calibration of 1px in um for 63x objective (WF2)
        self.FLUOR_THRESH = 2500
        self.ORIGINAL_DIR = os.getcwd()
        self.METHOD = method
        self.SUFFIX = suffix
        self.PHASES = phases
        self.WRITE_EXCEL = write_excel
        self.WRITE_PDF = write_pdf
        self.PRINT = print_data
        self.DEBUG = 1

    def three(self):
        for path in self.PATHS:
            self.process_three(path)

    def write_excel_row(self, rownum, *values):
        i = 0
        for val in values:
            self.excel_ws.write(rownum, i, val)
            i += 1

    def start(self):
        excel_wb = xlwt.Workbook()
        self.excel_ws = excel_wb.add_sheet("General")
        self.excel_ws2 = excel_wb.add_sheet("End Lengths")
        self.write_excel_row(0, "Name", "n", "Mean", "SD", "SEM", "95%", "unit")

        self.doubling_time = Storage("Doubling Time", "h")
        self.growth_rate = Storage("Elongation Rate", "\u03BCm/h")
        self.div_length = Storage("Division Length", "\u03BCm")
        self.end_length = []
        self.septum_placement = Storage("Septum Placement", "%")
        for path in self.PATHS:
            print("Processing {0}".format(path))
            self.process(path)

        self.decorate_tracks()
        if self.PHASES and self.PRINT:
            self.doubling_time.print_data()
            self.growth_rate.print_data()
            self.div_length.print_data()
            self.septum_placement.print_data()
        elif not self.PHASES and self.PRINT:
            self.doubling_time.print_all_data()
            self.growth_rate.print_all_data()
            self.div_length.print_all_data()
            self.septum_placement.print_all_data()

        r = 1
        for x in [self.doubling_time, self.growth_rate, self.div_length]:
            if self.PHASES:
                data = x.get_data()
            else:
                data = x.get_all_data()

            for label, n, mean, std, sem, ci, unit in data:
                if label:
                    desc_cell = "{0} ({1})".format(x.description, label)
                else:
                    desc_cell = x.description
                self.write_excel_row(
                    r,
                    desc_cell,
                    int(n),
                    float(mean),
                    float(std),
                    float(sem),
                    float(ci),
                    unit
                )
                r += 1

        # get mini-cells
        cl = np.array(self.end_length)
        n = len(cl)
        m = cl.mean()
        sem = scipy.stats.sem(cl).flatten()[0]
        ci = sem * scipy.stats.t.ppf(1.95/2, n - 1)
        self.write_excel_row(
            r, "Cell length (endpoint)", n, m, std, sem, ci, "\u03BCm"
        )
        cell_length_row = 0
        for cell_length in cl:
            self.excel_ws2.write(cell_length_row, 0, cell_length)
            cell_length_row += 1

        r += 1
        nmini = len(cl[cl < 2.5])
        self.write_excel_row(
            r, "Mini-cells (< 2.5 \u03BCm)", nmini,
            (nmini / n) * 100, "", "", "", "%"
        )

        if self.PRINT:
            print("Cell lengths (endpoint): {0:.5f}\u03BCm \u00B1 {1:.5f} (n = {2}) [S.D. {3:.5f}]".format(
                m, ci, n, cl.std()
            ))
            print("Mini-cells (<2.5\u03BCm): {0} ({0} / {1}); {2:.1f}%".format(
                nmini, n, (nmini / n) * 100
            ))

        if self.WRITE_EXCEL:
            if not os.path.exists("data"):
                os.mkdir("data")
            excel_wb.save("data/data.xls")

        if self.PHASES:
            self.plot_histograms()

    def _set_topmid(self, ax):
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

    def _h(self, h, b):
        h = (h.astype("f64") / np.sum(h)) * 100
        x = (b[:-1] + b[1:])/2
        return h, x, b

    def set_limits(self):
        xmax = ((7 * self.PX) / 15) * 60
        plt.xlim([0, xmax])
        plt.ylim([0, plt.ylim()[1]])
        plt.xticks(np.arange(0, xmax, 0.5))

    def plot_bars(self, bins, c, h, t, w, col, leg, counts):
        b = bins[:-1]
        n1 = counts[c <= t]
        n2 = counts[c > t]
        if len(n1) > 0:
            l1 = plt.bar(b[c <= t], h[c <= t], w, color="0.5", linewidth=1)
            l2 = plt.bar(b[c > t], h[c > t], w, color=col, linewidth=1)
            plt.legend(
                [l1, l2],
                [
                    r"n = {0} ({1:.0f}\%)".format(n1.sum(), 100 * n1.sum() / counts.sum()),
                    r"n = {0} ({1:.0f}\%)".format(n2.sum(), 100 * n2.sum() / counts.sum())
                ],
                title=leg,
            )
        else:
            l = plt.bar(b, h, w, color=col, linewidth=1)
            plt.legend(
                [l],
                ["n = {0} (100%)".format(counts.sum())],
                title=leg,
            )

    def _f(self, x, *p):
        A, mu, sigma = p
        return A * np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))
        return (1 / (sigma * np.sqrt(2 * np.pi)) *
                np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))

    def plot_fit(self, c, h, t, counts):
        b = c[c > t]
        f = h[c > t]

        n = counts[c > t]
        if n.sum() < 30:
            return

        m = np.argmax(f)
        A = np.mean([f[m - 1], f[m], f[m + 1]])
        mu = np.mean([b[m - 1], b[m], b[m + 1]])

        sigma = np.sqrt(
            np.sum((b ** 2) * f) /
            np.sum(f) -
            ((np.sum(b * f) / np.sum(f)) ** 2)
        )

        mu = b[np.argmax(f)]
        guess = [A, mu, sigma]
        if 0 in guess:
            return
        try:
            coeff, matrix = scipy.optimize.curve_fit(self._f, b, f, p0=guess)
        except RuntimeError:
            return
        fit_x = np.linspace(
            0,
            ((7 * self.PX) / 15) * 60,
            500
        )
        fit = self._f(fit_x, *coeff)
        fit_mean = coeff[1]

        plt.plot(fit_x, fit, "k-")
        plt.vlines(fit_mean, 0, plt.ylim()[1], colors=["k"], linestyles=["dashed"])
        plt.text(fit_mean + 0.1, 9 * plt.ylim()[1] / 10, "$\mu =$ {0:.2f}".format(fit_mean))
        plt.text(fit_mean + 0.1, 7.5 * plt.ylim()[1] / 10, "$\sigma =$ {0:.2f}".format(coeff[2]))

    def plot_mean(self, d, phase, threshold=False):
        mean, std, sem, ci = d.mean_growth(phase, threshold=threshold, ci=True)
        if not mean:
            return

        plt.vlines(mean, 0, plt.ylim()[1], colors=["k"], linestyles=["dashed"])
        plt.text(
            mean + 0.02,
            95 * plt.ylim()[1] / 100,
            r"\footnotesize {0:.2f} $\pm$ \SI{{{1:.2f}}}{{\micro\metre\per\hour}}".format(mean, ci)
        )

    def plot_histograms(self):
        plt.figure()
        x1 = np.array(self.growth_rate.phase1)
        x2 = np.array(self.growth_rate.phase2)
        x3 = np.array(self.growth_rate.phase3)

        ax = plt.subplot(3, 1, 1)
        self._set_topmid(ax)

        hist, bin_centres, bins, threshold, width, counts = self.get_histogram(x1)

        self.plot_bars(bins, bin_centres, hist, threshold, width, "r", "Before RIF", counts)
        # self.plot_fit(bin_centres, hist, threshold, counts)
        self.set_limits()
        self.plot_mean(self.growth_rate, 1, threshold=threshold)

        ax = plt.subplot(3, 1, 2)
        self._set_topmid(ax)

        d = self.get_histogram(x2, bins)
        hist = d[0]
        counts = d[-1]

        self.plot_bars(bins, bin_centres, hist, threshold, width, "g", "With RIF", counts)
        # self.plot_fit(bin_centres, hist, threshold, counts)
        self.set_limits()
        self.plot_mean(self.growth_rate, 2, threshold=threshold)
        plt.ylabel(r"Frequency (\%)")

        ax = plt.subplot(3, 1, 3)
        self._set_topmid(ax)

        d = self.get_histogram(x3, bins)
        hist = d[0]
        counts = d[-1]

        self.plot_bars(bins, bin_centres, hist, threshold, width, "y", "After RIF", counts)
        # self.plot_fit(bin_centres, hist, threshold, counts)
        self.set_limits()
        self.plot_mean(self.growth_rate, 3, threshold=threshold)

        plt.xlabel("Elongation Rate (\si{\micro\metre\per\hour})")

        if self.WRITE_PDF:
            if self.SUFFIX:
                plt.savefig("growth-rates-{0}.pdf".format(self.SUFFIX))
            else:
                plt.savefig("growth-rates.pdf")
            plt.close()

    def decorate_tracks(self):
        ax = plt.gca()
        if self.PHASES:
            d = self.RIF_REMOVE - self.RIF_ADD
            s = d.days * 24 * 60 * 60
            s += d.seconds
            w = (s / 60) // 60

            xd = self.RIF_ADD - self.T0
            s = xd.days * 24 * 60 * 60
            s += xd.seconds
            x = (s / 60) // 60

            ax.axvspan(
                x, x + w,
                facecolor="y",
                edgecolor=None,
                alpha=.3,
            )

        plt.xlabel(r"Time (\si{\hour})")
        plt.ylabel(r"Cell size (\si{\micro\metre})")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        if self.PHASES:
            plt.xlim([0, 22])
        else:
            plt.xlim([0, plt.xlim()[1]])

        if self.WRITE_PDF:
            if self.SUFFIX:
                plt.savefig("growth-traces-{0}.pdf".format(self.SUFFIX))
            else:
                plt.savefig("growth-traces.pdf")
            plt.close()

    def _gettimestamp(self, day, time, *args):
        return datetime.datetime.strptime(
            "{0} {1}".format(day, time),
            "%d.%m.%y %H:%M"
        )

    def _timediff(self, day, time, t0):
        t1 = self._gettimestamp(day, time)
        td = t1 - t0
        s = td.days * 24 * 60 * 60
        s += td.seconds
        m = s // 60
        return m

    def get_timings(self, t0=False):
        if os.path.exists("timing.json"):
            return json.loads(open("timing.json").read())["timing"]
        elif os.path.exists("timings.json"):
            timing_data = json.loads(open("timings.json").read())
            if "add" in timing_data and "remove" in timing_data:
                rif_add = self._gettimestamp(*timing_data["add"])
                rif_remove = self._gettimestamp(*timing_data["remove"])
            else:
                rif_add, rif_remove = None, None
            timings = timing_data["timings"]
            T = []
            if "start" in timing_data:
                t0 = self._gettimestamp(*timing_data["start"])
            else:
                t0 = self._gettimestamp(*timings[0])

            for d1, t1, frames in timings:
                sm = self._timediff(d1, t1, t0)
                for _ in range(frames):
                    T.append(sm)
                    sm += self.PASS_DELAY

            if t0:
                return T, rif_add, rif_remove, t0
            else:
                return T, rif_add, rif_remove

    def get_fluoresence(self):
        return []
        if os.path.exists("fluorescence.json"):
            return json.loads(open("fluorescence.json").read())
        else:
            return []

    def get_histogram(self, data, bins=None):
        data[data < 0] = 0
        if bins is None:
            n_bins = 30
            bins = np.linspace(0, 2, n_bins)
        counts, bins = np.histogram(data, bins)
        hist, bin_centres, bins = self._h(counts, bins)
        width = bins[1] - bins[0]
        threshold = width * 2
        return hist, bin_centres, bins, threshold, width, counts

    def process_three(self, path):
        os.chdir(path)
        L = track.Lineage()
        self.T, self.RIF_ADD, self.RIF_REMOVE, self.T0 = self.get_timings(t0=True)

        start_delta = self.RIF_ADD - self.T0
        start = start_delta.days * 24 * 60 * 60
        start += start_delta.seconds
        start /= 60
        t_array = np.array(self.T) - start
        t_array[t_array < 0] = np.inf
        p2 = np.argmin(t_array)

        end_delta = self.RIF_REMOVE - self.RIF_ADD
        end = end_delta.days * 24 * 60 * 60
        end += end_delta.seconds
        end /= 60
        t_array = np.array(self.T) - end - start
        t_array[t_array < 0] = np.inf
        p3 = np.argmin(t_array)

        fig = plt.figure()
        ax_all = fig.add_subplot(111, projection="3d")
        # bins = None
        bins = np.linspace(0, 4, 20)
        bin_centres = None
        threshold = None
        width = None
        for F in L.frames:
            growth_events = []
            for cell in F.cells:
                if type(cell.children) is str:
                    child = L.frames[cell.frame].cell(cell.children)
                    assert cell.id == child.parent
                    assert cell.children == child.id

                    # get change in length
                    l1, l2 = cell.length[0][0], child.length[0][0]
                    f1, f2 = cell.frame, child.frame
                    t1, t2 = self.T[f1 - 1], self.T[f2 - 1]
                    delta_time = t2 - t1 / 60  # h^-1
                    delta_length = (l2 - l1) * self.PX  # um
                    growth_rate = delta_length / delta_time  # um/h
                    growth_rate = ((l2 - l1) / ((t2 - t1) / 60)) * self.PX
                    growth_events.append(growth_rate)

            if F.frame < p2:
                col = "r"
            elif F.frame >= p2 and F.frame < p3:
                col = "g"
            elif F.frame >= p3:
                col = "y"

            if len(growth_events) >= 1:
                if bins is not None:
                    data = self.get_histogram(np.array(growth_events), bins)
                    bin_centres = data[1]
                    threshold = data[3]
                    width = data[4]
                else:
                    data = self.get_histogram(np.array(growth_events))
                    bin_centres = data[1]
                    bins = data[2]
                    threshold = data[3]
                    width = data[4]
                hist = data[0]
                counts = data[5]

                b = bins[:-1]
                n1 = counts[bin_centres <= threshold]
                # n2 = counts[bin_centres >= threshold]
                colours = [col] * len(b)
                colours[0] = "0.5"
                colours[1] = "0.5"
                if len(n1) > 0:
                    l1 = ax_all.bar(
                        b[bin_centres <= threshold],
                        hist[bin_centres <= threshold],
                        zs=self.T[F.frame] / 60,
                        width=width,
                        color="0.5",
                        # linewidth=0.5,
                        alpha=0.8,
                        zdir="y",
                    )
                    l2 = ax_all.bar(
                        b[bin_centres > threshold],
                        hist[bin_centres > threshold],
                        zs=self.T[F.frame] / 60,
                        width=width,
                        color=col,
                        # linewidth=0.5,
                        alpha=0.8,
                        zdir="y",
                    )

        plt.show()

        os.chdir(self.ORIGINAL_DIR)

    def process(self, path):
        os.chdir(path)

        if os.path.exists(".WF1_100"):
            self.PX = 0.062893  # calibration for 1px in um for 100x objective (WF1)

        L = track.Lineage()
        self.T, self.RIF_ADD, self.RIF_REMOVE, self.T0 = self.get_timings(t0=True)
        F = self.get_fluoresence()

        start_delta = self.RIF_ADD - self.T0
        start = start_delta.days * 24 * 60 * 60
        start += start_delta.seconds
        start /= 60
        t_array = np.array(self.T) - start
        t_array[t_array < 0] = np.inf
        p2 = np.argmin(t_array) + 1

        end_delta = self.RIF_REMOVE - self.RIF_ADD
        end = end_delta.days * 24 * 60 * 60
        end += end_delta.seconds
        end /= 60
        t_array = np.array(self.T) - end - start
        t_array[t_array < 0] = np.inf
        p3 = np.argmin(t_array) + 1

        self.doubling_time.set_phase_boundaries(p2, p3)
        self.growth_rate.set_phase_boundaries(p2, p3)
        self.div_length.set_phase_boundaries(p2, p3)

        cell_queue = L.frames[0].cells
        while True:
            try:
                cell = cell_queue.pop(0)
            except IndexError:
                break
            frame_idx = cell.frame - 1
            end_type = None

            lineage = [
                (cell.frame, cell.length)
            ]
            dead_lineage = []

            while True:
                if type(cell.children) is str:
                    # growth event
                    if F:
                        fluor = F[cell.id]
                    else:
                        fluor = 0

                    frame_idx += 1
                    cell = L.frames[frame_idx].cell(cell.children)

                    if fluor > self.FLUOR_THRESH:
                        if not dead_lineage:
                            dead_lineage.append(lineage[-1])
                        dead_lineage.append(
                            (cell.frame, cell.length)
                        )
                    else:
                        lineage.append(
                            (cell.frame, cell.length)
                        )

                elif type(cell.children) is list:
                    # division event
                    end_type = 1
                    dt = (self.T[lineage[-1][0] - 1] - self.T[lineage[0][0] - 1]) / 60
                    if dt == 0:
                        break

                    if lineage[0][0] != 1:
                        self.doubling_time.append(dt)
                        self.doubling_time.append_p(cell, dt)

                    if cell.frame == 13:
                        lt = self.get_growth_rate(lineage, self.METHOD)
                        self.DEBUG += 1
                    else:
                        lt = self.get_growth_rate(lineage, self.METHOD)
                    dl = lineage[-1][1] * self.PX
                    self.growth_rate.append(lt)
                    self.div_length.append(dl)

                    # get septum placement
                    child_lengths = np.array([
                        L.frames.cell(cell.children[0]).length[0][0],
                        L.frames.cell(cell.children[1]).length[0][0]
                    ])
                    child_deviation = np.abs(
                        child_lengths[0] - child_lengths[1]
                    ) / 2
                    placement = (child_deviation / child_lengths.sum()) * 100
                    self.septum_placement.append(placement)

                    self.divideandconquer(lineage, p2, p3, div=True)
                    break
                else:
                    # death event
                    end_type = 2
                    dt = (self.T[lineage[-1][0] - 1] - self.T[lineage[0][0] - 1]) / 60
                    if dt == 0:
                        break
                    lt = self.get_growth_rate(lineage, self.METHOD)

                    self.growth_rate.append(lt)
                    self.divideandconquer(lineage, p2, p3, div=False)
                    break

            l = np.array(lineage)
            plt.plot(self.ftt(l[:, 0]), l[:, 1] * self.PX)
            if dead_lineage:
                d = np.array(dead_lineage)
                plt.plot(self.ftt(d[:, 0]), d[:, 1] * self.PX, "k-", alpha=0.4)

            if end_type == 1:
                cell_queue.append(
                    L.frames[frame_idx + 1].cell(
                        cell.children[0]
                    )
                )
                cell_queue.append(
                    L.frames[frame_idx + 1].cell(
                        cell.children[1]
                    )
                )

        final_cells = L.frames[-1].cells
        for final_cell in final_cells:
            self.end_length.append(float(final_cell.length) * self.PX)
        os.chdir(self.ORIGINAL_DIR)
        return L

    def ftt(self, x):
        return np.array([self.T[int(z) - 1] / 60 for z in x])

    def divideandconquer(self, lineage, p2, p3, div):
        before = [
            x for x in lineage
            if x[0] <= p2
        ]
        self.assign_t(before, 1)

        during = [
            x for x in lineage
            if x[0] <= p3 and x[0] > p2
        ]
        self.assign_t(during, 2)

        after = [
            x for x in lineage
            if x[0] > p3
        ]
        self.assign_t(after, 3)

        if div:
            if after:
                dl = after[-1][1] * self.PX
                self.div_length.append_p3(dl)
            elif during:
                dl = during[-1][1] * self.PX
                self.div_length.append_p2(dl)
            elif before:
                dl = before[-1][1] * self.PX
                self.div_length.append_p1(dl)

    def assign_t(self, data, phase):
        if len(data) <= 1:
            return
        lt = self.get_growth_rate(data, self.METHOD)

        if phase == 1:
            self.growth_rate.append_p1(lt)
        elif phase == 2:
            self.growth_rate.append_p2(lt)
        elif phase == 3:
            self.growth_rate.append_p3(lt)

    def get_growth_rate(self, data, method, debug=False):
        if debug:
            plt.figure()
            ax = plt.subplot(111)
            trans = matplotlib.transforms.blended_transform_factory(
                ax.transAxes, ax.transAxes
            )
            l = np.array([(
                self.T[_[0] - 1] / 60,
                _[1][0][0] * self.PX
            ) for _ in data])
            plt.xlabel("Time (h)")
            plt.ylabel("Cell length (\si{\micro\metre})")
            self._set_topmid(ax)
            plt.plot(l[:, 0], l[:, 1], marker=".")

            # endpoint
            plt.plot(
                [
                    l[0][0],
                    l[-1][0]
                ], [
                    l[0][1],
                    l[-1][1]
                ],
                "ro",
                markersize=10,
                markeredgewidth=2,
                fillstyle="none",
            )

            plt.text(
                0.1,
                1,
                r"ENDPOINT: (length $B$ - length $A$) / $\Delta t = {0:.2f}$".format(
                    (l[-1][1] - l[0][1]) / (l[-1][0] - l[0][0])
                ),
                color="red",
                transform=trans,
            )

            # mean
            n = 0
            lts = []
            for curr in data:
                if n == 0:
                    pass
                else:
                    prev = data[n - 1]
                    time_change = (
                        self.T[prev[0] - 1] -
                        self.T[curr[0] - 1]
                    ) / 60
                    length_change = (
                        (
                            prev[1][0][0] - curr[1][0][0]
                        ) / time_change
                    ) * self.PX
                    lts += [length_change]
                n += 1
            lt = np.mean(lts)
            components = r"""
                $\frac{{{0}}}{{{1}}}
            """.format("+".join(["{0:.2f}".format(_) for _ in lts]), len(lts)).strip()
            plt.text(
                0.1,
                0.9,
                r"MEAN: {1} = {0:.2f}$".format(lt, components),
                color="blue",
                transform=trans,
            )

            # gradient
            pf = np.polyfit(l[:, 0], l[:, 1], 1)
            x = np.linspace(l[0, 0], l[-1, 0], 50)
            y = pf[0] * x + pf[1]
            plt.plot(x, y, "k-")

            plt.text(
                0.1,
                0.8,
                r"GRAD: $y = {0:.2f}x + {1:.2f}$, growth $= m = {0:.2f}$".format(
                    pf[0], pf[1]
                ),
                color="black",
                transform=trans,
            )
            plt.savefig("debug{0}.pdf".format(self.DEBUG))

        if method == "endpoint":
            # take length and start and end and divide by time
            dt = (
                self.T[data[-1][0] - 1] -
                self.T[data[0][0] - 1]
            ) / 60
            l = ((data[-1][1][0][0] - data[0][1][0][0]) * self.PX)
            lt = l / dt

        elif method == "mean":
            # take change in length for each adjacent timepoint
            # and divide by time difference
            # return the mean of all these changes
            n = 0
            lts = []
            for curr in data:
                if n == 0:
                    pass
                else:
                    prev = data[n - 1]
                    time_change = (
                        self.T[prev[0] - 1] -
                        self.T[curr[0] - 1]
                    ) / 60
                    length_change = (
                        (
                            prev[1][0][0] - curr[1][0][0]
                        ) / time_change
                    ) * self.PX
                    lts += [length_change]
                n += 1
            lt = np.mean(lts)

        elif method == "gradient":
            # take gradient of the line of best fit of
            # cell lengths over time
            l = np.array([
                (self.T[_[0] - 1] / 60,
                 _[1][0][0] * self.PX)
                for _ in data
            ])
            lt = np.polyfit(l[:, 0], l[:, 1], 1)[0]

        return lt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot lineage information."
    )
    parser.add_argument(
        "-t", "--three", default=False, action="store_true",
        help="""
            Process as pairwise frames to generate 3d
            histograms of growth rates.
        """
    )

    parser.add_argument(
        "-m", "--method", default="gradient",
        help="""
            Set growth rate determination method. Options are: mean - use
            growth rate between each adjacent frame, and use the mean of all
            rates.  endpoint - use initial and final cell lengths. gradient
            (default) - use the gradient of the line of best fit for cell
            lengths.
        """,
        choices=["mean", "endpoint", "gradient"]
    )

    parser.add_argument(
        "-s", "--suffix", default="",
        help="""
            Add suffix to filenames outputted.
        """
    )

    parser.add_argument(
        "-n", "--nophase", default=False, action="store_true",
        help="""
            Disable determination of phases (specific for Rif experiments)
        """
    )

    parser.add_argument(
        "directories", metavar="dirs", type=str, nargs="*",
        help="""
            Specify which directories to process. Combines each dataset into
            single plots. If omitted, only the current directory is processed.
        """
    )

    args = parser.parse_args()

    print("Using growth rate determination method: {0}".format(args.method))
    if args.directories:
        print("Processing multiple datasets: {0}".format(
            " ".join(filter(
                lambda x: os.path.exists(x),
                args.directories
            ))
        ))
        paths = [
            os.path.abspath(_)
            for _ in args.directories
            if os.path.exists(_)
        ]
    else:
        paths = [os.path.abspath(".")]

    P = Plotter(paths, method=args.method, suffix=args.suffix, phases=not args.nophase)
    if args.three:
        P.three()
    else:
        P.start()
