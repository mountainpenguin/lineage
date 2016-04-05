import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

PX = 0.12254


def exp_growth(lin):
    L = [x.length[0][0] * PX for x in lin]
    t = lin[0].t

    logL = np.log(L)
    lamb, logL0 = np.polyfit(t - t[0], logL, 1)
    model = np.exp(logL0 + lamb * (t - t[0]))

    return t, L, lamb, model


def lin_growth(lin):
    L = [x.length[0][0] * PX for x in lin]
    t = lin[0].t

    pf = np.polyfit(t, L, 1)
    y = pf[0] * t + pf[1]
    return pf[0], y


def main():
    lins = glob.glob("/home/miles/Work/Iria/wanted/14-7-15/9-1/data/cell_lines/lineage*.npy")
    plt.figure()
    lambs = []
    ms = []

    for l in lins:
        testlin = np.load(l)
        t, L, lamb, model = exp_growth(testlin)
        lambs.append(lamb)
        m, linear_model = lin_growth(testlin)
        ms.append(m)

        plt.subplot(221)
        plt.plot(t, L, label=os.path.basename(l).split(".npy")[0])
        plt.plot(t, model, "k-")
        plt.ylabel("Cell Length (um)")
        plt.xlabel("Time (min)")
        plt.title("Exponential")

        plt.subplot(222)
        plt.plot(t, L, label=os.path.basename(l).split(".npy")[0])
        plt.plot(t, linear_model, "k-")
        plt.ylabel("Cell Length (um)")
        plt.xlabel("Time (min)")
        plt.title("Linear")

    plt.subplot(223)
    for x in range(len(lambs)):
        plt.plot(x, lambs[x], marker=".", ms=10)
    plt.ylabel("lambda")
    plt.xlim([-.1, 5.1])

    plt.subplot(224)
    for x in range(len(ms)):
        plt.plot(x, ms[x], marker=".", ms=10)
    plt.ylabel("Elongation rate (um/h)")
    plt.xlim([-.1, 5.1])

    sns.despine()
    plt.tight_layout()
    plt.savefig("testexp.pdf")

if __name__ == "__main__":
    main()
