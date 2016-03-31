import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")


def exp_growth(lin):
    L = [x.length[0][0] for x in lin]
    t = lin[0].t
    L0 = L[0]
    L1 = L[-1]

    dt = t[-1] - t[0]
    dl = L1 / L0
    alpha = np.log(dl) / dt

    model = L0 * (np.exp(alpha * (t - t[0])))
    return t, L, alpha, model


def main():
    lins = glob.glob("/home/miles/Work/Iria/wanted/14-7-15/9-1/data/cell_lines/lineage*.npy")
    plt.figure()
    for l in lins:
        testlin = np.load(l)
        t, L, alpha, model = exp_growth(testlin)
        plt.plot(t, L, label=os.path.basename(l).split(".npy")[0])
        plt.plot(t, model, "k-")
    plt.show()

if __name__ == "__main__":
    main()
