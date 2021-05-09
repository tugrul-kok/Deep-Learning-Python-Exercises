import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pisagor(num1, num2):
    return np.sqrt(num2*num2 + num1*num1)


def plotter(plotindex):

    related_columns = [plotindex, plotindex + 10, plotindex + 20, plotindex + 1, plotindex + 11, plotindex + 21]
    fig = plt.figure(figsize=(20, 10))
    if plotindex == 1:
        fig.suptitle("Kinematic Analysis For Point A", fontsize='xx-large')
    elif plotindex == 3:
        fig.suptitle("Kinematic Analysis For Point B", fontsize='xx-large')
    elif plotindex == 5:
        fig.suptitle("Kinematic Analysis For Point C", fontsize='xx-large')
    elif plotindex == 7:
        fig.suptitle("Kinematic Analysis For Point D", fontsize='xx-large')
    elif plotindex == 9:
        fig.suptitle("Kinematic Analysis For Point D0", fontsize='xx-large')

    pltcount = 1

    for i in related_columns:
        plt.subplot(3, 3, pltcount)
        plt.plot(df["Theta"], df[df.columns[i]])
        plt.xlabel('Θ12', fontsize=10)
        plt.ylabel(df.columns[i])
        plt.grid()
        pltcount += 1

    for j in [plotindex, plotindex + 10, plotindex + 20]:
        dist = []
        for i in range(28, 148):
            dist.append(pisagor(df[df.columns[j]][i], df[df.columns[j + 1]][i]))
        plt.subplot(3, 3, pltcount)
        plt.plot(df["Theta"], dist)
        plt.xlabel('Θ12', fontsize=10)
        if pltcount == 7:
            plt.ylabel("Magnitude of Position Vector (m)")
        elif pltcount == 8:
            plt.ylabel("Magnitude of Velocity Vector (m/s)")
        elif pltcount == 9:
            plt.ylabel("Magnitude of Acceleration Vector (m/s**2)")

        plt.ylim([-1, 4])
        plt.grid()
        pltcount += 1

    if plotindex == 1:
        fig.savefig("A.png")
    elif plotindex == 3:
        fig.savefig("B.png")
    elif plotindex == 5:
        fig.savefig("C.png")
    elif plotindex == 7:
        fig.savefig("D.png")
    elif plotindex == 9:
        fig.savefig("D0.png")

    plt.show()


df = pd.read_csv('data.csv')
# Next lines saves only [28:148], drops other elements.
df = df.drop(df.index[range(148, df.shape[0])])
df = df.drop(df.index[range(0, 28)])

plotindex = [1, 3, 5, 7, 9] # 1 = A, 2 = B, 3 = C, 4 = D, 5 = D0
for i in plotindex:
    plotter(i)
