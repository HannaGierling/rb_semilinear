import numpy as np
import matplotlib.pyplot as plt

def plot_mu_trained(folder:str, P_train_opt:str, plot:bool=False):
    tmp          = np.loadtxt(f"{folder}/P_used_flags.csv", delimiter=',')
    P_train      = tmp[:,0]
    P_used_flags = tmp[:,1]

    plt.scatter(P_train,P_used_flags)
    if P_train_opt == "log":
        plt.xscale('log')
    plt.xlabel("µ"); plt.ylabel("used (bool)")
    plt.title("Param. values µ used by Greedy-algorithm")
    plt.savefig(f"{folder}/P_used.png")
    plt.show() if plot else plt.close()