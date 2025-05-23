import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def errRedSolsPlot(path:str, csvfile:str, xlabel="log", plot = False):
    """
    Plot and export error norms of reduced solutions from file 'mapping.log' in 
    directory 'path'.

    Parameters
    ----------
    path : str
        Directory where file 'mapping.log' is located and where plot will be
        saved
    csvfile : str
        Path for CSV file.
    xlabel : str, optional
        If 'lin', plots with linear x-axis. if 'log', plots whith
        locarithmic x-axis, by default "log".
    plot : bool, optional
        If True, plot is shown, by default False.
    """

    if not os.path.exists(f"{path}/mapping.log"):
        raise FileNotFoundError(f"'mapping.log' not found in path {path}")

    df = pd.read_csv("%s/mapping.log"%(path), sep=r'\s+', engine='python')

    Mus     = pd.to_numeric(df["μ"], errors="raise")
    errL2  = df["||u_h-u_N||_L2"]
    PerrL2 = df["||u_h-P_N(u_h)||_L2"]
    conv   = df["converged"]

    csvdf = pd.DataFrame({"μ":Mus,"errL2":errL2, "PerrL2":PerrL2, "conv":conv})
    csvdf.to_csv(f"{csvfile}", sep='\t', encoding='utf-8',
                    index=False, header=True)

    fig, ax = plt.subplots(figsize=(8, 6))  

    if xlabel=="lin":
        width = (Mus[1]-Mus[0])*0.5
        ax.bar(Mus, errL2, width=width, color='steelblue', label="errL2")
        ax.bar(Mus, PerrL2, width=width, color='red', label="PerrL2")
    else : 
        log_x = np.log10(Mus)
        width = min(np.diff(log_x))*0.7

        for i,(lx, val) in enumerate(zip(log_x, errL2)):
            left = 10**(lx - width/2)
            bar_width = 10**(lx + width/2) - left
            ax.bar(left, val, width=bar_width, align='edge', color='steelblue', 
                   label="err" if i==0 else "")
            ax.bar(left, PerrL2[i], width=bar_width, align='edge', color='red', 
                   label="errP" if i == 0 else "")
        ax.set_xscale('log')

    ax.set_xlabel("μ")
    ax.set_ylabel('error')
    ax.set_yscale('log')
    plt.title(f"sumPerrsq={np.sum(PerrL2**2):.3e}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/errBarPlot.png")
    plt.show() if plot == True else plt.close()

