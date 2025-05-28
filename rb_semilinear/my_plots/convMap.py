import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def convMap(path, plot=False):
    """
    Generates and saves convergence map from file 'mapping.log' in 
    directory 'path'.

    Parameters
    ----------
    path : str
        Directory path where 'mapping.log' file is located
        and where convergence map 'convMap.png' will be saved.
    plot : bool
        If True, convergence map is shown after saving it.
    """
    if not os.path.exists(f"{path}/mapping.log"):
        raise FileNotFoundError(f"'mapping.log' not found in path {path}")

    df = pd.read_csv("%s/mapping.log"%(path), sep=r'\s+', engine='python')

    # Werte für N und D
    Ns = sorted(df['N'].unique())
    Ds = sorted(df['μ'].unique())

    cmap = ListedColormap(['red', 'green'])
    red_rgb = cmap(0)[:3]
    green_rgb = cmap(1)[:3]

    grid = np.zeros((len(Ds), len(Ns), 3))
    for idx, row in df.iterrows():
        n_idx = Ns.index(row['N'])
        d_idx = Ds.index(row['μ'])
        grid[d_idx, n_idx] = green_rgb if row['converged'] else red_rgb

    plt.figure(figsize=(10, 16))
    plt.imshow(grid, cmap=cmap, origin='lower', aspect='auto')
    plt.xticks(ticks=range(len(Ns)), labels=[f"{N:.2e}" for N in Ns])
    plt.yticks(ticks=range(len(Ds)), labels=[f"{D:.2e}" for D in Ds])
    #plt.xticks(ticks=range(len(Ns)), labels=Ns)
    #plt.yticks(ticks=range(len(Ds)), labels=Ds)
    plt.grid(True)
    plt.xlabel('N_h')
    plt.ylabel('μ')
    plt.title('Convergence Map')

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='green', label='Converged = True'),
        mpatches.Patch(color='red', label='Converged = False')
    ]
    plt.legend(handles=legend_handles, loc='upper right')
    plt.savefig("%s/%s"%(path,"convMap.png"))
    if plot == True:
        plt.show()
    else:
        plt.close()
