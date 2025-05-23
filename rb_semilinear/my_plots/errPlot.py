import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def _str2pathstr(string:str):
    """
    modifies given string s.t. it becomes suitable for a path-string

    This function removes/replaces characters that are problematic in paths
    """
    s = f"{string}"\
        .replace(" ","").replace("'","").replace("{","").replace(":","")\
        .replace("}","").replace(".","").replace("-","m")
    return s

def errPlot(path, mu, plot=False):
    """
    Generates and saves error plot from file 'mapping.log' in 
    directory 'path' for parameter 'mu'.

    Parameters
    ----------
    path : str
        Directory path where 'mapping.log' file is located
        and where error plot 'err_mu{mu}.png' will be saved.
    mu : float 
        Parameter value for which the error plot is generated.
    plot : bool
        If True, convergence map is shown after saving it.
    """
    if not os.path.exists(f"{path}/mapping.log"):
        raise FileNotFoundError(f"'mapping.log' not found in path {path}")

    df = pd.read_csv("%s/mapping.log"%(path), sep=r'\s+', engine='python')

    dfD = df[df["μ"]==mu]

    Ns = dfD['N'].unique()
    err_L2 = dfD['||u-u_e||_L2'].array.tolist()
    err_inf = dfD['||u_dofs-u_e_dofs||_inf'].array.tolist()
    err_H10 = dfD['|u-u_e|_H10'].array.tolist()

    hs = [1.0 / N for N in Ns]
    r_L2_values = []
    r_inf_values = []
    r_H10_values = []
    for i in range(1, len(hs)):
        r_L2_values.append(np.log(err_L2[i]/ err_L2[i-1])/np.log(hs[i]/ hs[i-1]))
        r_inf_values.append(np.log(err_inf[i]/ err_inf[i-1])/np.log(hs[i]/ hs[i-1]))
        r_H10_values.append(np.log(err_H10[i]/ err_H10[i-1])/np.log(hs[i]/ hs[i-1]))

    tmp_L2 = [h*h*(err_L2[0]-0.001)/hs[0]**2 for h in hs]

    average_r_L2 = np.mean(r_L2_values)
    average_r_H10 = np.mean(r_H10_values)
    average_r_inf = np.mean(r_inf_values)

    csv_df = pd.DataFrame({
    "N":Ns,
    "L2":err_L2,
    "inf":err_inf,
    "H10":err_H10,
    "ref":tmp_L2
    })

    csv_df.to_csv(f"{path}/err.csv", sep='\t', encoding='utf-8',\
                index=False, header=True)

    plt.figure(figsize=(8, 6))
    plt.loglog(Ns,err_L2, 'o-', label='err_L2, r=%.3f'%average_r_L2)
    plt.loglog(Ns,err_inf, 'o-', label='err_inf, r=%.3f'%average_r_inf)
    plt.loglog(Ns,err_H10, 'o-', label='err_H10, r=%.3f'%average_r_H10)
    plt.loglog(Ns,tmp_L2, label="convergence rate 2")
    plt.xlabel('N_h'); plt.ylabel('err')
    plt.title(f'Error for D={mu:.0e}')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(_str2pathstr(f"{path}/err_mu{mu}.png"))
    if plot == True:
        plt.show()
    else:
        plt.close()


