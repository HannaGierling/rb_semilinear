# rb_semilinear

## 📘 [Documentation](https://hannagierling.github.io/rb_semilinear/)

Open Documentation: [https://hannagierling.github.io/rb_semilinear/](https://hannagierling.github.io/rb_semilinear/)

or

 ```bash
 cd ./docs/_build/html
 xdg-open index.html  # linux
 open index.html      # macOS
 ```

## 🔧 Requirements

This project uses [FEniCS](https://fenicsproject.org/), so please make sure it is installed correctly before running any scripts.

> ⚠️ Note:  
> FEniCS runs natively on **Linux**.  
> If you're using **Windows**, you must install and use the **Windows Subsystem for Linux (WSL)**.  
> 👉 See the [official WSL installation guide](https://docs.microsoft.com/en-us/windows/wsl/install)


## ⬇️ Installing FEniCS
The following instructions are copied from [https://fenicsproject.org/download/archive/](https://fenicsproject.org/download/archive/).

> ⚠️ Info: 
> I used the Linux-Distribution Pop!_OS and [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).
---

### Anaconda
The easiest (and suggested) way to start using FEniCS on Linux or MacOS systems is to use
[conda](https://docs.conda.io/projects/conda/en/stable/index.html):
```bash
conda create -n fenics_env -c conda-forge fenics
```
➡️ Creates a new Conda environment named fenics_env and installs FEniCS from conda-forge.
```bash
conda activate fenics_env
```
➡️ Activates the environment so you can use FEniCS within it.

### Windows
To install FEniCS on Windows, use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) and install the Ubuntu distribution. Then follow the instructions for Ubuntu (or Anaconda).

### Ubuntu 
To install FEniCS on Ubuntu, run the following commands:

```bash
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics
```

## ▶️  Running the `test_ns.py` and `test_rbm.py` Script

To run the `test_ns.py` and `test_rbm.py` script:

1. **Activate your Conda environment (If Anaconda was used for installing)**:

    ```bash
    conda activate fenics_env
    ```

2. **Run the script**:

    ```bash
    python3 ./test_ns.py
    ```

