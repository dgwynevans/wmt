# WMT code installation

## Introduction

Below we outline some general essentials for setting up workflows and new projects in Python using virtual environments, templates for directory structure and version control. Our focus is on earth and environmental sciences where we typically use large gridded datasets. However, the outlined approach is relevant beyond earth sciences and can adapted easily.

If you are new to Python take a look at some of the following resources to get yourself up to speed:

[The Scipy lecture notes](https://scipy-lectures.org/) - a great step-by-step tutorial to get you familiar with the python programming environment.</br>
[Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) - a comprehensive guide to IPython, NumPy, Pandas and Matplotlib.</br>
[Earth and Environmental Data Science](https://earth-env-data-science.github.io/) - an online book for built for researchers in earth and environmental science by Ryan Abernathey.

## Initial set-up

Install [Visual Studio Code](https://code.visualstudio.com/)</br>
 Using Visual Studio Code you can install various extensions for specific programming languages to help with coding errors etc. You can also open your project folder as a workspace, so that you can easily navigate through the different parts of the project.</br>

Install [iTerm2](https://www.iterm2.com/)</br>

## Setting up a virtual environment for Python

Using Conda you can set up a virtual environment in which the dependencies for the specific project don't change unless you tell them to. Therefore, you can come back to the project after some time and everything should work as you left it.

To setup a virtual environment run the following command in a terminal:</br>

```bash
conda create --name wmt_env python=3.12
```

Here `python=3.12` points to the version of python running via Anaconda. You can check this before outside of the virtual environment using `which python`. Check that python used by the virtual environment stems from the new virtual environment using the same command. </br>

Running the above code will essentially install the desired version of python, pip...etc. within the virtual environment.

Once installed you can activate the environment using

```bash
conda activate wmt_env
```

And deactivate an active environment using

```bash
conda deactivate
```

To remove a virtual environment you can simply delete it:

```bash
rm -rf ~/anaconda3/envs/wmt_env
```

## Install the desired python dependencies

Now you need to install the relevant packages within the new virtual environment. For example:

```bash
conda install ipython matplotlib scipy dask
```
```bash
pip install cartopy
```

Then install xarray and it's (and some other) dependencies:

```bash
conda install -c conda-forge xarray dask netCDF4 bottleneck cmocean gsw
```

You can add anymore modules to the list as required.

You can check the contents of each of the installed packages within python, for example start by opening iPython in a terminal window:

```bash
ipython
```

Then within iPython enter the following:

```python
import sys
sys.path
```

This should give you the list of paths along which python will search when importing modules and should include something like this:

```bash
'/Users/dgwynevans/anaconda3/envs/osnap_env/lib/python3.12/site-packages'
```

You can then check the version of each package within your virtual environment to ensure they're satisfactory.

## Setting up JupyterLab and within the virtual environment

To install JupyterLab within the virtual environment run the following command:

```bash
conda install -c conda-forge jupyterlab
```

So that JupyterLab works within the virtual environment you need to setup and install [IPython kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the virtual environment. To install IPython kernal you run the following command within your virtual environment:

```bash
conda install ipykernel
```

And then to setup the kernel for that virtual environment you run the following:

```bash
python -m ipykernel install --user --name wmt_env --display-name "Python (wmt_env)"
```

Here you replace `wmt_env` with the name of your environment.

To run JupyterLab from a terminal window enter `jupyter lab`, and in the top right-hand corner you should see the display name of the kernel you just installed.

## Version control: Git and GitHub

Version control is useful for keeping track of changes to files and projects, particularly when collaborating or sharing code. Here we will show some examples of the software Git and the online host [GitHub](https://github.com/).

### Git

We begin by creating a local Git repository within your project folder, for example:

```bash
cd project_name
git init
```

You then need to stage files for addition to the repository as follows:

```bash
git add <filenames>
```

This step will need to be repeated every time you change and commit a file.

To stage all the files from in your project directory you can use:

```bash
git add .
```

You then can check the status of your local Git repository:

```bash
git status
```

This will tell you which files are ready to commit to your repository and those that are yet to be added. When you are ready to commit these staged files, run the following:

```bash
git commit -m "a short comment to describe the file you are adding or the changes you've made"
```

You can get more information about your repository using the following commands:

```bash
git diff  # View file differences
git log   # View the commit log
```

To revert back to an earlier version of your file, use the following:

```bash
git checkout <commit tag> <filename>
```

### GitHub

At this point you can push your files to [GitHub](https://github.com/). To do this you need to new repository on [GitHub](https://github.com/). Then you follow the instructions and run within your local Git repository:

```bash
git remote add origin <repo url>
```

To upload files and changes to GitHub use the command:

```bash
git push origin master
```

If a collaborator makes changes to the GitHub repository you can update your local repository as follows:

```bash
get pull origin master
```

## Setting up custom paths for Python

You need to set the `PYTHONPATH` variable in `~\.bash_profile` (or equivalent) to point to the new project directory. That way any functions created as part of the project can be imported. The file `__init__.py` placed throughout the project directory structure tells python where to look for the functions. Ensure you set the parent directory of your project directory to the `PYTHONPATH` variable, so that each the functions defined in each project stem from a unique branch.

To open `~\.bash_profile` using visual studio code, enter the following in a terminal window:

```bash
code ~/.bash_profile
```

You can then add or update the `PYTHONPATH` variable as follows:

```bash
PYTHONPATH="/Users/dgwynevans/Dropbox/Python_modules:/Users/dgwynevans/Dropbox/work_general/dev/programming_workshop"
export PYTHONPATH
```

Then to run `~\.bash_profile` without restarting terminal run the following command:

```bash
source ~/.bash_profile
```

This will deactivate the virtual environment, so you'll need to reactivate it as described above. You can then open iPython to check everything works by checking the path:

```python
import sys
sys.path
```

