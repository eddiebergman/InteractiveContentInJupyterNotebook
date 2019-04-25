# An example of creating interactive content in Jupyter Notebook
This notebook serves as an example on how to make interactive and animated notebooks.

## Note:
The Notebook isn't complete but goes through the prerequestites to understand the Fourier Series and Transform.

There is two 'debug' cells at the begging, both of these need to be run for functional purporses, namely CSS, importing and configuring how IPython handles generated figures.

## Installation

#### Jupyter Notebook
Please refer to https://jupyter.readthedocs.io/en/latest/install.html on how to install Jupyter.

#### Dependancies
This notebook uses Numpy, SciPy and Bokeh

If you've installed using the Anaconda and conda packages, you should have Numpy and SciPy installed by default.

If you decided to install jupyter using the manual way we advice using https://virtualenv.pypa.io/en/latest/ to create a virtual environment and installing into a there for handling dependances.

Once a virtualenv is installed:

```bash
virtualenv /path/to/virtualenv
source /path/to/virtuelenv/bin/activate
pip install Numpy SciPy Bokeh
```

If you haven't already:
```bash
cd /path/to/download
git clone https://github.com/eddiebergman/InteractiveContentInJupyterNotebook
cd InteractiveContentInJupyterNotebook
jupyter notebook
```

This will open a browser tab, click on the notebook contained to view.
