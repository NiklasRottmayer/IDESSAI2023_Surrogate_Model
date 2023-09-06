# IDEESAI2023-Surrogate-Model
This repository contains the material for course 5 of the IDESSAI 2023 with the topic "Surrogate model for Monte-Carlo simulation of electron matter interaction".
The authors of this interactive session are Tim Dahmen, Katja Schladitz and Niklas Rottmayer. 

## Prerequisites
For running code, you require a python environment. If you do not have a working environment yet, we recommend you to install and use [anaconda](https://www.anaconda.com/download). Open Anaconda Prompt after installation.
The working environment can either be set up manually (not recommended) or by installing it directly from the yml-file in this repository. For this to work use the command
```conda env create --name envname --file=environment.yml``` and replace ```envname``` by the name you want to call your environment. All necessary packages will be installed automatically. Afterwards, you can activate the environment with ```conda activate envname``` and open an instance of jupyter notebook by typing ```jupyter notebook```. 

Linux packages:
* numpy
* scipy
* matplotlib
* configparser
* tifffile
* vtk
* notebook

We will perform the generation of synthetic geometrical data using this environment as interactive 3D plots are not supported well in Google Colab which will be used for the remainder. 

## Generating Synthetic Geometrical Data for the Surrogate Model
The first part of our interactive session is about the geometric modelling and generation of data for the surrogate model. Simply navigate to and open the notebook 'Generator-Notebook.ipynb' in the instance that opened in your browser to begin with the first subject. The notebook contains all information on this topic and will introduce you to Boolean models. 

