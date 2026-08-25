All files should be run on Lorien. This project uses conda 25.11.1.

Install coffea in conda.

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh 
source ~/.bashrc 
conda  --version
python --version
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n coffea python=3.10 -y
conda activate coffea
conda install -c conda-forge coffea uproot awkward vector dask hist -y


While in conda, activate coffea 'conda activate coffea'. 
Run 'run.py' for processing all the files and obtain file 'output.coffea'. 
Run 'plot.py' to plot the histograms along with the yields table.