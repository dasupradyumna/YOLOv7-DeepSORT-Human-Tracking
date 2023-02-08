# pre-execution checks
if ! command -v conda >/dev/null 2>&1 ; then
  echo "ERROR: Conda command not found"
  exit
elif [ $# -ne 1 ] ; then
  echo "ERROR: Invalid arguments"
  echo "  Usage: ./setup_conda.sh <env_name>"
  exit
fi

# Ensure conda works in current shell
. "$HOME/miniconda3/etc/profile.d/conda.sh"

# create and activate conda environment
conda create -yn "$1" python=3.8 || exit
conda activate "$1"

# install the conda packages
conda install -yc conda-forge tensorflow-gpu=2.8 || exit
conda install -yc pytorch pytorch==1.10.1 torchvision==0.11.2 || exit
conda install -yc conda-forge opencv==4.6.0 || exit
conda install -yc conda-forge matplotlib || exit

echo "Finished setting up $1 conda environment"
