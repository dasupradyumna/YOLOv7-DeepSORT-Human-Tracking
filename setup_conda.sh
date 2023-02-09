# pre-execution checks
if ! command -v conda >/dev/null 2>&1 ; then
  echo "ERROR: 'conda' command not found"
  exit
elif [ $# -gt 1 ] ; then
  echo "ERROR: Too many arguments"
  echo "  Usage: ./setup_conda.sh [<envname>]"
  echo "    <envname> defaults to 'yolo7ds' if absent"
  exit
fi

# Ensure conda works in current shell
CONDA_ROOT="$HOME/miniconda3"
if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ] ; then
  echo "ERROR: Check if CONDA_ROOT directory is correct"
  echo "  setup_conda.sh:13: CONDA_ROOT='$CONDA_ROOT'"
  exit
fi
. "$CONDA_ROOT/etc/profile.d/conda.sh"

# create and activate conda environment
envname="$1"
[ -z "$envname" ] && envname="yolo7ds"
conda create -yn "$envname" python=3.8 || exit
conda activate "$envname"

# install the conda packages
conda install -yc pytorch pytorch==1.10.1 torchvision==0.11.2 \
  cudatoolkit==10.2.89 cudnn==7.6.5.32 || exit
conda install -yc conda-forge tensorflow-gpu==2.8.1 || exit
conda install -yc conda-forge opencv==4.6.0 || exit
conda install -yc conda-forge matplotlib==3.6.3 || exit
conda install -yc conda-forge pandas==1.5.3 || exit

echo "Finished setting up '$envname' conda environment"
