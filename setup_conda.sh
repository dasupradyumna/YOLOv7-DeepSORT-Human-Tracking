#!/usr/bin/env bash
# this script creates a conda environment with specified name (argument) and installs all the
# required packages for running this repository's python scripts

run_checks() {
  # check if conda exists
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 'conda' command not found"
    exit
    # check the number of arguments to this script
  elif [ $# -gt 1 ]; then
    echo "ERROR: Too many arguments"
    echo "  Usage: ./setup_conda.sh [<envname>]"
    echo "    <envname> defaults to 'yolo7ds' if absent"
    exit
  fi
}

# ensure conda works in current shell
ensure_conda_shell() {
  CONDA_ROOT="$HOME/miniconda3"
  if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "ERROR: Check if CONDA_ROOT directory is correct"
    echo "  setup_conda.sh:13: CONDA_ROOT='$CONDA_ROOT'"
    exit
  fi
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
}

# create environment and install packages
create_and_install() {
  # create
  conda create -yn "$1" python=3.8 || return 1
  conda activate "$1"

  # install
  conda install -yc pytorch pytorch==1.10.1 torchvision==0.11.2 \
    cudatoolkit==10.2.89 cudnn==7.6.5.32 || return 1
  conda install -yc conda-forge tensorflow-gpu==2.8.1 || return 1
  conda install -yc conda-forge opencv==4.6.0 || return 1
  conda install -yc conda-forge matplotlib==3.6.3 || return 1
  conda install -yc conda-forge pandas==1.5.3 || return 1

  return 0
}

# main function which runs all the tasks
main() {
  run_checks "$@"
  ensure_conda_shell

  envname="$1"
  [ -z "$envname" ] && envname="yolo7ds"
  if create_and_install "$envname"; then
    echo "INFO: Finished setting up '$envname' conda environment"
  else
    echo "WARN: '$envname' conda environment setup incomplete"
  fi
}

main "$@"
