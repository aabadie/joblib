#!/bin/bash

# Change the list according to your local conda env.
# Use command 'conda env list' to get your list.
CONDA_ENVS="py26-np16 py27-np17 py33-np18 py34-np19 py35-np19"

for i in $CONDA_ENVS
do
  . activate $i
  # Generate non compressed pickles.
  python create_numpy_pickle.py
  # Generate compressed pickles using pickle for numpy arrays.
  python create_numpy_pickle.py --compress
  # Generate compressed pickles with numpy arrays in companion files.
  python create_numpy_pickle.py --compress --cache_size
done
. deactivate
