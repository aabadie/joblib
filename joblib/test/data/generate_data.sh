#!/bin/bash

for i in py26-np16 py27-np17 py33-np18 py34-np19 py35-np19
do 
  . activate $i
  python create_numpy_pickle.py
  . deactivate
done
