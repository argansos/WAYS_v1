#!/bin/bash

files=($(< filelist.txt))

for file in "${files[@]}"; do
wget -b -c -r -np -nd -k -L -p -A.hdf "$file" -P /Volumes/LIU/DATA/Auxiliary_Data/MODIS09A1
echo "$file"
done
