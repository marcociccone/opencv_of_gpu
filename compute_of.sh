#!/bin/bash

filename=$1
echo $filename
while IFS='' read -r line || [[ -n "$line" ]]; do
    stringarray=($line)
    ./of_gpu ${stringarray[0]} ${stringarray[1]}
done < $filename
