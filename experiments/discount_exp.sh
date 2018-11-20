#!/bin/bash

CACHE_SIZE=0.005
FILES=(madmax-110108-112108.3.blkparse)
ALGORITHMS=(LeCaR)
BLOCKSIZE=512

for ((i=0;i<${#FILES[@]};++i)); do
    python ../run.py "${CACHE_SIZE}" "${FILES[i]}" "${BLOCKSIZE}" "${ALGORITHMS[@]}"
done
