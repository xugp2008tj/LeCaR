#!/bin/bash

CACHE_SIZE=0.002
FILES=(topgun-110108-112108.3.blkparse)
ALGORITHMS=(LRU LFU ARC LeCaR)
BLOCKSIZE=512

for ((i=0;i<${#FILES[@]};++i)); do
    python ../run.py "${CACHE_SIZE}" "${FILES[i]}" "${BLOCKSIZE}" "${ALGORITHMS[@]}"
	python  visualize.py "${FILES[i]}_${CACHE_SIZE}"
done



