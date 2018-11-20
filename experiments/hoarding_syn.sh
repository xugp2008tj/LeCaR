#!/bin/bash

CACHE_SIZE=0.05
FILES=(1000_Exp_ARC_LFU_ARC_LFUResult.txt)
ALGORITHMS=(LRU LFU ARC LeCaR)
BLOCKSIZE=512

for ((i=0;i<${#FILES[@]};++i)); do
    python ../run.py "${CACHE_SIZE}" "${FILES[i]}" "${BLOCKSIZE}" "${ALGORITHMS[@]}"
	python  visualize.py "${FILES[i]}_${CACHE_SIZE}"
done



