#!/bin/bash

CACHE_SIZE=0.005
FILES=(madmax-110108-112108.3.blkparse ikki-110108-112108.3.blkparse topgun-110108-112108.3.blkparse casa-110108-112108.3.blkparse online.cs.fiu.edu-110108-113008.3.blkparse webusers-030409-033109.3.blkparse webresearch-030409-033109.3.blkparse)
ALGORITHMS=(LeCaR)
BLOCKSIZE=512



#for ((i=0;i<${#FILES[@]};++i)); do
#    for ((j=0; j< 1; j+=0.00000125)); do
#        python ../run.py "${CACHE_SIZE}" "${FILES[i]}" "${BLOCKSIZE}" "${ALGORITHMS[@]}" "${j}"
#    done
#done


for ((i=0;i<${#FILES[@]};++i)); do
    for j in 0.00025 0.000375 0.0005 do
        python ../run.py "${CACHE_SIZE}" "${FILES[i]}" "${BLOCKSIZE}" "${ALGORITHMS[@]}" "${j}"
    done
done






