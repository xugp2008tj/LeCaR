#!/bin/bash

CONFIG=

for ((j=0;j<5;++j)); do
	python ../run2.py "${CONFIG}"
done
