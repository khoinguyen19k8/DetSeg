#!/bin/bash

for i in {1..4}
do
    python eval.py --config-file scratch_holdout_$(i).yaml --model runs/holdout_$(i)/best_model_scratch.h5 --threshold 0.5 --device 1
done