#!/bin/bash

for layer in {0..11}; do
    python run_POS.py True $layer >> ../results/POS_random_ablation.txt
    python run_POS.py False $layer >> ../results/POS_target_ablation.txt
done
