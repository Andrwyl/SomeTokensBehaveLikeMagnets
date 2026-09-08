#!/bin/bash

for layer in {0..11}; do
    python run_sentiment.py True $layer >> ../results/sentiment_random_ablation.txt
    python run_sentiment.py False $layer >> ../results/sentiment_targeted_ablation.txt
done