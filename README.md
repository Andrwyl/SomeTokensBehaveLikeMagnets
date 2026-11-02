# Some Tokens Behave like Magnets: Revealing Linguistic Organization in the Layers of Language Models
This repository provides the code for our paper Some Tokens Behave like Magnets: Revealing Linguistic Organization in the Layers of Language Models published at EMNLP 2026

**Links**:

[Paper](https://arxiv.org/abs/2609.05743)

**Abstract**:

We identify a special group of token vectors inside large language models (LLMs) - which we term magnetic vectors - that organize the surrounding tokens by either attracting or repelling them. Particularly, tokens pointing the same way as an attracting magnet are elongated; tokens pointing the same way as a repelling magnet are compressed. Just as physical magnets pull or push away the iron filings around them, these vectors organize their surroundings through two opposing polarities. Moreover, we identify a statistically significant pattern in linguistic category where function words consistently act as repelling magnets in early layers, and we also find magnets consistently reorganize their polarities in unique ways deeper in the model. In a further case study we find this observation may unveil a deliberate, layer-wise organization in how LLMs process language.

This pattern is consistent across different LLM architectures, sizes, and layer configurations. It is also causally relevant. When the LLM is fine-tuned for a downstream task, the task-functional tokens emerge as magnets. E.g., in question answering, the answer-span tokens become uniquely repelling magnets in the final layer, geometrically carving the answer out of the surrounding context. Furthermore, removing early-layer repelling magnets devastates syntactic tasks (POS tagging accuracy drops from 91% to below 10%) while sparing semantic ones, and removing late-layer attracting magnets does the reverse. We believe this phenomenon warrants further investigation, as it opens the first probe-free path to understanding how language models geometrically organize linguistic computation across their layers.

## Repository Layout

Our repository layout is very simple. At the top level

- `/datasets`
contains `paragraph.txt`, the 100 hand-selected paragraph dataset as described in Section 4.1

- `/models`
contains `bert-pos`, `bert-sentiment`, and `bert-qa`, three fine-tuned BERT models as described in Section 5.1

- `/scripts` contains the code for reproducing the tables and charts throughout the paper, described in detail following this

## Visualizing Magnets

`/scripts/visualizations` contains the scripts used to generate the color coded magnets (representing their polarity and strength) on an input paragraph, which is what is used to generate Figure 3 and Figure 10-17. We encourage people to test with different paragraphs to see what patterns emerge.

TODO: Explain the individual files when this is filed in

## Linguistic Striations

`/scripts/linguistic_striations` contains the code to generate the results in Section 4.2, where we find that early magnets create a function/content split, disappearing moving deeper through the model. For example, `bert.py` will output the point biserial correlations at each layer into `results/bert.json` and a graph of those numbers into `results/bert.pdf`. Similar naming conventions for `qwen`, `llama`, `gemma`, and `gpt`. This recreates Figure 4.

## BERT and Downstream Tasks

`/scripts/downstream_experiments` contains the code to recreate the results of Section 5 (Tables 1,2,3)

- `/downstream_experiments/results` stores the outputs of the scripts as outlined below

- `/downstream_experiments/POS_Tagging` contains the code for the POS tagging results
    
    * `run_POS.py` takes as input `True` (to indicate random ablation) or `False` (to indicate magnet ablation) and a layer, then ablates under the paradigm at that layer and reports the POS task accuracy
    * `run_POS.sh` is a wrapper script that sweeps across the layers of BERT and outputs the POS taggin accuracy at each layer under random ablations into `/downstream_experiments/results/POS_random_ablation.txt` and under magnet ablation into `/downstream_experiments/results/POS_target_ablation.txt`

- `/downstream_experiments/Sentiment_Analysis` contqains the code for the sentiment analysis results

    * `run_sentiment.py` takes as input `True` (to indicate random ablation) or `False` (to indicate magnet ablation) and a layer, then ablates under the paradigm at that layer and reports the sentiment classification accuracy

    * `run_sentiment.sh` is a wrapper script that sweeps across the layers of BERT and outputs the sentiment classification accuracy at each layer under random ablations into `/downstream_experiments/results/sentiment_random_ablation.txt` and under magnet ablation into `/downstream_experiments/results/sentiment_target_ablation.txt`

- `/downstream_experiments/Question_Asnwering`
TODO

## Citation

```bibtex
@inproceedings{liu-etal-2026-tokens-behave,
    title = "Some Tokens Behave like Magnets: Revealing Linguistic Organization in the Layers of Language Models",
    author = "Liu, Andrew  and
      Srinivasan, Devan  and
      Penn, Gerald",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.690/",
    doi = "10.18653/2026.emnlp-main.690",
    pages = "15867--15886",
    ISBN = "979-8-89176-459-0",
}
```

