This is heavily modified fork of the Text2KGBench repository for the purpose of testing various information and relation extraction models.  The code is being refactored to improve maintainability, add aditional corpora, prompt and encoder techniques, and to make it easier to run experiments.

# Text2KG: A Benchmark for Ontology Driven Knowledge Graph Generation from Text
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This is the repository for ISWC 2023 Resource Track submission for `Text2KGBench: Benchmark for
Ontology-Driven Knowledge Graph Generation from Text`.
Text2KGBench is a benchmark to evaluate the capabilities of language models to generate KGs
from natural language text guided by an ontology. Given an input ontology and a set of sentences, the task is to extract facts from the text
while complying to the given ontology (concepts, relations, domain/range constraints) and being faithful to the input sentences.

It contains two datasets (i) Wikidata-TekGen with 10 ontologies and 13,474 sentences
and (ii) DBpedia-WebNLG with 19 ontologies and 4,860 sentences.

## An example

An example test sentence:
```
Test Sentence:
{"id": "ont_music_test_n", "sent": "\"The Loco-Motion\" is a 1962 pop song written by
American songwriters Gerry Goffin and Carole King."}
```

An example ontology:

Ontology: [Music Ontology](data/wikidata_tekgen/ontologies/owl/ont_2_music.ttl)

<img width="1258" alt="music3" src="https://github.com/nandana/iswc-2023/assets/204855/1ff0bfa3-3b2f-4908-9d1b-074d0698485c">

Expected Output:
```
{
 "id": "ont_k_music_test_n",
 "sent": "\"The Loco-Motion\" is a 1962 pop song written by American songwriters Gerry Goffin and Carole King.",
 "triples": [
  {
    "sub": "The Loco-Motion",
    "rel": "publication date",
    "obj": "01 January 1962"
  },{
    "sub": "The Loco-Motion",
    "rel": "lyrics by",
    "obj": "Gerry Goffin"
  },{
    "sub": "The Loco-Motion",
    "rel": "lyrics by",
    "obj": "Carole King"
  },]
}
```

The data is released under under a Creative Commons Attribution-ShareAlike 4.0 International (CC BY 4.0) License.

The structure of the repo is as the following.

- Text2KGBench
  - [assets](assets) contains raw source data, papers, and other assets
  - [config](config) contains the experiment configuration files for running the experiments
  - [src](src) contains the source code used for nlp, llm calls, and other support functions
  - [scripts](scripts) contains scripts used for generation and evaluation, and baseline
    - [evaluate_responses.py](scripts/evaluate_responses.py) evaluation script for calculating the results
    - [generate_prompts.py](scripts/generate_prompts.py) script for generating the baselines including prompts, and sentence similarities
    - [generate_responses.py](scripts/generate_responses.py) script used to generate the benchmark results
  - [data](data) : the benchmark corpora. The original repo had two datasets: wikidata_tekgen and dbpedia_webnlg.
      - [wikidata_tekgen](data/wikidata_tekgen) Wikidata-TekGen Dataset
        - [ontologies](data/wikidata_tekgen/ontologies) ontologies used by this dataset
        - [train](data/wikidata_tekgen/train) training data
        - [test](data/wikidata_tekgen/test) test data
        - [ground_truth](data/wikidata_tekgen/ground_truth) ground truth for the test data
        - [test_train_sent_similarity](data/wikidata_tekgen/test_train_sent_similarity) for each test case, 5 most similar train sentences generated using SBERT T5-XXL model.
        - [prompts](data/wikidata_tekgen/prompts) prompts corresponding to each test file
        - [baselines](data/wikidata_tekgen/baselines) data related to running the baselines.
          - [OpenAI-GPT-4o](data/wikidata_tekgen/baselines/OpenAI-GPT-4o) data related to the OpenAI-GPT-4o model
            - [llm_responses](data/wikidata_tekgen/baselines/OpenAI-GPT-4o/llm_responses) raw LLM responses and extracted triples
            - [eval_metrics](data/wikidata_tekgen/baselines/OpenAI-GPT-4o/eval_metrics) ontology-level and aggregated evaluation results
          - [Qwen2_5-32B-Instruct-Q4KM](data/dbpedia_webnlg/baselines/Qwen2_5-32B-Instruct-Q4KM)
            - The structure of this and other baseline folders is the same as [OpenAI-GPT-4o](data/wikidata_tekgen/baselines/OpenAI-GPT-4o)
          - ...other model baselines
      - [dbpedia_webnlg](data/dbpedia_webnlg)
        - The structure of DBpedia dataset and others is the same as [wikidata_tekgen](data/wikidata_tekgen)
    - ...other corpora

This benchmark contains data derived from TekGen corpus (part of  the KELM corpus) [1] released under CC BY-SA 2.0 license
and WebNLG 3.0 corpus [2] released under CC BY-NC-SA 4.0 license.

[1] Oshin Agarwal, Heming Ge, Siamak Shakeri, and Rami Al-Rfou. 2021. Knowledge Graph Based Synthetic Corpus Generation
for Knowledge-Enhanced Language Model Pre-training. In Proceedings of the 2021 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies, pages 3554–3565, Online.
Association for Computational Linguistics.

[2] Claire Gardent, Anastasia Shimorina, Shashi Narayan, and Laura Perez-Beltrachini. 2017. Creating Training Corpora
for NLG Micro-Planners. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 179–188, Vancouver, Canada. Association for Computational Linguistics.
