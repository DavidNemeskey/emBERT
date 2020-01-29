# emBERT

[`emtsv`](https://github.com/dlt-rilmta/emtsv) module for pre-trained Transfomer-based
models. It provides tagging modelsbased on
[Huggingface's `transformers`](https://github.com/huggingface/transformers) package.

`emBERT` defines the following tools:

| Name(s) | Task | Training corpus | F1 score |
| ------- | ---- | --------------- | -------- |
| `bert-ner` | NER | Szeged NER corpus | 97.08\% |
| `bert-basenp` | base NP chunking | Szeged TreeBank 2.0 | **95.58\%** |
| `bert-np` (or `bert-chunk`) | maximal NP chunking | Szeged TreeBank 2.0 | **95.05\%** |

(The results in **bold** are state-of-the-art for Hungarian.)

Due to their size (a little over 700M apiece), the models are stored in a separate
repository. [emBERT-models](https://github.com/dlt-rilmta/emBERT-models)
is a submodule of this repository, so if cloned recursively with `git` LFS,
the models will be downloaded as well:
```
git clone --recursive https://github.com/DavidNemeskey/emBERT.git
```

Alternatively, the models can be obtained via `emtsv`'s `download_models.py` script.

If you use `emBERT` in your work, please cite the following paper
([see link for bib](https://hlt.bme.hu/en/publ/embert_2020); Hungarian):

Nemeskey Dávid Márk 2020. Egy emBERT próbáló feladat. In Proceedings of the
16th Conference on Hungarian Computational Linguistics (MSZNY 2020). pp. 409-418.
