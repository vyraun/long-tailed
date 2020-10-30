## On Long-Tailed Phenomena in NMT. Findings of EMNLP 2020.

Code release date: Nov 1. 2020.

##### Setup
We use fairseq to train the models. Our code is tested on Ubuntu 18.04, with a Conda installation of Python 3.6.

```bash
git clone https://github.com/vyraun/long-tailed.git
pip install .
```

Other Repositories Used (thanks!): 

* https://github.com/neulab/compare-mt
* https://github.com/mjpost/sacrebleu

### Steps to Replicate
Below are the steps to replicate each section of the paper.

#### Section 1: Train the Cross-Entropy Baseline Transformer

The scripts with the prefix 'run' provides the code, from data preparation to evaluation. For example: 

```bash
bash run_iwslt14_de_en.sh
```

Compute the Spearman's Rank Correlation between Norms and Frequencies:

```bash
python norm.py
```

#### Section 2: Characterizing the Long Tail

```bash
cd analysis
bash evauate_splits.sh [model_dir]
bash evauate_model_on_splits.sh [model_dir]
```

The plot can be generated using compare-mt

#### Section 3: Analze Beam Search

```bash
bash evaluate.sh model_dir data_dir
python probs_new.py beam_search.pkl
python probs_all.py [beam_search_*.pkl]
```

#### Section 4: Train Transformer using Focal and Anti-Focal Losses

The loss functions are implemented in the Criterions Directory.

```bash
bash run_iwslt14_de_fc.sh
bash run_iwslt14_de_afc.sh
```

#### Section 5: Tau Normalization Baseline

```bash
cd analysis
bash normalization.sh
```

## Citation
```bibtex
@inproceedings{raunak2020longtailed,
  title = {On Long-Tailed Phenomena in Neural Machine Translation},
  author = {Raunak, Vikas and Dalmia, Siddharth and Gupta, Vivek and Metze, Florian},
  booktitle = {Findings of EMNLP},
  year = 2020,
}
```
