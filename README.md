# compsumm
Code, Datasets and Supplementary Appendix for AAAI-19 paper [**Comparative Document Summarisation via Classification**](https://arxiv.org/abs/1812.02171)

**Supplementary Appendix**: [pdf](/appendix.pdf)

## How to use this repository ?

### 1. Installing
If you have miniconda or anaconda, please use `install.sh` to install new env `compsumm` that has all dependencies; otherwise the dependencies are listed in `environment.yml`

### 2. Dataset
The dataset are in directory `dataset` in `HDF5` format. There are three files for each of the three news topics used in paper. Each file has following structure:
```
-- data: Averaged GLOVE vectors of title and first 3 sentences, 300 dimensional
-- y: labels created by dividing timeranges into two groups
-- yn: labels created using month for beefban and wee for capital punishment and guncontrol.
-- title: title of article
-- text: first three sentences
-- datetime: date of publication

The dataset was split 70-20-10 as train-test-val sets 51 times. The precomputed splits are available in:
-- train_idxs: Matrix with each row i containing training indexes of split i.
-- test_idxs: Matrix with each row i containing test indexes of split i.
-- val_idxs: Matrix with each row i containing val indexes of split i.
```
The paper uses first 10 splits to compute error bars in automatic evaluations results. 
Please see `news.py` for example loading of this dataset.

### 3. Code
Please see [demo notebook](/demo.ipynb) for example use of `subm.py` and `grad.py`
- `subm.py` has utility functions and greedy optimiser for discrete optimisation.
- `grad.py` has utility functions and SGD optimiser for continuous optimisation. SGD optimised wasn't used, LBFGS from scipy was used instead.

`models.py` has several models for summarisation as classifiers. Models were abstracted into `Summ` class. This is particularly useful in creating common pattern for different summariser methods and in tuning hyperparameters. Please see [models notebook](/models.ipynb) for demo of `news.py` and `models.py`.

`utils.py` has common functions such as `balanced accuracy`, which is used for evaluation.

### 4. Crowd-sourced evaluations results
The crowd-sourced evaluations results is in file `crowdflower.csv`. The design and settings for this experiment is explained [in the paper](https://arxiv.org/abs/1812.02171).

## Citing
If you use this dataset, please cite this work at
```
@inproceedings{bista2019compsumm,
  title={Comparative Document Summarisation via Classification},
  author={Bista, Umanga and Mathews, Alexander and Shin, Minjeong and Menon, Aditya Krishna and Xie, Lexing},
  booktitle={AAAI},
  volume={},
  pages={},
  year={2019}
}
```
The paper can be [found in arxiv](https://arxiv.org/abs/1812.02171)
