# Code for Joint Evaluation of Fairness and Relevance in Recommender Systems with Pareto Frontier ⚖ (TheWebConf/WWW'25 Full Paper - Oral) 

This repository contains the code used for the experiments and analysis in "Joint Evaluation of Fairness and Relevance in Recommender Systems with Pareto Frontier" by Theresia Veronika Rampisela, Tuukka Ruotsalo, Maria Maistro, and Christina Lioma. This work has been accepted to TheWebConf/WWW 2025 for oral presentation.

[[ACM]](https://doi.org/10.1145/3696410.3714589) [[arXiv]](https://arxiv.org/abs/2502.11921)

# Abstract
Fairness and relevance are two important aspects of recommender systems (RSs). Typically, they are evaluated either (i) separately by individual measures of fairness and relevance, or (ii) jointly using a single measure that accounts for fairness with respect to relevance. However, approach (i) often does not provide a reliable joint estimate of the goodness of the models, as it has two different best models: one for fairness and another for relevance. Approach (ii) is also problematic because these measures tend to be ad-hoc and do not relate well to traditional relevance measures, like NDCG. Motivated by this, we present a new approach for jointly evaluating fairness and relevance in RSs: Distance to Pareto Frontier (DPFR). Given some user-item interaction data, we compute their Pareto frontier for a pair of existing relevance and fairness measures, and then use the distance from the frontier as a measure of the jointly achievable fairness and relevance. Our approach is modular and intuitive as it can be computed with existing measures. Experiments with 4 RS models, 3 re-ranking strategies, and 6 datasets show that existing metrics have inconsistent associations with our Pareto-optimal solution, making DPFR a more robust and theoretically well-founded joint measure for assessing fairness and relevance.

# License and Terms of Usage
The code is usable under the MIT License. Please note that RecBole may have different terms of usage (see [their page](https://github.com/RUCAIBox/RecBole) for updated information).

# Citation
If you use the code for the fairness-only measures in `metrics.py`, please cite our paper and the original papers proposing the measures.
```BibTeX
@article{Rampisela2024EvaluationStudy,
author = {Rampisela, Theresia Veronika and Maistro, Maria and Ruotsalo, Tuukka and Lioma, Christina},
title = {Evaluation Measures of Individual Item Fairness for Recommender Systems: A Critical Study},
year = {2024},
issue_date = {June 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {3},
number = {2},
url = {https://doi.org/10.1145/3631943},
doi = {10.1145/3631943},
journal = {ACM Trans. Recomm. Syst.},
month = nov,
articleno = {18},
numpages = {52},
keywords = {Item fairness, individual fairness, fairness measures, evaluation measures, recommender systems}
}
```
If you use the code outside of RecBole's original code, please cite the following:
```BibTeX
@inproceedings{Rampisela2025Pareto,
author = {Rampisela, Theresia Veronika and Ruotsalo, Tuukka and Maistro, Maria and Lioma, Christina},
title = {Joint Evaluation of Fairness and Relevance in Recommender Systems with Pareto Frontier},
year = {2025},
isbn = {9798400712746},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3696410.3714589},
doi = {10.1145/3696410.3714589},
pages = {1548–1566},
numpages = {19},
keywords = {evaluation, fairness, pareto frontier, recommendation, relevance},
location = {Sydney NSW, Australia},
series = {WWW '25}
}
```
# Datasets

## Downloads
We use the following datasets, that can be downloaded from the Google Drive folder provided by [RecBole](https://recbole.io/dataset_list.html), under ProcessedDatasets:
- Amazon-lb: this dataset can be found in the Amazon2018 folder. The name of the folder is Amazon_Luxury_Beauty
- Lastfm is under the LastFM folder
- Jester is under the Jester folder
- ML-10M and ML-20M are under the MovieLens folder; the files are ml-10m.zip and ml-20m.zip respectively

1. Download the zip files corresponding to the full datasets (not the examples) and place them inside an empty `dataset` folder in the main folder.
2. Unzip the files.
3. Ensure that the name of the folder and the .inter files are the same as in the [dataset properties](https://github.com/theresiavr/DPFR-recsys-evaluation/tree/main/RecBole/recbole/properties/dataset).

QK-video can be downloaded from the link on the [Tenrec repository](https://github.com/yuangh-x/2022-NIPS-Tenrec).

## Preprocessing
Please follow the instructions in the notebooks under the `preproc_data` folder.
Further preprocessing settings are configured under `Recbole/recbole/properties/dataset`

# Model training and hyperparameter tuning
Please find the hyperparameter tuning script in the `scripts` folder.
The hyperparameter search space can be found in  `Recbole/hyperchoice`.

To get the output file (struct) corresponding to the best model, run `cluster/get_best_struct_to_eval.py`.

# Reranking
To perform reranking, run `reranking/rerank_subset.py`.

# Evaluation

## Groundwork Evaluation
To compute the evaluation measures for the best model, run `eval/base_eval.py`.

## Pareto frontier generation and DPFR computation
To generate the Pareto Frontier (Oracle2Fair algorithm), run `pareto/generate_pareto.py`.
To generate an estimation of the Pareto Frontier, run `pareto/estimate_pareto.py`.

Example bash scripts for running the Python scripts are provided in the `scripts` folder. Please use the dataset name as an argument.

To compute Distance to Pareto Frontier (DPFR), please refer to `experiments/02_pareto_pair_plot.ipynb`.
