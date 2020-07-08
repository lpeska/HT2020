# HT2020
Source codes, raw data and complete results of  paper Off-line vs. On-line Evaluation of Recommender Systems in Small E-commerce accepted for HT2020 conference

The paper as well as this repository is partially based on the preliminary version (https://arxiv.org/abs/1809.03186) presented at REVEAL 2018 vorkshop. For instructions on running the recommending algorithms, please refer to https://github.com/lpeska/REVEAL2018 repository. This repository contains scripts to process on-line and off-line evaluation, regression models aiming to predict the on-line results based on the off-line ones and results of the second on-line experiment aiming to evaluate these models.

- original on-line results are processed in OnlineResultsEvaluation.ipynb file
- several off-line to on-line prediction models are evaluated there as well
- results of scedond on-line experiment are within the online_results2 folder

## Abstract
In this paper, we present our work towards comparing on-line and off-line evaluation metrics in the context of small e-commerce recommender systems. Recommending on small e-commerce enterprises is rather challenging due to the lower volume of interactions and low user loyalty, rarely extending beyond a single session. On the other hand, we usually have to deal with lower volumes of objects, which are easier to discover by users through various browsing/searching GUIs.

The main goal of this paper is to determine applicability of off-line evaluation metrics in learning true usability of recommender systems (evaluated on-line in A/B testing). In total 800 variants of recommenders were evaluated off-line w.r.t. 18 metrics covering rating-based, ranking-based, novelty and diversity evaluation. The off-line results were afterwards compared with on-line evaluation of 12 selected recommender variants and based on the results, we tried to learn and utilize an off-line to on-line results prediction model.

Off-line results shown a great variance in performance w.r.t. different metrics with the Pareto front covering 64\% of the approaches. Furthermore, we observed that on-line results are considerably affected by the seniority of users. On-line metrics correlates positively with ranking-based metrics (AUC, MRR, nDCG) for novice users, while too high values of  novelty had a negative impact on the on-line results for them.

## Novelty compared to the previous version
Finally, this paper contains numerous extensions as compared to our preliminary work in this field (https://arxiv.org/abs/1809.03186). The main ones are re-defined VRR and novelty metrics, more thorough results analysis including the effect of user profile size, evaluated regression algorithms aiming to learn on-line CTR and VRR results from the off-line metrics and additional on-line evaluations.
