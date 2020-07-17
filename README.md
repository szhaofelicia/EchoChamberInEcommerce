# Understanding Echo Chambers in E-commerce Recommender Systems (https://arxiv.org/abs/2007.02474)

The codebase is useful for measuring Echo Chamber in e-commerce recommender system. It was used in the SIGIR 2020 paper Understanding Echo Chambers in E-commerce Recommender Systems (https://arxiv.org/abs/2007.02474). This method could be also used for other RS scenarios.

# Datasets

1. User embeddings:

    * Click embeddings of Following Group (jsons/pos_user_click_embed.json) and Ignoring Group (jsons/neg_user_click_embed.json)
    * Purchase embeddings of Following group (jsons/pos_user_purchase_embed.json) and Ignoring group (jsons/neg_user_purchase_embed.json)

2. Content diversity based on item embeddings:
    * Following Group (jsons/pos_user_display_diversity.json) and Ignoring Group (jsons/neg_user_display_diversity.json)
# Measures for Echo Chamber
We measure the effect in two steps:

1. Measure reinforcement in user interests via clustering analysis.

    * Detect clustering tendency (hopkins.py)
    * Select proper clustering settings (bic.py) and plot results for each user group (plot/bic_plot.py).
    * Measure internal validity index: Calinski-Harabasz (ch.py), boxplots of eacho group (plot/ch_plot.py).
    * Measure external validity index: Adjusted Rand Index(ari.py), boxplots of eacho group (plot/ari_plot.py).
    * Results are saved in pickle files (pickle/*.pickle).
2. Measure changes of content diversity in recommendations.

    * Computer average content diversity in each group (diversity.py), plot distribution of content diversityn (plot/diversity_plot.py).


## Presentation
The video (ind0007.mp4) and slides (presentation.pdf) are for Our presentation on SIGIR 2020.


## Citation

If you found our paper/code useful in your research, please consider citing
our paper:

```
@article{2007.02474,
Author = {Yingqiang Ge and Shuya Zhao and Honglu Zhou and Changhua Pei and Fei Sun and Wenwu Ou and Yongfeng Zhang},
Title = {Understanding Echo Chambers in E-commerce Recommender Systems},
Year = {2020},
Eprint = {arXiv:2007.02474},
Doi = {10.1145/3397271.3401431},
}
```