# On the Relation between Sensitivity and Accuracy in In-context Learning

This is the implementation of the paper [On the Relation between Sensitivity and Accuracy in In-context Learning](https://aclanthology.org/2023.findings-emnlp.12/). 

## Overview
In-context learning (ICL) suffers from oversensitivity to the prompt, making it unreliable in real-world scenarios. We study the sensitivity of ICL with respect to multiple perturbation types. First, we find that label bias obscures the true sensitivity, and therefore prior work may have significantly underestimated ICL sensitivity. Second, we observe a strong **negative correlation between ICL sensitivity and accuracy**: predictions sensitive to perturbations are less likely to be correct. 

Motivated by these findings, we propose ***SenSel***, a few-shot selective prediction method that abstains from sensitive predictions. Experiments on ten classification datasets show that SenSel consistently outperforms two commonly used confidence-based and entropy-based baselines on abstention decisions.

You could find more details of this work in our [paper](https://aclanthology.org/2023.findings-emnlp.12/).

## Questions?

If you have any questions related to the code or the paper, feel free to reach out to us at `yanda.chen@cs.columbia.edu`.

## Citation

```bibtex
@inproceedings{chen-etal-2023-relation,
    title = "On the Relation between Sensitivity and Accuracy in In-Context Learning",
    author = "Chen, Yanda  and
      Zhao, Chen  and
      Yu, Zhou  and
      McKeown, Kathleen  and
      He, He",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.12",
    doi = "10.18653/v1/2023.findings-emnlp.12",
    pages = "155--167"
}
```