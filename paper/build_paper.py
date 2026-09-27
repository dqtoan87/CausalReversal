#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build paper.md from paper_src.md: replace [@key] citations by numbers in order of first appearance and append
the reference list. Keys that are defined but never cited are dropped; an unknown key is an error."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

REFS = {
    "balachandar": "Balachandar, S., Garg, N., & Pierson, E. (2024). Domain constraints improve risk prediction when outcome data is missing. In *International Conference on Learning Representations (ICLR)*.",
    "rambachan": "Rambachan, A., Coston, A., & Kennedy, E. H. (2022). Robust design and evaluation of predictive algorithms under unobserved confounding. *arXiv preprint* arXiv:2212.09844.",
    "coston": "Coston, A., Rambachan, A., & Chouldechova, A. (2021). Characterizing fairness over the set of good models under selective labels. In *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139 (pp. 2144–2155).",
    "mullainathan": "Mullainathan, S., & Obermeyer, Z. (2022). Diagnosing physician error: A machine learning approach to low-value health care. *The Quarterly Journal of Economics*, *137*(2), 679–727.",
    "fithian": "Fithian, W., & Hastie, T. (2014). Local case-control sampling: Efficient subsampling in imbalanced data sets. *The Annals of Statistics*, *42*(5), 1693–1724.",
    "lakkaraju": "Lakkaraju, H., Kleinberg, J., Leskovec, J., Ludwig, J., & Mullainathan, S. (2017). The selective labels problem: Evaluating algorithmic predictions in the presence of unobservables. In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 275–284).",
    "kleinberg": "Kleinberg, J., Lakkaraju, H., Leskovec, J., Ludwig, J., & Mullainathan, S. (2018). Human decisions and machine predictions. *The Quarterly Journal of Economics*, *133*(1), 237–293.",
    "dearteaga": "De-Arteaga, M., Dubrawski, A., & Chouldechova, A. (2018). Learning under selective labels in the presence of expert consistency. Presented at the Workshop on Fairness, Accountability, and Transparency in Machine Learning (FAT/ML). *arXiv preprint* arXiv:1807.00905.",
    "bekkersar": "Bekker, J., Robberechts, P., & Davis, J. (2019). Beyond the selected completely at random assumption for learning from positive and unlabeled data. In *Machine Learning and Knowledge Discovery in Databases: ECML PKDD 2019*, LNCS 11907 (pp. 71–85). Springer.",
    "nguyen": "Nguyen, T. Q., Dafoe, A., & Ogburn, E. L. (2019). The magnitude and direction of collider bias for binary variables. *Epidemiologic Methods*, *8*(1), 20170013.",
    "banack": "Banack, H. R., & Kaufman, J. S. (2014). The obesity paradox: Understanding the effect of obesity on mortality among individuals with cardiovascular disease. *Preventive Medicine*, *62*, 96–102.",
    "kurtansky": "Kurtansky, N. R., D'Alessandro, B. M., Gillis, M. C., Betz-Stablein, B., Cerminara, S. E., Garcia, R., … & Rotemberg, V. (2024). The SLICE-3D dataset: 400,000 skin lesion image crops extracted from 3D TBP for skin cancer detection. *Scientific Data*, *11*(1), 884.",
    "esteva": "Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, *542*(7639), 115–118.",
    "begg": "Begg, C. B., & Greenes, R. A. (1983). Assessment of diagnostic tests when disease verification is subject to selection bias. *Biometrics*, *39*(1), 207–215.",
    "hernan": "Hernán, M. A., Hernández-Díaz, S., & Robins, J. M. (2004). A structural approach to selection bias. *Epidemiology*, *15*(5), 615–625.",
    "berkson": "Berkson, J. (1946). Limitations of the application of fourfold table analysis to hospital data. *Biometrics Bulletin*, *2*(3), 47–53.",
    "kleinbaum": "Kleinbaum, D. G., Kupper, L. L., & Morgenstern, H. (1982). *Epidemiologic Research: Principles and Quantitative Methods*. Lifetime Learning Publications.",
    "greenland": "Greenland, S. (1996). Basic methods for sensitivity analysis of biases. *International Journal of Epidemiology*, *25*(6), 1107–1116.",
    "prentice": "Prentice, R. L., & Pyke, R. (1979). Logistic disease incidence models and case-control studies. *Biometrika*, *66*(3), 403–411.",
    "pacheco": "Pacheco, A. G. C., Lima, G. R., Salomão, A. S., Krohling, B., Biral, I. P., de Angelo, G. G., … & de Barros, L. F. (2020). PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*, *32*, 106221.",
    "lash": "Lash, T. L., Fox, M. P., & Fink, A. K. (2009). *Applying Quantitative Bias Analysis to Epidemiologic Data*. Springer.",
    "griffith": "Griffith, G. J., Morris, T. T., Tudball, M. J., Herbert, A., Mancano, G., Pike, L., … & Hemani, G. (2020). Collider bias undermines our understanding of COVID-19 disease risk and severity. *Nature Communications*, *11*(1), 5749.",
    "alonzo": "Alonzo, T. A., & Pepe, M. S. (2005). Assessing accuracy of a continuous screening test in the presence of verification bias. *Journal of the Royal Statistical Society: Series C*, *54*(1), 173–190.",
    "bareinboim": "Bareinboim, E., & Pearl, J. (2012). Controlling selection bias in causal inference. In *Proceedings of the 15th International Conference on Artificial Intelligence and Statistics* (pp. 100–108).",
    "mohan": "Mohan, K., & Pearl, J. (2021). Graphical models for processing missing data. *Journal of the American Statistical Association*, *116*(534), 1023–1037.",
    "zadrozny": "Zadrozny, B. (2004). Learning and evaluating classifiers under sample selection bias. In *Proceedings of the 21st International Conference on Machine Learning* (p. 114).",
    "heckman": "Heckman, J. J. (1979). Sample selection bias as a specification error. *Econometrica*, *47*(1), 153–161.",
    "elkan": "Elkan, C., & Noto, K. (2008). Learning classifiers from only positive and unlabeled data. In *Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 213–220).",
    "bekker": "Bekker, J., & Davis, J. (2020). Learning from positive and unlabeled data: A survey. *Machine Learning*, *109*(4), 719–760.",
    "winkler": "Winkler, J. K., Fink, C., Toberer, F., Enk, A., Deinlein, T., Hofmann-Wellenhof, R., … & Haenssle, H. A. (2019). Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning convolutional neural network for melanoma recognition. *JAMA Dermatology*, *155*(10), 1135–1141.",
    "oakden": "Oakden-Rayner, L., Dunnmon, J., Carneiro, G., & Ré, C. (2020). Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. In *Proceedings of the ACM Conference on Health, Inference, and Learning* (pp. 151–159).",
    "geirhos": "Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, *2*(11), 665–673.",
    "castro": "Castro, D. C., Walker, I., & Glocker, B. (2020). Causality matters in medical imaging. *Nature Communications*, *11*(1), 3673.",
    "manski": "Manski, C. F. (2003). *Partial Identification of Probability Distributions*. Springer.",
    "tamer": "Tamer, E. (2010). Partial identification in econometrics. *Annual Review of Economics*, *2*, 167–195.",
    "kallus": "Kallus, N., & Zhou, A. (2018). Confounding-robust policy improvement. In *Advances in Neural Information Processing Systems*, *31*, 9269–9279.",
    "alain": "Alain, G., & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. In *International Conference on Learning Representations, Workshop Track*.",
    "kim": "Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., & Sayres, R. (2018). Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In *Proceedings of the 35th International Conference on Machine Learning* (pp. 2668–2677).",
    "kornblith": "Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 3519–3529).",
    "nachbar": "Nachbar, F., Stolz, W., Merkle, T., Cognetta, A. B., Vogt, T., Landthaler, M., … & Plewig, G. (1994). The ABCD rule of dermatoscopy: High prospective value in the diagnosis of doubtful melanocytic skin lesions. *Journal of the American Academy of Dermatology*, *30*(4), 551–559.",
    "he": "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770–778).",
    "platt": "Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In A. J. Smola, P. Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), *Advances in Large Margin Classifiers* (pp. 61–74). MIT Press.",
    "zadroznyelkan": "Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. In *Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 694–699).",
    "friedman": "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, *29*(5), 1189–1232.",
    "efron": "Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.",
}


def main():
    src = open(os.path.join(HERE, "paper_src.md"), encoding="utf-8").read()
    import paper_numbers as PN                                     # mọi số chờ từ JSON, không chép tay
    src = re.sub(r"\{\{N:(\w+)\}\}", lambda m: PN.NUM[m.group(1)](), src)
    src = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda m: PN.TABLE[m.group(1)](), src)
    body, rest = src.split("<!-- REFERENCES -->")
    order = []
    for m in re.finditer(r"\[@([^\]]+)\]", body):
        for k in [x.strip().lstrip("@") for x in m.group(1).split(";")]:
            if k not in REFS:
                raise KeyError(k)
            if k not in order:
                order.append(k)
    num = {k: i + 1 for i, k in enumerate(order)}
    body = re.sub(r"\[@([^\]]+)\]",
                  lambda m: ", ".join(f"[{num[x.strip().lstrip('@')]}]" for x in m.group(1).split(";")), body)
    refs = "\n\n".join(f"[{num[k]}] {REFS[k]}" for k in order)
    open(os.path.join(HERE, "paper.md"), "w", encoding="utf-8").write(body + "## References\n\n" + refs + "\n" + rest)
    print(f"built paper.md with {len(order)} references")


if __name__ == "__main__":
    main()
