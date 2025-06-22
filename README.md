# Explainable Differential Diagnosis With Dual-Inference Large Language Models (Dual-Inf)

## 1. Introduction
This repository contains code for the paper "[Explainable Differential Diagnosis With Dual-Inference Large Language Models](https://www.nature.com/articles/s44401-025-00015-6)" (NPJ Health Systems 2025).

![Figure: Dual-Inference Model](Figure_Dual-Inf.jpg)

## 2. Usage
### Requirements:
+ openai==0.28.0
+ python==3.9.19
+ ast
+ pandas


### Datasets:
The annotated dataset Open-XDDx is available in this repo. 
Each sample is annotated with a set of differential diagnoses (i.e., possible diseases) along with their corresponding diagnostic explanations.

---

#### **Figure 1: OpenXDDx Dataset**

<img src="Figure_OpenXDDx_dataset.jpg" alt="Figure: Overview of the investigated scope" width="700px" />

*An example of the annotated data. It contains a set of possible diagnoses and the corresponding evidence that supports the diagnosis.*

---


### Example:
Please modify the 'file_path' in the code to adapt to the path of your data folder.

+ python Code_Dual-Inf.py



For research cooperation, please contact zhou2219 AT umn.edu


## 3. Citation
Please kindly cite the paper if you use the code or any resources in this repo:

Zhou, S., Lin, M., Ding, S. et al. Explainable differential diagnosis with dual-inference large language models. npj Health Syst. 2, 12 (2025). https://doi.org/10.1038/s44401-025-00015-6

or

```bib
@article{zhou2025explainable,
  title={Explainable differential diagnosis with dual-inference large language models},
  author={Zhou, Shuang and Lin, Mingquan and Ding, Sirui and Wang, Jiashuo and Chen, Canyu and Melton, Genevieve B and Zou, James and Zhang, Rui},
  journal={npj Health Systems},
  volume={2},
  number={1},
  pages={12},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

