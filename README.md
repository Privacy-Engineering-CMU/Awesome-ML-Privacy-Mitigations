# Awesome ML Privacy Mitigation [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A community-driven collection of privacy-preserving machine learning techniques, tools, and practical evaluations

This repository serves as a living catalog of privacy-preserving machine learning (PPML) techniques and tools. Building on the [NIST Adversarial Machine Learning Taxonomy (2025)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf), our goal is to create a comprehensive resource where practitioners can find, evaluate, and implement privacy-preserving solutions in their ML workflows.

<details>
<summary>About Our Team</summary>

Our team actively maintains and evaluates the repository by:
- Testing and benchmarking each framework/tool
- Documenting pros, cons, and integration challenges
- Providing practical examples and use cases
- Maintaining an evaluation website with detailed analyses
- Keeping the collection updated with the latest PPML developments
</details>

<details>
<summary>How to Contribute</summary>

We welcome contributions from the community! Whether you're a researcher, practitioner, or enthusiast, you can help by:
- Adding new privacy-preserving tools and frameworks
- Sharing your experiences with existing tools
- Contributing evaluation results
- Suggesting improvements to our documentation
</details>

<details>
<summary>Repository Structure</summary>

Each section includes:
1. **Libraries & Tools**: Practical implementations and frameworks you can use
2. **References**: Research papers, tutorials, and resources for deeper understanding

The techniques covered include:
- Data minimization and synthetic data generation
- Local differential privacy and secure multi-party computation
- Differentially private training and federated learning
- Private inference and model protection
- Privacy governance and evaluation
</details>

## Contents

- [1. Data Collection Phase](#1-data-collection-phase)
  - [1.1 Data Minimization](#11-data-minimization)
  - [1.2 Synthetic Data Generation](#12-synthetic-data-generation)
- [2. Data Processing Phase](#2-data-processing-phase)
  - [2.1 Local Differential Privacy (LDP)](#21-local-differential-privacy-ldp)
  - [2.2 Secure Multi-Party Computation (SMPC)](#22-secure-multi-party-computation-smpc)
- [3. Model Training Phase](#3-model-training-phase)
  - [3.1 Differentially Private Training](#31-differentially-private-training)
  - [3.2 Federated Learning](#32-federated-learning)
- [4. Model Deployment Phase](#4-model-deployment-phase)
  - [4.1 Private Inference](#41-private-inference)
  - [4.2 Model Anonymization and Protection](#42-model-anonymization-and-protection)
- [5. Privacy Governance](#5-privacy-governance)
  - [5.1 Privacy Budget Management](#51-privacy-budget-management)
  - [5.2 Privacy Impact Evaluation](#52-privacy-impact-evaluation)
- [6. Evaluation & Metrics](#6-evaluation--metrics)
  - [6.1 Privacy Metrics](#61-privacy-metrics)
  - [6.2 Utility Metrics](#62-utility-metrics)
- [7. Libraries & Tools](#7-libraries--tools)
  - [7.1 Differential Privacy](#71-differential-privacy)
  - [7.2 Federated Learning](#72-federated-learning)
  - [7.3 Secure Computation](#73-secure-computation)
  - [7.4 Synthetic Data](#74-synthetic-data)
  - [7.5 Privacy Evaluation](#75-privacy-evaluation)
- [8. Tutorials & Resources](#8-tutorials--resources)
  - [8.1 Differential Privacy](#81-differential-privacy)
  - [8.2 Federated Learning](#82-federated-learning)
  - [8.3 Secure Computation](#83-secure-computation)
  - [8.4 Synthetic Data](#84-synthetic-data)
  - [8.5 Privacy Evaluation](#85-privacy-evaluation)
- [Contribute](#contribute)

## 1. Data Collection Phase

### 1.1 Data Minimization

**Libraries & Tools**:
- [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [scikit-learn](https://scikit-learn.org/)
- [SHAP](https://github.com/slundberg/shap)
- [Data Shapley](https://github.com/amiratag/DataShapley)

**References**:
- [The Data Minimization Principle in Machine Learning (Ganesh et al., 2024)](https://arxiv.org/abs/2405.19471) - Empirical exploration of data minimization and its misalignment with privacy, along with potential solutions
- [Data Minimization for GDPR Compliance in Machine Learning Models (Goldsteen et al., 2022)](https://link.springer.com/article/10.1007/s43681-021-00095-8) - Method to reduce personal data needed for ML predictions while preserving model accuracy through knowledge distillation
- [From Principle to Practice: Vertical Data Minimization for Machine Learning (Staab et al., 2023)](https://arxiv.org/abs/2311.10500) - Comprehensive framework for implementing data minimization in machine learning with data generalization techniques
- [Data Shapley: Equitable Valuation of Data for Machine Learning (Ghorbani & Zou, 2019)](https://proceedings.mlr.press/v97/ghorbani19c.html) - Introduces method to quantify the value of individual data points to model performance, enabling systematic data reduction
- [Algorithmic Data Minimization for ML over IoT Data Streams (Kil et al., 2024)](https://arxiv.org/abs/2503.05675) - Framework for minimizing data collection in IoT environments while balancing utility and privacy
- [Membership Inference Attacks Against Machine Learning Models (Shokri et al., 2017)](https://arxiv.org/abs/1610.05820) - Pioneering work on membership inference attacks that can be used to audit privacy leakage in ML models
- [Selecting critical features for data classification based on machine learning methods (Dewi et al., 2020)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00327-4) - Demonstrates that feature selection improves model accuracy and performance while reducing dimensionality

### 1.2 Synthetic Data Generation

**Libraries & Tools**:
- [CTGAN](https://github.com/sdv-dev/CTGAN)
- [PATE-GAN](https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/pategan)
- [Gretel Synthetics](https://github.com/gretelai/gretel-synthetics)
- [SDV](https://github.com/sdv-dev/SDV)
- [Ydata-Synthetic](https://github.com/ydataai/ydata-synthetic)
- [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)
- [Copulas](https://github.com/sdv-dev/Copulas)

**References**:
- [Synthetic Data: Revisiting the Privacy-Utility Trade-off (Sarmin et al., 2024)](https://arxiv.org/abs/2407.07926) - Analysis of privacy-utility trade-offs between synthetic data and traditional anonymization
- [Machine Learning for Synthetic Data Generation: A Review (Zhao et al., 2023)](https://arxiv.org/abs/2302.04062) - Comprehensive review of synthetic data generation techniques and their applications
- [Modeling Tabular Data using Conditional GAN (Xu et al., 2019)](https://arxiv.org/abs/1907.00503) - Introduces CTGAN, designed specifically for mixed-type tabular data generation
- [Tabular and latent space synthetic data generation: a literature review (Garcia-Gasulla et al., 2023)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00792-7) - Review of data generation methods for tabular data
- [Synthetic data for enhanced privacy: A VAE-GAN approach against membership inference attacks (Yan et al., 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124015338) - Novel hybrid approach combining VAE and GAN
- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813) - Classic approach for generating synthetic samples for minority classes
- [Empirical privacy metrics: the bad, the ugly... and the good, maybe? (Desfontaines, 2024)](https://desfontain.es/privacy/empirical-privacy-metrics.html) - Critical analysis of common empirical privacy metrics in synthetic data
- [Challenges of Using Synthetic Data Generation Methods for Tabular Microdata (Winter & Tolan, 2023)](https://www.mdpi.com/2076-3417/14/14/5975) - Empirical study of trade-offs in different synthetic data generation methods
- [Privacy Auditing of Machine Learning using Membership Inference Attacks (Yaghini et al., 2021)](https://openreview.net/forum?id=EG5Pgd7-MY) - Framework for privacy auditing in ML models
- [PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees (Jordon et al., 2019)](https://openreview.net/forum?id=S1zk9iRqF7) - Integrates differential privacy into GANs using the PATE framework
- [A Critical Review on the Use (and Misuse) of Differential Privacy in Machine Learning (Domingo-Ferrer & Soria-Comas, 2022)](https://dl.acm.org/doi/10.1145/3547139) - Analysis of privacy in ML including synthetic data approaches
- [Protect and Extend - Using GANs for Synthetic Data Generation of Time-Series Medical Records (2024)](https://arxiv.org/html/2402.14042v1) - Application and evaluation of synthetic data in healthcare domain

## 2. Data Processing Phase

### 2.1 Local Differential Privacy (LDP)

**Libraries & Tools**:
- [OpenDP](https://github.com/opendp/opendp)
- [IBM Differential Privacy Library](https://github.com/IBM/differential-privacy-library)
- [Tumult Analytics](https://github.com/tumult-labs/analytics)
- [Google's Differential Privacy Library](https://github.com/google/differential-privacy)
- [TensorFlow Privacy](https://github.com/tensorflow/privacy)
- [Opacus](https://github.com/pytorch/opacus)
- [Microsoft SmartNoise](https://github.com/opendifferentialprivacy/smartnoise-core)

**References**:
- [A friendly introduction to differential privacy (Desfontaines)](https://desfontain.es/privacy/friendly-intro-to-differential-privacy.html) - Accessible explanation of differential privacy concepts and fundamentals
- [Local Differential Privacy: a tutorial (Xiong et al., 2020)](https://arxiv.org/abs/2008.03083) - Comprehensive overview of LDP theory and applications
- [RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response (Erlingsson et al., 2014)](https://arxiv.org/abs/1407.6981) - Google's LDP system for Chrome usage statistics
- [The Algorithmic Foundations of Differential Privacy (Dwork & Roth, 2014)](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) - Comprehensive textbook on differential privacy
- [Approximate Differential Privacy (Programming Differential Privacy)](https://programming-dp.com/ch6.html) - Detailed guide to approximate DP implementation
- [Rényi Differential Privacy (Mironov, 2017)](https://arxiv.org/abs/1702.07476) - Original paper introducing RDP
- [Gaussian Differential Privacy (Dong et al., 2022)](https://academic.oup.com/jrsssb/article/84/1/3/7056089) - Framework connecting DP to hypothesis testing
- [Getting more useful results with differential privacy (Desfontaines)](https://desfontain.es/privacy/more-useful-results-dp.html) - Practical advice for improving utility in DP systems
- [A reading list on differential privacy (Desfontaines)](https://desfontain.es/blog/differential-privacy-reading-list.html) - Curated list of papers and resources for learning DP
- [Rényi Differential Privacy of the Sampled Gaussian Mechanism (Mironov et al., 2019)](https://arxiv.org/abs/1908.10530) - Analysis of privacy guarantees for subsampled data
- [On the Rényi Differential Privacy of the Shuffle Model (Wang et al., 2021)](https://dl.acm.org/doi/10.1145/3460120.3484794) - Analysis of shuffling for privacy amplification
- [Differential Privacy: An Economic Method for Choosing Epsilon (Hsu et al., 2014)](https://www.researchgate.net/publication/260211494_Differential_Privacy_An_Economic_Method_for_Choosing_Epsilon) - Framework for epsilon selection based on economic principles
- [Functional Rényi Differential Privacy for Generative Modeling (Jalko et al., 2023)](https://dl.acm.org/doi/10.5555/3666122.3666774) - Extension of RDP to functional outputs
- [Precision-based attacks and interval refining: how to break, then fix, differential privacy (Haney et al., 2022)](https://desfontain.es/serious.html) - Analysis of vulnerabilities in DP implementations
- [Differential Privacy: A Primer for a Non-technical Audience (Wood et al., 2018)](https://journalprivacyconfidentiality.org/index.php/jpc/article/view/659) - Accessible introduction for non-technical readers
- [Using differential privacy to harness big data and preserve privacy (Brookings, 2020)](https://www.brookings.edu/articles/using-differential-privacy-to-harness-big-data-and-preserve-privacy/) - Overview of real-world applications

### 2.2 Secure Multi-Party Computation (SMPC)

**Libraries & Tools**:
- [MP-SPDZ](https://github.com/data61/MP-SPDZ)
- [PySyft](https://github.com/OpenMined/PySyft)
- [CrypTen](https://github.com/facebookresearch/CrypTen)
- [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted)

**References**:
- [Secure Multiparty Computation (Lindell, 2020)](https://dl.acm.org/doi/10.1145/3387108) - Comprehensive overview of SMPC theory and applications

## 3. Model Training Phase

### 3.1 Differentially Private Training

**Libraries & Tools**:
- [Opacus](https://github.com/pytorch/opacus)
- [TensorFlow Privacy](https://github.com/tensorflow/privacy)
- [JAX Privacy](https://github.com/deepmind/jax_privacy)
- [FastDP](https://github.com/awslabs/fast-differential-privacy)

**References**:
- [Deep Learning with Differential Privacy (Abadi et al., 2016)](https://arxiv.org/abs/1607.00133) - Introduces DP-SGD algorithm for training deep neural networks with differential privacy
- [Differentially Private Model Publishing for Deep Learning (Yu et al., 2018)](https://arxiv.org/abs/1904.02200) - Methods for publishing deep learning models with privacy guarantees

### 3.2 Federated Learning

**Libraries & Tools**:
- [TensorFlow Federated](https://github.com/tensorflow/federated)
- [PySyft](https://github.com/OpenMined/PySyft)
- [FATE](https://github.com/FederatedAI/FATE)
- [Flower](https://github.com/adap/flower)

**References**:
- [Communication-Efficient Learning of Deep Networks from Decentralized Data (McMahan et al., 2017)](https://arxiv.org/abs/1602.05629) - Introduces Federated Averaging (FedAvg) algorithm for efficient federated learning
- [Practical Secure Aggregation for Federated Learning on User-Held Data (Bonawitz et al., 2017)](https://arxiv.org/abs/1611.04482) - Cryptographic protocol for secure aggregation in federated learning
- [Federated Learning: Strategies for Improving Communication Efficiency (Konečný et al., 2016)](https://arxiv.org/abs/1610.05492) - Techniques for reducing communication costs in federated learning

## 4. Model Deployment Phase

### 4.1 Private Inference

**Libraries & Tools**:
- [TenSEAL](https://github.com/OpenMined/TenSEAL)
- [Microsoft SEAL](https://github.com/microsoft/SEAL)
- [HElib](https://github.com/homenc/HElib)
- [CrypTen](https://github.com/facebookresearch/CrypTen)

**References**:
- [CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy (Gilad-Bachrach et al., 2016)](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) - Early work on running neural networks on homomorphically encrypted data
- [GAZELLE: A Low Latency Framework for Secure Neural Network Inference (Juvekar et al., 2018)](https://www.usenix.org/conference/usenixsecurity18/presentation/juvekar) - Efficient framework for secure neural network inference using homomorphic encryption

### 4.2 Model Anonymization and Protection

**Libraries & Tools**:
- [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

**References**:
- [Distillation as a Defense to Adversarial Perturbations Against Deep Neural Networks (Papernot et al., 2016)](https://arxiv.org/abs/1511.04508) - Uses model distillation to improve robustness and privacy
- [Membership Inference Attacks Against Machine Learning Models (Shokri et al., 2017)](https://arxiv.org/abs/1610.05820) - Introduces membership inference attacks and defenses

## 5. Privacy Governance

### 5.1 Privacy Budget Management

**Libraries & Tools**:
- [TensorFlow Privacy Accountant](https://github.com/tensorflow/privacy)
- [Opacus Accounting](https://github.com/pytorch/opacus)
- [Google's dp-accounting](https://github.com/google/differential-privacy/tree/main/python/dp_accounting)

**References**:
- [The Algorithmic Foundations of Differential Privacy (Dwork & Roth, 2014)](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) - Comprehensive textbook on differential privacy theory and practice
- [Renyi Differential Privacy (Mironov, 2017)](https://arxiv.org/abs/1702.07476) - Introduces Rényi differential privacy for better privacy accounting

### 5.2 Privacy Impact Evaluation

**Libraries & Tools**:
- [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)
- [Privacy-Preserving Machine Learning in TF](https://github.com/tensorflow/privacy)
- [IMIA](https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021)

**References**:
- [Evaluating Differentially Private Machine Learning in Practice (Jayaraman & Evans, 2019)](https://arxiv.org/abs/1902.08874) - Empirical evaluation of privacy-utility trade-offs in DP ML
- [Machine Learning with Membership Privacy using Adversarial Regularization (Nasr et al., 2018)](https://arxiv.org/abs/1807.05852) - Uses adversarial training to improve membership privacy

## 7. Libraries & Tools

### 7.1 Differential Privacy

- [PyDP](https://github.com/OpenMined/PyDP)
- [Opacus](https://github.com/pytorch/opacus)
- [TensorFlow Privacy](https://github.com/tensorflow/privacy)
- [Diffprivlib](https://github.com/IBM/differential-privacy-library)
- [Tumult Analytics](https://github.com/tumult-labs/analytics)
- [Microsoft SmartNoise](https://github.com/opendifferentialprivacy/smartnoise-core)

### 7.2 Federated Learning

- [TensorFlow Federated](https://github.com/tensorflow/federated)
- [Flower](https://github.com/adap/flower)
- [PySyft](https://github.com/OpenMined/PySyft)
- [FATE](https://github.com/FederatedAI/FATE)
- [FedML](https://github.com/FedML-AI/FedML)
- [NVFlare](https://github.com/NVIDIA/NVFlare)

### 7.3 Secure Computation

- [TenSEAL](https://github.com/OpenMined/TenSEAL)
- [Microsoft SEAL](https://github.com/microsoft/SEAL)
- [CrypTen](https://github.com/facebookresearch/CrypTen)
- [MP-SPDZ](https://github.com/data61/MP-SPDZ)
- [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted)

### 7.4 Synthetic Data

- [SDV](https://github.com/sdv-dev/SDV)
- [Gretel Synthetics](https://github.com/gretelai/gretel-synthetics)
- [CTGAN](https://github.com/sdv-dev/CTGAN)
- [Ydata-Synthetic](https://github.com/ydataai/ydata-synthetic)

### 7.5 Privacy Evaluation

- [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [TensorFlow Privacy Attacks](https://github.com/tensorflow/privacy)

## 8. Tutorials & Resources

### 8.1 Differential Privacy Tutorials

- [Google's Differential Privacy Tutorial](https://github.com/google/differential-privacy/tree/main/examples)
- [OpenDP Tutorial Series](https://docs.opendp.org/en/stable/user/tutorials/index.html)
- [Opacus Tutorials](https://github.com/pytorch/opacus/tree/main/tutorials)
- [TensorFlow Privacy Tutorials](https://github.com/tensorflow/privacy/tree/master/tutorials)
- [IBM Differential Privacy Library Tutorials](https://github.com/IBM/differential-privacy-library/tree/main/notebooks)

### 8.2 Federated Learning Tutorials

- [TensorFlow Federated Tutorials](https://www.tensorflow.org/federated/tutorials)
- [Flower Federated Learning Tutorials](https://flower.dev/docs/tutorial/quickstart-pytorch.html)
- [PySyft Tutorials](https://github.com/OpenMined/PySyft/tree/dev/examples)
- [FedML Tutorials](https://doc.fedml.ai/starter/examples.html)
- [NVFlare Examples](https://github.com/NVIDIA/NVFlare/tree/main/examples)

### 8.3 Secure Computation Tutorials

- [Microsoft SEAL Examples](https://github.com/microsoft/SEAL/tree/main/native/examples)
- [TenSEAL Tutorials](https://github.com/OpenMined/TenSEAL/tree/main/tutorials)
- [CrypTen Tutorials](https://github.com/facebookresearch/CrypTen/tree/main/tutorials)
- [TF Encrypted Examples](https://github.com/tf-encrypted/tf-encrypted/tree/master/examples)

### 8.4 Synthetic Data Tutorials

- [SDV Tutorials](https://docs.sdv.dev/sdv/tutorials)
- [CTGAN Examples](https://github.com/sdv-dev/CTGAN/tree/master/examples)
- [Gretel Tutorials](https://github.com/gretelai/gretel-synthetics/tree/main/examples)
- [Ydata-Synthetic Examples](https://github.com/ydataai/ydata-synthetic/tree/dev/examples)

### 8.5 Privacy Evaluation Tutorials

- [ML Privacy Meter Tutorial](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/examples)
- [Adversarial Robustness Toolbox Tutorials](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main/notebooks/privacy)
- [TensorFlow Privacy Attacks](https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/membership_inference_attack/demos)

## Contribute

Contributions welcome! Read the [contribution guidelines](CONTRIBUTING.md) first.

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0)
