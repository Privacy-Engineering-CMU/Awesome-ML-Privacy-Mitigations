---
layout: default
title: Awesome ML Privacy Mitigations
---

# Awesome ML Privacy Mitigation [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of practical privacy-preserving techniques for machine learning

This repository aims to bridge the gap between theoretical privacy research and practical implementation in machine learning. Unlike other resources that only provide high-level overviews, we focus on actionable techniques with code examples, specific parameter recommendations, and realistic privacy-utility trade-offs.

## Contents

- [Introduction](#introduction)
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
  - [8.1 Differential Privacy Tutorials](#81-differential-privacy)
  - [8.2 Federated Learning Tutorials](#82-federated-learning)
  - [8.3 Secure Computation Tutorials](#83-secure-computation)
  - [8.4 Synthetic Data Tutorials](#84-synthetic-data)
  - [8.5 Privacy Evaluation Tutorials](#85-privacy-evaluation)
- [Contribute](#contribute)

## Introduction

Machine learning systems increasingly handle sensitive data, making privacy protection essential. Building on the [NIST Adversarial Machine Learning Taxonomy (2025)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf), this repository provides implementation-focused guidance for ML practitioners.

Primary goals:
- ✅ Provide code examples for privacy-preserving techniques
- ✅ Document realistic privacy-utility trade-offs
- ✅ Help practitioners select appropriate techniques for their use case
- ✅ Maintain up-to-date links to libraries, tools, and research

## 1. Data Collection Phase {#1-data-collection-phase}

### 1.1 Data Minimization {#11-data-minimization}

**Description**: 
* Collecting only the data necessary for the intended purpose
* Built on two core privacy pillars: purpose limitation and data relevance
* Different from anonymization - focuses on reducing data collection upfront rather than transforming collected data
* Mandated by regulations like GDPR and CCPA as a fundamental privacy principle [[1]](#s1-ref1)

**NIST AML Attack Mappings**:
* **Primary Mitigation**: [NISTAML.032] Data Reconstruction
* **Additional Protection**:
  * [NISTAML.033] Membership Inference
  * [NISTAML.034] Property Inference

**Why It Matters for ML**:
* Machine learning systems often collect excessive data "just in case," creating unnecessary privacy risks
* Reduces the attack surface and potential harm from data breaches [[2]](#s1-ref2)
* Prevents feature creep that can lead to model overfitting and privacy vulnerabilities
* Simplifies compliance with privacy regulations and builds user trust

**Implementation Approach**:

* **Pre-collection Phase**
  - Conduct a data necessity audit before collection
  - Define explicit variables needed for model functionality based on domain expertise
  - Document justification for each feature's necessity relative to the model's objective
  - Avoid collecting indirect identifiers where possible
  * **Libraries**:
    - [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter) - Privacy risk assessment
    - [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) - Feature importance analysis

* **Feature Selection and Evaluation**
  - Apply feature importance ranking to identify non-essential features [[3]](#s1-ref3)
  - Evaluate correlation between features to avoid redundant data collection
  - Measure model performance impact when removing features
  - Test different feature subsets to find minimal viable feature set
  * **Libraries**:
    - [scikit-learn](https://scikit-learn.org/) - Feature selection utilities
    - [SHAP](https://github.com/slundberg/shap) - Feature importance analysis
    - [Data Shapley](https://github.com/amiratag/DataShapley) - Data valuation

* **Ongoing Governance**
  - Implement data expiration policies to remove data that's no longer needed
  - Review feature requirements when model objectives change
  - Conduct periodic audits to identify and eliminate feature creep

**Algorithms and Tools**:

* **Feature Selection Methods**
  - Column-based filtering with domain expertise validation
  - Feature importance analysis using permutation importance or Shapley values [[4]](#s1-ref4)
  - Privacy Impact Assessment (PIA) frameworks for systematic evaluation
  
* **Privacy Risk Assessment**
  - Evaluate uniqueness of feature combinations
  - Analyze correlation between features and user identifiability [[5]](#s1-ref5)
  - Measure how much each feature contributes to model performance vs. privacy risk

* **Privacy Auditing**
  - Use Membership Inference Attacks (MIAs) to evaluate privacy leakage in models [[6]](#s1-ref6)
  - Test if models trained with minimal data are more resilient to privacy attacks
  - Iteratively adjust feature selection based on audit results

**Utility/Privacy Trade-off**: 

* Minimal impact on model utility if properly implemented with domain expertise
* Models can maintain accuracy with significantly reduced feature sets [[7]](#s1-ref7)
* Impact varies by domain and use case - requires empirical testing

**Important Considerations**:
* Data minimization is not a full fix since various features are inherently correlated
* Proper correlation analysis should be conducted to understand feature relationships
* Domain expertise is crucial for effective minimization without harming model utility
* Regular reassessment is needed as data relevance may change over time

**References**:

<a id="s1-ref1">[1]</a> [The Data Minimization Principle in Machine Learning (Ganesh et al., 2024)](https://arxiv.org/abs/2405.19471) / [Blog](https://medium.com/data-science/data-minimization-does-not-guarantee-privacy-544ca15c7193) - Empirical exploration of data minimization and its misalignment with privacy, along with potential solutions

<a id="s1-ref2">[2]</a> [Data Minimization for GDPR Compliance in Machine Learning Models (Goldsteen et al., 2022)](https://link.springer.com/article/10.1007/s43681-021-00095-8) - Method to reduce personal data needed for ML predictions while preserving model accuracy through knowledge distillation

<a id="s1-ref3">[3]</a> [From Principle to Practice: Vertical Data Minimization for Machine Learning (Staab et al., 2023)](https://arxiv.org/abs/2311.10500) - Comprehensive framework for implementing data minimization in machine learning with data generalization techniques

<a id="s1-ref4">[4]</a> [Data Shapley: Equitable Valuation of Data for Machine Learning (Ghorbani & Zou, 2019)](https://proceedings.mlr.press/v97/ghorbani19c.html) - Introduces method to quantify the value of individual data points to model performance, enabling systematic data reduction

<a id="s1-ref5">[5]</a> [Algorithmic Data Minimization for ML over IoT Data Streams (Kil et al., 2024)](https://arxiv.org/abs/2503.05675) - Framework for minimizing data collection in IoT environments while balancing utility and privacy

<a id="s1-ref6">[6]</a> [Membership Inference Attacks Against Machine Learning Models (Shokri et al., 2017)](https://arxiv.org/abs/1610.05820) - Pioneering work on membership inference attacks that can be used to audit privacy leakage in ML models

<a id="s1-ref7">[7]</a> [Selecting critical features for data classification based on machine learning methods (Dewi et al., 2020)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00327-4) - Demonstrates that feature selection improves model accuracy and performance while reducing dimensionality

### 1.2 Synthetic Data Generation {#12-synthetic-data-generation}

**Description**: 
* Creating artificial data that preserves statistical properties without containing real individual information
* Alternative to traditional anonymization for enabling data sharing and machine learning
* Generates new data points rather than transforming existing ones
* Increasingly adopted for privacy-sensitive applications [[1]](#s2-ref1)

**NIST AML Attack Mappings**:
* **Primary Mitigation**:
  * [NISTAML.037] Training Data Attacks
  * [NISTAML.038] Data Extraction
* **Additional Protection**:
  * [NISTAML.033] Membership Inference

**Why It Matters for ML**:
* Provides training data for models without exposing real individuals' information directly
* Helps address data scarcity and imbalance issues in specialized domains
* Enables experimentation and development while minimizing privacy risks
* Facilitates data sharing across organizations or with researchers [[2]](#s2-ref2)

**Implementation Approaches**:

1. **Generative Adversarial Networks (GANs)**
   * **Mechanism**: Two-network architecture where generator creates samples and discriminator evaluates authenticity
   * **Best For**: Complex, high-dimensional data including tabular, time-series, and images
   * **Variants**: CTGAN for tabular data, PATE-GAN for enhanced privacy guarantees [[3]](#s2-ref3)
   * **Libraries**:
     - [CTGAN](https://github.com/sdv-dev/CTGAN) - Tabular data generation
     - [PATE-GAN](https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/pategan) - Privacy-preserving GAN
     - [Gretel Synthetics](https://github.com/gretelai/gretel-synthetics) - GAN-based synthesis

2. **Variational Autoencoders (VAEs)**
   * **Mechanism**: Encoder-decoder architecture with probabilistic latent space
   * **Best For**: Tabular data with mixed numerical and categorical variables
   * **Variants**: TVAE specifically designed for tabular data [[4]](#s2-ref4)
   * **Libraries**:
     - [SDV](https://github.com/sdv-dev/SDV) - TVAE implementation
     - [Ydata-Synthetic](https://github.com/ydataai/ydata-synthetic) - VAE-based synthesis
     - [Gretel Synthetics](https://github.com/gretelai/gretel-synthetics) - VAE support

3. **Hybrid Approaches**
   * **Mechanism**: Combines VAE's encoding capabilities with GAN's generation abilities
   * **Best For**: Applications requiring both high fidelity and enhanced privacy protection
   * **Recent Advances**: VAE-GAN models with improved membership inference resistance [[5]](#s2-ref5)
   * **Libraries**:
     - [Gretel Synthetics](https://github.com/gretelai/gretel-synthetics) - Hybrid models
     - [Ydata-Synthetic](https://github.com/ydataai/ydata-synthetic) - Advanced synthesis

4. **Traditional Statistical Methods**
   * **Bayesian Networks**: Model conditional dependencies between variables
   * **Copula Methods**: Capture complex correlation structures
   * **SMOTE**: Generate synthetic minority samples for imbalanced data
   * **Libraries**:
     - [SDV](https://github.com/sdv-dev/SDV) - Statistical methods
     - [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) - SMOTE implementation
     - [Copulas](https://github.com/sdv-dev/Copulas) - Copula-based synthesis

**Critical Privacy Evaluation**:

* **Common Evaluation Approaches** [[7]](#s2-ref7)
   * Measuring similarity between synthetic data and original data
   * Testing for successful membership inference attacks
   * Analyzing model performance when trained on synthetic versus real data

* **Limitations of Current Metrics** [[8]](#s2-ref8)
   * Distance-based metrics may not capture actual privacy risks
   * Simple attacker models don't reflect sophisticated real-world attacks
   * Averaged metrics can miss vulnerabilities affecting minority groups or outliers
   * Results often vary significantly with different random initializations

* **Beyond Empirical Metrics**
   * Complementing testing with formal privacy guarantees like differential privacy 
   * Adopting adversarial mindset when evaluating privacy claims
   * Considering multiple attack vectors beyond basic membership inference [[9]](#s2-ref9)

**Important Considerations**:

* **Privacy-Utility Trade-off**
   * Higher privacy protection typically reduces data utility and vice versa
   * Optimal balance depends on specific use case and sensitivity of the data
   * Quantitative measurement of both aspects is essential for decision-making [[10]](#s2-ref10)

* **Technical Challenges**
   * Handling categorical variables effectively
   * Preserving complex relationships between features
   * Scaling to high-dimensional data
   * Computational resources required for training [[11]](#s2-ref11)

* **Deployment Guidance**
   * Validate both utility and privacy before use
   * Consider complementary privacy techniques alongside synthetic data
   * Be cautious of overstated privacy claims from vendors
   * Match evaluation rigor to application sensitivity [[12]](#s2-ref12)

**Implementation Example**:
#need to add sample
[SDV tutorial notebooks](https://docs.sdv.dev/sdv/tutorials)
[Privacy Auditing using ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)
[]

**Best Practices**:

* **Data Preprocessing**
  * Remove direct identifiers before synthetic data generation
  * Consider dimensionality reduction for very high-dimensional data
  * Address class imbalance issues at preprocessing stage

* **Model Selection and Configuration**
  * Choose generation method based on data type and privacy requirements
  * Consider differential privacy mechanisms when possible
  * Tune hyperparameters to balance utility and privacy

* **Evaluation and Validation**
  * Test with multiple privacy metrics at different thresholds
  * Evaluate utility for specific downstream tasks
  * Pay special attention to outliers and minority groups
  * Document privacy evaluation methodology alongside synthetic data

**References**:

<a id="s2-ref1">[1]</a> [Synthetic Data: Revisiting the Privacy-Utility Trade-off (Sarmin et al., 2024)](https://arxiv.org/abs/2407.07926) - Analysis of privacy-utility trade-offs between synthetic data and traditional anonymization

<a id="s2-ref2">[2]</a> [Machine Learning for Synthetic Data Generation: A Review (Zhao et al., 2023)](https://arxiv.org/abs/2302.04062) - Comprehensive review of synthetic data generation techniques and their applications

<a id="s2-ref3">[3]</a> [Modeling Tabular Data using Conditional GAN (Xu et al., 2019)](https://arxiv.org/abs/1907.00503) - Introduces CTGAN, designed specifically for mixed-type tabular data generation

<a id="s2-ref4">[4]</a> [Tabular and latent space synthetic data generation: a literature review (Garcia-Gasulla et al., 2023)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00792-7) - Review of data generation methods for tabular data

<a id="s2-ref5">[5]</a> [Synthetic data for enhanced privacy: A VAE-GAN approach against membership inference attacks (Yan et al., 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124015338) - Novel hybrid approach combining VAE and GAN

<a id="s2-ref6">[6]</a> [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813) - Classic approach for generating synthetic samples for minority classes

<a id="s2-ref7">[7]</a> [Empirical privacy metrics: the bad, the ugly... and the good, maybe? (Desfontaines, 2024)](https://desfontain.es/privacy/empirical-privacy-metrics.html) - Critical analysis of common empirical privacy metrics in synthetic data

<a id="s2-ref8">[8]</a> [Challenges of Using Synthetic Data Generation Methods for Tabular Microdata (Winter & Tolan, 2023)](https://www.mdpi.com/2076-3417/14/14/5975) - Empirical study of trade-offs in different synthetic data generation methods

<a id="s2-ref9">[9]</a> [Privacy Auditing of Machine Learning using Membership Inference Attacks (Yaghini et al., 2021)](https://openreview.net/forum?id=EG5Pgd7-MY) - Framework for privacy auditing in ML models

<a id="s2-ref10">[10]</a> [PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees (Jordon et al., 2019)](https://openreview.net/forum?id=S1zk9iRqF7) - Integrates differential privacy into GANs using the PATE framework

<a id="s2-ref11">[11]</a> [A Critical Review on the Use (and Misuse) of Differential Privacy in Machine Learning (Domingo-Ferrer & Soria-Comas, 2022)](https://dl.acm.org/doi/10.1145/3547139) - Analysis of privacy in ML including synthetic data approaches

<a id="s2-ref12">[12]</a> [Protect and Extend - Using GANs for Synthetic Data Generation of Time-Series Medical Records (2024)](https://arxiv.org/html/2402.14042v1) - Application and evaluation of synthetic data in healthcare domain

**Libraries**:
- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV)
- [CTGAN](https://github.com/sdv-dev/CTGAN)
- [PATE-GAN](https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/pategan)
- [Gretel Synthetics](https://github.com/gretelai/gretel-synthetics)

## 2. Data Processing Phase {#2-data-processing-phase}

### 2.1 Local Differential Privacy (LDP) {#21-local-differential-privacy-ldp}

**NIST AML Attack Mappings**:
* **Primary Mitigation**:
  * [NISTAML.032] Data Reconstruction
  * [NISTAML.033] Membership Inference
* **Additional Protection**:
  * [NISTAML.034] Property Inference

**Description**: 
* Adding calibrated noise to data on the user's device before it leaves their control
* Provides strong privacy guarantees without requiring a trusted central aggregator
* Each user independently applies a randomization mechanism to their own data
* Allows organizations to collect sensitive data while maintaining formal privacy guarantees [[1]](#ldp-ref1r)

**Key Concepts**:

* **Definition**: Algorithm M satisfies ε-LDP if for all possible inputs x, x' and all possible outputs y: 
  ```
  Pr[M(x) = y] ≤ e^ε × Pr[M(x') = y]
  ```

* **Versus Central DP**: LDP typically requires more noise than central DP for the same privacy level but eliminates the need for a trusted data collector [[2]](#ldp-ref2r)

* **Privacy Budget Management**: 
  * ε value controls privacy-utility trade-off
  * Lower ε = stronger privacy but greater accuracy loss
  * Composition: Multiple LDP queries consume privacy budget cumulatively [[3]](#ldp-ref3r)

**Variants of Differential Privacy**:

1. **Pure ε-Differential Privacy**
   * **Definition**: The strictest form, defined by the inequality above
   * **Properties**: 
     - No probability of failure
     - Strict worst-case guarantees
     - Typically requires more noise than relaxed versions
   * **Local Application**: Randomized response, RAPPOR in high-privacy settings [[4]](#ldp-ref4r)

2. **Approximate (ε,δ)-Differential Privacy**
   * **Definition**: Relaxes pure DP by allowing small probability δ of exceeding the privacy bound
   * **Properties**:
     - More practical for many applications
     - Allows δ probability of information leakage
     - Enables more efficient mechanisms
   * **Local Application**: Gaussian mechanism, discrete Laplace in local settings [[5]](#ldp-ref5r)

3. **Rényi Differential Privacy (RDP)**
   * **Definition**: Based on Rényi divergence between output distributions
   * **Properties**:
     - Better handles composition of mechanisms
     - More precise accounting of privacy loss
     - Particularly useful for iterative algorithms
   * **Local Application**: Advanced LDP systems with multiple rounds of communication [[6]](#ldp-ref6r)

4. **Gaussian Differential Privacy (GDP)**
   * **Definition**: Special form that connects DP to hypothesis testing
   * **Properties**:
     - Elegant handling of composition via central limit theorem
     - Natural framework for analyzing mechanisms with Gaussian noise
     - Tighter bounds than (ε,δ)-DP in many cases
   * **Local Application**: Modern private federated learning systems [[7]](#ldp-ref7r)

**Implementation Approaches**:

1. **Randomized Response** (for binary/categorical data)
   * **Mechanism**: Random perturbation of true value based on privacy parameter
   * **Use Case**: Surveys with sensitive yes/no or categorical questions
   * **Variants**: Unary encoding, RAPPOR, Generalized Randomized Response [[8]](#ldp-ref8r)
   * **Libraries**:
     - [OpenDP](https://github.com/opendp/opendp) - Supports randomized response and RAPPOR
     - [IBM Differential Privacy Library](https://github.com/IBM/differential-privacy-library) - Implements RAPPOR and variants
     - [Tumult Analytics](https://github.com/tumult-labs/analytics) - Includes RAPPOR implementation

2. **Laplace Mechanism** (for numerical data)
   * **Mechanism**: Adds noise calibrated to L1 sensitivity
   * **Properties**: 
     - Achieves pure ε-DP
     - Noise proportional to sensitivity/ε
     - Simple to implement
   * **Use Case**: Count queries, sums, averages with bounded sensitivity [[9]](#ldp-ref9r)
   * **Libraries**:
     - [Google's Differential Privacy Library](https://github.com/google/differential-privacy) - Core implementation
     - [OpenDP](https://github.com/opendp/opendp) - Python bindings with Laplace mechanism
     - [IBM Differential Privacy Library](https://github.com/IBM/differential-privacy-library) - Laplace mechanism with utilities

3. **Gaussian Mechanism** (for numerical data)
   * **Mechanism**: Adds noise calibrated to L2 sensitivity
   * **Properties**:
     - Achieves (ε,δ)-DP
     - Better for vector-valued functions (lower noise in high dimensions)
     - Allows leveraging L2 sensitivity
   * **Use Case**: ML model training, high-dimensional statistics [[10]](#ldp-ref10r)
   * **Libraries**:
     - [TensorFlow Privacy](https://github.com/tensorflow/privacy) - DP-SGD implementation
     - [Opacus](https://github.com/pytorch/opacus) - PyTorch-based DP training
     - [Microsoft SmartNoise](https://github.com/opendifferentialprivacy/smartnoise-core) - Core implementation

4. **Advanced Techniques**
   * **Amplification by Shuffling**: Improving privacy by anonymizing source of contributions
   * **Sampled Gaussian Mechanism**: Subsampling data before applying Gaussian noise
   * **Discrete Gaussian**: Better handling of integer-valued functions [[11]](#ldp-ref11r)
   * **Libraries**:
     - [OpenDP](https://github.com/opendp/opendp) - Supports composition and amplification
     - [Tumult Analytics](https://github.com/tumult-labs/analytics) - Advanced composition utilities
     - [IBM Differential Privacy Library](https://github.com/IBM/differential-privacy-library) - Composition tools


**Privacy Budget Considerations**:

* **Selecting Appropriate Parameters**:
  * **ε value**: Controls privacy-utility trade-off in all variants
  * **δ parameter**: Should be smaller than 1/n (n = number of users) for (ε,δ)-DP
  * **α parameter**: Order of Rényi divergence for RDP [[12]](#ldp-ref12r)

* **Composition Advantages of Variants**:
  * **Pure ε-DP**: Simple linear composition (privacy loss adds up)
  * **(ε,δ)-DP**: Better composition via advanced composition theorems
  * **RDP**: Precise tracking of privacy loss under composition
  * **GDP**: Natural composition via central limit theorem [[13]](#ldp-ref13r)

* **Real-World Considerations**:
  * Theoretical guarantees can be undermined by implementation issues
  * Floating-point vulnerabilities can affect all variants
  * Consider robustness to side-channel attacks
  * Balance between formal guarantees and practical utility [[14]](#ldp-ref14r)

**Use Cases by Variant**:

* **Pure ε-DP**: 
  * Simple counts and statistics
  * One-time data collection
  * Highly sensitive applications requiring strict guarantees

* **(ε,δ)-DP with Gaussian Mechanism**:
  * Vector-valued queries (where L2 sensitivity << L1 sensitivity)
  * Applications where moderate relaxation of privacy is acceptable
  * Machine learning with high-dimensional gradients

* **RDP and Advanced Variants**:
  * Iterative algorithms with many composed mechanisms
  * Private machine learning (especially SGD-based)
  * Complex federated analytics systems [[15]](#ldp-ref15r)

**Few Real-World Applications (more available on [Damien's blog](https://desfontain.es/blog/real-world-differential-privacy.html))**:

* **Apple**: iOS/macOS telemetry and emoji suggestions
* **Google**: Chrome browser usage statistics via RAPPOR
* **Microsoft**: Windows telemetry data collection
* **Meta**: Ad delivery optimization without cross-site tracking 

**Libraries and Tools**:

* **[PyDP (OpenMined)](https://github.com/OpenMined/PyDP)**: Python wrapper around Google's C++ DP library
* **[Tumult Analytics](https://github.com/tumult-labs/analytics)**: Open-source DP library with LDP support
* **[IBM Differential Privacy Library](https://github.com/IBM/differential-privacy-library)**: Comprehensive DP toolkit
* **[Microsoft SmartNoise](https://github.com/opendifferentialprivacy/smartnoise-core)**: Extensible DP framework
* **[TensorFlow Privacy](https://github.com/tensorflow/privacy)**: DP for machine learning 

**Resources**:

1. <a id="ldp-ref1r"></a>[A friendly introduction to differential privacy (Desfontaines)](https://desfontain.es/privacy/friendly-intro-to-differential-privacy.html) - Accessible explanation of differential privacy concepts and fundamentals

2. <a id="ldp-ref2r"></a>[Local Differential Privacy: a tutorial (Xiong et al., 2020)](https://arxiv.org/abs/2008.03083) - Comprehensive overview of LDP theory and applications

3. <a id="ldp-ref3r"></a>[RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response (Erlingsson et al., 2014)](https://arxiv.org/abs/1407.6981) - Google's LDP system for Chrome usage statistics

4. <a id="ldp-ref4r"></a>[The Algorithmic Foundations of Differential Privacy (Dwork & Roth, 2014)](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) - Comprehensive textbook on differential privacy

5. <a id="ldp-ref5r"></a>[Approximate Differential Privacy (Programming Differential Privacy)](https://programming-dp.com/ch6.html) - Detailed guide to approximate DP implementation

6. <a id="ldp-ref6r"></a>[Rényi Differential Privacy (Mironov, 2017)](https://arxiv.org/abs/1702.07476) - Original paper introducing RDP

7. <a id="ldp-ref7r"></a>[Gaussian Differential Privacy (Dong et al., 2022)](https://academic.oup.com/jrsssb/article/84/1/3/7056089) - Framework connecting DP to hypothesis testing

8. <a id="ldp-ref8r"></a>[Getting more useful results with differential privacy (Desfontaines)](https://desfontain.es/privacy/more-useful-results-dp.html) - Practical advice for improving utility in DP systems

9. <a id="ldp-ref9r"></a>[A reading list on differential privacy (Desfontaines)](https://desfontain.es/blog/differential-privacy-reading-list.html) - Curated list of papers and resources for learning DP

10. <a id="ldp-ref10r"></a>[Rényi Differential Privacy of the Sampled Gaussian Mechanism (Mironov et al., 2019)](https://arxiv.org/abs/1908.10530) - Analysis of privacy guarantees for subsampled data

11. <a id="ldp-ref11r"></a>[On the Rényi Differential Privacy of the Shuffle Model (Wang et al., 2021)](https://dl.acm.org/doi/10.1145/3460120.3484794) - Analysis of shuffling for privacy amplification

12. <a id="ldp-ref12r"></a>[Differential Privacy: An Economic Method for Choosing Epsilon (Hsu et al., 2014)](https://www.researchgate.net/publication/260211494_Differential_Privacy_An_Economic_Method_for_Choosing_Epsilon) - Framework for epsilon selection based on economic principles

13. <a id="ldp-ref13r"></a>[Functional Rényi Differential Privacy for Generative Modeling (Jalko et al., 2023)](https://dl.acm.org/doi/10.5555/3666122.3666774) - Extension of RDP to functional outputs

14. <a id="ldp-ref14r"></a>[Precision-based attacks and interval refining: how to break, then fix, differential privacy (Haney et al., 2022)](https://desfontain.es/serious.html) - Analysis of vulnerabilities in DP implementations

15. <a id="ldp-ref15r"></a>[Differential Privacy: A Primer for a Non-technical Audience (Wood et al., 2018)](https://journalprivacyconfidentiality.org/index.php/jpc/article/view/659) - Accessible introduction for non-technical readers

16. <a id="ldp-ref16r"></a>[Using differential privacy to harness big data and preserve privacy (Brookings, 2020)](https://www.brookings.edu/articles/using-differential-privacy-to-harness-big-data-and-preserve-privacy/) - Overview of real-world applications

17. <a id="ldp-ref17r"></a>[Tumult Analytics tutorials](https://docs.tmlt.io/analytics/latest/tutorials/tutorial.html) - Practical guide to implementing DP in real-world scenarios

### 2.2 Secure Multi-Party Computation (SMPC) {#22-secure-multi-party-computation-smpc}

**NIST AML Attack Mappings**:
* **Primary Mitigation**:
  * [NISTAML.031] Model Extraction
  * [NISTAML.032] Data Reconstruction

**Description**: Enable multiple parties to jointly compute a function over their inputs while keeping those inputs private.


**Libraries**:
- [MP-SPDZ](https://github.com/data61/MP-SPDZ)
- [PySyft](https://github.com/OpenMined/PySyft)
- [CrypTen](https://github.com/facebookresearch/CrypTen)
- [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted)

**Papers**:
- [Secure Multiparty Computation (Lindell, 2020)](https://dl.acm.org/doi/10.1145/3387108)

## 3. Model Training Phase {#3-model-training-phase}

### 3.1 Differentially Private Training {#31-differentially-private-training}

**NIST AML Attack Mappings**:
* **Primary Mitigation**: [NISTAML.033] Membership Inference
* **Additional Protection**:
  * [NISTAML.032] Data Reconstruction
  * [NISTAML.034] Property Inference

**Description**: Train ML models with mathematical privacy guarantees by adding carefully calibrated noise during optimization.

**Code Example with FastDP by Amazon**:
```python
from fastDP import PrivacyEngine
optimizer = SGD(model.parameters(), lr=0.05)
privacy_engine = PrivacyEngine(
    model,
    batch_size=256,
    sample_size=50000,
    epochs=3,
    target_epsilon=2,
    clipping_fn='automatic',
    clipping_mode='MixOpt',
    origin_params=None,
    clipping_style='all-layer',
)
# attaching to optimizers is not needed for multi-GPU distributed learning
privacy_engine.attach(optimizer) 

#----- standard training pipeline
loss = F.cross_entropy(model(batch), labels)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

**Code Example with TensorFlow Privacy**:
```python
import tensorflow as tf
import tensorflow_privacy as tfp

# Create optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# Make optimizer differentially private
dp_optimizer = tfp.DPKerasSGDOptimizer(
    optimizer,
    noise_multiplier=1.1,
    l2_norm_clip=1.0,
    num_microbatches=1,
    sample_rate=256/60000  # batch_size/dataset_size
)

# Compile model with DP optimizer
model.compile(
    optimizer=dp_optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE
    ),
    metrics=['accuracy']
)
```

**Parameter Selection Guide**:
- Noise multiplier: 0.5-3.0 (higher = more privacy)
- Gradient clipping: 0.1-5.0 (domain dependent)
- Privacy budget: ε = 1-10 (lower = more privacy)

**Libraries**:
- [Opacus (PyTorch)](https://github.com/pytorch/opacus)
- [TensorFlow Privacy](https://github.com/tensorflow/privacy)
- [JAX Privacy](https://github.com/deepmind/jax_privacy)
- [FastDP](https://github.com/awslabs/fast-differential-privacy)

**Privacy-Utility Trade-offs**:
- For ε = 1.0: ~5-15% accuracy drop
- For ε = 3.0: ~2-7% accuracy drop
- For ε = 8.0: ~1-3% accuracy drop
- (Depends heavily on dataset size and task complexity)

**Papers**:
- [Deep Learning with Differential Privacy (Abadi et al., 2016)](https://arxiv.org/abs/1607.00133)
- [Differentially Private Model Publishing for Deep Learning (Yu et al., 2018)](https://arxiv.org/abs/1904.02200)

### 3.2 Federated Learning {#32-federated-learning}

**NIST AML Attack Mappings**:
* **Primary Mitigation**:
  * [NISTAML.038] Data Extraction
  * [NISTAML.037] Training Data Attacks

**Description**: Train models across multiple devices or servers without exchanging raw data.

**Code Example with TensorFlow Federated**:
```python
import tensorflow as tf
import tensorflow_federated as tff

# Define model and optimization
def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(output_classes, activation='softmax')
    ])

def model_fn():
    model = create_model()
    return tff.learning.from_keras_model(
        model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Build the federated training process
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0)
)

# Train the model
state = iterative_process.initialize()
for round_num in range(num_rounds):
    # Select clients for this round
    sample_clients = np.random.choice(client_ids, num_clients_per_round)
    client_datasets = [client_data[client_id] for client_id in sample_clients]
    
    # Run one round of training
    state, metrics = iterative_process.next(state, client_datasets)
    print(f'Round {round_num}: {metrics}')
```

**Libraries**:
- [TensorFlow Federated](https://github.com/tensorflow/federated)
- [PySyft](https://github.com/OpenMined/PySyft)
- [FATE (Federated AI Technology Enabler)](https://github.com/FederatedAI/FATE)
- [Flower](https://github.com/adap/flower)

**Privacy Enhancements**:
- Secure Aggregation: Cryptographic protocol to protect individual updates
- Differential Privacy: Add noise to updates to prevent memorization
- Update Compression: Reduce information content of transmitted updates

**Papers**:
- [Communication-Efficient Learning of Deep Networks from Decentralized Data (McMahan et al., 2017)](https://arxiv.org/abs/1602.05629)
- [Practical Secure Aggregation for Federated Learning on User-Held Data (Bonawitz et al., 2017)](https://arxiv.org/abs/1611.04482)
- [Federated Learning: Strategies for Improving Communication Efficiency (Konečný et al., 2016)](https://arxiv.org/abs/1610.05492)

## 4. Model Deployment Phase {#4-model-deployment-phase}

### 4.1 Private Inference {#41-private-inference}

**NIST AML Attack Mappings**:
* **Primary Mitigation**:
  * [NISTAML.031] Model Extraction
  * [NISTAML.038] Data Extraction

**Description**: Protect privacy during model inference, where both the model and user inputs need protection.

**Code Example with Homomorphic Encryption (TenSEAL)**:
```python
import tenseal as ts
import numpy as np

# Client-side code
# Create context for BFV homomorphic encryption scheme
context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192, plain_modulus=1032193)
context.generate_galois_keys()

# Encrypt input data
x = np.array([[0.1, 0.2, 0.3, 0.4]])
encrypted_x = ts.ckks_vector(context, x)

# Send encrypted_x to server for inference

# Server-side code (computing inference on encrypted data)
def private_inference(encrypted_input, model_weights):
    # First layer computation - matrix multiplication
    weights1 = model_weights[0]
    bias1 = model_weights[1]
    layer1_out = encrypted_input.matmul(weights1) + bias1
    
    # Apply approximate activation function
    # (usually polynomial approximation of ReLU, sigmoid, etc.)
    activated = approximate_activation(layer1_out)
    
    # Additional layers...
    
    # Return encrypted prediction
    return encrypted_prediction

# Client receives and decrypts the result
decrypted_result = encrypted_prediction.decrypt()
```

**Libraries**:
- [TenSEAL](https://github.com/OpenMined/TenSEAL)
- [Microsoft SEAL](https://github.com/microsoft/SEAL)
- [HElib](https://github.com/homenc/HElib)
- [CrypTen](https://github.com/facebookresearch/CrypTen)

**Performance Trade-offs**:
- Homomorphic Encryption: 1000-100000x slowdown, strongest privacy
- Secure Multi-Party Computation: 10-1000x slowdown, balanced approach
- Trusted Execution Environments: 1.1-2x slowdown, weaker guarantees

**Papers**:
- [CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy (Gilad-Bachrach et al., 2016)](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/)
- [GAZELLE: A Low Latency Framework for Secure Neural Network Inference (Juvekar et al., 2018)](https://www.usenix.org/conference/usenixsecurity18/presentation/juvekar)

### 4.2 Model Anonymization and Protection {#42-model-anonymization-and-protection}

**NIST AML Attack Mappings**:
* **Primary Mitigation**: [NISTAML.031] Model Extraction
* **Additional Protection**:
  * [NISTAML.023] Backdoor Poisoning (security-related)

**Description**: Protect the model itself from attacks that aim to extract training data or reverse-engineer model functionality.

**Code Example of Prediction Purification**:
```python
# Prediction purification with calibrated noise
def purify_predictions(model_output, epsilon=1.0, sensitivity=1.0):
    # Calculate noise scale based on sensitivity and privacy parameter
    scale = sensitivity / epsilon
    
    # Add calibrated noise
    noise = np.random.laplace(0, scale, size=model_output.shape)
    purified_output = model_output + noise
    
    # Normalize if probability distribution
    if np.all(model_output >= 0) and np.isclose(np.sum(model_output), 1.0):
        purified_output = np.clip(purified_output, 0, 1)
        purified_output = purified_output / np.sum(purified_output)
        
    return purified_output

# Use in inference pipeline
def private_inference(input_data):
    raw_predictions = model.predict(input_data)
    private_predictions = purify_predictions(raw_predictions, epsilon=2.0)
    return private_predictions
```

**Techniques**:
- Model Distillation: Training a student model on the outputs of a teacher model
- Prediction Purification: Adding noise to model outputs
- Adversarial Regularization: Adding regularization during training to reduce information leakage
- Model Watermarking: Adding imperceptible watermarks to detect model theft

**Libraries**:
- [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

**Papers**:
- [Distillation as a Defense to Adversarial Perturbations Against Deep Neural Networks (Papernot et al., 2016)](https://arxiv.org/abs/1511.04508)
- [Membership Inference Attacks Against Machine Learning Models (Shokri et al., 2017)](https://arxiv.org/abs/1610.05820)

## 5. Privacy Governance {#5-privacy-governance}

### 5.1 Privacy Budget Management {#51-privacy-budget-management}

**NIST AML Attack Mappings**:
* **Risk Management**:
  * [NISTAML.033] Membership Inference
  * [NISTAML.032] Data Reconstruction

**Description**: Track and allocate privacy loss across the ML pipeline.

**Code Example**:
```python
# Using RDP accountant for DP-SGD with budget management
from prv_accountant import PRVAccountant

# Initialize privacy accountant
accountant = PRVAccountant(noise_multiplier=1.1, 
                         sampling_probability=256/50000)

# Track training iterations
for epoch in range(epochs):
    # Update the accountant with batch training
    accountant.step(noise_multiplier=1.1, 
                  sampling_probability=256/50000)
    
    # Check current privacy spent
    epsilon = accountant.get_epsilon(delta=1e-5)
    
    # If budget exceeded, stop training
    if epsilon > privacy_budget:
        print(f"Privacy budget {privacy_budget} exceeded at epoch {epoch}")
        break
```

**Libraries**:
- [TensorFlow Privacy Accountant](https://github.com/tensorflow/privacy)
- [Opacus Accounting](https://github.com/pytorch/opacus)
- [Google's dp-accounting](https://github.com/google/differential-privacy/tree/main/python/dp_accounting)

**Papers**:
- [The Algorithmic Foundations of Differential Privacy (Dwork & Roth, 2014)](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [Renyi Differential Privacy (Mironov, 2017)](https://arxiv.org/abs/1702.07476)

### 5.2 Privacy Impact Evaluation {#52-privacy-impact-evaluation}

**NIST AML Attack Mappings**:
* **Vulnerability Assessment**:
  * [NISTAML.033] Membership Inference
  * [NISTAML.034] Property Inference

**Description**: Quantitatively measure privacy risks in ML systems.

**Code Example**:
```python
from privacy_meter.audit import MembershipInferenceAttack

# Configure the attack
attack = MembershipInferenceAttack(
    target_model=model,
    target_train_data=x_train,
    target_test_data=x_test,
    attack_type='black_box'
)

# Run the attack
attack_results = attack.run()

# Analyze results
accuracy = attack_results.get_attack_accuracy()
auc = attack_results.get_auc_score()
print(f"Attack accuracy: {accuracy}, AUC: {auc}")

# Comparative evaluation
if auc > 0.6:
    print("Privacy protection INSUFFICIENT - model vulnerable to membership inference")
elif auc > 0.55:
    print("Privacy protection MARGINAL - consider additional mitigations")
else:
    print("Privacy protection ADEQUATE against membership inference")
```

**Libraries**:
- [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)
- [Privacy-Preserving Machine Learning in TF](https://github.com/tensorflow/privacy)
- [IMIA (Indirect Membership Inference Attack)](https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021)

**Papers**:
- [Evaluating Differentially Private Machine Learning in Practice (Jayaraman & Evans, 2019)](https://arxiv.org/abs/1902.08874)
- [Machine Learning with Membership Privacy using Adversarial Regularization (Nasr et al., 2018)](https://arxiv.org/abs/1807.05852)

## 6. Evaluation & Metrics {#6-evaluation--metrics}

### 6.1 Privacy Metrics {#61-privacy-metrics}

**NIST AML Attack Mappings**:
* **Comprehensive Coverage**:
  * [NISTAML.033] Membership Inference
  * [NISTAML.032] Data Reconstruction
  * [NISTAML.031] Model Extraction
  * [NISTAML.034] Property Inference

- **Differential Privacy (ε, δ)**: Smaller values indicate stronger privacy
- **KL Divergence**: Measures information gain from model about training data
- **AUC of Membership Inference**: How well attacks can identify training data (closer to 0.5 is better)
- **Maximum Information Leakage**: Maximum information an adversary can extract

### 6.2 Utility Metrics {#62-utility-metrics}

- **Privacy-Utility Curves**: Plot of accuracy vs. privacy parameter
- **Performance Gap**: Difference between private and non-private model metrics
- **Privacy-Constrained Accuracy**: Best accuracy achievable under privacy budget constraint

## 7. Libraries & Tools {#7-libraries--tools}

### 7.1 Differential Privacy {#71-differential-privacy}

- [PyDP (Google's Differential Privacy)](https://github.com/OpenMined/PyDP) - Python wrapper for Google's Differential Privacy library
- [Opacus](https://github.com/pytorch/opacus) - PyTorch-based library for differential privacy in deep learning
- [TensorFlow Privacy](https://github.com/tensorflow/privacy) - TensorFlow-based library for differential privacy
- [Diffprivlib](https://github.com/IBM/differential-privacy-library) - IBM's library for differential privacy
- [Tumult Analytics](https://github.com/tumult-labs/analytics) - Open-source DP library with LDP support
- [Microsoft SmartNoise](https://github.com/opendifferentialprivacy/smartnoise-core) - Extensible DP framework

### 7.2 Federated Learning {#72-federated-learning}

- [TensorFlow Federated](https://github.com/tensorflow/federated) - Google's framework for federated learning
- [Flower](https://github.com/adap/flower) - A friendly federated learning framework
- [PySyft](https://github.com/OpenMined/PySyft) - Library for secure and private ML with federated learning
- [FATE](https://github.com/FederatedAI/FATE) - Industrial-grade federated learning framework
- [FedML](https://github.com/FedML-AI/FedML) - Research-oriented federated learning framework
- [NVFlare](https://github.com/NVIDIA/NVFlare) - NVIDIA's federated learning framework

### 7.3 Secure Computation {#73-secure-computation}

- [TenSEAL](https://github.com/OpenMined/TenSEAL) - Library for homomorphic encryption with tensor operations
- [Microsoft SEAL](https://github.com/microsoft/SEAL) - Homomorphic encryption library
- [CrypTen](https://github.com/facebookresearch/CrypTen) - Framework for privacy-preserving machine learning based on PyTorch
- [MP-SPDZ](https://github.com/data61/MP-SPDZ) - Secure multi-party computation framework
- [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted) - Privacy-preserving machine learning in TensorFlow

### 7.4 Synthetic Data {#74-synthetic-data}

- [SDV](https://github.com/sdv-dev/SDV) - Synthetic data generation ecosystem of libraries
- [Gretel Synthetics](https://github.com/gretelai/gretel-synthetics) - Synthetic data generation with privacy guarantees
- [CTGAN](https://github.com/sdv-dev/CTGAN) - GAN-based tabular data synthesis
- [Ydata-Synthetic](https://github.com/ydataai/ydata-synthetic) - Synthetic data generation for tabular and time-series data

### 7.5 Privacy Evaluation {#75-privacy-evaluation}

- [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter) - Tool for quantifying privacy risks in ML
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) - For evaluating model robustness including privacy attacks
- [TensorFlow Privacy Attacks](https://github.com/tensorflow/privacy) - Implementation of privacy attacks in TensorFlow

## 8. Tutorials & Resources {#8-tutorials--resources}

### 8.1 Differential Privacy Tutorials {#81-differential-privacy}

- [Google's Differential Privacy Tutorial](https://github.com/google/differential-privacy/tree/main/examples)
  - **Language**: C++, Go, Java
  - **Highlights**: Count-min sketch, quantiles, bounded mean and sum implementations

- [OpenDP Tutorial Series](https://docs.opendp.org/en/stable/user/tutorials/index.html)
  - **Language**: Python
  - **Highlights**: Step-by-step tutorials on measurements, transformations, composition

- [Opacus Tutorials](https://github.com/pytorch/opacus/tree/main/tutorials)
  - **Language**: Python (PyTorch)
  - **Highlights**: DP-SGD implementation, privacy accounting, CIFAR-10 training

- [TensorFlow Privacy Tutorials](https://github.com/tensorflow/privacy/tree/master/tutorials)
  - **Language**: Python (TensorFlow)
  - **Highlights**: DP-SGD, membership inference attacks, privacy accounting

- [IBM Differential Privacy Library Tutorials](https://github.com/IBM/differential-privacy-library/tree/main/notebooks)
  - **Language**: Python
  - **Highlights**: DP with scikit-learn integration, classification, regression

### 8.2 Federated Learning Tutorials {#82-federated-learning}

- [TensorFlow Federated Tutorials](https://www.tensorflow.org/federated/tutorials)
  - **Language**: Python (TensorFlow)
  - **Highlights**: Image classification, custom aggregations, federated analytics

- [Flower Federated Learning Tutorials](https://flower.dev/docs/tutorial/quickstart-pytorch.html)
  - **Language**: Python (framework-agnostic)
  - **Highlights**: PyTorch, TensorFlow, scikit-learn integrations, simulation

- [PySyft Tutorials](https://github.com/OpenMined/PySyft/tree/dev/examples)
  - **Language**: Python
  - **Highlights**: Privacy-preserving federated learning, secure aggregation

- [FedML Tutorials](https://doc.fedml.ai/starter/examples.html)
  - **Language**: Python
  - **Highlights**: Cross-device FL, cross-silo FL, mobile device examples

- [NVFlare Examples](https://github.com/NVIDIA/NVFlare/tree/main/examples)
  - **Language**: Python
  - **Highlights**: Medical imaging, federated analytics, custom aggregation

### 8.3 Secure Computation Tutorials {#83-secure-computation}

- [Microsoft SEAL Examples](https://github.com/microsoft/SEAL/tree/main/native/examples)
  - **Language**: C++
  - **Highlights**: Basic operations, encoding, encryption, performance

- [TenSEAL Tutorials](https://github.com/OpenMined/TenSEAL/tree/main/tutorials)
  - **Language**: Python
  - **Highlights**: Encrypted neural networks, homomorphic operations on tensors

- [CrypTen Tutorials](https://github.com/facebookresearch/CrypTen/tree/main/tutorials)
  - **Language**: Python (PyTorch)
  - **Highlights**: Secure multi-party computation for machine learning models

- [TF Encrypted Examples](https://github.com/tf-encrypted/tf-encrypted/tree/master/examples)
  - **Language**: Python (TensorFlow)
  - **Highlights**: Private predictions, secure training, encrypted computations

### 8.4 Synthetic Data Tutorials {#84-synthetic-data}

- [SDV Tutorials](https://docs.sdv.dev/sdv/tutorials)
  - **Language**: Python
  - **Highlights**: Tabular data generation, relational data synthesis, evaluation

- [CTGAN Examples](https://github.com/sdv-dev/CTGAN/tree/master/examples)
  - **Language**: Python
  - **Highlights**: GAN-based tabular data synthesis, training and sampling

- [Gretel Tutorials](https://github.com/gretelai/gretel-synthetics/tree/main/examples)
  - **Language**: Python
  - **Highlights**: Synthetic data with privacy guarantees, quality evaluation

- [Ydata-Synthetic Examples](https://github.com/ydataai/ydata-synthetic/tree/dev/examples)
  - **Language**: Python
  - **Highlights**: GAN models for tabular and time-series data

### 8.5 Privacy Evaluation Tutorials {#85-privacy-evaluation}

- [ML Privacy Meter Tutorial](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/examples)
  - **Language**: Python (TensorFlow)
  - **Highlights**: Membership inference attacks, measuring model privacy leaks

- [Adversarial Robustness Toolbox Tutorials](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main/notebooks/privacy)
  - **Language**: Python
  - **Highlights**: Membership inference, attribute inference, model inversion attacks

- [TensorFlow Privacy Attacks](https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/membership_inference_attack/demos)
  - **Language**: Python (TensorFlow)
  - **Highlights**: Membership inference attack implementation and evaluation

## Contribute {#contribute}

Contributions welcome! Read the [contribution guidelines](CONTRIBUTING.md) first.

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0)
