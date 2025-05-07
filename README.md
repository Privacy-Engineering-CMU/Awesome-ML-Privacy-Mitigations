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
- [Libraries & Tools](#libraries--tools)
- [Tutorials & Notebooks](#tutorials--notebooks)
- [Contribute](#contribute)

## Introduction

Machine learning systems increasingly handle sensitive data, making privacy protection essential. Building on the [NIST Adversarial Machine Learning Taxonomy (2025)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf), this repository provides implementation-focused guidance for ML practitioners.

Primary goals:
- ✅ Provide code examples for privacy-preserving techniques
- ✅ Document realistic privacy-utility trade-offs
- ✅ Help practitioners select appropriate techniques for their use case
- ✅ Maintain up-to-date links to libraries, tools, and research

## 1. Data Collection Phase

### 1.1 Data Minimization

**Description**: 
* Collecting only the data necessary for the intended purpose
* Built on two core privacy pillars: purpose limitation and data relevance
* Different from anonymization - focuses on reducing data collection upfront rather than transforming collected data
* Mandated by regulations like GDPR and CCPA as a fundamental privacy principle [[1]](#s1-ref1)

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

* **Feature Selection and Evaluation**
  - Apply feature importance ranking to identify non-essential features [[3]](#s1-ref3)
  - Evaluate correlation between features to avoid redundant data collection
  - Measure model performance impact when removing features
  - Test different feature subsets to find minimal viable feature set

* **Ongoing Governance**
  - Implement data expiration policies to remove data that's no longer needed
  - Review feature requirements when model objectives change
  - Conduct periodic audits to identify and eliminate feature creep

**Algorithms and Tools**:

* **Feature Selection Methods**
  - Column-based filtering with domain expertise validation
  - PDCA (Plan-Do-Check-Act) cycle for iterative minimization
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
* Prevents feature creep and reduces risk surface
* Research shows some models can maintain accuracy with significantly reduced feature sets [[7]](#s1-ref7)
* Impact varies by domain and use case - requires empirical testing

**Important Considerations**:
* Data minimization is not a full fix since various features are inherently correlated
* Proper correlation analysis should be conducted to understand feature relationships
* Domain expertise is crucial for effective minimization without harming model utility
* Regular reassessment is needed as data relevance may change over time

**Code Example**:
```python
# INSTEAD OF:
user_data = collect_all_user_attributes()

# DO THIS:
required_features = ['age_range', 'purchase_history', 'item_interactions']
user_data = collect_specific_attributes(required_features)
```

**References**:

<a id="s1-ref1">[1]</a> [The Data Minimization Principle in Machine Learning (Ganesh et al., 2024)](https://arxiv.org/abs/2405.19471) / [Blog](https://medium.com/data-science/data-minimization-does-not-guarantee-privacy-544ca15c7193) - Empirical exploration of data minimization and its misalignment with privacy, along with potential solutions

<a id="s1-ref2">[2]</a> [Data Minimization for GDPR Compliance in Machine Learning Models (Goldsteen et al., 2022)](https://link.springer.com/article/10.1007/s43681-021-00095-8) - Method to reduce personal data needed for ML predictions while preserving model accuracy through knowledge distillation

<a id="s1-ref3">[3]</a> [From Principle to Practice: Vertical Data Minimization for Machine Learning (Staab et al., 2023)](https://arxiv.org/abs/2311.10500) - Comprehensive framework for implementing data minimization in machine learning with data generalization techniques

<a id="s1-ref4">[4]</a> [Data Shapley: Equitable Valuation of Data for Machine Learning (Ghorbani & Zou, 2019)](https://proceedings.mlr.press/v97/ghorbani19c.html) - Introduces method to quantify the value of individual data points to model performance, enabling systematic data reduction

<a id="s1-ref5">[5]</a> [Algorithmic Data Minimization for ML over IoT Data Streams (Kil et al., 2024)](https://arxiv.org/abs/2503.05675) - Framework for minimizing data collection in IoT environments while balancing utility and privacy

<a id="s1-ref6">[6]</a> [Membership Inference Attacks Against Machine Learning Models (Shokri et al., 2017)](https://arxiv.org/abs/1610.05820) - Pioneering work on membership inference attacks that can be used to audit privacy leakage in ML models

<a id="s1-ref7">[7]</a> [Selecting critical features for data classification based on machine learning methods (Dewi et al., 2020)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00327-4) - Demonstrates that feature selection improves model accuracy and performance while reducing dimensionality

### 1.2 Synthetic Data Generation

**Description**: Create artificial data that preserves statistical properties without containing real individual information.

**Approaches**:
1. Generative Adversarial Networks (GANs)
    * Implementation: Train generator and discriminator networks on original data
    * Best for: Complex, high-dimensional data like images
    * Tools: CTGAN, TableGAN
2. Variational Autoencoders (VAEs)
    * Implementation: Train encoder-decoder architecture with KL divergence
    * Best for: Tabular data with mixed types
    * Tools: SDV (Synthetic Data Vault)
3. SMOTE and Variations
    * Implementation: Generate synthetic minority samples for imbalanced data
    * Best for: Addressing class imbalance
    * Tools: imbalanced-learn Python package

**Code Examples**:

**1. Tabular Data (CTGAN)**:
```python
from sdv.tabular import CTGAN

# Train model
model = CTGAN()
model.fit(real_data)

# Generate synthetic data
synthetic_data = model.sample(num_rows=10000)
```

**2. Differentially Private GAN**:
```python
from pate_gan import PATEGAN

# Initialize model with privacy parameters
model = PATEGAN(
    epsilon=3.0,  # Privacy budget
    delta=1e-5    # Privacy failure probability
)

# Train and generate
model.train(real_data, epochs=100)
synthetic_data = model.generate(1000)
```

**Libraries**:
- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV)
- [CTGAN](https://github.com/sdv-dev/CTGAN)
- [PATE-GAN](https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/pategan)
- [Gretel Synthetics](https://github.com/gretelai/gretel-synthetics)

**Privacy-Utility Trade-offs**:
- Accuracy drop: 5-15% for complex datasets
- Privacy: No direct mapping to original data, but still vulnerable to certain attacks

**Papers**:
- [Modeling Tabular Data using Conditional GAN (Xu et al., 2019)](https://arxiv.org/abs/1907.00503)
- [PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees (Jordon et al., 2019)](https://openreview.net/forum?id=S1zk9iRqF7)

## 2. Data Processing Phase

### 2.1 Local Differential Privacy (LDP)

**Description**: Add calibrated noise to data before it leaves the user's device.

**Code Example**:
```python
# Basic randomized response for binary attributes
def randomized_response(true_value, epsilon=1.0):
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    if true_value == 1:
        return np.random.choice([0, 1], p=[1-p, p])
    else:
        return np.random.choice([0, 1], p=[p, 1-p])
        
# Apply to data before collection
private_data = [randomized_response(value, epsilon=2.0) for value in raw_data]
```

**Privacy Budget Guide**:
- ε = 0.1: Very strong privacy (40-60% accuracy loss)
- ε = 1.0: Strong privacy (15-30% accuracy loss)
- ε = 4.0: Moderate privacy (5-15% accuracy loss)

**Libraries**:
- [PyDP (Google's Differential Privacy)](https://github.com/OpenMined/PyDP)
- [Tumult Analytics](https://github.com/tumult-labs/analytics)
- [IBM Differential Privacy Library](https://github.com/IBM/differential-privacy-library)

**Papers**:
- [RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response (Erlingsson et al., 2014)](https://arxiv.org/abs/1407.6981)
- [Local Differential Privacy: a tutorial (Xiong et al., 2020)](https://arxiv.org/abs/2008.03083)

### 2.2 Secure Multi-Party Computation (SMPC)

**Description**: Enable multiple parties to jointly compute a function over their inputs while keeping those inputs private.

**Code Example**:
```python
# Using MP-SPDZ for secret sharing-based computation
# Party 1 code:
from mpspdz.client import MPSPDZClient

client = MPSPDZClient(party_id=0)
secret_data = client.share_secret(local_features)
aggregated_model = client.run_computation("secure_model_training")
predictions = client.predict(aggregated_model, new_data)
```

**Libraries**:
- [MP-SPDZ](https://github.com/data61/MP-SPDZ)
- [PySyft](https://github.com/OpenMined/PySyft)
- [CrypTen](https://github.com/facebookresearch/CrypTen)
- [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted)

**Papers**:
- [Secure Multiparty Computation (Lindell, 2020)](https://dl.acm.org/doi/10.1145/3387108)

## 3. Model Training Phase

### 3.1 Differentially Private Training

**Description**: Train ML models with mathematical privacy guarantees by adding carefully calibrated noise during optimization.

**Code Example with PyTorch and Opacus**:
```python
import torch
from opacus import PrivacyEngine

model = MyNeuralNetwork()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,     # Higher value = more privacy
    max_grad_norm=1.0         # Gradient clipping threshold
)

# Train with privacy guarantees
for epoch in range(epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Get the privacy spent
epsilon, delta = privacy_engine.get_privacy_spent()
print(f"Privacy budget spent: (ε = {epsilon:.2f}, δ = {delta})")
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

**Privacy-Utility Trade-offs**:
- For ε = 1.0: ~5-15% accuracy drop
- For ε = 3.0: ~2-7% accuracy drop
- For ε = 8.0: ~1-3% accuracy drop
- (Depends heavily on dataset size and task complexity)

**Papers**:
- [Deep Learning with Differential Privacy (Abadi et al., 2016)](https://arxiv.org/abs/1607.00133)
- [Differentially Private Model Publishing for Deep Learning (Yu et al., 2018)](https://arxiv.org/abs/1904.02200)

### 3.2 Federated Learning

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

## 4. Model Deployment Phase

### 4.1 Private Inference

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

### 4.2 Model Anonymization and Protection

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

## 5. Privacy Governance

### 5.1 Privacy Budget Management

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

### 5.2 Privacy Impact Evaluation

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

## 6. Evaluation & Metrics

### 6.1 Privacy Metrics

- **Differential Privacy (ε, δ)**: Smaller values indicate stronger privacy
- **KL Divergence**: Measures information gain from model about training data
- **AUC of Membership Inference**: How well attacks can identify training data (closer to 0.5 is better)
- **Maximum Information Leakage**: Maximum information an adversary can extract

### 6.2 Utility Metrics

- **Privacy-Utility Curves**: Plot of accuracy vs. privacy parameter
- **Performance Gap**: Difference between private and non-private model metrics
- **Privacy-Constrained Accuracy**: Best accuracy achievable under privacy budget constraint

## Libraries & Tools

### Differential Privacy

- [PyDP (Google's Differential Privacy)](https://github.com/OpenMined/PyDP) - Python wrapper for Google's Differential Privacy library
- [Opacus](https://github.com/pytorch/opacus) - PyTorch-based library for differential privacy in deep learning
- [TensorFlow Privacy](https://github.com/tensorflow/privacy) - TensorFlow-based library for differential privacy
- [Diffprivlib](https://github.com/IBM/differential-privacy-library) - IBM's library for differential privacy

### Federated Learning

- [TensorFlow Federated](https://github.com/tensorflow/federated) - Google's framework for federated learning
- [Flower](https://github.com/adap/flower) - A friendly federated learning framework
- [PySyft](https://github.com/OpenMined/PySyft) - Library for secure and private ML with federated learning
- [FATE](https://github.com/FederatedAI/FATE) - Industrial-grade federated learning framework

### Secure Computation

- [TenSEAL](https://github.com/OpenMined/TenSEAL) - Library for homomorphic encryption with tensor operations
- [Microsoft SEAL](https://github.com/microsoft/SEAL) - Homomorphic encryption library
- [CrypTen](https://github.com/facebookresearch/CrypTen) - Framework for privacy-preserving machine learning based on PyTorch
- [MP-SPDZ](https://github.com/data61/MP-SPDZ) - Secure multi-party computation framework

### Synthetic Data

- [SDV](https://github.com/sdv-dev/SDV) - Synthetic data generation ecosystem of libraries
- [Gretel Synthetics](https://github.com/gretelai/gretel-synthetics) - Synthetic data generation with privacy guarantees
- [Synthetic Data Vault](https://github.com/sdv-dev/SDV) - Framework for generating synthetic data

### Privacy Evaluation

- [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter) - Tool for quantifying privacy risks in ML
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) - For evaluating model robustness including privacy attacks

## Tutorials & Notebooks

- [Differential Privacy in TensorFlow (Google)](https://github.com/tensorflow/privacy/tree/master/tutorials)
- [Federated Learning with TFF (Google)](https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification)
- [Private Deep Learning with PyTorch and Opacus (Facebook)](https://github.com/pytorch/opacus/tree/main/examples)
- [Homomorphic Encryption for ML (Microsoft)](https://github.com/microsoft/SEAL/tree/main/examples)
- [Privacy-Preserving Medical Image Analysis (PyMedPhys)](https://github.com/pymedphys/pymedphys/tree/main/examples)


## Contribute

Contributions welcome! Read the [contribution guidelines](CONTRIBUTING.md) first.

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0)
