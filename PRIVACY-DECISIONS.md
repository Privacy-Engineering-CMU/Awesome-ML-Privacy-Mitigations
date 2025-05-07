# ML Privacy Technique Decision Tree

This decision tree helps practitioners select appropriate privacy techniques based on their specific requirements, constraints, and use cases.

## Step 1: Determine Your Primary Privacy Risk

Start by identifying your primary privacy concerns:

```
Is your main concern:
├── Protecting individual training data? → Go to Step 2
├── Protecting the model itself? → Go to Step 7
├── Protecting inference data? → Go to Step 8
└── Regulatory compliance? → Go to Step 9
```

## Step 2: Evaluate Your Data Sensitivity

Assess how sensitive your data is:

```
How sensitive is your data?
├── High sensitivity (medical, financial, biometric) → Go to Step 3, consider ε ≤ 1.0
├── Medium sensitivity (behavioral, preferences) → Go to Step 3, consider ε = 3.0-5.0
└── Low sensitivity (non-personal, aggregate) → Go to Step 3, consider ε = 8.0-10.0
```

## Step 3: Identify Your Deployment Environment

Consider where your system will be deployed:

```
Where will your ML system be deployed?
├── On client devices (phones, browsers, IoT) → Go to Step 4A (Federated Learning Path)
├── Trusted centralized server → Go to Step 4B (Central DP Path)
├── Untrusted server/cloud → Go to Step 4C (Crypto Path)
└── Multi-party collaboration → Go to Step 4D (SMPC Path)
```

## Step 4A: Federated Learning Path

For systems running on client devices:

```
Federated Learning Path:
├── Communication-constrained? → Consider model compression, efficient FL algorithms
├── Want protection from server? → Add secure aggregation protocol
├── Need formal privacy guarantees? → Add local differential privacy
└── Client trust concerns? → Add verifiable computation or TEEs
```

**Recommended Techniques:**
- TensorFlow Federated or PyTorch Flower for implementation
- Local differential privacy with ε based on sensitivity from Step 2
- Secure aggregation protocol for enhanced security
- Model compression for efficiency

## Step 4B: Central DP Path

For systems with a trusted central server:

```
Central DP Path:
├── Training neural networks? → Use DP-SGD with appropriate parameters
├── Working with structured data? → Use differentially private query mechanisms
├── Need high accuracy? → Consider DP with tight composition bounds
└── Concerned about memorization? → Add privacy auditing techniques
```

**Recommended Techniques:**
- Opacus (PyTorch) or TensorFlow Privacy for DP-SGD implementation
- Privacy budget (ε) from Step 2
- Rényi DP for tighter composition
- Privacy unit definition based on domain (user-level vs. sample-level)

## Step 4C: Crypto Path

For systems on untrusted servers:

```
Crypto Path:
├── Simple operations only? → Use homomorphic encryption (CKKS/BFV)
├── Complex model needed? → Consider hybrid approaches (partial HE + MPC)
├── Performance critical? → Consider TEEs (e.g., Intel SGX, AMD SEV)
└── Need both training and inference? → Combine FL with crypto techniques
```

**Recommended Techniques:**
- Microsoft SEAL or TenSEAL for homomorphic encryption
- CrypTen for secure multi-party computation
- Model simplification to work with encryption constraints
- TEEs where appropriate with side-channel protections

## Step 4D: SMPC Path

For multi-party collaboration scenarios:

```
SMPC Path:
├── Few parties (<10)? → Use garbled circuits or secret sharing
├── Many parties? → Use federated learning with secure aggregation
├── High communication cost acceptable? → Full SMPC with malicious security
└── Low communication needed? → Hybrid approaches with trusted parties
```

**Recommended Techniques:**
- MP-SPDZ or CrypTen for generic secure computation
- Secret sharing-based protocols for linear operations
- Garbled circuits for complex non-linear functions
- Define appropriate threat model (semi-honest vs. malicious)

## Step 5: Consider Computational Constraints

Assess your computational resources:

```
What are your computational constraints?
├── High-performance computing available → Can use more complex privacy techniques
├── Limited computing resources → Need to optimize for efficiency
├── Edge/mobile deployment → Lightweight techniques essential
└── Can afford preprocessing → Consider hybrid approaches with offline preparation
```

## Step 6: Balance Privacy-Utility Trade-offs

Determine your acceptable utility loss:

```
What utility loss can you tolerate?
├── Minimal loss acceptable → Use weaker privacy parameters, utility-optimized algorithms
├── Moderate loss acceptable → Use standard recommendations
└── High loss acceptable → Use strongest privacy guarantees
```

## Step 7: Model Protection Path

If protecting the model is your primary concern:

```
Model Protection Path:
├── Preventing model stealing? → Implement watermarking, distillation, or API limitations
├── Preventing membership inference? → Use differential privacy during training
├── Preventing model inversion? → Add output perturbation and confidence masking
└── Protecting model parameters? → Consider encrypted models or TEEs
```

**Recommended Techniques:**
- Model distillation with public or synthetic data
- Output prediction calibration or purification
- Defense against membership inference attacks
- Watermarking techniques for model protection

## Step 8: Private Inference Path

If protecting inference data is your primary concern:

```
Private Inference Path:
├── Need provable security? → Use fully homomorphic encryption (slow)
├── Balance of security/performance? → Use hybrid cryptographic approaches
├── Performance critical? → Use TEEs with appropriate side-channel protections
└── Local inference possible? → Run model on client device
```

**Recommended Techniques:**
- CryptoNets-style architecture for homomorphic inference
- DELPHI or GAZELLE for hybrid protocols
- Intel SGX or AMD SEV for TEE-based solutions
- Model partitioning between client and server

## Step 9: Regulatory Compliance Path

If meeting specific regulations is your primary concern:

```
Regulatory Compliance Path:
├── GDPR compliance? → Focus on transparency, purpose limitation, data minimization
├── HIPAA compliance? → Focus on de-identification and access controls
├── CCPA/CPRA compliance? → Focus on disclosure and purpose limitation
└── Industry-specific regulations? → Consult domain experts and legal counsel
```

**Recommended Techniques:**
- Differential privacy with formal guarantees
- Synthetic data generation with privacy protections
- Federated analytics for aggregate insights
- Privacy impact assessments and documentation

## Decision Tree Example

To illustrate how to use this decision tree, consider the following example:

**Example: Healthcare prediction model using data from multiple hospitals**

1. **Primary Privacy Risk**: Protecting individual training data
2. **Data Sensitivity**: High (medical data) → ε ≤ 1.0
3. **Deployment Environment**: Multi-party collaboration → SMPC Path
4. **SMPC Path**: Multiple hospitals → Federated learning with secure aggregation
5. **Computational Constraints**: High-performance computing available
6. **Privacy-Utility Trade-offs**: Moderate loss acceptable

**Final Recommendation**:
- Implement federated learning across hospitals
- Add differential privacy (ε = 1.0) during local training
- Use secure aggregation protocol for model updates
- Implement regular privacy auditing with membership inference attacks
- Apply model distillation before deployment