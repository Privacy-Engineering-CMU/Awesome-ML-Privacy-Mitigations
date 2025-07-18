import React, { useState, useMemo } from 'react';
import { Shield, Database, BrainCircuit, FileText, Scale, Cpu, Users, Code, KeyRound, CheckCircle, ArrowRight, RefreshCw, Info, Smartphone, Server, Lock, Group } from 'lucide-react';

// --- Helper Components ---
const Tooltip = ({ text, children }) => {
  const [isVisible, setIsVisible] = useState(false);
  return (
    <div className="relative inline-flex items-center" onMouseEnter={() => setIsVisible(true)} onMouseLeave={() => setIsVisible(false)}>
      {children}
      {isVisible && (
        <div className="absolute bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-sm rounded-lg shadow-lg z-10">
          {text}
        </div>
      )}
    </div>
  );
};

const ProgressBar = ({ current, total }) => {
  const percentage = total > 1 ? ((current - 1) / (total - 1)) * 100 : 0;
  return (
    <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 mb-8">
      <div className="bg-blue-600 h-2.5 rounded-full transition-all duration-500 ease-in-out" style={{ width: `${percentage}%` }}></div>
    </div>
  );
};

// --- Main Application Component ---
const App = () => {
  const [currentStep, setCurrentStep] = useState('start');
  const [history, setHistory] = useState([]);
  const [context, setContext] = useState({});

  const handleAnswer = (nextStep, newContext = {}) => {
    setHistory([...history, { step: currentStep, context: context }]);
    setCurrentStep(nextStep);
    setContext({ ...context, ...newContext });
  };

  const handleBack = () => {
    const lastState = history.pop();
    if (lastState) {
        setHistory([...history]);
        setCurrentStep(lastState.step);
        setContext(lastState.context);
    }
  };
  
  const handleReset = () => {
    setCurrentStep('start');
    setHistory([]);
    setContext({});
  }

  const decisionTree = useMemo(() => ({
    'start': {
      question: "What is your primary privacy risk?",
      icon: <Scale className="h-12 w-12 mx-auto text-gray-500" />,
      options: [
        { text: "Protecting Individual Training Data", next: 'dataSensitivity', icon: <Database className="h-8 w-8" /> },
        { text: "Protecting the Model Itself", next: 'modelProtectionPath', icon: <BrainCircuit className="h-8 w-8" /> },
        { text: "Protecting Inference Data", next: 'privateInferencePath', icon: <Users className="h-8 w-8" /> },
        { text: "Ensuring Regulatory Compliance", next: 'compliancePath', icon: <FileText className="h-8 w-8" /> },
      ]
    },
    'dataSensitivity': {
      question: "How sensitive is your training data?",
      icon: <Shield className="h-12 w-12 mx-auto text-gray-500" />,
      tooltip: "Data sensitivity determines the level of noise needed. More sensitive data requires stronger privacy guarantees (lower ε).",
      options: [
        { text: "High (medical, financial)", next: 'deploymentEnvironment', context: { sensitivity: 'High', epsilon: 'ε ≤ 1.0' }, icon: <div className="text-red-500 font-bold">High</div> },
        { text: "Medium (behavioral, preferences)", next: 'deploymentEnvironment', context: { sensitivity: 'Medium', epsilon: 'ε = 3.0-5.0' }, icon: <div className="text-yellow-500 font-bold">Medium</div> },
        { text: "Low (non-personal, aggregate)", next: 'deploymentEnvironment', context: { sensitivity: 'Low', epsilon: 'ε = 8.0-10.0' }, icon: <div className="text-green-500 font-bold">Low</div> },
      ]
    },
    'deploymentEnvironment': {
        question: "Where will your ML system be deployed?",
        icon: <Cpu className="h-12 w-12 mx-auto text-gray-500" />,
        options: [
            { text: "On client devices (phones, IoT)", next: 'solution_federated', icon: <Smartphone className="h-8 w-8" /> },
            { text: "Trusted centralized server", next: 'solution_central_dp', icon: <Server className="h-8 w-8" /> },
            { text: "Untrusted server/cloud", next: 'solution_crypto', icon: <Lock className="h-8 w-8" /> },
            { text: "Multi-party collaboration", next: 'solution_smpc', icon: <Group className="h-8 w-8" /> },
        ]
    },
    // --- Solution Nodes ---
    'solution_federated': {
        isSolution: true,
        title: "Federated Learning Path",
        icon: <Smartphone className="h-12 w-12 mx-auto text-green-500" />,
        description: "For systems running on client devices, Federated Learning (FL) is the ideal architecture. Data stays on the user's device, preserving privacy by design.",
        recommendations: [
            context.epsilon && `**Privacy Context**: You selected **${context.sensitivity}** sensitivity data. Aim for a privacy budget of **${context.epsilon}** when applying local DP.`,
            "**Need formal privacy guarantees?** Add Local Differential Privacy. Each client adds noise to its model update before sending it to the server. This protects individual user data even from the server.",
            "**Want protection from a curious server?** Implement a Secure Aggregation protocol. This cryptographic technique ensures the server can only learn the sum of all model updates, not individual contributions.",
            "**Communication-constrained?** FL can be bandwidth-intensive. Consider model compression techniques (e.g., quantization, sparsification) or more communication-efficient FL algorithms.",
            "**Recommended Tools**: Use frameworks like TensorFlow Federated (TFF) or Flower (PyTorch-based) to implement your FL system."
        ].filter(Boolean)
    },
    'solution_central_dp': {
        isSolution: true,
        title: "Central Differential Privacy Path",
        icon: <Server className="h-12 w-12 mx-auto text-green-500" />,
        description: "When you have a trusted server that can access raw data, Central DP provides strong, formal privacy guarantees for the entire dataset.",
        recommendations: [
            context.epsilon && `**Privacy Context**: You selected **${context.sensitivity}** sensitivity data. The central privacy budget for the entire training process should be **${context.epsilon}**.`,
            "**Training neural networks?** Use Differentially Private Stochastic Gradient Descent (DP-SGD). It clips the influence of each data point and adds noise during training.",
            "**Working with structured data?** Use differentially private query mechanisms or algorithms designed for tabular data (e.g., releasing private histograms or synthetic data).",
            "**Need high accuracy?** DP introduces a privacy-utility trade-off. Use tight composition bounds (e.g., Rényi DP) to more accurately track the privacy budget over many training steps, often leading to better utility.",
            "**Recommended Tools**: Use Opacus (for PyTorch) or TensorFlow Privacy for robust and tested implementations of DP-SGD."
        ].filter(Boolean)
    },
    'solution_crypto': {
        isSolution: true,
        title: "Cryptography Path for Untrusted Servers",
        icon: <Lock className="h-12 w-12 mx-auto text-green-500" />,
        description: "For systems on untrusted servers, cryptographic methods allow computation on data without revealing it to the server.",
        recommendations: [
            "**Simple operations or linear models?** Use Homomorphic Encryption (HE). It allows the server to compute on encrypted data. This is computationally very expensive and best suited for simpler models.",
            "**Complex model needed?** The performance of pure HE may be prohibitive. Consider hybrid approaches that combine HE with Secure Multi-Party Computation (SMPC) or run parts of the model on the client.",
            "**Performance critical?** Trusted Execution Environments (TEEs) like Intel SGX or AMD SEV can be an option. They create a secure enclave on the server's CPU where data can be decrypted and processed in isolation. Be aware of potential side-channel attacks.",
            "**Recommended Tools**: Microsoft SEAL or TenSEAL for Homomorphic Encryption; CrypTen for secure multi-party computation."
        ].filter(Boolean)
    },
    'solution_smpc': {
        isSolution: true,
        title: "Secure Multi-Party Computation (SMPC) Path",
        icon: <Group className="h-12 w-12 mx-auto text-green-500" />,
        description: "SMPC is designed for scenarios where multiple organizations want to collaborate on training a model without sharing their private datasets.",
        recommendations: [
            context.epsilon && `**Privacy Context**: You selected **${context.sensitivity}** sensitivity data. Consider combining SMPC with DP techniques, applying a budget of **${context.epsilon}** to each party's data.`,
            "**Few parties (<10)?** Protocols based on Garbled Circuits or Secret Sharing (like GMW) are often suitable and can provide strong security guarantees (malicious vs. semi-honest).",
            "**Many parties?** As the number of parties grows, traditional SMPC becomes difficult. Federated Learning with Secure Aggregation is a more scalable form of SMPC designed for this scenario.",
            "**Communication costs**: SMPC protocols are communication-heavy. The choice of protocol depends on the trade-off between communication rounds, computational cost, and the desired security model.",
            "**Recommended Tools**: MP-SPDZ or CrypTen for general-purpose secure computation; TensorFlow Federated for large-scale collaboration."
        ].filter(Boolean)
    },
    'modelProtectionPath': {
        isSolution: true,
        title: "Model Protection Path",
        icon: <BrainCircuit className="h-12 w-12 mx-auto text-green-500" />,
        description: "If protecting the model itself as intellectual property or from specific attacks is the primary goal.",
        recommendations: [
            "**Preventing model stealing?** Implement watermarking by embedding a secret trigger-response pattern into your model. This helps prove ownership if it's stolen. Rate-limiting API access can also deter theft.",
            "**Preventing membership inference?** This attack aims to determine if a specific user's data was in the training set. Training with Differential Privacy is the most effective defense.",
            "**Preventing model inversion/extraction?** These attacks try to reconstruct training data or the model's logic. Defenses include adding noise to model outputs (output perturbation), reducing the confidence scores reported, and again, using Differential Privacy.",
            "**Protecting model parameters?** If the model parameters themselves must remain secret, consider using encrypted models (with HE/SMPC) or deploying the model within a Trusted Execution Environment (TEE)."
        ]
    },
    'privateInferencePath': {
        isSolution: true,
        title: "Private Inference Path",
        icon: <Users className="h-12 w-12 mx-auto text-green-500" />,
        description: "If protecting the user's input data during prediction (inference) is the primary concern.",
        recommendations: [
            "**Local inference possible?** The best solution is to run the model directly on the client's device (e.g., using TensorFlow Lite, Core ML). The data never leaves the device, providing maximum privacy.",
            "**Need provable security on a server?** Use Fully Homomorphic Encryption (FHE). The client sends encrypted data, the server computes the prediction on the ciphertext, and returns an encrypted result. It is very secure but also very slow.",
            "**Balance of security/performance?** Use hybrid cryptographic approaches like those from DELPHI or GAZELLE, which smartly combine HE with SMPC to achieve better performance than pure FHE.",
            "**Performance critical?** Use Trusted Execution Environments (TEEs) like Intel SGX. The client establishes a secure channel with the enclave on the server, sends its data, the model computes inside the enclave, and sends the result back. This is much faster than FHE but relies on hardware security."
        ]
    },
    'compliancePath': {
        isSolution: true,
        title: "Regulatory Compliance Path",
        icon: <FileText className="h-12 w-12 mx-auto text-green-500" />,
        description: "If your primary driver is meeting specific legal or regulatory requirements for data privacy.",
        recommendations: [
            "**For GDPR (Europe)**: Focus on 'Data Protection by Design'. Key principles are data minimization, purpose limitation, and user rights (e.g., Right to be Forgotten). Differential Privacy can be a strong technical measure to demonstrate anonymization. You must have a 'machine unlearning' strategy.",
            "**For HIPAA (US Healthcare)**: Focus on the de-identification of Protected Health Information (PHI). The 'Safe Harbor' or 'Expert Determination' methods are required. Federated Learning is a powerful architecture for HIPAA, as PHI never leaves the hospital's control.",
            "**For CCPA/CPRA (California)**: Focus on transparency and user rights, particularly the right to opt-out of data 'sharing' for advertising purposes and the right to deletion. Your data pipelines must be able to honor these flags.",
            "**General Advice**: For all regulations, **Differential Privacy** provides strong, quantifiable evidence that your system protects individual privacy. **Synthetic Data** generated with privacy guarantees can also be a valuable tool for analysis without using real user data. Always pair technical solutions with legal counsel and thorough documentation (e.g., Privacy Impact Assessments)."
        ]
    }
  }), [context]);

  const node = decisionTree[currentStep];

  const renderContent = () => {
    if (node.isSolution) {
      return (
        <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl text-left animate-fade-in">
          <div className="text-center mb-6">
            {node.icon}
            <h2 className="text-3xl font-bold text-gray-800 mt-4">{node.title}</h2>
            <p className="text-md text-gray-600 mt-2">{node.description}</p>
          </div>
          <div className="space-y-4">
            {node.recommendations.map((rec, index) => (
              <div key={index} className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <p className="text-gray-700" dangerouslySetInnerHTML={{ __html: rec }}></p>
              </div>
            ))}
          </div>
        </div>
      );
    }

    return (
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl text-center animate-fade-in">
        <div className="mb-6">
            {node.icon}
            <h2 className="text-2xl font-semibold text-gray-800 mt-4">{node.question}</h2>
            {node.tooltip && (
                <div className="flex justify-center items-center mt-2">
                    <Tooltip text={node.tooltip}>
                        <Info className="h-5 w-5 text-gray-500 cursor-pointer" />
                    </Tooltip>
                </div>
            )}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {node.options.map((option, index) => (
            <button
              key={index}
              onClick={() => handleAnswer(option.next, option.context)}
              className="group flex flex-col items-center justify-center text-center p-6 bg-gray-50 rounded-lg border-2 border-gray-200 hover:border-blue-500 hover:bg-blue-50 transition-all duration-300 transform hover:scale-105"
            >
              <div className="text-blue-600 mb-3">{option.icon}</div>
              <span className="font-semibold text-gray-700 group-hover:text-blue-600">{option.text}</span>
            </button>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 font-sans flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-2xl mb-4">
          <div className="flex justify-between items-center mb-2">
            <h1 className="text-2xl font-bold text-gray-800">Privacy ML Guide</h1>
            <button 
                onClick={handleReset}
                className="flex items-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
            >
                <RefreshCw className="h-4 w-4 mr-2"/>
                Start Over
            </button>
          </div>
          {history.length > 0 && <ProgressBar current={history.length + 1} total={4} />}
      </div>
      
      {renderContent()}

      <div className="mt-8 flex space-x-4">
        {history.length > 0 && (
          <button
            onClick={handleBack}
            className="px-6 py-2 bg-white text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Back
          </button>
        )}
      </div>
    </div>
  );
};

export default App;
