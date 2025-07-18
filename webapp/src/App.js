import React, { useState } from 'react';
// Import all icons from lucide-react to be used dynamically
import * as icons from 'lucide-react';
// Import the JSON file that now holds our decision tree logic
import decisionTree from './decisionTree.json';

// --- Helper Components ---

// This component renders a Lucide icon based on a string name
const DynamicIcon = ({ name, className }) => {
  // Default to HelpCircle if an invalid icon name is provided
  const IconComponent = icons[name] || icons.HelpCircle;
  return <IconComponent className={className} />;
};

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

  // The decisionTree is now loaded directly from the imported JSON
  const node = decisionTree[currentStep];

  const renderContent = () => {
    // Check if the current node is a solution node
    if (node.type === 'solution') {
      // Filter and format recommendations based on the user's journey context
      const processedRecommendations = node.recommendations
        .filter(rec => !rec.requiresContext || context[rec.requiresContext])
        .map(rec => {
          let text = rec.text;
          // Replace placeholders like {{sensitivity}} with values from context
          if (rec.requiresContext) {
            text = text.replace(`{{${rec.requiresContext}}}`, context[rec.requiresContext]);
            // A common case is needing both sensitivity and epsilon
            text = text.replace(`{{sensitivity}}`, context.sensitivity);
            text = text.replace(`{{epsilon}}`, context.epsilon);
          }
          return text;
        });

      return (
        <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl text-left animate-fade-in">
          <div className="text-center mb-6">
            <DynamicIcon name={node.iconId} className="h-12 w-12 mx-auto text-green-500" />
            <h2 className="text-3xl font-bold text-gray-800 mt-4">{node.title}</h2>
            <p className="text-md text-gray-600 mt-2">{node.description}</p>
          </div>
          <div className="space-y-4">
            {processedRecommendations.map((rec, index) => (
              <div key={index} className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <p className="text-gray-700" dangerouslySetInnerHTML={{ __html: rec }}></p>
              </div>
            ))}
          </div>
        </div>
      );
    }

    // Otherwise, render a question node
    return (
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl text-center animate-fade-in">
        <div className="mb-6">
            <DynamicIcon name={node.iconId} className="h-12 w-12 mx-auto text-gray-500" />
            <h2 className="text-2xl font-semibold text-gray-800 mt-4">{node.questionText}</h2>
            {node.tooltipText && (
                <div className="flex justify-center items-center mt-2">
                    <Tooltip text={node.tooltipText}>
                        <icons.Info className="h-5 w-5 text-gray-500 cursor-pointer" />
                    </Tooltip>
                </div>
            )}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {node.options.map((option, index) => {
            let iconElement;
            // Handle special styled text "icons" defined in the JSON
            if (option.icon && option.icon.type === 'styledText') {
                iconElement = <div className={option.icon.className}>{option.icon.text}</div>
            } else {
            // Handle regular Lucide icons
                iconElement = <DynamicIcon name={option.iconId} className="h-8 w-8" />;
            }
            
            return (
                <button
                key={index}
                onClick={() => handleAnswer(option.nextStep, option.context)}
                className="group flex flex-col items-center justify-center text-center p-6 bg-gray-50 rounded-lg border-2 border-gray-200 hover:border-blue-500 hover:bg-blue-50 transition-all duration-300 transform hover:scale-105"
                >
                <div className="text-blue-600 mb-3">{iconElement}</div>
                <span className="font-semibold text-gray-700 group-hover:text-blue-600">{option.text}</span>
                </button>
            )
          })}
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
                <icons.RefreshCw className="h-4 w-4 mr-2"/>
                Start Over
            </button>
          </div>
          {/* A fixed total for the progress bar is a reasonable simplification */}
          {history.length > 0 && <ProgressBar current={history.length + 1} total={4} />}
      </div>
      
      {node ? renderContent() : <div>Loading...</div>}

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
