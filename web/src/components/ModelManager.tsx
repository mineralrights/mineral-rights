import React, { useState, useEffect } from 'react';

interface ModelManagerProps {
  isVisible: boolean;
  onClose: () => void;
  onModelChange?: (modelName: string) => void;
}

// Common Claude model names - Latest Claude 4.5 models (as of 2025)
// See: https://platform.claude.com/docs/en/about-claude/models/overview
const COMMON_MODELS = [
  {
    name: 'claude-opus-4-5-20251101',
    display: 'Opus 4.5',
    description: 'Most capable (default)'
  },
  {
    name: 'claude-sonnet-4-5-20250929',
    display: 'Sonnet 4.5',
    description: 'Best balance'
  },
  {
    name: 'claude-3-5-haiku-20241022',
    display: 'Haiku 3.5',
    description: 'Fastest, cheapest'
  },
  {
    name: 'claude-haiku-4-5-20251001',
    display: 'Haiku 4.5',
    description: 'Fastest, cheapest'
  },
  {
    name: 'claude-opus-4-5-20251101',
    display: 'Opus 4.5',
    description: 'Most capable'
  },
  {
    name: 'claude-opus-4-1-20250805',
    display: 'Opus 4.1',
    description: 'Most capable'
  }
];

export default function ModelManager({ isVisible, onClose, onModelChange }: ModelManagerProps) {
  const [modelName, setModelName] = useState('');
  const [currentModel, setCurrentModel] = useState('');
  const [isUpdating, setIsUpdating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<'success' | 'error' | ''>('');

  // Load current model name when component becomes visible
  useEffect(() => {
    if (isVisible) {
      loadCurrentModel();
    }
  }, [isVisible]);

  const loadCurrentModel = async () => {
    setIsLoading(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://mineral-rights-api-1081023230228.us-central1.run.app';
      const response = await fetch(`${apiUrl}/get-model-name`);
      
      if (response.ok) {
        const result = await response.json();
        setCurrentModel(result.model_name || '');
        setModelName(result.model_name || '');
      } else {
        // Default fallback
        setCurrentModel('claude-opus-4-5-20251101');
        setModelName('claude-opus-4-5-20251101');
      }
    } catch (error) {
      console.error('Error loading current model:', error);
      // Default fallback
      setCurrentModel('claude-3-5-sonnet-20241022');
      setModelName('claude-3-5-sonnet-20241022');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpdateModel = async () => {
    if (!modelName.trim()) {
      setMessage('Please enter a model name');
      setMessageType('error');
      return;
    }

    setIsUpdating(true);
    setMessage('');

    try {
      const formData = new FormData();
      formData.append('model_name', modelName.trim());

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://mineral-rights-api-1081023230228.us-central1.run.app';
      const response = await fetch(`${apiUrl}/update-model-name`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setMessage(`✅ Model updated successfully to ${modelName.trim()}! The service will now use this model.`);
        setMessageType('success');
        setCurrentModel(modelName.trim());
        if (onModelChange) {
          onModelChange(modelName.trim());
        }
        // Auto-close after 3 seconds
        setTimeout(() => {
          onClose();
        }, 3000);
      } else {
        setMessage(`❌ Error: ${result.detail || 'Failed to update model name'}`);
        setMessageType('error');
      }
    } catch (error) {
      setMessage(`❌ Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setMessageType('error');
    } finally {
      setIsUpdating(false);
    }
  };

  const handleClose = () => {
    setModelName('');
    setMessage('');
    setMessageType('');
    onClose();
  };

  const handleModelSelect = (model: string) => {
    setModelName(model);
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Update Claude Model</h3>
            <button
              onClick={handleClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="mb-4">
            <p className="text-sm text-gray-600 mb-4">
              If you're experiencing model-related errors (like "model not found" or "404"), you may need to update the Claude model name. 
              This usually happens when a model version is deprecated or unavailable.
            </p>
            
            {isLoading ? (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                  <span className="text-sm text-blue-800">Loading current model...</span>
                </div>
              </div>
            ) : (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
                <div className="flex items-start">
                  <svg className="w-5 h-5 text-blue-600 mt-0.5 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 011 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                  <div className="text-sm text-blue-800">
                    <p className="font-medium mb-1">Current Model:</p>
                    <p className="text-xs font-mono bg-white px-2 py-1 rounded border">{currentModel || 'Not loaded'}</p>
                    <p className="font-medium mt-3 mb-1">Recommended Models:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li><strong>claude-opus-4-5-20251101</strong> - Most capable (default, recommended)</li>
                      <li>claude-sonnet-4-5-20250929 - Best balance</li>
                      <li>claude-haiku-4-5-20251001 - Fastest, cheapest</li>
                      <li>claude-opus-4-5-20251101 - Most capable</li>
                    </ul>
                    <p className="text-xs text-gray-600 mt-2">
                      <a href="https://platform.claude.com/docs/en/about-claude/models/overview" target="_blank" rel="noopener noreferrer" className="underline">View all available models</a>
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="mb-4">
            <label htmlFor="model-name" className="block text-sm font-medium text-gray-700 mb-2">
              Claude Model Name
            </label>
            
            {/* Quick select buttons for common models */}
            <div className="grid grid-cols-2 gap-2 mb-3">
              {COMMON_MODELS.map((model) => {
                return (
                  <button
                    key={model.name}
                    onClick={() => handleModelSelect(model.name)}
                    className={`px-3 py-2 text-xs rounded-lg border transition-colors text-left ${
                      modelName === model.name
                        ? 'bg-indigo-100 border-indigo-500 text-indigo-800'
                        : 'bg-gray-50 border-gray-300 text-gray-700 hover:bg-gray-100'
                    }`}
                    disabled={isUpdating}
                    title={model.name}
                  >
                    <div className="font-medium">{model.display}</div>
                    <div className="text-xs text-gray-600 mt-0.5">{model.description}</div>
                  </button>
                );
              })}
            </div>
            
            <input
              id="model-name"
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="claude-opus-4-5-20251101"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 font-mono text-sm"
              disabled={isUpdating}
            />
            <p className="text-xs text-gray-500 mt-1">
              Model name should start with "claude-" and include the version date (e.g., "claude-3-5-haiku-20241022").
            </p>
          </div>

          {message && (
            <div className={`mb-4 p-3 rounded-lg text-sm ${
              messageType === 'success' 
                ? 'bg-green-50 border border-green-200 text-green-800' 
                : 'bg-red-50 border border-red-200 text-red-800'
            }`}>
              {message}
            </div>
          )}

          <div className="flex gap-3">
            <button
              onClick={handleUpdateModel}
              disabled={isUpdating || !modelName.trim() || isLoading}
              className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                isUpdating || !modelName.trim() || isLoading
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              {isUpdating ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Updating...
                </div>
              ) : (
                'Update Model'
              )}
            </button>
            <button
              onClick={handleClose}
              disabled={isUpdating}
              className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

