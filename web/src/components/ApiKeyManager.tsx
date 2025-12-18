import React, { useState } from 'react';

interface ApiKeyManagerProps {
  isVisible: boolean;
  onClose: () => void;
}

export default function ApiKeyManager({ isVisible, onClose }: ApiKeyManagerProps) {
  const [apiKey, setApiKey] = useState('');
  const [isUpdating, setIsUpdating] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<'success' | 'error' | ''>('');

  const handleUpdateApiKey = async () => {
    if (!apiKey.trim()) {
      setMessage('Please enter an API key');
      setMessageType('error');
      return;
    }

    setIsUpdating(true);
    setMessage('');

    try {
      const formData = new FormData();
      formData.append('api_key', apiKey.trim());

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/update-api-key`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setMessage('✅ API key updated successfully! The service will now use the new key.');
        setMessageType('success');
        setApiKey('');
        // Auto-close after 3 seconds
        setTimeout(() => {
          onClose();
        }, 3000);
      } else {
        setMessage(`❌ Error: ${result.detail || 'Failed to update API key'}`);
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
    setApiKey('');
    setMessage('');
    setMessageType('');
    onClose();
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Update API Key</h3>
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
              If you're experiencing authentication errors, you may need to update your Anthropic API key. 
              This usually happens when the key expires (typically monthly).
            </p>
            
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
              <div className="flex items-start">
                <svg className="w-5 h-5 text-blue-600 mt-0.5 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 011 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                <div className="text-sm text-blue-800">
                  <p className="font-medium mb-1">How to get a new API key:</p>
                  <ol className="list-decimal list-inside space-y-1 text-xs">
                    <li>Visit <a href="https://console.anthropic.com/" target="_blank" rel="noopener noreferrer" className="underline">console.anthropic.com</a></li>
                    <li>Sign in to your account</li>
                    <li>Go to API Keys section</li>
                    <li>Create a new key or copy an existing one</li>
                    <li>Paste it below and click "Update Key"</li>
                  </ol>
                </div>
              </div>
            </div>
          </div>

          <div className="mb-4">
            <label htmlFor="api-key" className="block text-sm font-medium text-gray-700 mb-2">
              Anthropic API Key
            </label>
            <input
              id="api-key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-ant-api03-..."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              disabled={isUpdating}
            />
            <p className="text-xs text-gray-500 mt-1">
              Your API key should start with "sk-ant-" and be about 50+ characters long.
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
              onClick={handleUpdateApiKey}
              disabled={isUpdating || !apiKey.trim()}
              className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                isUpdating || !apiKey.trim()
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
                'Update Key'
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
