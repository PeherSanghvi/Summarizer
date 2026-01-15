import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Key, Eye, EyeOff, CheckCircle, AlertCircle } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface ApiKeyInputProps {
  apiKey: string | null;
  onApiKeyChange: (key: string | null) => void;
}

export const ApiKeyInput = ({ apiKey, onApiKeyChange }: ApiKeyInputProps) => {
  const [inputValue, setInputValue] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [isSaved, setIsSaved] = useState(!!apiKey);

  useEffect(() => {
    // Load from localStorage on mount
    const savedKey = localStorage.getItem('cohere_api_key');
    if (savedKey) {
      onApiKeyChange(savedKey);
      setIsSaved(true);
    }
  }, [onApiKeyChange]);

  const handleSave = () => {
    if (inputValue.trim()) {
      localStorage.setItem('cohere_api_key', inputValue.trim());
      onApiKeyChange(inputValue.trim());
      setIsSaved(true);
      setInputValue('');
    }
  };

  const handleClear = () => {
    localStorage.removeItem('cohere_api_key');
    onApiKeyChange(null);
    setIsSaved(false);
    setInputValue('');
  };

  return (
    <Card className="p-4">
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-lg bg-primary/10">
          <Key className="h-5 w-5 text-primary" />
        </div>
        <div className="flex-1 space-y-3">
          <div>
            <h3 className="font-medium">Cohere API Key</h3>
            <p className="text-sm text-muted-foreground">
              Required for AI-powered summarization.{' '}
              <a 
                href="https://dashboard.cohere.com/api-keys" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                Get your key →
              </a>
            </p>
          </div>
          
          {isSaved ? (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-3"
            >
              <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-success/10 text-success">
                <CheckCircle className="h-4 w-4" />
                <span className="text-sm font-medium">API key saved locally</span>
              </div>
              <Button variant="ghost" size="sm" onClick={handleClear}>
                Clear
              </Button>
            </motion.div>
          ) : (
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Input
                  type={showKey ? 'text' : 'password'}
                  placeholder="Enter your Cohere API key"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  className="pr-10"
                />
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute right-1 top-1/2 -translate-y-1/2 h-8 w-8"
                  onClick={() => setShowKey(!showKey)}
                >
                  {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </Button>
              </div>
              <Button onClick={handleSave} disabled={!inputValue.trim()}>
                Save
              </Button>
            </div>
          )}
          
          <p className="text-xs text-muted-foreground flex items-center gap-1">
            <AlertCircle className="h-3 w-3" />
            Your API key is stored only in your browser's localStorage
          </p>
        </div>
      </div>
    </Card>
  );
};
