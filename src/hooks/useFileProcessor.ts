import { useState, useCallback } from 'react';
import { UploadedFile, ProcessingResult } from '@/types';

const API_BASE = 'http://localhost:4000/api';

export const useFileProcessor = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState<string>('');

  const sleep = (ms: number) => new Promise(res => setTimeout(res, ms));

  const processFile = useCallback(async (
    uploadedFile: UploadedFile
  ): Promise<Partial<ProcessingResult>> => {
    setIsProcessing(true);
    setProgress(0);
    setCurrentStep('Uploading file to server...');

    try {
      // 1. Upload file
      const form = new FormData();
      form.append('file', uploadedFile.file);

      const uploadRes = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: form,
      });

      if (!uploadRes.ok) {
        throw new Error('Upload failed');
      }

      const uploadJson = await uploadRes.json();
      const id = uploadJson.id;

      setProgress(20);
      setCurrentStep('File queued. Waiting for processing...');

      // 2. Poll status
      let done = false;
      let result: any = null;

      while (!done) {
        await sleep(1500);

        const statusRes = await fetch(`${API_BASE}/status/${id}`);
        if (!statusRes.ok) {
          throw new Error('Status check failed');
        }

        const statusJson = await statusRes.json();

        if (statusJson.status === 'processing') {
          setCurrentStep('Processing on server...');
          setProgress(p => Math.min(p + 10, 90));
        }

        if (statusJson.status === 'done') {
          done = true;
          result = statusJson.result;
          setProgress(100);
        }

        if (statusJson.status === 'error') {
          throw new Error(statusJson.message || 'Processing failed');
        }
      }

      setCurrentStep('Processing complete!');

      return {
        fileId: id,
        fileName: uploadedFile.name,
        extractedText: result?.text || '',
        summary: result?.summary || '',
        timestamp: new Date(),
      };
    } finally {
      setIsProcessing(false);
    }
  }, []);

  return {
    processFile,
    isProcessing,
    progress,
    currentStep,
  };
};
