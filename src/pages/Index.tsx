import { useState, useCallback, Suspense, lazy } from 'react';
import {
  Sparkles,
  ArrowRight,
  AlertCircle,
  Brain,
  Upload,
  FileText,
  Image,
  Presentation,
  X,
  Loader2,
  CheckCircle,
  Copy,
  Download,
  File,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import { useFileProcessor } from '@/hooks/useFileProcessor';
import { UploadedFile, ProcessingResult } from '@/types';
import ReactMarkdown from 'react-markdown';

const ConceptMapView = lazy(() =>
  import('@/components/ConceptMapView').then(m => ({ default: m.ConceptMapView }))
);

const Index = () => {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [results, setResults] = useState<ProcessingResult[]>([]);
  const { toast } = useToast();

  const { processFile, isProcessing, progress, currentStep } = useFileProcessor();

  const MAX_FILES = 5;

  const getFileType = (file: File): 'pdf' | 'ppt' | 'image' | 'doc' | 'other' => {
    const name = file.name.toLowerCase();
    const type = file.type.toLowerCase();

    if (type === 'application/pdf' || name.endsWith('.pdf')) return 'pdf';
    if (type.includes('presentation') || name.endsWith('.ppt') || name.endsWith('.pptx')) return 'ppt';
    if (type.startsWith('image/') || /\.(jpg|jpeg|png|gif|webp|bmp)$/.test(name)) return 'image';
    if (type.includes('word') || name.endsWith('.doc') || name.endsWith('.docx') || name.endsWith('.txt') || name.endsWith('.rtf')) return 'doc';

    return 'other';
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;

    const remainingSlots = MAX_FILES - files.length;
    if (remainingSlots <= 0) {
      toast({
        title: 'File limit reached',
        description: `Maximum ${MAX_FILES} files allowed`,
        variant: 'destructive',
      });
      return;
    }

    const selectedFiles = Array.from(e.target.files).slice(0, remainingSlots);
    const newFiles = selectedFiles.map(file => ({
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      name: file.name,
      type: getFileType(file),
      file,
      status: 'pending' as const,
    })) as UploadedFile[];

    setFiles(prev => [...prev, ...newFiles]);
    e.target.value = '';
  };

  const handleProcess = useCallback(async () => {
    if (files.length === 0) return;

    const newResults: ProcessingResult[] = [];

    for (const uploadedFile of files) {
      try {
        setFiles(prev =>
          prev.map(f =>
            f.id === uploadedFile.id ? { ...f, status: 'processing' } : f
          )
        );

        toast({ title: 'Uploading & processing...', description: uploadedFile.name });

        const result = await processFile(uploadedFile);

        const finalResult: ProcessingResult = {
          fileId: uploadedFile.id,
          fileName: uploadedFile.name,
          extractedText: result.extractedText || '',
          summary: (result as any).summary || (result as any)?.result?.summary || '',
          conceptMap: (result as any)?.conceptMap || (result as any)?.result?.conceptMap || null,
          timestamp: new Date(),
        };

        newResults.push(finalResult);

        setFiles(prev =>
          prev.map(f =>
            f.id === uploadedFile.id ? { ...f, status: 'completed' } : f
          )
        );

        toast({ title: 'Processing complete!', description: uploadedFile.name });
      } catch (error) {
        console.error('Processing error:', error);
        setFiles(prev =>
          prev.map(f =>
            f.id === uploadedFile.id ? { ...f, status: 'error', error: String(error) } : f
          )
        );
        toast({
          title: 'Processing failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      }
    }

    setResults(prev => [...prev, ...newResults]);
  }, [files, processFile, toast]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({ title: 'Copied to clipboard' });
  };

  const downloadText = (text: string, filename: string) => {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 to-amber-50">
      <div className="max-w-4xl mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex p-4 rounded-2xl bg-gradient-to-br from-teal-600 to-teal-700 shadow-lg mb-4">
            <Brain className="h-10 w-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-teal-900 mb-2">ConceptSummarizer</h1>
          <p className="text-teal-600">Transform documents into summaries and concept maps with AI</p>
        </div>

        {/* Upload */}
        <Card className="p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4">
            Upload Documents ({files.length}/{MAX_FILES})
          </h2>

          <div
            className={cn(
              'border-2 border-dashed rounded-xl p-8 text-center transition-colors',
              files.length >= MAX_FILES
                ? 'border-muted bg-muted/30 cursor-not-allowed'
                : 'border-teal-200 hover:border-teal-400 hover:bg-teal-50/50'
            )}
          >
            <Upload className="h-8 w-8 mx-auto mb-4 text-teal-400" />

            {files.length >= MAX_FILES ? (
              <>
                <p className="font-medium mb-1">Maximum {MAX_FILES} files reached</p>
                <p className="text-sm text-muted-foreground">Remove a file to add more</p>
              </>
            ) : (
              <>
                <p className="font-medium mb-1">Drag & drop or browse</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Any document type • PDF, PPT, Word, Images & more
                </p>
                <Button variant="outline" asChild>
                  <label className="cursor-pointer">
                    Browse Files
                    <input type="file" className="hidden" multiple onChange={handleFileSelect} />
                  </label>
                </Button>
              </>
            )}
          </div>

          {files.length > 0 && (
            <div className="mt-4 space-y-2">
              {files.map(file => (
                <div key={file.id} className="flex items-center gap-3 p-3 bg-muted rounded-lg">
                  {file.type === 'pdf' && <FileText className="h-5 w-5 text-red-500" />}
                  {file.type === 'ppt' && <Presentation className="h-5 w-5 text-orange-500" />}
                  {file.type === 'image' && <Image className="h-5 w-5 text-green-500" />}
                  {file.type === 'doc' && <FileText className="h-5 w-5 text-blue-500" />}
                  {file.type === 'other' && <File className="h-5 w-5 text-gray-500" />}
                  <span className="flex-1 truncate">{file.name}</span>
                  {file.status === 'processing' && <Loader2 className="h-4 w-4 animate-spin text-teal-600" />}
                  {file.status === 'completed' && <CheckCircle className="h-4 w-4 text-green-500" />}
                  {file.status === 'error' && <AlertCircle className="h-4 w-4 text-red-500" />}
                  <Button variant="ghost" size="icon" onClick={() => setFiles(f => f.filter(x => x.id !== file.id))}>
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ))}

              {isProcessing && (
                <div className="p-4 bg-teal-50 rounded-lg">
                  <p className="text-sm font-medium text-teal-700 mb-2">
                    {currentStep || 'Processing...'}
                  </p>
                  <Progress value={progress} className="h-2" />
                </div>
              )}

              <div className="flex justify-end mt-4">
                <Button
                  size="lg"
                  onClick={handleProcess}
                  disabled={isProcessing || files.every(f => f.status !== 'pending')}
                  className="bg-gradient-to-r from-teal-600 to-teal-700"
                >
                  {isProcessing ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Sparkles className="h-4 w-4 mr-2" />}
                  Process with AI
                  <ArrowRight className="h-4 w-4 ml-2" />
                </Button>
              </div>
            </div>
          )}
        </Card>

        {/* Results */}
        {results.length > 0 && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-teal-900">Results</h2>

            {results.map(result => (
              <div key={result.fileId} className="space-y-4">
                <Card className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold">
                      {result.fileName} - Summary
                    </h3>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" onClick={() => copyToClipboard(result.summary || '')}>
                        <Copy className="h-4 w-4 mr-1" /> Copy
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => downloadText(result.summary || '', `${result.fileName}_summary.txt`)}>
                        <Download className="h-4 w-4 mr-1" /> Download
                      </Button>
                    </div>
                  </div>

                  <div className="prose max-w-none text-muted-foreground">
                  <ReactMarkdown>
                  {result.summary || ''}
                </ReactMarkdown>
              </div>

                </Card>

                {result.conceptMap && (
                  <Suspense fallback={<Card className="p-6"><Loader2 className="h-6 w-6 animate-spin mx-auto" /></Card>}>
                    <ConceptMapView conceptMap={result.conceptMap} fileName={result.fileName} />
                  </Suspense>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Index;
