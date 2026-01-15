import { useCallback, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileText, Image, Presentation, X, CheckCircle, AlertCircle, Loader2, File } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { UploadedFile } from '@/types';
import { cn } from '@/lib/utils';

interface FileUploadProps {
  files: UploadedFile[];
  onFilesChange: (files: UploadedFile[]) => void;
  disabled?: boolean;
}

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

const FileIcon = ({ type }: { type: UploadedFile['type'] }) => {
  switch (type) {
    case 'pdf':
      return <FileText className="h-5 w-5 text-destructive" />;
    case 'ppt':
      return <Presentation className="h-5 w-5 text-warning" />;
    case 'image':
      return <Image className="h-5 w-5 text-success" />;
    case 'doc':
      return <FileText className="h-5 w-5 text-primary" />;
    default:
      return <File className="h-5 w-5 text-muted-foreground" />;
  }
};

const StatusIcon = ({ status }: { status: UploadedFile['status'] }) => {
  switch (status) {
    case 'pending':
      return null;
    case 'processing':
      return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
    case 'completed':
      return <CheckCircle className="h-4 w-4 text-success" />;
    case 'error':
      return <AlertCircle className="h-4 w-4 text-destructive" />;
  }
};

export const FileUpload = ({ files, onFilesChange, disabled }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const isAtLimit = files.length >= MAX_FILES;

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (disabled || isAtLimit) return;
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    const remainingSlots = MAX_FILES - files.length;
    const filesToAdd = droppedFiles.slice(0, remainingSlots);
    
    const validFiles = filesToAdd.map(file => ({
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      name: file.name,
      type: getFileType(file),
      file,
      status: 'pending' as const,
    })) as UploadedFile[];
    
    onFilesChange([...files, ...validFiles]);
  }, [files, onFilesChange, disabled, isAtLimit]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || disabled || isAtLimit) return;
    
    const selectedFiles = Array.from(e.target.files);
    const remainingSlots = MAX_FILES - files.length;
    const filesToAdd = selectedFiles.slice(0, remainingSlots);
    
    const validFiles = filesToAdd.map(file => ({
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      name: file.name,
      type: getFileType(file),
      file,
      status: 'pending' as const,
    })) as UploadedFile[];
    
    onFilesChange([...files, ...validFiles]);
    e.target.value = '';
  }, [files, onFilesChange, disabled, isAtLimit]);

  const removeFile = useCallback((id: string) => {
    onFilesChange(files.filter(f => f.id !== id));
  }, [files, onFilesChange]);

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <motion.div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={cn(
          "relative border-2 border-dashed rounded-xl p-8 transition-all duration-300",
          isDragging 
            ? "border-primary bg-primary/5 scale-[1.02]" 
            : "border-border hover:border-primary/50 hover:bg-muted/50",
          (disabled || isAtLimit) && "opacity-50 cursor-not-allowed"
        )}
        whileHover={!(disabled || isAtLimit) ? { scale: 1.01 } : {}}
      >
        <div className="flex flex-col items-center gap-4 text-center">
          <motion.div
            className={cn(
              "p-4 rounded-full transition-colors",
              isDragging ? "bg-primary/10" : "bg-muted"
            )}
            animate={isDragging ? { scale: 1.1, rotate: 5 } : { scale: 1, rotate: 0 }}
          >
            <Upload className={cn(
              "h-8 w-8 transition-colors",
              isDragging ? "text-primary" : "text-muted-foreground"
            )} />
          </motion.div>
          
          <div>
            <p className="text-lg font-medium">
              {isAtLimit 
                ? `Maximum ${MAX_FILES} files reached` 
                : isDragging 
                  ? "Drop your files here" 
                  : "Drag & drop your documents"}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              {isAtLimit 
                ? "Remove a file to add more" 
                : `Any document type • Max ${MAX_FILES} files (${files.length}/${MAX_FILES})`}
            </p>
          </div>
          
          {!isAtLimit && (
            <>
              <div className="flex items-center gap-4">
                <div className="h-px w-12 bg-border" />
                <span className="text-sm text-muted-foreground">or</span>
                <div className="h-px w-12 bg-border" />
              </div>
              
              <Button variant="outline" disabled={disabled} asChild>
                <label className="cursor-pointer">
                  Browse Files
                  <input
                    type="file"
                    className="hidden"
                    multiple
                    onChange={handleFileSelect}
                    disabled={disabled}
                  />
                </label>
              </Button>
            </>
          )}
        </div>
      </motion.div>

      {/* File List */}
      <AnimatePresence mode="popLayout">
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-2"
          >
            {files.map((file, index) => (
              <motion.div
                key={file.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20, height: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Card className="p-3 flex items-center gap-3 group hover:shadow-md transition-shadow">
                  <FileIcon type={file.type} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{file.name}</p>
                    {file.error && (
                      <p className="text-xs text-destructive">{file.error}</p>
                    )}
                  </div>
                  <StatusIcon status={file.status} />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => removeFile(file.id)}
                    disabled={file.status === 'processing'}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
