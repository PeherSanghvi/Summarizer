export interface UploadedFile {
  id: string;
  name: string;
  type: 'pdf' | 'ppt' | 'image' | 'doc' | 'other';
  file: File;
  status: 'pending' | 'processing' | 'completed' | 'error';
  extractedText?: string;
  error?: string;
}

export interface ConceptNode {
  id: string;
  label: string;
  type: 'concept' | 'detail' | 'example';
}

export interface ConceptEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}

export interface ConceptMap {
  nodes: ConceptNode[];
  edges: ConceptEdge[];
}

export interface ProcessingResult {
  fileId: string;
  fileName: string;
  extractedText: string;
  summary?: string;
  conceptMap?: ConceptMap;
  timestamp: Date;
}

export interface AppState {
  files: UploadedFile[];
  results: ProcessingResult[];
  isProcessing: boolean;
  apiKey: string | null;
}
