import { motion } from 'framer-motion';
import { FileText, Copy, Download, Check, Pin, Key } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';

interface SummaryDisplayProps {
  summary: string;
  fileName: string;
}

const parseSummary = (summary: string) => {
  const sections: { type: string; title: string; content: string }[] = [];
  
  // Find Document Summary section
  const summaryMatch = summary.match(/📌\s*(DOCUMENT SUMMARY|Document Summary)?\s*:?\s*([\s\S]*?)(?=🔑|$)/i);
  if (summaryMatch) {
    sections.push({
      type: 'summary',
      title: 'Document Summary',
      content: summaryMatch[2].trim(),
    });
  }
  
  // Find Key Highlights section
  const highlightsMatch = summary.match(/🔑\s*(KEY HIGHLIGHTS|Key Highlights|KEY POINTS|Key Points)?\s*:?\s*([\s\S]*?)$/i);
  if (highlightsMatch) {
    sections.push({
      type: 'highlights',
      title: 'Key Highlights',
      content: highlightsMatch[2].trim(),
    });
  }

  // If no sections found, return the whole summary
  if (sections.length === 0) {
    sections.push({
      type: 'summary',
      title: 'Document Summary',
      content: summary,
    });
  }

  return sections;
};

const renderContent = (content: string, type: string) => {
  if (type === 'summary') {
    // Render as paragraphs
    const paragraphs = content.split('\n\n').filter(p => p.trim());
    return (
      <div className="space-y-4">
        {paragraphs.map((paragraph, i) => (
          <p key={i} className="text-foreground leading-relaxed text-base">
            {paragraph.trim()}
          </p>
        ))}
      </div>
    );
  }

  if (type === 'highlights') {
    // Render as bullet points
    const lines = content.split('\n').filter(line => line.trim());
    return (
      <ul className="space-y-3">
        {lines.map((line, i) => (
          <li key={i} className="flex items-start gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-xs font-medium text-primary mt-0.5">
              {i + 1}
            </span>
            <span className="text-foreground leading-relaxed">
              {line.replace(/^[•\-\*\d.)\]]+\s*/, '').trim()}
            </span>
          </li>
        ))}
      </ul>
    );
  }

  return <p className="text-foreground leading-relaxed whitespace-pre-wrap">{content}</p>;
};

export const SummaryDisplay = ({ summary, fileName }: SummaryDisplayProps) => {
  const [copied, setCopied] = useState(false);
  const sections = parseSummary(summary);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(summary);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([summary], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${fileName.replace(/\.[^/.]+$/, '')}_summary.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="overflow-hidden">
        <div className="p-4 border-b flex items-center justify-between bg-muted/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              <FileText className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="font-semibold">Summary</h3>
              <p className="text-sm text-muted-foreground">{fileName}</p>
            </div>
          </div>
          
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleCopy}>
              {copied ? (
                <>
                  <Check className="h-4 w-4 mr-1" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-4 w-4 mr-1" />
                  Copy
                </>
              )}
            </Button>
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <Download className="h-4 w-4 mr-1" />
              Download
            </Button>
          </div>
        </div>
        
        <ScrollArea className="h-[450px]">
          <div className="p-6 space-y-6">
            {sections.map((section, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.15 }}
              >
                {index > 0 && <Separator className="mb-6" />}
                
                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    {section.type === 'summary' ? (
                      <Pin className="h-5 w-5 text-blue-500" />
                    ) : (
                      <Key className="h-5 w-5 text-amber-500" />
                    )}
                    <h3 className="font-semibold text-lg text-foreground">
                      {section.title}
                    </h3>
                  </div>
                  
                  <div className="pl-7">
                    {renderContent(section.content, section.type)}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </ScrollArea>
      </Card>
    </motion.div>
  );
};