import { motion } from 'framer-motion';
import { Loader2, FileText, Brain, Network } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { Card } from '@/components/ui/card';

interface ProcessingStatusProps {
  step: 'extracting' | 'summarizing' | 'mapping';
  progress: number;
  fileName?: string;
}

const steps = [
  { id: 'extracting', label: 'Extracting Text', icon: FileText },
  { id: 'summarizing', label: 'Generating Summary', icon: Brain },
  { id: 'mapping', label: 'Creating Concept Map', icon: Network },
];

export const ProcessingStatus = ({ step, progress, fileName }: ProcessingStatusProps) => {
  const currentIndex = steps.findIndex(s => s.id === step);

  return (
    <Card className="p-6 overflow-hidden relative">
      {/* Background animation */}
      <motion.div
        className="absolute inset-0 gradient-mesh opacity-50"
        animate={{ 
          backgroundPosition: ['0% 0%', '100% 100%'],
        }}
        transition={{ 
          duration: 10, 
          repeat: Infinity, 
          repeatType: 'reverse' 
        }}
      />
      
      <div className="relative z-10">
        <div className="flex items-center gap-3 mb-4">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          >
            <Loader2 className="h-6 w-6 text-primary" />
          </motion.div>
          <div>
            <h3 className="font-semibold">Processing Document</h3>
            {fileName && (
              <p className="text-sm text-muted-foreground truncate max-w-xs">{fileName}</p>
            )}
          </div>
        </div>
        
        <Progress value={progress} className="mb-6" />
        
        <div className="flex justify-between">
          {steps.map((s, index) => {
            const Icon = s.icon;
            const isActive = index === currentIndex;
            const isCompleted = index < currentIndex;
            
            return (
              <motion.div
                key={s.id}
                className="flex flex-col items-center gap-2"
                animate={isActive ? { scale: 1.1 } : { scale: 1 }}
              >
                <motion.div
                  className={`p-3 rounded-full transition-colors ${
                    isActive 
                      ? 'bg-primary text-primary-foreground shadow-glow' 
                      : isCompleted 
                        ? 'bg-success text-success-foreground' 
                        : 'bg-muted text-muted-foreground'
                  }`}
                  animate={isActive ? { 
                    boxShadow: [
                      '0 0 0 0 hsl(var(--primary) / 0.3)',
                      '0 0 0 10px hsl(var(--primary) / 0)',
                    ]
                  } : {}}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <Icon className="h-5 w-5" />
                </motion.div>
                <span className={`text-xs font-medium ${
                  isActive ? 'text-primary' : isCompleted ? 'text-success' : 'text-muted-foreground'
                }`}>
                  {s.label}
                </span>
              </motion.div>
            );
          })}
        </div>
      </div>
    </Card>
  );
};
