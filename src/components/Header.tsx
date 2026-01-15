import { motion } from 'framer-motion';
import { Brain, Sparkles } from 'lucide-react';

export const Header = () => {
  return (
    <header className="py-8 px-4">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex flex-col items-center text-center"
      >
        <motion.div
          className="relative mb-4"
          animate={{ 
            rotate: [0, 5, -5, 0],
          }}
          transition={{ 
            duration: 4, 
            repeat: Infinity, 
            ease: "easeInOut" 
          }}
        >
          <div className="p-4 rounded-2xl gradient-primary shadow-lg shadow-primary/20">
            <Brain className="h-10 w-10 text-primary-foreground" />
          </div>
          <motion.div
            className="absolute -top-1 -right-1"
            animate={{ 
              scale: [1, 1.2, 1],
              opacity: [0.7, 1, 0.7],
            }}
            transition={{ 
              duration: 2, 
              repeat: Infinity,
            }}
          >
            <Sparkles className="h-5 w-5 text-accent" />
          </motion.div>
        </motion.div>
        
        <h1 className="text-4xl font-bold tracking-tight mb-2">
          <span className="text-gradient">ConceptSummarizer</span>
        </h1>
        <p className="text-muted-foreground max-w-md">
          Transform documents into clear summaries and visual concept maps using AI
        </p>
        
        <div className="flex items-center gap-2 mt-4">
          <div className="flex -space-x-1">
            <div className="w-3 h-3 rounded-full bg-success animate-pulse" />
            <div className="w-3 h-3 rounded-full bg-warning animate-pulse delay-100" />
            <div className="w-3 h-3 rounded-full bg-primary animate-pulse delay-200" />
          </div>
          <span className="text-xs text-muted-foreground">
            Supports PDF, PowerPoint & Handwritten Notes
          </span>
        </div>
      </motion.div>
    </header>
  );
};
