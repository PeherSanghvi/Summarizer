import { useCallback, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  ConnectionMode,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { motion } from 'framer-motion';
import { Network, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ConceptMap } from '@/types';

interface ConceptMapViewProps {
  conceptMap: ConceptMap;
  fileName: string;
}

const nodeColors = {
  concept: { bg: 'hsl(180, 60%, 30%)', text: 'white' },
  detail: { bg: 'hsl(180, 30%, 95%)', text: 'hsl(180, 60%, 25%)' },
  example: { bg: 'hsl(38, 92%, 95%)', text: 'hsl(38, 90%, 30%)' },
};

export const ConceptMapView = ({ conceptMap, fileName }: ConceptMapViewProps) => {
  const initialNodes: Node[] = useMemo(() => {
    const centerX = 300;
    const centerY = 200;
    const radius = 180;
    
    return conceptMap.nodes.map((node, index) => {
      const angle = (index / conceptMap.nodes.length) * 2 * Math.PI - Math.PI / 2;
      const isMain = node.type === 'concept' && index === 0;
      
      return {
        id: node.id,
        data: { label: node.label },
        position: isMain 
          ? { x: centerX, y: centerY }
          : { x: centerX + radius * Math.cos(angle), y: centerY + radius * Math.sin(angle) },
        style: {
          background: nodeColors[node.type].bg,
          color: nodeColors[node.type].text,
          border: 'none',
          borderRadius: '12px',
          padding: '12px 20px',
          fontSize: isMain ? '14px' : '12px',
          fontWeight: isMain ? '600' : '500',
          boxShadow: isMain 
            ? '0 4px 20px hsla(180, 60%, 30%, 0.3)'
            : '0 2px 8px hsla(0, 0%, 0%, 0.1)',
          minWidth: '100px',
          textAlign: 'center' as const,
        },
      };
    });
  }, [conceptMap.nodes]);

  const initialEdges: Edge[] = useMemo(() => {
    return conceptMap.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      type: 'smoothstep',
      animated: true,
      style: { stroke: 'hsl(180, 30%, 70%)', strokeWidth: 2 },
      labelStyle: { 
        fontSize: '10px', 
        fill: 'hsl(180, 15%, 45%)',
        fontWeight: 500,
      },
      labelBgStyle: { 
        fill: 'hsl(180, 20%, 99%)', 
        fillOpacity: 0.9,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: 'hsl(180, 30%, 70%)',
      },
    }));
  }, [conceptMap.edges]);

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  const handleDownloadJSON = useCallback(() => {
    const data = JSON.stringify(conceptMap, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${fileName.replace(/\.[^/.]+$/, '')}_concept_map.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [conceptMap, fileName]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <Card className="overflow-hidden">
        <div className="p-4 border-b flex items-center justify-between bg-muted/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-accent/20">
              <Network className="h-5 w-5 text-accent-foreground" />
            </div>
            <div>
              <h3 className="font-semibold">Concept Map</h3>
              <p className="text-sm text-muted-foreground">
                {conceptMap.nodes.length} concepts, {conceptMap.edges.length} connections
              </p>
            </div>
          </div>
          
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleDownloadJSON}>
              <Download className="h-4 w-4 mr-1" />
              Export JSON
            </Button>
          </div>
        </div>
        
        <div className="h-[400px] bg-gradient-to-br from-muted/30 to-muted/10">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            connectionMode={ConnectionMode.Loose}
            fitView
            fitViewOptions={{ padding: 0.3 }}
            minZoom={0.5}
            maxZoom={2}
          >
            <Background color="hsl(180, 20%, 90%)" gap={20} />
            <Controls 
              showInteractive={false}
              style={{ 
                background: 'hsl(0, 0%, 100%)', 
                borderRadius: '8px',
                border: '1px solid hsl(180, 20%, 90%)',
              }}
            />
          </ReactFlow>
        </div>
        
        {/* Legend */}
        <div className="p-4 border-t bg-muted/30 flex items-center gap-6">
          <span className="text-sm text-muted-foreground">Legend:</span>
          <div className="flex gap-4">
            {Object.entries(nodeColors).map(([type, colors]) => (
              <div key={type} className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-sm" 
                  style={{ background: colors.bg }}
                />
                <span className="text-xs text-muted-foreground capitalize">{type}</span>
              </div>
            ))}
          </div>
        </div>
      </Card>
    </motion.div>
  );
};
