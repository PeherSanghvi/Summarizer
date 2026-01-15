import { useState, useCallback } from 'react';
import { ConceptMap, ConceptNode, ConceptEdge } from '@/types';

const COHERE_API_URL = 'https://api.cohere.com/v2';

export const useSummarizer = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const summarizeText = useCallback(async (
    text: string,
    apiKey: string
  ): Promise<string> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${COHERE_API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'command-a-03-2025',
          messages: [
            {
              role: 'user',
              content: `Analyze the following document and create a summary using this EXACT format:

📌 DOCUMENT SUMMARY
[Write 1-2 detailed paragraphs explaining:
- What the document is about
- Main purpose / objective  
- Key idea & context
- Overall meaning / takeaway
Use flowing prose like: "This document discusses… It explains… The primary goal is… Overall, it highlights…"]

🔑 KEY HIGHLIGHTS
• [Point 1 - clear and concise]
• [Point 2 - clear and concise]
• [Point 3 - clear and concise]
• [Point 4 - clear and concise]
• [Point 5 - clear and concise]

IMPORTANT: Do NOT use any markdown formatting like **bold**, *italics*, or other special characters. Write plain text only. No asterisks, no underscores for emphasis.

Document to analyze:
${text.slice(0, 100000)}

Create the summary following the EXACT format above. Be thorough and detailed in the summary paragraph, and use clear bullet points for key highlights. Remember: NO markdown formatting, plain text only.`
            }
          ],
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to summarize text');
      }
      
      const data = await response.json();
      return data.message?.content?.[0]?.text || data.text || 'No summary generated';
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Summarization failed';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const generateConceptMap = useCallback(async (
    text: string,
    apiKey: string
  ): Promise<ConceptMap> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${COHERE_API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'command-a-03-2025',
          messages: [
            {
              role: 'user',
              content: `Analyze the following text and extract the main concepts and their relationships. Return a JSON object with two arrays:
1. "nodes": array of objects with "id", "label" (concept name), and "type" (one of: "concept", "detail", "example")
2. "edges": array of objects with "id", "source" (node id), "target" (node id), and "label" (relationship description)

Extract 5-10 main concepts and their relationships. Keep labels concise.

Text to analyze:
${text.slice(0, 10000)}

Return ONLY valid JSON, no other text or markdown formatting:`
            }
          ],
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to generate concept map');
      }
      
      const data = await response.json();
      const generatedText = data.message?.content?.[0]?.text || data.text || '';
      
      // Try to parse the JSON from the response
      const jsonMatch = generatedText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('Could not parse concept map from response');
      }
      
      const conceptData = JSON.parse(jsonMatch[0]);
      
      // Validate and clean the data
      const nodes: ConceptNode[] = (conceptData.nodes || []).map((n: any, i: number) => ({
        id: n.id || `node-${i}`,
        label: n.label || `Concept ${i + 1}`,
        type: ['concept', 'detail', 'example'].includes(n.type) ? n.type : 'concept',
      }));
      
      const nodeIds = new Set(nodes.map(n => n.id));
      const edges: ConceptEdge[] = (conceptData.edges || [])
        .filter((e: any) => nodeIds.has(e.source) && nodeIds.has(e.target))
        .map((e: any, i: number) => ({
          id: e.id || `edge-${i}`,
          source: e.source,
          target: e.target,
          label: e.label || '',
        }));
      
      return { nodes, edges };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Concept map generation failed';
      setError(message);
      
      // Return a fallback simple concept map
      return {
        nodes: [
          { id: 'main', label: 'Main Concept', type: 'concept' },
          { id: 'detail1', label: 'Key Point 1', type: 'detail' },
          { id: 'detail2', label: 'Key Point 2', type: 'detail' },
        ],
        edges: [
          { id: 'e1', source: 'main', target: 'detail1', label: 'includes' },
          { id: 'e2', source: 'main', target: 'detail2', label: 'includes' },
        ],
      };
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    summarizeText,
    generateConceptMap,
    isLoading,
    error,
  };
};
