/**
 * In-memory queue with real async worker hook.
 */

const fs = require('fs');
const path = require('path');
const pdfParse = require('pdf-parse');
const JSZip = require('jszip');
const mammoth = require('mammoth');
const Tesseract = require('tesseract.js');
const { CohereClient } = require('cohere-ai');

const statuses = {};
const jobs = [];
let workerRunning = false;

let cohereClient = null;

function initCohere() {
  if (!cohereClient) {
    if (!process.env.COHERE_API_KEY) {
      throw new Error('COHERE_API_KEY is not set in environment');
    }
    cohereClient = new CohereClient({
      token: process.env.COHERE_API_KEY,
    });
  }
  return cohereClient;
}

function enqueue(job) {
  jobs.push(job);
  processNext();
}

function processNext() {
  if (workerRunning) return;
  const job = jobs.shift();
  if (!job) return;

  workerRunning = true;

  const { id } = job;
  statuses[id] = {
    ...(statuses[id] || {}),
    status: 'processing',
    startedAt: new Date().toISOString(),
  };

  processJob(job);
}

async function processJob(job) {
  const { id, savedLocation } = job;

  try {
    const relativePath = savedLocation.startsWith('/')
      ? savedLocation.slice(1)
      : savedLocation;

    const fullPath = path.join(process.cwd(), relativePath);
    const ext = fullPath.toLowerCase();

    let extractedText = '';

    // ---- PDF ----
    if (ext.endsWith('.pdf')) {
      const buffer = fs.readFileSync(fullPath);
      const data = await pdfParse(buffer);
      extractedText = data.text || '';
    }

    // ---- PPTX ----
    else if (ext.endsWith('.pptx')) {
      const buffer = fs.readFileSync(fullPath);
      const zip = await JSZip.loadAsync(buffer);

      const slideFiles = Object.keys(zip.files)
        .filter(name => name.startsWith('ppt/slides/slide') && name.endsWith('.xml'))
        .sort();

      let text = '';
      for (let i = 0; i < slideFiles.length; i++) {
        const slideXml = await zip.files[slideFiles[i]].async('text');
        const cleaned = slideXml
          .replace(/<[^>]+>/g, ' ')
          .replace(/\s+/g, ' ')
          .trim();

        if (cleaned) {
          text += `Slide ${i + 1}: ${cleaned}\n\n`;
        }
      }

      extractedText = text;
    }

    // ---- DOCX ----
    else if (ext.endsWith('.docx')) {
      const buffer = fs.readFileSync(fullPath);
      const result = await mammoth.extractRawText({ buffer });
      extractedText = result.value || '';
    }

    // ---- IMAGES (OCR) ----
    else if (ext.endsWith('.png') || ext.endsWith('.jpg') || ext.endsWith('.jpeg')) {
      const result = await Tesseract.recognize(fullPath, 'eng');
      extractedText = result.data.text || '';
    }

    // ---- FALLBACK ----
    else {
      extractedText = `
This document is of type "${path.extname(fullPath)}".
It likely contains educational or structured content.

Generate a clear academic-style summary and key concepts based on this context.
`;
    }

    // Guard
    if (!extractedText || extractedText.trim().length < 20) {
      throw new Error(
        'Not enough readable text found in this file to generate a meaningful summary.'
      );
    }

    const cohere = initCohere();

// ---- SUMMARY ----
const summaryRes = await cohere.chat({
  model: 'command-a-03-2025',
  message: `
You are an academic summarizer.

STRICT RULES:
- Do NOT say anything like "content not provided", "template", or "replace with..."
- Do NOT explain what you are doing.
- Do NOT add introductions about missing data.
- Start directly with the summary content.
- Maintain a clean academic tone.

Task:
Generate a **detailed, well-structured academic summary** of the content below for:
- Exam preparation
- Conceptual clarity
- Revision notes

Style Requirements:
- Use clear headings and subheadings
- Explain each concept in 2–4 lines
- Preserve definitions, procedures, and examples
- Use bullet points where appropriate
- Do NOT over-compress
- End the summary cleanly (no abrupt cutoff)

Content:
${extractedText}
`,
  maxTokens: 1400, // slightly higher to avoid cutoffs
});

    const summary = summaryRes.text;

    // ---- CONCEPT MAP ----
    const conceptPrompt = `
Return a concept map derived from the content below.

You MUST respond with a valid JSON object only.
Do not include any explanation, heading, or commentary.

Schema:
{
  "nodes": [
    { "id": "A", "label": "Main Topic" }
  ],
  "edges": [
    { "from": "A", "to": "B", "label": "relates to" }
  ]
}

Rules:
- Response must start with '{' and end with '}'
- No markdown
- No extra text
- No comments

Content:
${extractedText}
`;

    const conceptRes = await cohere.chat({
      model: 'command-a-03-2025',
      message: conceptPrompt,
      maxTokens: 1400,
    });

    let conceptMap = null;
    try {
      const raw = conceptRes.text.trim();
      const start = raw.indexOf('{');
      const end = raw.lastIndexOf('}');
      if (start === -1 || end === -1) {
        throw new Error('No JSON boundaries found');
      }
      const jsonString = raw.slice(start, end + 1);
      conceptMap = JSON.parse(jsonString);
    } catch (e) {
      console.warn('Concept map JSON parse failed:\n', conceptRes.text);
    }

    statuses[id] = {
      ...(statuses[id] || {}),
      status: 'done',
      finishedAt: new Date().toISOString(),
      result: {
        summary,
        conceptMap,
      },
    };
  } catch (err) {
    console.error('[WorkerError]', err);
    statuses[id] = {
      ...(statuses[id] || {}),
      status: 'error',
      message: err.message || 'Processing failed',
    };
  } finally {
    workerRunning = false;
    if (jobs.length > 0) setTimeout(processNext, 100);
  }
}

module.exports = { enqueue, statuses };