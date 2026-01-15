````md
# ConceptSummarizer 🧠✨

ConceptSummarizer is an AI-powered web app that transforms documents into:

- 📄 Clear, exam-ready summaries  
- 🧩 Visual concept maps for fast understanding  

It supports **PDFs, PPT/PPTX, DOCX, and Images (PNG/JPG)** using a full-stack pipeline with file extraction, OCR, and AI processing.

Built as a real-world system:
- React + Vite frontend  
- Node.js + Express backend  
- Async processing queue  
- OCR + document parsing  
- Cohere AI for summarization & concept mapping  

---

## 🚀 Features

- Upload multiple documents  
- Automatic text extraction  
  - PDF → `pdf-parse`  
  - PPTX → slide XML parsing  
  - DOCX → `mammoth`  
  - Images → OCR with `tesseract.js`  
- AI-generated:
  - Detailed academic summaries  
  - Concept maps (JSON → visual graph)  
- Async background processing  
- Clean, modern UI  

---

## 🛠️ Tech Stack

### Frontend
- React (TypeScript)
- Vite
- Tailwind CSS
- Framer Motion

### Backend
- Node.js
- Express
- Multer (uploads)
- In-memory queue (MVP)
- Cohere AI
- pdf-parse, jszip, mammoth, tesseract.js

---

## ⚙️ Setup on a New Computer

### 1️⃣ Clone the repo

```bash
git clone https://github.com/maazzshaikh11/ConceptSummarizer.git
cd ConceptSummarizer
````

---

### 2️⃣ Backend setup

```bash
cd concept-weaver-backend
npm install
```

Create `.env`:

```bash
cp .env.example .env
```

Edit `.env`:

```env
PORT=4000
COHERE_API_KEY=your_cohere_api_key_here
MAX_FILE_SIZE=52428800
```

Start backend:

```bash
npm run dev
```

You should see:

```
Server listening on http://localhost:4000
```

---

### 3️⃣ Frontend setup

Open a new terminal:

```bash
cd ConceptSummarizer
npm install
npm run dev
```

You’ll get a URL like:

```
http://localhost:5173
```

Open it in your browser.

---

## 📐 Architecture

```
Frontend (React)
      |
      |  POST /api/upload
      v
Backend (Express)
      |
      |  Save file → Queue job
      v
Worker (in-memory)
      |
      |  Extract text (PDF/PPT/DOC/OCR)
      |  → Cohere AI
      v
Summary + Concept Map
      |
      v
Status API → Frontend renders result
```

---

## 🧪 Supported File Types

* `.pdf`
* `.ppt`, `.pptx`
* `.docx`
* `.png`, `.jpg`, `.jpeg`

```

