require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const path = require('path');
const fs = require('fs');

const uploadRouter = require('./routes/upload');

const app = express();

const PORT = process.env.PORT || 4000;
const STORAGE = process.env.STORAGE || 'local';
const UPLOAD_DIR = process.env.UPLOAD_DIR || 'uploads';
const CORS_ORIGIN = process.env.CORS_ORIGIN || '*';

// ensure upload dir exists for local storage
if (STORAGE === 'local') {
  const uploadPath = path.resolve(process.cwd(), UPLOAD_DIR);
  if (!fs.existsSync(uploadPath)) {
    fs.mkdirSync(uploadPath, { recursive: true });
    console.log('Created upload directory:', uploadPath);
  }
}

app.use(helmet());
app.use(cors({ origin: CORS_ORIGIN, methods: ['GET','POST','OPTIONS'] }));
app.use(express.json());

// Serve local uploaded files in dev (only when STORAGE=local)
if (STORAGE === 'local') {
  const staticPath = path.resolve(process.cwd(), UPLOAD_DIR);
  app.use('/uploads', express.static(staticPath));
}

// mount API
app.use('/api', uploadRouter);

// health
app.get('/health', (req, res) => res.json({ ok: true, time: new Date().toISOString() }));

// global error handler
app.use((err, req, res, next) => {
  console.error('[GlobalError]', err && err.stack ? err.stack : err);
  res.status(500).json({ success: false, status: 'error', message: 'Server error' });
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});
