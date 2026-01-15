const express = require('express');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const storage = require('../lib/storage');
const queue = require('../lib/queue');
// optional auth stub (not enforced by default)
const authStub = require('../middleware/authStub');

const router = express.Router();

// Use memory storage; we explicitly write via storage helper
// Use memory storage; we explicitly write via storage helper
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: parseInt(process.env.MAX_FILE_SIZE || '52428800', 10) // 50MB
  },
  fileFilter: (req, file, cb) => {
    const allowedMimeTypes = [
      'application/pdf',
      'application/vnd.ms-powerpoint', // .ppt
      'application/vnd.openxmlformats-officedocument.presentationml.presentation', // .pptx
      'application/msword', // .doc
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // .docx
      'text/plain',
      'image/jpeg',
      'image/png',
      'image/webp'
    ];

    if (allowedMimeTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      const err = new Error(`File type not allowed: ${file.mimetype}`);
      err.code = 'LIMIT_FILE_TYPE';
      cb(err);
    }
  }
});


// If you'd like to enforce auth, uncomment the next line
// router.use(authStub);

// POST /api/upload
router.post('/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, status: 'error', id: null, message: 'No file provided' });
    }

    const id = uuidv4();
    const originalName = req.file.originalname || 'file';
    // build a unique filename to avoid collisions
    const safeName = `${id}-${originalName.replace(/\s+/g, '_')}`;

    // Save using storage helper (local in MVP)
    const saved = await storage.save({ buffer: req.file.buffer, filename: safeName });

    // record status and enqueue for background processing (simulated)
    queue.statuses[id] = {
      id,
      status: 'queued',
      originalName,
      savedName: safeName,
      savedLocation: saved.url || saved.path || null,
      size: req.file.size,
      createdAt: new Date().toISOString()
    };

    // enqueue the job for simulated processing
    queue.enqueue({ id, savedName: safeName, savedLocation: saved.url || saved.path || null });

    // stable response
    return res.status(200).json({
      success: true,
      status: 'queued',
      id,
      file: {
        originalName,
        size: req.file.size,
        location: saved.url || null
      }
    });
  } catch (err) {
    console.error('[UploadError]', err);
    const errorId = uuidv4();
    queue.statuses[errorId] = { id: errorId, status: 'error', message: err.message };
    return res.status(500).json({ success: false, status: 'error', id: errorId, message: err.message });
  }
});

// GET /api/status/:id
router.get('/status/:id', (req, res) => {
  const { id } = req.params;
  const info = queue.statuses[id];
  if (!info) return res.status(404).json({ success: false, status: 'not_found', id });
  return res.json({ success: true, ...info });
});

module.exports = router;
