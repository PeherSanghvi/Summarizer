const fs = require('fs');
const path = require('path');

const STORAGE = process.env.STORAGE || 'local';
const UPLOAD_DIR = process.env.UPLOAD_DIR || 'uploads';

// ensure upload dir exists (extra safety)
if (STORAGE === 'local') {
  const uploadPath = path.resolve(process.cwd(), UPLOAD_DIR);
  if (!fs.existsSync(uploadPath)) {
    fs.mkdirSync(uploadPath, { recursive: true });
  }
}

/**
 * Save file to local disk.
 * @param {{buffer: Buffer, filename: string}} params
 * @returns {Promise<{path: string, url: string}>}
 */
async function saveFileLocal(params) {
  const { buffer, filename } = params;
  const filePath = path.join(uploadDirAbsolute(), filename);
  await fs.promises.writeFile(filePath, buffer);
  // url for dev: exposed under /uploads route in index.js
  const url = `/uploads/${encodeURIComponent(filename)}`;
  return { path: filePath, url };
}

function uploadDirAbsolute() {
  return path.resolve(process.cwd(), UPLOAD_DIR);
}

/**
 * Main save helper (switch by STORAGE)
 */
async function save(params) {
  if (STORAGE === 'local') {
    return saveFileLocal(params);
  }
  // placeholder for S3/GCS
  throw new Error('Cloud storage not implemented in this MVP. Set STORAGE=local.');
}

module.exports = { save };
