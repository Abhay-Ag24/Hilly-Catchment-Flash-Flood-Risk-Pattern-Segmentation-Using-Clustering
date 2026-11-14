/**
 * Development & Presentation Web Server
 * Serves the Flash-Flood Risk Segmentation Dashboard
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3000;
const PUBLIC_DIR = path.join(__dirname, '..', 'web_app');

const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.csv': 'text/csv',
  '.png': 'image/png',
  '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
  let relativePath = req.url.split('?')[0];
  if (relativePath === '/' || relativePath === '') {
    relativePath = '/index.html';
  }

  let filePath = path.join(PUBLIC_DIR, relativePath);

  // Fallback for data directory access (for CSV downloads)
  if (relativePath.startsWith('/data/')) {
    filePath = path.join(__dirname, '..', relativePath);
  }

  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';
    res.writeHead(200, {
      'Content-Type': contentType,
      'Access-Control-Allow-Origin': '*'
    });
    fs.createReadStream(filePath).pipe(res);
  } else {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('404 Not Found');
  }
});

server.listen(PORT, () => {
  console.log(`================================================================`);
  console.log(`Flash-Flood Risk Segmentation Dashboard running at:`);
  console.log(`http://localhost:${PORT}`);
  console.log(`Press Ctrl+C to terminate the server.`);
  console.log(`================================================================`);
});
