/**
 * Standalone Generator
 * Generates dataset.js and a single-file Flash_Flood_Dashboard.html
 * so it opens directly in ANY browser (Chrome, Edge, Firefox, Opera)
 * with zero command prompt and zero server setup.
 */

const fs = require('fs');
const path = require('path');

const projectDir = path.join(__dirname, '..');
const webDir = path.join(projectDir, 'web_app');

const jsonPath = path.join(webDir, 'dataset.json');
const jsDataPath = path.join(webDir, 'dataset.js');
const cssPath = path.join(webDir, 'styles.css');
const htmlPath = path.join(webDir, 'index.html');
const appJsPath = path.join(webDir, 'app.js');
const standalonePath = path.join(projectDir, 'Flash_Flood_Dashboard.html');

const datasetJson = fs.readFileSync(jsonPath, 'utf8');

// 1. Create dataset.js
const datasetJsContent = `// Global dataset variable for direct file:// browser execution\nwindow.appData = ${datasetJson};\n`;
fs.writeFileSync(jsDataPath, datasetJsContent, 'utf8');
console.log(`Created: ${jsDataPath}`);

// 2. Read HTML, CSS, app.js
let html = fs.readFileSync(htmlPath, 'utf8');
const css = fs.readFileSync(cssPath, 'utf8');
const appJs = fs.readFileSync(appJsPath, 'utf8');

// Update index.html to include dataset.js before app.js
if (!html.includes('dataset.js')) {
  html = html.replace('<script src="app.js"></script>', '<script src="dataset.js"></script>\n  <script src="app.js"></script>');
  fs.writeFileSync(htmlPath, html, 'utf8');
  console.log(`Updated index.html to include dataset.js`);
}

// 3. Create single-file standalone HTML
let standaloneHtml = html;
// Replace <link rel="stylesheet" href="styles.css"> with <style>...</style>
standaloneHtml = standaloneHtml.replace('<link rel="stylesheet" href="styles.css">', `<style>\n${css}\n</style>`);
// Replace scripts with inline script containing data and app logic
const inlineScripts = `
<script>
window.appData = ${datasetJson};
</script>
<script>
${appJs}
</script>
`;
standaloneHtml = standaloneHtml.replace(/<script src="dataset.js"><\/script>\s*<script src="app.js"><\/script>/, inlineScripts);

fs.writeFileSync(standalonePath, standaloneHtml, 'utf8');
console.log(`Created standalone single-file dashboard at: ${standalonePath}`);
