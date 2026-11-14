@echo off
echo Starting Flash-Flood Risk Segmentation Web Dashboard...
echo Opening browser at http://localhost:3000
start http://localhost:3000
"C:\Users\agraw\AppData\Roaming\Antigravity\bin\agy-node.cmd" "%~dp0scripts\serve.js"
pause
