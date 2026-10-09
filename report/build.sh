#!/bin/sh
# Build report/report.pdf from report/report.md (needs pandoc and Google Chrome).
set -e
cd "$(dirname "$0")"
pandoc report.md -s --embed-resources --css style.css --resource-path=. -o report.html
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu \
  --no-pdf-header-footer --print-to-pdf=report.pdf "file://$PWD/report.html" 2>/dev/null
rm report.html
echo "wrote report/report.pdf"
