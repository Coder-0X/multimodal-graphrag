# Automation script to create the GitHub repository and push commits
$ErrorActionPreference = "Stop"
$ProjectPath = "C:\Users\NITRO V 15\.gemini\antigravity\scratch\multimodal-graphrag"

Set-Location $ProjectPath
Write-Host "Creating GitHub repository Coder-0X/multimodal-graphrag and pushing commits..." -ForegroundColor Cyan

gh repo create multimodal-graphrag `
  --public `
  --source="." `
  --remote="origin" `
  --push `
  --description "Multimodal GraphRAG & Complex Document Ingestion Engine (Tables, Balance Sheets, Diagrams, LangGraph, NetworkX/Neo4j)"

Write-Host "Successfully pushed multimodal-graphrag to GitHub!" -ForegroundColor Green
gh repo view Coder-0X/multimodal-graphrag --web
