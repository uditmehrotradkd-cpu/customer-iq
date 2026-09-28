# Start the Customer Segmentation website locally: .\run_website.ps1 [-Port 8000] [-Dev] [-AiModel qwen3.5:4b]
# Settings in a local .env file (KEY=VALUE per line) are loaded first, e.g. SEG_DATABASE_URL (cloud accounts)
# and GEMINI_API_KEY (AI answers by Google Gemini). -AiModel uses a local Ollama model instead.
param(
    [int]$Port = 8000,
    [switch]$Dev,
    [string]$AiModel = ""
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (Test-Path ".env") {
    foreach ($line in Get-Content ".env") {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2].Trim('"')
        }
    }
    Write-Host "Loaded settings from .env"
}

if (-not (Test-Path "artifacts\segmenter.joblib")) {
    Write-Host "No trained model found - training once..."
    python run_pipeline.py train
}

if ($AiModel) {
    $env:SEG_LLM_BASE_URL = "http://localhost:11434/v1"
    $env:SEG_LLM_MODEL = $AiModel
    $env:SEG_LLM_REASONING = "none"
    $env:SEG_LLM_TIMEOUT = "120"
    try { Invoke-RestMethod -TimeoutSec 3 http://localhost:11434/api/tags | Out-Null }
    catch {
        Write-Host "Starting Ollama..."
        Start-Process -WindowStyle Hidden ollama -ArgumentList "serve"
    }
    Write-Host "AI answers: $AiModel via Ollama"
}

$env:SEG_ALLOWED_HOSTS = "localhost,127.0.0.1"
Write-Host "Website: http://localhost:$Port   API docs: http://localhost:$Port/docs"
if ($Dev) {
    python -m uvicorn web.main:app --host 127.0.0.1 --port $Port --reload
} else {
    python -m uvicorn web.main:app --host 127.0.0.1 --port $Port --workers 2
}
