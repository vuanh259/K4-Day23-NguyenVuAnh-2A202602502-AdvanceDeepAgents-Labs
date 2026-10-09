# Start a local-only Ollama service for the lab; reuse installed models.
$ErrorActionPreference = 'Stop'
$labPort = 'http://127.0.0.1:11435'
try {
    $null = Invoke-RestMethod "$labPort/api/version" -TimeoutSec 2
    Write-Output "Ollama is already available at $labPort"
    exit 0
} catch { }
$labCommand = Get-Command ollama -ErrorAction SilentlyContinue
$labExecutable = if ($labCommand) { $labCommand.Source } else { 'E:\AIData\Ollama\App\ollama.exe' }
if (-not (Test-Path -LiteralPath $labExecutable)) { throw 'Install Ollama and add ollama.exe to PATH.' }
$labModels = [Environment]::GetEnvironmentVariable('OLLAMA_MODELS', 'User')
if ($labModels) { $env:OLLAMA_MODELS = $labModels }
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_FLASH_ATTENTION = '1'
$env:OLLAMA_KV_CACHE_TYPE = 'q8_0'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_NO_CLOUD = '1'
$env:LLAMA_ARG_FIT_TARGET = '128'
$labLogDirectory = Join-Path $PSScriptRoot '.local'
$null = New-Item -ItemType Directory -Path $labLogDirectory -Force
$labProcess = Start-Process -FilePath $labExecutable -ArgumentList 'serve' -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $labLogDirectory 'ollama-local.out.log') `
    -RedirectStandardError (Join-Path $labLogDirectory 'ollama-local.err.log')
Write-Output "Started local Ollama process $($labProcess.Id) at $labPort"
