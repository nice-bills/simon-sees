# One-command playtest for Windows (PowerShell): install (first time only), start the server, give you the
# phone link, and when you press Ctrl+C at the end, upload the recorded session (or zip it without HF_TOKEN).
#
#   .\run_playtest.ps1            # real referee: SAM 3 -> DINOv3 -> d1-3B on this laptop
#   .\run_playtest.ps1 fast       # lighter models for CPU laptops
#   .\run_playtest.ps1 mock       # no AI: checks camera, phone link, recording and upload (random calls)
#   .\run_playtest.ps1 modal      # vision + d1 on your Modal GPU (needs $env:SIMON_VISION_URL)
#
# If Windows blocks the script: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# Guide: docs/playtest.md
param([ValidateSet("real", "fast", "mock", "modal")][string]$Mode = "real")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Port = if ($env:PORT) { $env:PORT } else { "8000" }

switch ($Mode) {
  "real"  { $Flags = @();                                                                  $Req = "requirements-model.txt" }
  "fast"  { $Flags = @("--segmenter", "fallback", "--embedder", "dinov2", "--model", "LiquidAI/d1-omni-600M"); $Req = "requirements-model.txt" }
  "mock"  { $Flags = @("--segmenter", "mock", "--embedder", "mock", "--decider", "mock");  $Req = "requirements.txt" }
  "modal" { if (-not $env:SIMON_VISION_URL) { Write-Host "modal needs SIMON_VISION_URL (docs/modal.md)"; exit 1 }
            $Flags = @("--backend", "modal");                                              $Req = "requirements.txt" }
}

if (-not (Test-Path .venv)) { python -m venv .venv }
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path ".venv\.installed-$Req")) {
  Write-Host "Installing $Req (first run only, can take a few minutes)..."
  & $Py -m pip install -q --upgrade pip
  & $Py -m pip install -q -r $Req
  New-Item ".venv\.installed-$Req" -ItemType File | Out-Null
}
if (-not $env:HF_TOKEN) {
  Write-Host "Note: HF_TOKEN is not set. The real models need it to download SAM 3, and upload needs it at the end."
  Write-Host '      $env:HF_TOKEN = "hf_..."   (docs/playtest.md, step 3)'
}

$Server = Start-Process -FilePath $Py -ArgumentList (@("-m", "simon_sees", "serve", "--port", $Port) + $Flags) -NoNewWindow -PassThru
$Tunnel = $null
try {
  if (Get-Command cloudflared -ErrorAction SilentlyContinue) {
    $Log = New-TemporaryFile
    $Tunnel = Start-Process cloudflared -ArgumentList @("tunnel", "--url", "http://localhost:$Port") -NoNewWindow -PassThru -RedirectStandardError $Log
    $Url = $null
    for ($i = 0; $i -lt 30 -and -not $Url; $i++) {
      Start-Sleep 1
      $m = Select-String -Path $Log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1
      if ($m) { $Url = $m.Matches[0].Value }
    }
    Write-Host "`n  Open this on your phone:  $Url`n"
  } else {
    Write-Host "`n  cloudflared is not installed, so there is no https link for the phone yet."
    Write-Host "  Install it (docs/playtest.md, step 4) or follow docs/phone-setup.md (option C).`n"
  }
  Write-Host "  When you are done playing, press Ctrl+C here: the session is uploaded (or zipped) for you."
  Wait-Process -Id $Server.Id
} finally {
  Write-Host "`nStopping..."
  if ($Tunnel) { Stop-Process -Id $Tunnel.Id -ErrorAction SilentlyContinue }
  Stop-Process -Id $Server.Id -ErrorAction SilentlyContinue
  if ($env:HF_TOKEN) {
    & $Py -m simon_sees upload
    if ($LASTEXITCODE -ne 0) { & $Py -m simon_sees bundle }
  } else { & $Py -m simon_sees bundle }
}
