<#
  Refresh.ps1 - the daily command. Run it once each morning.

      .\Refresh.ps1              refresh the inputs, rebuild the read, publish to GitHub
      .\Refresh.ps1 -NoPush      rebuild and stop (look at docs\index.html first)
      .\Refresh.ps1 -SkipCushion reuse this morning's cushion-repo refresh (if you already ran Daily.ps1 there)

  What it does, in order:
    1. cushion repo   .\Update.ps1 -SkipScore -NoPush  next door - pulls CANPOWER + AESO
                      (composition, weather, gencap, outage report, the 44-day outlook).
                      Skipped automatically if that repo already refreshed in the last 6 hours.
    2. this repo      python update.py - tops up gas and power forwards and settled price,
                      rebuilds the curve, the outlook, the contracts; writes docs\index.html
                      and docs\artifact.html; appends today's reads to history\reads.csv
    3. publish        git add / commit / push  ->  GitHub Pages serves docs\index.html

  Monthly.ps1 is separate: the things that only need doing once a month.
#>
param([switch]$NoPush, [switch]$SkipCushion, [string]$Message = "")
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$cushion = Join-Path (Split-Path $PSScriptRoot -Parent) 'aeso-cushion model'

# ---- 1. the inputs the cushion repo owns
if (-not $SkipCushion) {
    $stamp = Join-Path $cushion 'cache\composition.csv'
    $fresh = (Test-Path $stamp) -and (((Get-Date) - (Get-Item $stamp).LastWriteTime).TotalHours -lt 6)
    if ($fresh) {
        Write-Host "`n=== 1/3  cushion repo refreshed $([int]((Get-Date) - (Get-Item $stamp).LastWriteTime).TotalMinutes) min ago - reusing ===" -ForegroundColor DarkGray
    } elseif (Test-Path (Join-Path $cushion 'Update.ps1')) {
        Write-Host "`n=== 1/3  refreshing the cushion repo's inputs ===" -ForegroundColor Cyan
        Push-Location $cushion
        & .\Update.ps1 -SkipScore -NoPush
        Pop-Location
        if (-not (Test-Path $stamp)) { Write-Host "cushion repo did not write cache\composition.csv - carrying on with what is there" -ForegroundColor Yellow }
    } else {
        Write-Host "`n=== 1/3  cushion repo not found at $cushion - carrying on with cached inputs ===" -ForegroundColor Yellow
    }
} else { Write-Host "`n=== 1/3  cushion refresh skipped (-SkipCushion) ===" -ForegroundColor DarkGray }

# ---- 2. this model
Write-Host "`n=== 2/3  rebuilding the monthly read ===" -ForegroundColor Cyan
python update.py
if ($LASTEXITCODE -ne 0) { Write-Host "update.py failed - nothing published. Read the error above." -ForegroundColor Red; exit 1 }
if (-not (Test-Path 'docs\index.html')) { Write-Host "docs\index.html was not written." -ForegroundColor Red; exit 1 }
$age = ((Get-Date) - (Get-Item 'docs\index.html').LastWriteTime).TotalMinutes
if ($age -gt 10) { Write-Host "docs\index.html is stale ($([int]$age) min) - the rebuild did not take." -ForegroundColor Red; exit 1 }

if ($NoPush) { Write-Host "`n-NoPush: open docs\index.html. Publish later with .\Refresh.ps1 -SkipCushion" -ForegroundColor Green; exit 0 }

# ---- 3. publish
Write-Host "`n=== 3/3  publishing ===" -ForegroundColor Cyan
if (-not (Test-Path '.git')) { Write-Host "no git repo here yet - run .\Setup-GitHub.ps1 once." -ForegroundColor Yellow; exit 0 }
$lock = Join-Path $PSScriptRoot '.git\index.lock'
if ((Test-Path $lock) -and (((Get-Date) - (Get-Item $lock).LastWriteTime).TotalMinutes -gt 2)) { Remove-Item $lock -Force -ErrorAction SilentlyContinue }
$env:GIT_TERMINAL_PROMPT = '0'; $env:GCM_INTERACTIVE = 'Never'
if (-not $Message) { $Message = "refresh $(Get-Date -Format 'yyyy-MM-dd HH:mm')" }
git add -A
git commit -m $Message --quiet
if ($LASTEXITCODE -ne 0) { Write-Host "nothing new to commit." -ForegroundColor DarkGray; exit 0 }
git push
if ($LASTEXITCODE -ne 0) { Write-Host "git push failed - the read is rebuilt and committed; push by hand when the connection is back (git push)." -ForegroundColor Yellow; exit 0 }
Write-Host "`npublished. GitHub Pages picks it up in a minute or two." -ForegroundColor Green
