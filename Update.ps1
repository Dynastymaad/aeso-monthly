# Superseded: the daily command is now .\Refresh.ps1 (and .\Monthly.ps1 once a month).
param([switch]$NoPush, [switch]$NoPull)
& (Join-Path $PSScriptRoot 'Refresh.ps1') -NoPush:$NoPush
