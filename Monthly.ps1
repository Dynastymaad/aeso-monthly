<#
  Monthly.ps1 - once a month, after the previous month has settled (any day in the first week).

      .\Monthly.ps1

  1. Outage adder   the aeso-outage-adder needs last month's PLAN and ACTUAL gas outage
                    from NRGStream added to monthly_nrgstream.csv (one row: YYYY-MM,plan,actual).
                    This script opens that file for you and, once you have saved it, refreshes the
                    adder. The seasonal adder and its level factor feed every forward day's gas
                    availability in this model, so this is the one input that cannot be automated.
  2. Deep re-sync   python pull_history.py --days 120   re-pulls four months of forward settles in case
                    a daily top-up missed a late correction, and tops up settled pool price.
  3. Rebuild + publish   .\Refresh.ps1 -SkipCushion
#>
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$adder = Join-Path (Split-Path $PSScriptRoot -Parent) 'aeso-outage-adder'
$csv = Join-Path $adder 'monthly_nrgstream.csv'

Write-Host "`n=== 1/3  outage adder: last month's plan vs actual ===" -ForegroundColor Cyan
if (Test-Path $csv) {
    $last = (Get-Content $csv | Select-Object -Last 1)
    $want = (Get-Date).AddMonths(-1).ToString('yyyy-MM')
    Write-Host "last row on file: $last   (this month you want a $want row)"
    if ($last -like "$want,*") {
        Write-Host "already there." -ForegroundColor DarkGray
    } else {
        Write-Host "Add the row  $want,<plan MW>,<actual MW>  (AB - Gas Monthly Outage / AB - Gas Outage Forecast 6am averaged + Mothball) and save." -ForegroundColor Yellow
        Start-Process notepad.exe $csv -Wait
    }
    if (Test-Path (Join-Path $adder 'refresh_adder.py')) {
        Push-Location $adder; python refresh_adder.py --no-curve; Pop-Location
    }
} else { Write-Host "aeso-outage-adder not found beside this folder - the model keeps using its published adder constants." -ForegroundColor Yellow }

Write-Host "`n=== 2/3  deep re-sync of forwards and settled price ===" -ForegroundColor Cyan
python pull_history.py --days 120

Write-Host "`n=== 3/3  rebuild and publish ===" -ForegroundColor Cyan
& .\Refresh.ps1 -SkipCushion
