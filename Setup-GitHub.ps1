<#
  Setup-GitHub.ps1 - run ONCE to put this folder on GitHub and turn on Pages.

      .\Setup-GitHub.ps1                 uses the GitHub CLI (gh) if installed
      .\Setup-GitHub.ps1 -User dynastymaad -Repo aeso-monthly

  With gh:     creates a private repo, pushes, enables Pages from main:/docs.
  Without gh:  initialises and commits locally, then prints the three steps to finish by hand.
  Safe to re-run: it does nothing that already exists.
#>
param([string]$User = "", [string]$Repo = "aeso-monthly")
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

if (-not (Test-Path '.git')) { git init -b main | Out-Null; Write-Host "git repo initialised" }
git add -A
git commit -m "AESO monthly model" --quiet 2>$null
if ($LASTEXITCODE -eq 0) { Write-Host "first commit made" } else { Write-Host "nothing new to commit" -ForegroundColor DarkGray }

$hasRemote = (git remote) -contains 'origin'
$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $hasRemote -and $gh) {
    if (-not $User) { $User = (gh api user --jq .login 2>$null) }
    Write-Host "creating private repo $User/$Repo with gh ..."
    gh repo create "$User/$Repo" --private --source . --remote origin --push
    if ($LASTEXITCODE -ne 0) { Write-Host "gh repo create failed - is gh logged in? (gh auth login)" -ForegroundColor Yellow; exit 1 }
    Write-Host "enabling GitHub Pages from main:/docs ..."
    gh api -X POST "repos/$User/$Repo/pages" -f "source[branch]=main" -f "source[path]=/docs" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { gh api -X PUT "repos/$User/$Repo/pages" -f "source[branch]=main" -f "source[path]=/docs" 2>$null | Out-Null }
    Write-Host "`ndone. Your page will be at  https://$User.github.io/$Repo/  in a minute or two." -ForegroundColor Green
    exit 0
}
if ($hasRemote) {
    git push -u origin main
    Write-Host "pushed to the existing origin." -ForegroundColor Green
    exit 0
}
Write-Host @"

No GitHub CLI found, so finish these three steps by hand (two minutes):

  1. On github.com create an EMPTY private repo named  $Repo  (no README, no .gitignore).
  2. Back here:
         git remote add origin https://github.com/<you>/$Repo.git
         git push -u origin main
  3. On the repo: Settings -> Pages -> Source 'Deploy from a branch' -> branch main, folder /docs -> Save.

Your page will be at  https://<you>.github.io/$Repo/
After that, .\Refresh.ps1 publishes every morning on its own.
"@
