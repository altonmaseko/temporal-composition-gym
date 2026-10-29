$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = 'D:\miniconda3\envs\temporal_gym\python.exe'
$Lab = Join-Path "$Repo\run_scripts\tuning\results" ('r'+[guid]::NewGuid().ToString('N').Substring(0,5))
$ResultsRoot = Join-Path $Lab 'm'
$Bundles = Join-Path $Lab 'z'
$Mode = 'smoke'
$guide = Get-Content -LiteralPath "$PSScriptRoot\PRODUCTION_LAUNCH.md" -Raw
function Invoke-GuideBlock([string]$Name) {
    $pattern = '<!-- rehearsal:'+[regex]::Escape($Name)+' -->\s*```powershell\r?\n([\s\S]*?)```'
    $match = [regex]::Match($guide,$pattern)
    if (-not $match.Success) { throw "Guide block missing: $Name" }
    Invoke-Expression $match.Groups[1].Value
}
& $Python -B "$PSScriptRoot\rehearsal_data.py" --root $Lab --stage screen
if ($LASTEXITCODE -ne 0) { throw 'Fixture setup failed' }
Invoke-GuideBlock 'screen-merge'
# Dot source the block so its finalist variables remain in this scope.
$match = [regex]::Match($guide,'<!-- rehearsal:screen-check -->\s*```powershell\r?\n([\s\S]*?)```')
. ([scriptblock]::Create($match.Groups[1].Value))
foreach ($role in 'a','b','c') {
    $dest = Join-Path $Lab "$role\v3\smoke\finalists.json"
    Copy-Item -LiteralPath $Finalists -Destination $dest
    if ((Get-FileHash $dest).Hash.ToLower() -ne $FinalistHash) { throw 'Distribution hash mismatch' }
}
Write-Output 'FINALISTS DISTRIBUTED AND HASH VERIFIED'
& $Python -B "$PSScriptRoot\rehearsal_data.py" --root $Lab --stage confirm
if ($LASTEXITCODE -ne 0) { throw 'Confirmation fixtures failed' }
Invoke-GuideBlock 'final-merge'
Invoke-GuideBlock 'final-check'
Write-Output "REHEARSAL PASSED: $Lab"
