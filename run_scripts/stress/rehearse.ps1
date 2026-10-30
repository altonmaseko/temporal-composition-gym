# Local smoke rehearsal of the stress pipeline (pipeline check only; never part of the analysis).
#
#   powershell -ExecutionPolicy Bypass -File run_scripts\stress\rehearse.ps1
#
# Simulates the three machines on this laptop, one after another, each with its own results
# root (results\r<id>a, \r<id>b, \r<id>c; merged into r<id>m; bundles and logs in r<id>z; all
# git-ignored and only 7 characters deeper than production paths), all in smoke mode (64 training steps per run, labelled
# results/v3/smoke, W&B offline, CPU device policy):
#   1. preflight (smoke thresholds, CPU, learning check)
#   2. role A and C through their bash wrappers; role B with one injected failure, which must end
#      in BATCH FAILED, then the same command with --continue-batch, which must resume that run
#      and end in BATCH COMPLETE; a rerun without --continue-batch must be refused
#   3. per-role manifest checks, then collection (CSV/JSON zip + .sha256) for each role
#   4. laptop merge in the order C, A, B; a repeat merge (no change); a tampered bundle and a
#      wrong hash, both of which must be refused
#   5. merged manifest 154/154 plus --role and --level filters
#   6. final-policy packing per role outside the checkout, verification, and a refused pack
#      inside the checkout
# Uses training seeds 201-203 / 301-305 only through smoke configs. Takes about an hour.
#
#   powershell -ExecutionPolicy Bypass -File run_scripts\stress\rehearse.ps1 -GuideOnly [-From r6ff6]
#
# -GuideOnly runs only the laptop PowerShell blocks of PRODUCTION_LAUNCH.md marked
# <!-- rehearsal:NAME -->, against an earlier full rehearsal's bundles (results\<From>z) and model
# archives ($env:TEMP\stress-models-<From>), merging into a fresh folder results\g<id> that is
# deleted afterwards. Order: machine C, A, B (verify + models each), final merge, 154/154 check,
# and the staging block as a git add --dry-run. Takes a few minutes; nothing is trained or staged.
param([switch]$GuideOnly, [string]$From = 'r6ff6')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = 'D:\miniconda3\envs\temporal_gym\python.exe'

if ($GuideOnly) {
    $guide = Get-Content -LiteralPath "$PSScriptRoot\PRODUCTION_LAUNCH.md" -Raw
    $Bundles = Join-Path $Repo "run_scripts\stress\results\${From}z"
    $ModelDownloads = Join-Path $env:TEMP "stress-models-$From"
    foreach ($p in $Bundles, $ModelDownloads) { if (-not (Test-Path -LiteralPath $p)) { throw "Missing earlier rehearsal output: $p" } }
    $ResultsRoot = Join-Path $Repo ('run_scripts\stress\results\g' + [guid]::NewGuid().ToString('N').Substring(0, 4))
    $Mode = 'smoke'
    $StageDryRun = $true
    Write-Output "Guide rehearsal: bundles $Bundles ; models $ModelDownloads ; fresh merge folder $ResultsRoot"
    $order = 'verify-c', 'models-c', 'verify-a', 'models-a', 'verify-b', 'models-b', 'final-merge', 'final-check', 'stage'
    try {
        foreach ($name in $order) {
            $match = [regex]::Match($guide, '<!-- rehearsal:' + [regex]::Escape($name) + ' -->\s*```powershell\r?\n([\s\S]*?)```')
            if (-not $match.Success) { throw "Guide block missing: $name" }
            Write-Output ''
            Write-Output "=== guide block: $name"
            . ([scriptblock]::Create($match.Groups[1].Value))
        }
        $staged = @(git -C $Repo diff --cached --name-only)
        if ($staged | Where-Object { $_ -like "*$(Split-Path $ResultsRoot -Leaf)*" }) { throw 'The dry run staged files' }
        Write-Output ''
        Write-Output 'GUIDE REHEARSAL PASSED: merge order C, A, B; per-role and 154/154 manifests; model hashes; staging dry run CSV/JSON only'
    } finally {
        if (Test-Path -LiteralPath $ResultsRoot) { Remove-Item -LiteralPath $ResultsRoot -Recurse -Force }
        Set-Location -LiteralPath $Repo
    }
    exit 0
}
$Bash = 'C:\Program Files\Git\bin\bash.exe'
$PyForBash = $Python.Replace('\', '/')
$Id = 'r' + [guid]::NewGuid().ToString('N').Substring(0, 4)
$Results = "$Repo\run_scripts\stress\results"
function RoleRoot([string]$Role) { Join-Path $Results "$Id$Role" }
$Merged = RoleRoot 'm'
$Zips = RoleRoot 'z'
$Models = Join-Path $env:TEMP "stress-models-$Id"
$env:WANDB_MODE = 'offline'
$env:PYTHONDONTWRITEBYTECODE = '1'
$FailRun = 'maintenance/recurrent_ppo/default/s302@40'
New-Item -ItemType Directory -Force $Zips | Out-Null
Write-Output "Rehearsal id: $Id (roots $(RoleRoot 'a') ... $(RoleRoot 'z'))"

function Step([string]$Text) { Write-Output ''; Write-Output "=== $Text" }
function Expect([int]$Code, [int]$Want, [string]$What) {
    if ($Code -ne $Want) { throw "$What exited $Code (expected $Want)" }
    Write-Output "OK: $What (exit $Code)"
}
function Py([string[]]$ArgList) { & $Python -B @ArgList | Out-Host; return $LASTEXITCODE }
function Launch([string]$Role, [string[]]$Extra) {
    $log = Join-Path $Zips "launch_$Role.log"
    $script = "$Repo\run_scripts\stress\run_${Role}_stress.sh".Replace('\', '/')
    $root = (RoleRoot $Role).Replace('\', '/')
    & $Bash $script --python $PyForBash --threads 1 --mode smoke --device-policy cpu --wandb-mode offline `
        --results-root $root @Extra | Tee-Object -FilePath $log -Append |
        Select-String -Pattern '^(ESTIMATE|FAILED|BATCH|PAUSED|role )' | Out-Host
    return $LASTEXITCODE
}

Step 'Preflight (smoke thresholds, CPU, learning check)'
$code = Py @("$Repo\run_scripts\stress\preflight.py", '--role', 'a', '--threads', '1', '--mode', 'smoke',
             '--device-policy', 'cpu', '--wandb-mode', 'offline', '--results-root', (RoleRoot 'a'))
if ($code -ne 0) {
    Write-Output 'WARNING: preflight reported a failure on this laptop (see [FAIL] lines above). Production machines must pass it; the rehearsal continues.'
}

Step 'Role A'
Expect (Launch 'a' @()) 0 'role A batch'
Step "Role B with an injected failure in $FailRun"
Expect (Launch 'b' @('--debug-fail-run', $FailRun)) 1 'role B batch with injected failure'
Step 'Role B restart with --continue-batch'
Expect (Launch 'b' @('--continue-batch')) 0 'role B --continue-batch'
$resumed = Get-ChildItem -Recurse -Filter status.json (Join-Path (RoleRoot 'b') 'v3\smoke\maintenance\recurrent_ppo\default\s302') |
    ForEach-Object { Get-Content -Raw $_.FullName | ConvertFrom-Json }
if (@($resumed).Count -ne 1 -or $resumed.state -ne 'complete' -or @($resumed.resumes).Count -ne 1) {
    throw 'injected-failure run was not resumed exactly once in its original attempt'
}
Write-Output 'OK: the failed run was resumed once from its restart checkpoint and completed'
Step 'Role C'
Expect (Launch 'c' @()) 0 'role C batch'
Step 'Rerun role A without --continue-batch (must be refused)'
Expect (Launch 'a' @()) 1 'role A rerun without --continue-batch'

Step 'Per-role manifests and collection'
foreach ($r in 'a', 'b', 'c') {
    Expect (Py @("$Repo\run_scripts\stress\check_manifest.py", '--results-root', (RoleRoot $r), '--mode', 'smoke', '--role', $r)) 0 "role $r manifest"
    Expect (Py @("$Repo\run_scripts\stress\collect.py", '--results-root', (RoleRoot $r), '--mode', 'smoke', '--role', $r,
                 '--out', (Join-Path $Zips "stress_${r}_smoke.zip"))) 0 "role $r collection"
}
foreach ($z in Get-ChildItem $Zips -Filter *.zip) {
    $names = [IO.Compression.ZipFile]::OpenRead($z.FullName)
    try {
        $bad = $names.Entries | Where-Object { [IO.Path]::GetExtension($_.FullName) -notin '.csv', '.json' }
    } finally { $names.Dispose() }
    if ($bad) { throw "non CSV/JSON member in $($z.Name)" }
}
Write-Output 'OK: bundles contain CSV/JSON only'

Step 'Laptop merge in order C, A, B; repeat merge; tampered and wrong-hash bundles'
$order = 'c', 'a', 'b' | ForEach-Object { Join-Path $Zips "stress_${_}_smoke.zip" }
& "$PSScriptRoot\merge_bundles.ps1" -ResultsRoot $Merged -Bundles $order
& "$PSScriptRoot\merge_bundles.ps1" -ResultsRoot $Merged -Bundles $order
Write-Output 'OK: repeat merge of identical bundles changed nothing'
$tampered = Join-Path $Zips 'tampered.zip'
$helper = Join-Path $Zips 'tamper.py'
Set-Content -LiteralPath $helper -Encoding ascii -Value @'
import hashlib, sys, zipfile
src, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(src) as z, zipfile.ZipFile(out, "x") as t:
    name = next(n for n in z.namelist() if n.endswith("/final_summary.json"))
    t.writestr(name, z.read(name).replace(b'"global_step"', b'"global_step" ', 1))
digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
with open(out + ".sha256", "wb") as f:
    f.write((digest + "  tampered.zip").encode() + bytes([10]))
'@
$code = Py @($helper, $order[0], $tampered)
Expect $code 0 'tampered bundle built'
$refused = $false
try { & "$PSScriptRoot\merge_bundles.ps1" -ResultsRoot $Merged -Bundles $tampered } catch { $refused = $_.Exception.Message -like 'Conflicting evidence*' }
if (-not $refused) { throw 'conflicting bundle was not refused' }
Write-Output 'OK: conflicting evidence refused'
Copy-Item $order[1] (Join-Path $Zips 'wronghash.zip')
Set-Content -LiteralPath (Join-Path $Zips 'wronghash.zip.sha256') -Value ('0' * 64 + '  wronghash.zip') -Encoding ascii
$refused = $false
try { & "$PSScriptRoot\merge_bundles.ps1" -ResultsRoot $Merged -Bundles (Join-Path $Zips 'wronghash.zip') } catch { $refused = $_.Exception.Message -like 'Hash mismatch*' }
if (-not $refused) { throw 'wrong-hash bundle was not refused' }
Write-Output 'OK: hash mismatch refused'

Step 'Merged manifest'
Expect (Py @("$Repo\run_scripts\stress\check_manifest.py", '--results-root', $Merged, '--mode', 'smoke')) 0 'merged manifest 154/154'
Expect (Py @("$Repo\run_scripts\stress\check_manifest.py", '--results-root', $Merged, '--mode', 'smoke', '--level', 'hard1')) 0 'merged manifest --level hard1'
Expect (Py @("$Repo\run_scripts\stress\check_manifest.py", '--results-root', $Merged, '--mode', 'smoke', '--role', 'b')) 0 'merged manifest --role b'

Step 'Final-policy packing outside the checkout'
foreach ($r in 'a', 'b', 'c') {
    $out = Join-Path $Models "stress_models_${r}_smoke.tar.gz"
    Expect (Py @("$Repo\run_scripts\stress\pack_models.py", 'pack', '--results-root', (RoleRoot $r), '--mode', 'smoke', '--role', $r, '--out', $out)) 0 "role $r models packed"
    Expect (Py @("$Repo\run_scripts\stress\pack_models.py", 'verify', '--archive', $out)) 0 "role $r models verified"
}
Expect (Py @("$Repo\run_scripts\stress\pack_models.py", 'pack', '--results-root', (RoleRoot 'a'), '--mode', 'smoke', '--role', 'a', '--out', (Join-Path $Zips 'inside.tar.gz'))) 1 'pack inside the checkout refused'

Write-Output ''
Write-Output "REHEARSAL PASSED: results $Results\$Id(a|b|c|m|z) ; models $Models"
Write-Output 'Rehearsal folders are kept for inspection; delete them when done (git-ignored, or outside the checkout).'
