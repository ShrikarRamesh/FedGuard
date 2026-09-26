# All main experiments (3 seeds) + privacy sweep + baselines, resumable (completed runs are skipped).
# Usage: .\scripts\run_all_main.ps1 [-Workers 2] [-Fast]
param([int]$Workers = 2, [switch]$Fast)
$Py = if ($env:FEDGUARD_PY) { $env:FEDGUARD_PY } else { 'C:\Users\Shrikar\.venvs\fedguard\Scripts\python.exe' }
$f = if ($Fast) { @('--fast') } else { @() }
foreach ($stage in 'tune', 'tune_dp', 'main', 'baselines', 'sweep') {
    & $Py (Join-Path $PSScriptRoot 'run_experiments.py') --stage $stage --workers $Workers @f
}
