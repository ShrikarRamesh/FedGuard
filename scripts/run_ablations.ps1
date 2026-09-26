# Ablations (seed 0) + post-hoc analyses, resumable.
# Usage: .\scripts\run_ablations.ps1 [-Workers 2] [-Fast]
param([int]$Workers = 2, [switch]$Fast)
$Py = if ($env:FEDGUARD_PY) { $env:FEDGUARD_PY } else { 'C:\Users\Shrikar\.venvs\fedguard\Scripts\python.exe' }
$f = if ($Fast) { @('--fast') } else { @() }
& $Py (Join-Path $PSScriptRoot 'run_experiments.py') --stage ablations --workers $Workers @f
& $Py -m fedguard.cli mc-ablation -e fedguard --seeds 0 @f
