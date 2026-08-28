# Vendor analysis-pipeline artifacts into report/assets/ so the results report
# compiles from a clean clone (outputs/ is gitignored).
#
# Emitted .tex files are full `table` floats carrying their own \caption and
# \label. Four datasets would therefore contribute four copies of every label.
# Only the tabular body is kept here; caption and label stay in the report.

$ErrorActionPreference = "Stop"
$assets = $PSScriptRoot
$outputs = Join-Path $assets "..\..\outputs"

if (-not (Test-Path $outputs)) {
    Write-Host "No outputs/ directory yet - nothing to vendor."
    exit 0
}

foreach ($dataset in @("camelyon16", "tcga_ut", "bracs", "panda")) {
    $src = Join-Path $outputs "$dataset\patch"
    if (-not (Test-Path $src)) {
        Write-Host "skip $dataset (no results)"
        continue
    }
    $dst = Join-Path $assets $dataset
    New-Item -ItemType Directory -Force $dst | Out-Null

    # Tables: strip the float wrapper, keep the tabular body.
    Get-ChildItem -Path $src -Recurse -Filter "*.tex" | ForEach-Object {
        $body = Get-Content $_.FullName | Where-Object {
            $_ -notmatch '^\s*%' -and
            $_ -notmatch '^\s*\\(begin|end)\{table\}' -and
            $_ -notmatch '^\s*\\centering\s*$' -and
            $_ -notmatch '^\s*\\caption\{' -and
            $_ -notmatch '^\s*\\label\{'
        }
        # split=<i> artifacts keep the split in the name so they do not collide
        # with the base-level aggregate of the same name.
        $prefix = if ($_.FullName -match 'split=(\d)') { "split$($Matches[1])_" } else { "" }
        $body | Set-Content (Join-Path $dst "$prefix$($_.Name)")
    }

    Get-ChildItem -Path $src -Recurse -Filter "*.png" | ForEach-Object {
        $prefix = if ($_.FullName -match 'split=(\d)') { "split$($Matches[1])_" } else { "" }
        Copy-Item $_.FullName (Join-Path $dst "$prefix$($_.Name)") -Force
    }

    Write-Host "vendored $dataset"
}
