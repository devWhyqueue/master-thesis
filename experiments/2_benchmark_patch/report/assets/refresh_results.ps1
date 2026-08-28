# Vendor analysis artifacts into report/assets/ so the results report compiles
# from a clean clone (outputs/ is gitignored).
#
# Tables come from the `report-tables` command, which reads every dataset root
# listed in rq3_combined.yaml and writes one fragment per float in the results
# report. Each fragment is a bare tabular or longtable body: one float in the
# report spans all four datasets, so caption and label stay with the float.
#
# Figures are per split and per dataset; the report shows the first locked
# partition, so only split=0 is vendored.
#
# Run `report-tables` on the cluster first:
#   python __main__.py --config ../configs/rq3_combined.yaml report-tables

$ErrorActionPreference = "Stop"
$assets = $PSScriptRoot
$outputs = Join-Path $assets "..\..\outputs"

if (-not (Test-Path $outputs)) {
    Write-Host "No outputs/ directory yet - nothing to vendor."
    exit 0
}

$fragments = Join-Path $outputs "rq3_combined\tables\report"
if (Test-Path $fragments) {
    $destination = Join-Path $assets "tables"
    New-Item -ItemType Directory -Force $destination | Out-Null
    Copy-Item (Join-Path $fragments "*") $destination -Force
    Write-Host "vendored report tables"
} else {
    Write-Host "skip tables (run report-tables first)"
}

foreach ($dataset in @("camelyon16", "tcga_ut", "bracs", "panda")) {
    $figures = Join-Path $outputs "$dataset\patch\split=0\figures"
    if (-not (Test-Path $figures)) {
        Write-Host "skip $dataset (no figures)"
        continue
    }
    $destination = Join-Path $assets $dataset
    New-Item -ItemType Directory -Force $destination | Out-Null
    Copy-Item (Join-Path $figures "*.png") $destination -Force
    Write-Host "vendored $dataset figures"
}
