[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimePackageKey,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$dryRunRoot = Join-Path $RepositoryRoot "artifacts\linux-dry-run\$RuntimePackageKey"
$jsonPath = Join-Path $dryRunRoot "linux-runtime-dry-run.json"
if (-not (Test-Path -LiteralPath $jsonPath)) {
  throw "Linux dry-run summary was not found: $jsonPath"
}

$summary = Get-Content -LiteralPath $jsonPath -Raw -Encoding utf8 | ConvertFrom-Json
$outputPath = Join-Path $dryRunRoot "linux-preflight-summary.md"

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Linux Preflight Summary")
$lines.Add("")
$lines.Add("Runtime key: $($summary.runtimeKey)")
$lines.Add("")
$lines.Add("Package ID: $($summary.packageId)")
$lines.Add("")
$lines.Add("Distribution tier: $($summary.distributionTier)")
$lines.Add("")
$lines.Add("Validation state: $($summary.validationState)")
$lines.Add("")
$lines.Add("## Readiness")
$lines.Add("")
$lines.Add("- Dry-run artifacts are present.")
$lines.Add("- Input roots were accepted by the dry-run step.")
$lines.Add("- Real build/pack execution is still pending on a Linux x64 self-hosted runner unless separately validated.")
$lines.Add("")
$lines.Add("## Required labels")
$lines.Add("")
foreach ($label in $summary.requiredRunnerLabels) {
  $lines.Add("- $label")
}
$lines.Add("")
$lines.Add("## Required tools")
$lines.Add("")
foreach ($tool in $summary.requiredTools) {
  $lines.Add("- $tool")
}
$lines.Add("")
$lines.Add("## Expected artifacts after a real run")
$lines.Add("")
foreach ($artifact in $summary.expectedBuildArtifacts) {
  $lines.Add("- build: $artifact")
}
foreach ($artifact in $summary.expectedPackArtifacts) {
  $lines.Add("- pack: $artifact")
}
$lines.Add("")
$lines.Add("## First blockers to inspect")
$lines.Add("")
foreach ($failure in $summary.failureScenarios) {
  $lines.Add("- $failure")
}

$lines | Set-Content -LiteralPath $outputPath -Encoding utf8
Write-Host "Linux preflight summary written to $outputPath"
