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

function Get-ExpectedNativeFileNames {
  param(
    [object]$Package
  )

  $files = @($Package.bridgeFile)
  foreach ($relativePath in @($Package.tensorRtFiles + $Package.cudaFiles)) {
    $files += [System.IO.Path]::GetFileName($relativePath)
  }

  return @($files | Sort-Object -Unique)
}

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$package = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $package) {
  throw "Runtime package key '$RuntimePackageKey' was not found."
}

if ($package.platform -ne "linux") {
  throw "Runtime package key '$RuntimePackageKey' is not a Linux package."
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\linux-dry-run\$RuntimePackageKey"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$expectedNativeFiles = @(Get-ExpectedNativeFileNames -Package $package)
$commands = @(
  "dotnet pack ./pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj -c Release -o ./artifacts/managed",
  "dotnet pack ./pack/runtime/$RuntimePackageKey/$($package.packageId).csproj -c Release -o ./artifacts/runtime-nupkg",
  "pwsh -File ./eng/Test-PackageConsumer.ps1 -RuntimePackageKey $RuntimePackageKey",
  "pwsh -File ./eng/Test-PackageConsumer.ps1 -RuntimePackageKey $RuntimePackageKey -RunSmoke"
)

$summary = [ordered]@{
  runtimeKey = $package.key
  packageId = $package.packageId
  rid = $package.rid
  validationState = $package.validationState
  distributionTier = $package.distributionTier
  managedPackageDirectory = "artifacts/managed"
  runtimePackageDirectory = "artifacts/runtime-nupkg"
  consumerReportDirectory = "artifacts/package-consumer"
  expectedNativeFiles = $expectedNativeFiles
  commands = $commands
  smokeNotes = @(
    "Run the non-smoke consumer validation first to prove restore, build, and native asset copy.",
    "Run -RunSmoke only when the Linux runner has a usable NVIDIA driver, CUDA runtime compatibility, and GPU access.",
    "Do not change validationState from dry-run-only unless the non-smoke consumer validation passes on Linux; GPU smoke is additional evidence."
  )
  promotionCriteria = @(
    "Managed package nupkg exists under artifacts/managed.",
    "Linux runtime package nupkg exists under artifacts/runtime-nupkg.",
    "Test-PackageConsumer.ps1 restores both packages from local package sources.",
    "The consumer project builds for RID linux-x64.",
    "The output contains JYPPX.Shared.dll, JYPPX.TensorRtSharp.dll, and JYPPX.CudaSharp.dll.",
    "The output contains libjyppxtrtbridge.so plus every declared TensorRT/CUDA .so asset pattern.",
    "Optional smoke loads the bridge and queries TensorRT/CUDA if the runner has GPU access."
  )
}

$jsonPath = Join-Path $outputRoot "linux-package-consumer-plan.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Linux Package Consumer Plan")
$lines.Add("")
$lines.Add("Runtime key: $($package.key)")
$lines.Add("")
$lines.Add("Package ID: $($package.packageId)")
$lines.Add("")
$lines.Add("RID: $($package.rid)")
$lines.Add("")
$lines.Add("Validation state: $($package.validationState)")
$lines.Add("")
$lines.Add("Distribution tier: $($package.distributionTier)")
$lines.Add("")
$lines.Add("## Purpose")
$lines.Add("")
$lines.Add("This plan is the handoff from Linux build/pack to a real consumer-project validation on the same Linux self-hosted runner.")
$lines.Add("")
$lines.Add("## Commands")
$lines.Add("")
foreach ($command in $commands) {
  $lines.Add("1. $command")
}
$lines.Add("")
$lines.Add("## Expected native files")
$lines.Add("")
foreach ($file in $expectedNativeFiles) {
  $lines.Add("- $file")
}
$lines.Add("")
$lines.Add("## Smoke policy")
$lines.Add("")
foreach ($note in $summary.smokeNotes) {
  $lines.Add("- $note")
}
$lines.Add("")
$lines.Add("## Promotion criteria")
$lines.Add("")
foreach ($criteria in $summary.promotionCriteria) {
  $lines.Add("- $criteria")
}

$markdownPath = Join-Path $outputRoot "linux-package-consumer-plan.md"
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Linux package consumer plan written to $jsonPath"
Write-Host "Linux package consumer plan written to $markdownPath"
