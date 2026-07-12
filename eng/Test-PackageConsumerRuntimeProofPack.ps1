[CmdletBinding()]
param(
  [string]$PackPath = ".\artifacts\final-release\package-consumer-runtime-proof-execution-pack.json",
  [string]$RepositoryRoot,
  [switch]$FailOnNotProof
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($PackPath)) {
  $PackPath = Join-Path $RepositoryRoot $PackPath
}

if (-not (Test-Path -LiteralPath $PackPath -PathType Leaf)) {
  throw "Missing package consumer proof pack: $PackPath"
}

function Get-Value {
  param([object]$Object, [string]$Name)
  if ($null -eq $Object -or $Object.PSObject.Properties.Name -notcontains $Name) {
    return $null
  }

  return $Object.PSObject.Properties[$Name].Value
}

function Test-Sha256Text {
  param([object]$Value)
  if ($null -eq $Value) { return $false }
  return ([string]$Value) -match "^[0-9A-Fa-f]{64}$"
}

$pack = Get-Content -LiteralPath $PackPath -Raw -Encoding utf8 | ConvertFrom-Json
$missing = New-Object System.Collections.Generic.List[string]
$invalid = New-Object System.Collections.Generic.List[string]

foreach ($field in @("runtimePackageKey", "cleanConsumerRoot", "packageSourceKind", "packageSourcePathOrUrl", "managedNupkgSha256", "runtimeNupkgSha256", "restoreLogPath", "buildLogPath", "runtimeSmokeLogPath", "runtimeSmokeLogSha256", "noProjectReference", "validatorCommand")) {
  $value = [string](Get-Value -Object $pack -Name $field)
  if ([string]::IsNullOrWhiteSpace($value) -or $value -match "owner-action-required|template|TODO|TBD") {
    $missing.Add($field)
  }
}

foreach ($field in @("managedNupkgSha256", "runtimeNupkgSha256", "runtimeSmokeLogSha256")) {
  if (-not (Test-Sha256Text (Get-Value -Object $pack -Name $field))) {
    $invalid.Add($field)
  }
}

$cleanRoot = [string](Get-Value -Object $pack -Name "cleanConsumerRoot")
$cleanRootInsideRepository = $false
if (-not [string]::IsNullOrWhiteSpace($cleanRoot) -and $cleanRoot -notmatch "owner-action-required|template|TODO|TBD") {
  $fullCleanRoot = if ([System.IO.Path]::IsPathRooted($cleanRoot)) { [System.IO.Path]::GetFullPath($cleanRoot) } else { [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $cleanRoot)) }
  $fullRepo = [System.IO.Path]::GetFullPath($RepositoryRoot)
  $cleanRootInsideRepository = $fullCleanRoot.StartsWith($fullRepo, [System.StringComparison]::OrdinalIgnoreCase)
  if ($cleanRootInsideRepository) {
    $invalid.Add("cleanConsumerRootInsideRepository")
  }
}

$noProjectReference = [string](Get-Value -Object $pack -Name "noProjectReference")
if ($noProjectReference -notmatch "^(true|True|TRUE)$") {
  $invalid.Add("noProjectReference")
}

$canPromote = $missing.Count -eq 0 -and $invalid.Count -eq 0 -and -not $cleanRootInsideRepository
$validationState = if ($canPromote) { "validated-package-consumer-runtime-candidate" } else { "blocked-owner-action-required" }

$validation = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "package-consumer-runtime-proof-pack-validation"
  sourcePackPath = $PackPath
  validationState = $validationState
  missingOwnerInputCount = $missing.Count
  missingOwnerInputs = @($missing.ToArray())
  invalidFieldCount = $invalid.Count
  invalidFields = @($invalid.ToArray())
  cleanConsumerRootInsideRepository = $cleanRootInsideRepository
  canPromotePackageConsumerRuntime = $canPromote
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  boundary = "Validation never publishes and never converts local feed, ProjectReference, dependency-probe-only, build-only, sidecar, template, runbook, blocked-by-cuda-driver, managed-readiness-only, precheck-only, dry-run-only, schema-only, or CallbackAllocatorReadinessSnapshot records into package-consumer-runtime proof."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "package-consumer-runtime-proof-pack-validation.json"
$markdownPath = Join-Path $artifactRoot "package-consumer-runtime-proof-pack-validation.md"

$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$missingLines = if ($missing.Count -gt 0) { $missing | ForEach-Object { "- ``$_``" } } else { @("- none") }
$invalidLines = if ($invalid.Count -gt 0) { $invalid | ForEach-Object { "- ``$_``" } } else { @("- none") }

$markdown = @"
# Package Consumer Runtime Proof Pack Validation

生成时间：$($validation.generatedAtUtc)

## Summary

- validation state: ``$($validation.validationState)``
- missing owner input count: ``$($validation.missingOwnerInputCount)``
- invalid field count: ``$($validation.invalidFieldCount)``
- clean consumer root inside repository: ``$($validation.cleanConsumerRootInsideRepository)``
- can promote package-consumer-runtime: ``$($validation.canPromotePackageConsumerRuntime)``
- canPublishPublicly=false
- canCloseReleaseIssue=false

## Missing Owner Inputs

$($missingLines -join "`r`n")

## Invalid Fields

$($invalidLines -join "`r`n")

## Boundary

$($validation.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Package consumer runtime proof pack validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$validationState"
Write-Output "CanPromotePackageConsumerRuntime=$canPromote"
Write-Output "CanPublishPublicly=False"

if ($FailOnNotProof -and -not $canPromote) {
  throw "Package consumer runtime proof pack cannot promote runtime proof: $validationState"
}
