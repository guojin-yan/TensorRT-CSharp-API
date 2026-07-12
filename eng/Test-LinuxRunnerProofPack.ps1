[CmdletBinding()]
param(
  [string]$PackPath = ".\artifacts\final-release\linux-runner-proof-execution-pack.json",
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
  throw "Missing Linux runner proof pack: $PackPath"
}

function Get-Value {
  param([object]$Object, [string]$Name)
  if ($null -eq $Object -or $Object.PSObject.Properties.Name -notcontains $Name) {
    return $null
  }

  return $Object.PSObject.Properties[$Name].Value
}

function Test-OwnerValue {
  param([object]$Value)
  if ($null -eq $Value) { return $false }
  $text = ([string]$Value).Trim()
  return -not [string]::IsNullOrWhiteSpace($text) -and $text -notmatch "owner-action-required|template|TODO|TBD"
}

function Test-Sha256Text {
  param([object]$Value)
  if ($null -eq $Value) { return $false }
  return ([string]$Value) -match "^[0-9A-Fa-f]{64}$"
}

$pack = Get-Content -LiteralPath $PackPath -Raw -Encoding utf8 | ConvertFrom-Json
$missing = New-Object System.Collections.Generic.List[string]
$invalid = New-Object System.Collections.Generic.List[string]

foreach ($field in @("linuxRuntimePackageKey", "targetDistro", "targetArch", "runtimePackageSha256", "linuxRunnerCommand", "linuxRunnerLogPath", "linuxRunnerLogSha256", "hostOs", "kernelVersion", "gpuName", "nvidiaDriverVersion", "cudaVersion", "tensorRtVersion", "validatorCommand")) {
  if (-not (Test-OwnerValue (Get-Value -Object $pack -Name $field))) {
    $missing.Add($field)
  }
}

foreach ($field in @("runtimePackageSha256", "linuxRunnerLogSha256")) {
  if (-not (Test-Sha256Text (Get-Value -Object $pack -Name $field))) {
    $invalid.Add($field)
  }
}

$canPromote = $missing.Count -eq 0 -and $invalid.Count -eq 0
$validationState = if ($canPromote) { "validated-linux-runner-proof-candidate" } else { "blocked-owner-action-required" }

$validation = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "linux-runner-proof-pack-validation"
  sourcePackPath = $PackPath
  validationState = $validationState
  missingOwnerInputCount = $missing.Count
  missingOwnerInputs = @($missing.ToArray())
  invalidFieldCount = $invalid.Count
  invalidFields = @($invalid.ToArray())
  canPromoteLinuxRunnerProof = $canPromote
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  boundary = "Validation never publishes and never converts Windows dry-run, WSL without real GPU/CUDA/TensorRT proof, templates, runbooks, or blocked-by-cuda-driver records into Linux runner proof."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "linux-runner-proof-pack-validation.json"
$markdownPath = Join-Path $artifactRoot "linux-runner-proof-pack-validation.md"

$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$missingLines = if ($missing.Count -gt 0) { $missing | ForEach-Object { "- ``$_``" } } else { @("- none") }
$invalidLines = if ($invalid.Count -gt 0) { $invalid | ForEach-Object { "- ``$_``" } } else { @("- none") }

$markdown = @"
# Linux Runner Proof Pack Validation

生成时间：$($validation.generatedAtUtc)

## Summary

- validation state: ``$($validation.validationState)``
- missing owner input count: ``$($validation.missingOwnerInputCount)``
- invalid field count: ``$($validation.invalidFieldCount)``
- can promote Linux runner proof: ``$($validation.canPromoteLinuxRunnerProof)``
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

Write-Output "Linux runner proof pack validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$validationState"
Write-Output "CanPromoteLinuxRunnerProof=$canPromote"
Write-Output "CanPublishPublicly=False"

if ($FailOnNotProof -and -not $canPromote) {
  throw "Linux runner proof pack cannot promote Linux runner proof: $validationState"
}
