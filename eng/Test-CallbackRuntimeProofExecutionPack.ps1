[CmdletBinding()]
param(
  [string]$PackPath = ".\artifacts\final-release\callback-runtime-proof-execution-pack.json",
  [switch]$FailOnPromotable,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$packFullPath = if ([System.IO.Path]::IsPathRooted($PackPath)) { $PackPath } else { Join-Path $RepositoryRoot $PackPath }
if (-not (Test-Path -LiteralPath $packFullPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Export-CallbackRuntimeProofExecutionPack.ps1") -RuntimePackageKey "win-x64-trt11.0-cuda13.2-cudnn9.22" -RepositoryRoot $RepositoryRoot | Out-Null
}

$pack = Get-Content -LiteralPath $packFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$requiredNonSubstitutes = @(
  "managed-readiness",
  "TensorRtCallbackAllocatorReadinessSnapshot",
  "precheck-only",
  "dry-run-only",
  "schema-only"
)

$nonSubstitutes = @($pack.nonSubstituteProofKinds)
$missingRules = @($requiredNonSubstitutes | Where-Object { $nonSubstitutes -notcontains $_ })
$canPromote = [bool]$pack.canPromoteRealCallbackRuntime -and [bool]$pack.isRealCallbackRuntimeProof

$validation = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "callback-runtime-proof-execution-pack-validation"
  sourcePackPath = $packFullPath
  validationState = if ($canPromote -and $missingRules.Count -eq 0) { "validated-real-callback-runtime-candidate" } else { "blocked-owner-action-required" }
  canPromoteRealCallbackRuntime = $canPromote
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  missingOwnerInputCount = [int]$pack.missingOwnerInputCount
  missingNonSubstituteRuleCount = $missingRules.Count
  missingNonSubstituteRules = @($missingRules)
  boundary = "Validation never converts managed-readiness, CallbackAllocatorReadinessSnapshot, precheck-only, dry-run-only, schema-only, blocked-by-cuda-driver, dependency-probe-only, build-only, template, or runbook records into real callback runtime proof."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "callback-runtime-proof-execution-pack-validation.json"
$markdownPath = Join-Path $artifactRoot "callback-runtime-proof-execution-pack-validation.md"

$validation | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$missingLines = $validation.missingNonSubstituteRules | ForEach-Object { "- ``$_``" }
$markdown = @"
# Callback Runtime Proof Execution Pack Validation

- validation state: ``$($validation.validationState)``
- can promote real callback runtime: ``$($validation.canPromoteRealCallbackRuntime)``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- missing owner input count: ``$($validation.missingOwnerInputCount)``

## Missing Non-Substitute Rules

$($missingLines -join "`r`n")

## Boundary

$($validation.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Callback runtime proof execution pack validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$($validation.validationState)"
Write-Output "CanPromoteRealCallbackRuntime=$($validation.canPromoteRealCallbackRuntime)"

if ($FailOnPromotable -and $canPromote) {
  throw "Callback runtime proof execution pack unexpectedly became promotable."
}
