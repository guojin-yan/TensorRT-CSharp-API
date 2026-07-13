[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($ArtifactDirectory)) {
  $ArtifactDirectory = Join-Path $RepositoryRoot "artifacts\final-release"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

New-Item -ItemType Directory -Path $ArtifactDirectory -Force | Out-Null

$recordPath = Join-Path $ArtifactDirectory "cuda-device-initialization-local-smoke-classification.json"
if (-not (Test-Path -LiteralPath $recordPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-CudaDeviceInitializationLocalSmokeClassification.ps1") -OutputDirectory $ArtifactDirectory -RepositoryRoot $RepositoryRoot
}

$record = Get-Content -LiteralPath $recordPath -Raw -Encoding utf8 | ConvertFrom-Json
$findings = New-Object System.Collections.Generic.List[object]

function Add-Finding {
  param([string]$Id, [string]$Message)
  $script:findings.Add([pscustomobject]@{
      id = $Id
      severity = "blocker"
      message = $Message
    })
}

function Assert-FalseField {
  param([string]$Name)
  if ([bool]$record.$Name) {
    Add-Finding $Name "$Name must remain false."
  }
}

if ([string]$record.recordKind -ne "cuda-device-initialization-local-smoke-classification") {
  Add-Finding "record-kind" "recordKind must be cuda-device-initialization-local-smoke-classification."
}

if ([string]$record.proofKind -ne "local-smoke-not-external-proof") {
  Add-Finding "proof-kind" "ProofKind must remain local-smoke-not-external-proof."
}

foreach ($field in @(
    "isPackageConsumerRuntimeProof",
    "canPromoteRuntimeProof",
    "isRuntimeExecutionProof",
    "isPostPublishProof",
    "isReleaseCloseProof",
    "performsPublish",
    "canPublishPublicly",
    "canCloseReleaseIssue",
    "skippedCanPromoteRuntimeProof",
    "successCanPromoteWithoutStrictExternalValidator"
  )) {
  Assert-FalseField $field
}

foreach ($field in @(
    "sourceContainsProofKindMarker",
    "sourceContainsPackageConsumerFalseMarker",
    "sourceContainsPromotionFalseMarker",
    "preInitCallOrderReady",
    "skippedTrueIsForbiddenSubstitute"
  )) {
  if (-not [bool]$record.$field) {
    Add-Finding $field "$field must be true."
  }
}

if ([string]$record.sourceSmokeRunner -ne "smoke/CudaDeviceInitializationProofRunner/Program.cs") {
  Add-Finding "source-smoke-runner" "sourceSmokeRunner must point to CudaDeviceInitializationProofRunner/Program.cs."
}

$combined = ($record | ConvertTo-Json -Depth 8)
foreach ($marker in @("Skipped=True", "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof", "not runtime proof", "not post-publish proof", "not publish approval", "not release close approval", "not package push")) {
  if ($combined.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
    Add-Finding "missing-marker" "Record is missing marker: $marker"
  }
}

$validationState = if ($findings.Count -eq 0) { "cuda-device-initialization-local-smoke-classification-validation-passed-non-proof" } else { "cuda-device-initialization-local-smoke-classification-validation-blocked" }
$validation = [pscustomobject]@{
  recordKind = "cuda-device-initialization-local-smoke-classification-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  sourceRecord = "artifacts/final-release/cuda-device-initialization-local-smoke-classification.json"
  findingCount = $findings.Count
  canPromoteRuntimeProof = $false
  isPackageConsumerRuntimeProof = $false
  isRuntimeExecutionProof = $false
  performsPublish = $false
  findings = @($findings.ToArray())
  boundary = "Validation confirms local CUDA initialization smoke classification only: not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$validationJsonPath = Join-Path $ArtifactDirectory "cuda-device-initialization-local-smoke-classification-validation.json"
$validationMarkdownPath = Join-Path $ArtifactDirectory "cuda-device-initialization-local-smoke-classification-validation.md"
$validation | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $validationJsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# CUDA Device Initialization Local Smoke Classification Validation")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| validationState | ``$validationState`` |")
$lines.Add("| findingCount | ``$($findings.Count)`` |")
$lines.Add("| isPackageConsumerRuntimeProof | ``$($validation.isPackageConsumerRuntimeProof)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |")
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($validation.boundary)
if ($findings.Count -gt 0) {
  $lines.Add("")
  $lines.Add("## Findings")
  foreach ($finding in $findings) {
    $lines.Add("- ``$($finding.id)`` $($finding.message)")
  }
}

$lines | Set-Content -LiteralPath $validationMarkdownPath -Encoding utf8

Write-Host "CUDA device initialization local smoke classification validation written: $validationJsonPath"
Write-Host "ValidationState=$validationState FindingCount=$($findings.Count)"

if ($Strict -and $findings.Count -ne 0) {
  throw "CUDA device initialization local smoke classification validation failed with $($findings.Count) finding(s)."
}
