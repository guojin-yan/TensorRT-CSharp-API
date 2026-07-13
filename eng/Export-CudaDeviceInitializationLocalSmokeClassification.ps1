[CmdletBinding()]
param(
  [string]$OutputDirectory,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\final-release"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$sourceSmokeRunner = "smoke/CudaDeviceInitializationProofRunner/Program.cs"
$sourceProject = "smoke/CudaDeviceInitializationProofRunner/CudaDeviceInitializationProofRunner.csproj"
$sourcePath = Join-Path $RepositoryRoot ($sourceSmokeRunner -replace "/", "\")
if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
  throw "CUDA device initialization proof runner source not found: $sourcePath"
}

$source = Get-Content -LiteralPath $sourcePath -Raw -Encoding utf8
$setValidDevicesIndex = $source.IndexOf("CudaDevice.SetValidDevices(new[] { deviceOrdinal })", [StringComparison]::Ordinal)
$initDeviceIndex = $source.IndexOf("CudaDevice.InitDevice(deviceOrdinal, CudaDeviceRuntimeFlags.ScheduleAuto)", [StringComparison]::Ordinal)
$chooseDeviceIndex = $source.IndexOf("CudaDevice.ChooseDevice(requirements)", [StringComparison]::Ordinal)

$record = [pscustomobject]@{
  recordKind = "cuda-device-initialization-local-smoke-classification"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  classificationState = "cuda-device-initialization-local-smoke-classified-non-proof"
  proofKind = "local-smoke-not-external-proof"
  evidenceKind = "cuda-device-initialization-local-smoke-classification"
  runtimeEvidenceKind = "local-smoke"
  sourceSmokeRunner = $sourceSmokeRunner
  sourceProject = $sourceProject
  sourceContainsProofKindMarker = $source.Contains("ProofKind=local-smoke-not-external-proof", [StringComparison]::Ordinal)
  sourceContainsPackageConsumerFalseMarker = $source.Contains("IsPackageConsumerRuntimeProof=False", [StringComparison]::Ordinal)
  sourceContainsPromotionFalseMarker = $source.Contains("CanPromoteRuntimeProof=False", [StringComparison]::Ordinal)
  preInitCallOrderReady = ($setValidDevicesIndex -ge 0 -and $initDeviceIndex -gt $setValidDevicesIndex -and $chooseDeviceIndex -gt $initDeviceIndex)
  isPackageConsumerRuntimeProof = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  skippedTrueIsForbiddenSubstitute = $true
  skippedCanPromoteRuntimeProof = $false
  successCanPromoteWithoutStrictExternalValidator = $false
  forbiddenSubstitutes = @(
    "Skipped=True",
    "local-smoke",
    "ProjectReference",
    "direct nupkg",
    "local feed",
    "build-only",
    "dependency probe",
    "blocked-by-cuda-driver"
  )
  promotionRequires = @(
    "clean consumer project outside repository or owner-designated proof root",
    "no ProjectReference",
    "managed/runtime packages restored from package source",
    "native asset listing and SHA256",
    "runtime smoke log SHA256",
    "host metadata including OS/GPU/driver/CUDA/TensorRT/cuDNN",
    "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof passes"
  )
  boundary = "CudaDeviceInitializationProofRunner is local smoke classification only: not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. Skipped=True is a forbidden substitute and cannot promote package-consumer-runtime proof."
}

$jsonPath = Join-Path $OutputDirectory "cuda-device-initialization-local-smoke-classification.json"
$markdownPath = Join-Path $OutputDirectory "cuda-device-initialization-local-smoke-classification.md"
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# CUDA Device Initialization Local Smoke Classification

| Field | Value |
|---|---|
| classificationState | ``$($record.classificationState)`` |
| proofKind | ``$($record.proofKind)`` |
| isPackageConsumerRuntimeProof | ``$($record.isPackageConsumerRuntimeProof)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| preInitCallOrderReady | ``$($record.preInitCallOrderReady)`` |
| sourceSmokeRunner | ``$($record.sourceSmokeRunner)`` |

## Boundary

$($record.boundary)

## Promotion Requirements

$($record.promotionRequires | ForEach-Object { "- $_" } | Out-String)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "CUDA device initialization local smoke classification written: $jsonPath"
Write-Host "CUDA device initialization local smoke classification markdown written: $markdownPath"
Write-Host "ClassificationState=$($record.classificationState) ProofKind=$($record.proofKind) CanPromoteRuntimeProof=$($record.canPromoteRuntimeProof)"
