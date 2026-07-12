[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-release-pre-publish-audit-matrix.json",
  [string]$OutputRoot = "artifacts\final-release",
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-Array {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final release pre-publish audit matrix not found: $resolvedInputPath"
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$auditItems = ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "auditItems" -DefaultValue @())
$sourceArtifacts = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())) | ForEach-Object { [string]$_ })
$itemIds = @($auditItems | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$requiredItemIds = @(
  "release-evidence-current-bundle",
  "final-dry-run-blocked-by-owner-proof",
  "close-blocker-dashboard-current",
  "owner-runtime-smoke-field-alignment",
  "tensorrtexec-cli-winforms-parity",
  "tensorrtexec-gap-list-non-proof",
  "tensorrtexec-trtexec-parity-boundary",
  "yolovision-family-task-matrix",
  "onnxtoengine-trtexec-like-boundary",
  "docs-articles-readiness-boundary"
)
$requiredSourceArtifacts = @(
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/final-release-dry-run-summary.json",
  "artifacts/final-release/final-release-close-blocker-dashboard.json",
  "applications/TensorRtExec/tensor-rt-exec-feature-matrix.json",
  "applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json",
  "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json",
  "applications/TensorRtExec/README.md",
  "samples/YoloVision/yolo-model-matrix.json",
  "samples/YoloVision/README.md",
  "samples/OnnxToEngine/trtexec-parity-matrix.json",
  "samples/OnnxToEngine/README.md",
  "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json",
  "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json"
)
$requiredFamilies = @("yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov26", "custom")
$requiredTasks = @("det", "cls", "seg", "obb", "pose", "sem")
$families = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "yoloVisionFamilies" -DefaultValue @())) | ForEach-Object { [string]$_ })
$tasks = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "yoloVisionTasks" -DefaultValue @())) | ForEach-Object { [string]$_ })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$readmeSignals = Get-PropertyOrDefault -Object $record -Name "readmeBoundarySignals" -DefaultValue $null

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-release-pre-publish-audit-matrix") -Severity "blocker" -Detail "recordKind must be final-release-pre-publish-audit-matrix.")) | Out-Null
$items.Add((New-ValidationItem -Id "audit-state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "auditState" -DefaultValue "") -eq "blocked-final-pre-publish-owner-proof-required") -Severity "blocker" -Detail "Final pre-publish matrix must stay blocked until real owner proof and post-publish evidence exist.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-audit-items" -Passed (@($requiredItemIds | Where-Object { $itemIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Matrix must include TensorRtExec, YoloVision, OnnxToEngine, docs, dry-run, and close blocker audit items.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-source-artifacts" -Passed (@($requiredSourceArtifacts | Where-Object { $sourceArtifacts -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Matrix must name all release, TensorRtExec, YoloVision, and OnnxToEngine source artifacts.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-or-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Pre-publish audit must not publish, close, or promote proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "clean-owner-proof-remains-required" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "cleanOwnerInputReady" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "ownerInputCanPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "externalRuntimeProofReady" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "postPublishVerificationReady" -DefaultValue $true)) -Severity "blocker" -Detail "Current pre-publish state must still require owner clean consumer, external runtime, and post-publish proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-runtime-smoke-field-alignment-projected" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke-field-alignment" -and [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke-field-alignment-valid" -and [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus" -DefaultValue "") -eq "Smoke=not-requested" -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount" -DefaultValue 0) -ge 30 -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount" -DefaultValue -1) -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Pre-publish matrix must project owner runtime smoke field alignment as zero-missing non-proof field coverage.")) | Out-Null
$items.Add((New-ValidationItem -Id "tensorrtexec-non-proof" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "tensorRtExecRuntimeProofItems" -DefaultValue 1) -eq 0 -and ([string](Get-PropertyOrDefault -Object $record -Name "tensorRtExecBoundary" -DefaultValue "")).Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "TensorRtExec parity/gap outputs must not contain runtime proof items.")) | Out-Null
$items.Add((New-ValidationItem -Id "yolovision-family-task-coverage" -Passed (@($requiredFamilies | Where-Object { $families -notcontains $_ }).Count -eq 0 -and @($requiredTasks | Where-Object { $tasks -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "YoloVision matrix must cover YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom and det/cls/seg/obb/pose/sem.")) | Out-Null
$items.Add((New-ValidationItem -Id "yolovision-non-package-consumer-proof" -Passed (([string](Get-PropertyOrDefault -Object $record -Name "yoloVisionBoundary" -DefaultValue "")).Contains("not package-consumer-runtime proof", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "YoloVision matrix must not be package-consumer-runtime proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "onnxtoengine-build-boundary" -Passed (([string](Get-PropertyOrDefault -Object $record -Name "onnxToEngineBoundary" -DefaultValue "")).Contains("build/report", [StringComparison]::OrdinalIgnoreCase) -and ([string](Get-PropertyOrDefault -Object $record -Name "onnxToEngineBoundary" -DefaultValue "")).Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "OnnxToEngine parity matrix must explicitly keep build/report/preflight output outside runtime proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "readme-boundary-signals" -Passed ([bool](Get-PropertyOrDefault -Object $readmeSignals -Name "tensorRtExecMentionsBuildReportNotProof" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $readmeSignals -Name "yoloVisionMentionsRequiredFamilies" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $readmeSignals -Name "yoloVisionMentionsRequiredTasks" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $readmeSignals -Name "onnxToEngineMentionsBuildOnlyBoundary" -DefaultValue $false)) -Severity "blocker" -Detail "README boundary language must cover TensorRtExec, YoloVision, and OnnxToEngine non-proof states.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-retired-sample-live-project" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "retiredSampleLiveProjectPathPresent" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "retiredSampleLiveProjectFilePresent" -DefaultValue $true)) -Severity "blocker" -Detail "The retired detection-only sample path and project file must not reappear as a live sample identity.")) | Out-Null
$items.Add((New-ValidationItem -Id "audit-items-non-promotable" -Passed (@($auditItems | Where-Object {
  [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "isRuntimeProof" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "isPackageConsumerRuntimeProof" -DefaultValue $true)
}).Count -eq 0) -Severity "blocker" -Detail "Every audit item must remain non-promotable and non-proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary-text" -Passed ($boundary.Contains("does not publish packages", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("clean external package-consumer-runtime proof", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary text must explicitly block publish side effects and proof substitution.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "final-release-pre-publish-audit-matrix-ready" } else { "blocked-final-release-pre-publish-audit-matrix-invalid" }
$validationItems = @($items.ToArray())

$validation = [pscustomobject]@{
  recordKind = "final-release-pre-publish-audit-matrix-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  inputPath = $resolvedInputPath
  validationItemCount = $validationItems.Count
  failedBlockerCount = $failedBlockers.Count
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  performsPublish = $false
  validationItems = $validationItems
  boundary = "Validation confirms matrix shape and non-proof boundaries only; it is not runtime execution proof, public publish approval, or release close approval."
}

$jsonPath = Join-Path $OutputRoot "final-release-pre-publish-audit-matrix-validation.json"
$markdownPath = Join-Path $OutputRoot "final-release-pre-publish-audit-matrix-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $items | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# Final Release Pre-Publish Audit Matrix Validation

| Field | Value |
| --- | --- |
| validationState | ``$($validation.validationState)`` |
| validationItemCount | ``$($validation.validationItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
| --- | ---: | --- | --- |
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final release pre-publish audit matrix validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  foreach ($failure in $failedBlockers) {
    Write-Error "$($failure.id): $($failure.detail)"
  }

  exit 1
}
