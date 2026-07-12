[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-ChecklistArtifact {
  param(
    [object]$Checklist,
    [string]$JsonFileName,
    [string]$MarkdownFileName
  )

  $jsonPath = Join-Path $OutputRoot $JsonFileName
  $markdownPath = Join-Path $OutputRoot $MarkdownFileName
  $Checklist | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

  $fieldRows = foreach ($field in $Checklist.requiredFields) {
    "| ``$($field.name)`` | ``$($field.requiredEvidence)`` | ``$($field.boundary)`` |"
  }
  $forbiddenRows = $Checklist.forbiddenSubstitutes | ForEach-Object { "- ``$_``" }

  $markdown = @"
# $($Checklist.title)

Generated at: ``$($Checklist.generatedAtUtc)``

## Summary

- recordKind: ``$($Checklist.recordKind)``
- checklistState: ``$($Checklist.checklistState)``
- proofClassification: ``$($Checklist.proofClassification)``
- performsPublish: ``False``
- canPromotePackageConsumerRuntime: ``False``
- canCloseReleaseIssue: ``False``

## Required Fields

| Field | Required Evidence | Boundary |
| --- | --- | --- |
$($fieldRows -join "`r`n")

## Forbidden Substitutes

$($forbiddenRows -join "`r`n")

## Boundary

$($Checklist.boundary)
"@

  Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8
}

$generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
$forbidden = @("parse/report-only", "build-only", "dry-run", "sidecar-only report", "template", "local feed", "ProjectReference", "direct .nupkg", "package-consumer-runtime substitute")

$timingCacheChecklist = [ordered]@{
  recordKind = "tensorrtexec-timing-cache-owner-proof-checklist-template"
  title = "TensorRtExec Timing Cache Owner Proof Checklist Template"
  generatedAtUtc = $generatedAtUtc
  checklistState = "template-owner-input-required"
  proofClassification = "parse/report-only"
  performsPublish = $false
  canPromotePackageConsumerRuntime = $false
  canCloseReleaseIssue = $false
  requiredFields = @(
    [pscustomobject]@{ name = "timingCacheInputPath"; requiredEvidence = "owner real path when importing cache"; boundary = "path alone is not runtime proof" }
    [pscustomobject]@{ name = "timingCacheInputSha256"; requiredEvidence = "cache content hash"; boundary = "hash supports audit but not inference correctness" }
    [pscustomobject]@{ name = "timingCacheOutputPath"; requiredEvidence = "owner real path when exporting cache"; boundary = "export path alone is not proof" }
    [pscustomobject]@{ name = "timingCacheOutputSha256"; requiredEvidence = "cache content hash"; boundary = "must be paired with build log" }
    [pscustomobject]@{ name = "nativeImportExportSmokeLogPath"; requiredEvidence = "native import/export smoke log"; boundary = "native import/export smoke is required before promotion" }
    [pscustomobject]@{ name = "tensorRtVersion"; requiredEvidence = "TensorRT version metadata"; boundary = "metadata cannot replace runtime proof" }
    [pscustomobject]@{ name = "cudaVersion"; requiredEvidence = "CUDA version metadata"; boundary = "metadata cannot replace runtime proof" }
    [pscustomobject]@{ name = "gpuName"; requiredEvidence = "GPU name"; boundary = "metadata cannot replace runtime proof" }
    [pscustomobject]@{ name = "driverVersion"; requiredEvidence = "driver version"; boundary = "metadata cannot replace runtime proof" }
  )
  forbiddenSubstitutes = $forbidden
  boundary = "--timingCacheFile and --exportTimingCache remain parse/report-only until cache content hash, native import/export smoke, TensorRT/CUDA/GPU/driver metadata, build logs, and model runtime validation are provided; this checklist is not package-consumer-runtime proof."
}

$int8Checklist = [ordered]@{
  recordKind = "tensorrtexec-int8-calibration-owner-proof-checklist-template"
  title = "TensorRtExec INT8 Calibration Owner Proof Checklist Template"
  generatedAtUtc = $generatedAtUtc
  checklistState = "template-owner-input-required"
  proofClassification = "parse/report-only"
  performsPublish = $false
  canPromotePackageConsumerRuntime = $false
  canCloseReleaseIssue = $false
  requiredFields = @(
    [pscustomobject]@{ name = "int8Enabled"; requiredEvidence = "--int8 command and report field"; boundary = "INT8 flag alone is not proof" }
    [pscustomobject]@{ name = "calibrationCachePath"; requiredEvidence = "--calib cache path"; boundary = "path alone is not proof" }
    [pscustomobject]@{ name = "calibrationCacheSha256"; requiredEvidence = "cache content hash"; boundary = "hash supports audit but not accuracy proof" }
    [pscustomobject]@{ name = "calibrationDatasetProvenance"; requiredEvidence = "dataset source, version, sample count, license"; boundary = "dataset provenance is mandatory before promotion" }
    [pscustomobject]@{ name = "calibratorOwnership"; requiredEvidence = "owner-provided-cache or native-calibrator-run"; boundary = "calibrator ownership must be explicit" }
    [pscustomobject]@{ name = "callbackOwnership"; requiredEvidence = "not-used, managed-callback-owned, or native-owned"; boundary = "callback ownership must be explicit before callback paths promote" }
    [pscustomobject]@{ name = "modelSpecificInt8AccuracyEvidence"; requiredEvidence = "FP32/FP16/INT8 comparison log"; boundary = "model-specific INT8 accuracy evidence is required" }
    [pscustomobject]@{ name = "tensorRtVersion"; requiredEvidence = "TensorRT version metadata"; boundary = "metadata cannot replace runtime proof" }
    [pscustomobject]@{ name = "cudaVersion"; requiredEvidence = "CUDA version metadata"; boundary = "metadata cannot replace runtime proof" }
  )
  forbiddenSubstitutes = $forbidden
  boundary = "--int8 and --calib remain parse/report-only until calibration dataset provenance, calibrator ownership, callback ownership, cache hash, native smoke, and model-specific INT8 accuracy evidence are provided; this checklist is not package-consumer-runtime proof."
}

$combined = [ordered]@{
  recordKind = "tensorrtexec-advanced-proof-readiness-checklist"
  generatedAtUtc = $generatedAtUtc
  checklistState = "template-owner-input-required"
  performsPublish = $false
  canPromotePackageConsumerRuntime = $false
  canCloseReleaseIssue = $false
  checklists = @($timingCacheChecklist, $int8Checklist)
  boundary = "TensorRtExec advanced readiness checklists collect owner fields only. They do not execute native risky ownership paths and do not replace package-consumer-runtime proof."
}

Write-ChecklistArtifact -Checklist $timingCacheChecklist -JsonFileName "tensorrtexec-timing-cache-owner-proof-checklist.template.json" -MarkdownFileName "tensorrtexec-timing-cache-owner-proof-checklist.template.md"
Write-ChecklistArtifact -Checklist $int8Checklist -JsonFileName "tensorrtexec-int8-calibration-owner-proof-checklist.template.json" -MarkdownFileName "tensorrtexec-int8-calibration-owner-proof-checklist.template.md"

$combinedJsonPath = Join-Path $OutputRoot "tensorrtexec-advanced-proof-readiness-checklist.json"
$combinedMarkdownPath = Join-Path $OutputRoot "tensorrtexec-advanced-proof-readiness-checklist.md"
$combined | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $combinedJsonPath -Encoding utf8

$markdown = @"
# TensorRtExec Advanced Proof Readiness Checklist

Generated at: ``$($combined.generatedAtUtc)``

## Summary

- recordKind: ``$($combined.recordKind)``
- checklistState: ``$($combined.checklistState)``
- performsPublish: ``False``
- canPromotePackageConsumerRuntime: ``False``
- canCloseReleaseIssue: ``False``

## Checklists

- ``tensorrtexec-timing-cache-owner-proof-checklist.template.json``
- ``tensorrtexec-int8-calibration-owner-proof-checklist.template.json``

## Boundary

$($combined.boundary)
"@

Set-Content -LiteralPath $combinedMarkdownPath -Value $markdown -Encoding utf8

Write-Output "TensorRtExec advanced proof readiness checklist written:"
Write-Output "  Json=$combinedJsonPath"
Write-Output "  Markdown=$combinedMarkdownPath"
Write-Output "ChecklistState=$($combined.checklistState) ChecklistCount=$($combined.checklists.Count)"
