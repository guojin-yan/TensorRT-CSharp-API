[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$requiredOwnerFields = @(
  "caseId",
  "modelSource",
  "modelLicense",
  "onnxSha256",
  "engineSha256",
  "inputArtifactSha256",
  "outputArtifactSha256",
  "commandLine",
  "stdoutLogPath",
  "stderrLogPath",
  "screenshotPath",
  "hostOs",
  "gpuName",
  "nvidiaDriverVersion",
  "cudaVersion",
  "tensorRtVersion",
  "runtimePackageId",
  "runtimePackageVersion",
  "runtimePackageSource",
  "ownerReviewedBy",
  "ownerReviewedAtUtc"
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "real-case-evidence-record-template"
  templateState = "owner-action-required"
  templateOnly = $true
  proofClassification = "template-not-proof"
  performsRuntimeExecution = $false
  performsPublish = $false
  canPromoteRealModelRuntime = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  requiredOwnerFields = $requiredOwnerFields
  caseId = "owner-action-required"
  caseName = "owner-action-required"
  sampleProject = "owner-action-required"
  articleIds = @()
  modelSource = "owner-action-required"
  modelLicense = "owner-action-required"
  onnxPath = "owner-action-required"
  enginePath = "owner-action-required"
  inputArtifactPath = "owner-action-required"
  outputArtifactPath = "owner-action-required"
  labelsPath = "owner-action-required"
  screenshotPath = "owner-action-required"
  stdoutLogPath = "owner-action-required"
  stderrLogPath = "owner-action-required"
  reportPath = "owner-action-required"
  sidecarPath = "owner-action-required"
  onnxSha256 = "owner-action-required"
  engineSha256 = "owner-action-required"
  inputArtifactSha256 = "owner-action-required"
  outputArtifactSha256 = "owner-action-required"
  labelsSha256 = "owner-action-required"
  screenshotSha256 = "owner-action-required"
  stdoutLogSha256 = "owner-action-required"
  stderrLogSha256 = "owner-action-required"
  reportSha256 = "owner-action-required"
  sidecarSha256 = "owner-action-required"
  commandLine = "owner-action-required"
  stdoutSummary = "owner-action-required"
  stderrSummary = "owner-action-required"
  expectedEvidenceLines = @("owner-action-required")
  hostOs = "owner-action-required"
  gpuName = "owner-action-required"
  nvidiaDriverVersion = "owner-action-required"
  cudaVersion = "owner-action-required"
  tensorRtVersion = "owner-action-required"
  cudnnVersion = "owner-action-required-if-applicable"
  runtimePackageId = "owner-action-required"
  runtimePackageVersion = "owner-action-required"
  runtimePackageSource = "owner-action-required"
  ownerReviewedBy = "owner-action-required"
  ownerReviewedAtUtc = "owner-action-required"
  ownerDecision = "owner-action-required"
  nonSubstituteProofKinds = @("template", "draft", "runbook", "article planning", "build-only", "sidecar-only", "local feed", "ProjectReference", "dependency-probe-only", "blocked-by-cuda-driver")
  boundary = "This template is not proof. Empty templates, drafts, runbooks, sidecars, build-only reports, screenshots, and article plans cannot promote real-model-runtime or release publication."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "real-case-evidence-record-template.json"
$markdownPath = Join-Path $artifactRoot "real-case-evidence-record-template.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$fieldLines = $requiredOwnerFields | ForEach-Object { "- ``$_``" }
$markdown = @"
# Real Case Evidence Record Template

生成时间：$($record.generatedAtUtc)

## Summary

- record kind: ``$($record.recordKind)``
- template state: ``$($record.templateState)``
- template only: ``True``
- proof classification: ``$($record.proofClassification)``
- can promote real-model-runtime: ``False``
- canPublishPublicly=false
- canCloseReleaseIssue=false

## Required Owner Fields

$($fieldLines -join "`r`n")

## Boundary

$($record.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Real case evidence record template written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "TemplateOnly=True"
Write-Output "CanPromoteRealModelRuntime=False"
