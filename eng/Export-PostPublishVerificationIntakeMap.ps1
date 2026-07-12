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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$requiredFields = @(
  "publicChannelUrl",
  "publishedVersion",
  "packageSource",
  "installCommand",
  "runCommand",
  "smokeLogPath",
  "smokeLogSha256",
  "stdoutSummary",
  "stderrSummary",
  "machineName",
  "osVersion",
  "gpuName",
  "nvidiaDriverVersion",
  "cudaRuntimeVersion",
  "tensorRtVersion",
  "cudnnVersion",
  "ownerReviewer",
  "reviewedAtUtc",
  "ownerDecision",
  "rollbackOrDeprecationPlanReference"
)

$intakeItems = foreach ($field in $requiredFields) {
  [pscustomobject]@{
    field = $field
    state = "owner-input-required"
    requiredFor = "post-publish-verification"
  }
}

$map = [pscustomobject]@{
  recordKind = "post-publish-verification-intake-map"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  intakeState = "blocked-owner-public-channel-verification-required"
  requiredFieldCount = @($requiredFields).Count
  publicChannelUrl = "<owner-input-required>"
  publishedVersion = "<owner-input-required>"
  packageSource = "<owner-input-required>"
  installCommand = "dotnet add package JYPPX.TensorRT.CSharp.API --version <published-version> --source <public-package-source>"
  runCommand = "dotnet run -c Release -- --runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22"
  smokeLogPath = "<owner-input-required>"
  smokeLogSha256 = "<owner-input-required>"
  stdoutSummary = "<owner-input-required>"
  stderrSummary = "<owner-input-required>"
  installMachineMetadata = [pscustomobject]@{
    machineName = "<owner-input-required>"
    osVersion = "<owner-input-required>"
    gpuName = "<owner-input-required>"
    nvidiaDriverVersion = "<owner-input-required>"
    cudaRuntimeVersion = "<owner-input-required>"
    tensorRtVersion = "<owner-input-required>"
    cudnnVersion = "<owner-input-required>"
    dotnetSdkVersion = "<owner-input-required>"
  }
  ownerReview = [pscustomobject]@{
    ownerReviewer = "<owner-input-required>"
    reviewedAtUtc = "<owner-input-required>"
    decision = "<owner-input-required>"
  }
  rollbackOrDeprecationPlanReference = "<owner-input-required>"
  validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  ownerInputArtifact = "artifacts/final-release/post-publish-verification-record.template.json"
  expectedRecord = "artifacts/final-release/post-publish-verification-record.json"
  intakeItems = @($intakeItems)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This map lists required owner inputs after real public channel publication. It is not post-publish proof and cannot close release issues until strict validation passes on real logs and hashes."
}

$jsonPath = Join-Path $OutputRoot "post-publish-verification-intake-map.json"
$markdownPath = Join-Path $OutputRoot "post-publish-verification-intake-map.md"
$map | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$fieldRows = foreach ($item in $intakeItems) {
  "| ``$(ConvertTo-MarkdownCell $item.field)`` | ``$(ConvertTo-MarkdownCell $item.state)`` | ``$(ConvertTo-MarkdownCell $item.requiredFor)`` |"
}

$markdown = @"
# Post Publish Verification Intake Map

Generated at: ``$($map.generatedAtUtc)``

## Summary

- intakeState: ``$($map.intakeState)``
- requiredFieldCount: ``$($map.requiredFieldCount)``
- validatorCommand: ``$($map.validatorCommand)``
- isPostPublishProof: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Required Fields

| Field | State | Required For |
| --- | --- | --- |
$($fieldRows -join "`r`n")

## Boundary

$($map.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Post-publish verification intake map written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "IntakeState=$($map.intakeState) RequiredFieldCount=$($map.requiredFieldCount)"
