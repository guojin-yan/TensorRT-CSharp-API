[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-candidate-final-freeze-manifest.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

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

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release candidate final freeze manifest not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$artifacts = @((Get-PropertyOrDefault -Object $record -Name "freezeArtifacts" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-candidate-final-freeze-manifest") -Severity "blocker" -Detail "recordKind must be release-candidate-final-freeze-manifest.")) | Out-Null
$items.Add((New-ValidationItem -Id "freeze-state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "freezeState" -DefaultValue "") -eq "blocked-owner-public-publish-required") -Severity "blocker" -Detail "Freeze manifest must remain blocked until owner public publish happens externally.")) | Out-Null
$items.Add((New-ValidationItem -Id "artifact-count" -Passed ($artifacts.Count -ge 8) -Severity "blocker" -Detail "Freeze manifest must include release docs, evidence bundle, final audits, and targeted test artifacts.")) | Out-Null
$items.Add((New-ValidationItem -Id "classification-clean" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "classificationAuditFindingCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Classification audit must remain clean before freezing release-facing evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-proof-state" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "finalOwnerDecisionBlockedGateCount" -DefaultValue 0) -gt 0 -and [int](Get-PropertyOrDefault -Object $record -Name "finalPostPublishBlockedLaneCount" -DefaultValue 0) -gt 0) -Severity "action-required" -Detail "Freeze manifest must continue to show owner/post-publish blockers until real proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Freeze manifest must not publish, prove runtime, prove post-publish, or close release issue.")) | Out-Null

foreach ($artifact in $artifacts) {
  $relativePath = [string](Get-PropertyOrDefault -Object $artifact -Name "relativePath" -DefaultValue "")
  $exists = [bool](Get-PropertyOrDefault -Object $artifact -Name "exists" -DefaultValue $false)
  $sha = [string](Get-PropertyOrDefault -Object $artifact -Name "sha256" -DefaultValue "")
  $boundary = [string](Get-PropertyOrDefault -Object $artifact -Name "boundary" -DefaultValue "")
  $items.Add((New-ValidationItem -Id "artifact-$relativePath" -Passed ($exists -and $sha.Length -eq 64 -and $boundary.Contains("not", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Freeze artifact $relativePath must exist, have SHA256, and carry a non-proof boundary.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-release-candidate-final-freeze-manifest"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-owner-public-publish-required"
}
else {
  "release-candidate-final-freeze-manifest-ready-for-owner-handoff"
}

$validation = [pscustomobject]@{
  recordKind = "release-candidate-final-freeze-manifest-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  artifactCount = $artifacts.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Release candidate final freeze manifest validation only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$jsonPath = Join-Path $OutputRoot "release-candidate-final-freeze-manifest-validation.json"
$markdownPath = Join-Path $OutputRoot "release-candidate-final-freeze-manifest-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Release Candidate Final Freeze Manifest Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| artifactCount | ``$($validation.artifactCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate final freeze manifest validation written to $jsonPath"
Write-Host "ValidationState=$validationState Artifacts=$($validation.artifactCount) FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Release candidate final freeze manifest has blocker validation failures."
}
