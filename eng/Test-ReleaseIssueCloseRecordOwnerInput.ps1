[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-issue-close-record-owner-input.template.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)

  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Test-FileHashMatches {
  param(
    [AllowNull()][object]$Path,
    [AllowNull()][object]$Sha256
  )

  $pathText = [string]$Path
  $shaText = [string]$Sha256
  if ((Test-IsPlaceholder -Value $pathText) -or -not (Test-Sha256Format -Value $shaText)) {
    return $false
  }

  $resolvedPath = Resolve-InputPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $false
  }

  $actual = (Get-FileHash -LiteralPath $resolvedPath -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release issue close record owner input not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$postPublishState = [string](Get-PropertyOrDefault -Object $record -Name "postPublishProofValidationState" -DefaultValue "")
$postPublishStateIsReal = -not (Test-IsPlaceholder -Value $postPublishState) -and
  -not $postPublishState.Contains("template", [StringComparison]::OrdinalIgnoreCase) -and
  -not $postPublishState.Contains("draft", [StringComparison]::OrdinalIgnoreCase) -and
  -not $postPublishState.Contains("preflight", [StringComparison]::OrdinalIgnoreCase) -and
  -not $postPublishState.Contains("missing", [StringComparison]::OrdinalIgnoreCase)

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "release-issue-close-record-owner-input") -Severity "blocker" -Detail "recordKind must be release-issue-close-record-owner-input.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-close-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Owner input validation must not publish, approve publication, or close the release issue.")) | Out-Null

foreach ($pair in @(
  @("release-evidence-bundle-hash", "releaseEvidenceBundlePath", "releaseEvidenceBundleSha256", "Release evidence bundle path and SHA256 must exist and match."),
  @("release-close-preflight-hash", "releaseClosePreflightPath", "releaseClosePreflightSha256", "Release close preflight path and SHA256 must exist and match."),
  @("stale-claims-audit-hash", "staleClaimsAuditPath", "staleClaimsAuditSha256", "Stale claims audit path and SHA256 must exist and match."),
  @("post-publish-validation-hash", "postPublishProofValidationPath", "postPublishProofValidationSha256", "Post-publish proof validation path and SHA256 must exist and match.")
)) {
  $items.Add((New-ValidationItem -Id $pair[0] -Passed (Test-FileHashMatches -Path (Get-PropertyOrDefault -Object $record -Name $pair[1] -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name $pair[2] -DefaultValue "")) -Severity "action-required" -Detail $pair[3])) | Out-Null
}

$items.Add((New-ValidationItem -Id "post-publish-validation-real-proof" -Passed $postPublishStateIsReal -Severity "action-required" -Detail "postPublishProofValidationState must represent real post-publish proof, not template/draft/preflight-only guidance.")) | Out-Null

foreach ($field in @("rollbackPlan", "rollbackOwner", "rollbackTrigger", "ownerFinalCloseDecision", "ownerDecisionTimestamp", "releaseIssueId", "releaseIssueUrl")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be real non-placeholder owner input.")) | Out-Null
}

$strictCommand = [string](Get-PropertyOrDefault -Object $record -Name "strictCloseValidatorCommand" -DefaultValue "")
$items.Add((New-ValidationItem -Id "strict-close-validator-command" -Passed ($strictCommand.Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $strictCommand.Contains("-FailOnNotCloseReady", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "strictCloseValidatorCommand must use Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "owner-input-ready-for-close-candidate-overlay"
}
else {
  "blocked-owner-input-required"
}

$validation = [pscustomobject]@{
  recordKind = "release-issue-close-record-owner-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  isValidOwnerInputShape = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  postPublishProofValidationState = $postPublishState
  postPublishProofValidationIsReal = $postPublishStateIsReal
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates owner close input readiness only. It cannot close the release issue without the final strict close record validator."
}

$jsonPath = Join-Path $OutputRoot "release-issue-close-record-owner-input-validation.json"
$markdownPath = Join-Path $OutputRoot "release-issue-close-record-owner-input-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Release Issue Close Record Owner Input Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| isValidOwnerInputShape | ``$($validation.isValidOwnerInputShape)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Safety Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Release issue close record owner input validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Release issue close record owner input validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount) PerformsPublish=$($validation.performsPublish) CanPublishPublicly=$($validation.canPublishPublicly) CanCloseReleaseIssue=$($validation.canCloseReleaseIssue)"
