[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-issue-close-record-candidate.json",
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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
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

function Test-RelativeFileHash {
  param(
    [AllowNull()][object]$RelativePath,
    [AllowNull()][object]$Sha256
  )

  $relativeText = [string]$RelativePath
  $shaText = [string]$Sha256
  if ((Test-IsPlaceholder -Value $relativeText) -or -not (Test-Sha256Format -Value $shaText)) {
    return $false
  }

  $path = if ([System.IO.Path]::IsPathRooted($relativeText)) {
    $relativeText
  }
  else {
    Join-Path $RepositoryRoot $relativeText
  }

  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $false
  }

  $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
}

$resolvedInputPath = if ([System.IO.Path]::IsPathRooted($InputPath)) {
  $InputPath
}
else {
  Join-Path $RepositoryRoot $InputPath
}

if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release issue close record candidate not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$candidateState = [string](Get-PropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "")
$proofLineId = [string](Get-PropertyOrDefault -Object $record -Name "proofLineId" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$rules = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "requiredRealInputRules" -DefaultValue @())

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "release-issue-close-record-candidate") -Severity "blocker" -Detail "recordKind must be release-issue-close-record-candidate.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-line-id" -Passed ($proofLineId -eq "release-issue-close-record") -Severity "blocker" -Detail "proofLineId must be release-issue-close-record.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-state" -Passed ($candidateState -eq "blocked-release-close-real-proof-required") -Severity "blocker" -Detail "Candidate must remain blocked until real close proof and owner decision are available.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-close-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Candidate must not publish, approve publication, or close the release issue.")) | Out-Null

foreach ($requiredRule in @(
  "releaseEvidenceBundleSha256",
  "releaseClosePreflightPathAndHash",
  "staleClaimsAuditPathAndHash",
  "postPublishProofValidationPathAndHash",
  "rollbackPlan",
  "ownerFinalCloseDecision",
  "strictCloseValidatorCommand"
)) {
  $items.Add((New-ValidationItem -Id "rule-$requiredRule" -Passed ($rules -contains $requiredRule) -Severity "blocker" -Detail "Candidate must contain rule '$requiredRule'.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "release-evidence-bundle-hash" -Passed (Test-RelativeFileHash -RelativePath (Get-PropertyOrDefault -Object $record -Name "releaseEvidenceBundlePath" -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name "releaseEvidenceBundleSha256" -DefaultValue "")) -Severity "blocker" -Detail "Release evidence bundle path and SHA256 must exist and match.")) | Out-Null
$items.Add((New-ValidationItem -Id "release-close-preflight-hash" -Passed (Test-RelativeFileHash -RelativePath (Get-PropertyOrDefault -Object $record -Name "releaseClosePreflightPath" -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name "releaseClosePreflightSha256" -DefaultValue "")) -Severity "action-required" -Detail "Release close preflight path and SHA256 must exist and match.")) | Out-Null
$items.Add((New-ValidationItem -Id "stale-claims-audit-hash" -Passed (Test-RelativeFileHash -RelativePath (Get-PropertyOrDefault -Object $record -Name "staleClaimsAuditPath" -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name "staleClaimsAuditSha256" -DefaultValue "")) -Severity "action-required" -Detail "Stale claims audit path and SHA256 must exist and match.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-validation-hash" -Passed (Test-RelativeFileHash -RelativePath (Get-PropertyOrDefault -Object $record -Name "postPublishProofValidationPath" -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name "postPublishProofValidationSha256" -DefaultValue "")) -Severity "action-required" -Detail "Post-publish proof validation path and SHA256 must exist and match.")) | Out-Null

$postPublishState = [string](Get-PropertyOrDefault -Object $record -Name "postPublishProofValidationState" -DefaultValue "")
$postPublishStateIsReal = -not (Test-IsPlaceholder -Value $postPublishState) -and
  -not $postPublishState.Contains("template", [StringComparison]::OrdinalIgnoreCase) -and
  -not $postPublishState.Contains("draft", [StringComparison]::OrdinalIgnoreCase) -and
  -not $postPublishState.Contains("preflight", [StringComparison]::OrdinalIgnoreCase) -and
  -not $postPublishState.Contains("missing", [StringComparison]::OrdinalIgnoreCase)
$items.Add((New-ValidationItem -Id "post-publish-validation-real-proof" -Passed $postPublishStateIsReal -Severity "action-required" -Detail "Post-publish validation must represent real post-publish proof, not template/draft/preflight-only guidance.")) | Out-Null

$rollbackPlanReady = -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "rollbackPlan" -DefaultValue "")) -and
  -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "rollbackOwner" -DefaultValue "")) -and
  -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "rollbackTrigger" -DefaultValue ""))
$ownerDecisionReady = -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "ownerFinalCloseDecision" -DefaultValue "")) -and
  -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "ownerDecisionTimestamp" -DefaultValue ""))
$strictCommand = [string](Get-PropertyOrDefault -Object $record -Name "strictCloseValidatorCommand" -DefaultValue "")

$items.Add((New-ValidationItem -Id "rollback-plan-ready" -Passed $rollbackPlanReady -Severity "action-required" -Detail "Rollback plan, owner, and trigger must be filled with real values.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-final-close-decision-ready" -Passed $ownerDecisionReady -Severity "action-required" -Detail "Owner final close decision and timestamp must be real non-placeholder values.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-close-validator-command" -Passed ($strictCommand.Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $strictCommand.Contains("-FailOnNotCloseReady", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "Strict close validator command must use Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "candidate-ready-for-close-proof-review"
}
else {
  "blocked-release-close-real-proof-required"
}

$validation = [pscustomobject]@{
  recordKind = "release-issue-close-record-candidate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  isValidCandidate = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  canCloseReleaseIssue = $false
  performsPublish = $false
  canPublishPublicly = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validation reviews close candidate input readiness only. It cannot close the release issue without real post-publish proof, rollback plan, and owner final close decision."
}

$jsonPath = Join-Path $OutputRoot "release-issue-close-record-candidate-validation.json"
$markdownPath = Join-Path $OutputRoot "release-issue-close-record-candidate-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Release Issue Close Record Candidate Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| isValidCandidate | ``$($validation.isValidCandidate)`` |
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
  throw "Release issue close record candidate validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Release issue close record candidate validation written to $jsonPath"
Write-Host "Release issue close record candidate validation written to $markdownPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount) PerformsPublish=$($validation.performsPublish) CanPublishPublicly=$($validation.canPublishPublicly) CanCloseReleaseIssue=$($validation.canCloseReleaseIssue)"
