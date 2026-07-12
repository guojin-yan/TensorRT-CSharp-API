[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-issue-final-close-decision.template.json",
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

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

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

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)

  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
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

  $resolvedPath = Resolve-RepositoryPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $false
  }

  $actual = (Get-FileHash -LiteralPath $resolvedPath -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release issue final close decision input not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$ownerFinalCloseDecision = [string](Get-PropertyOrDefault -Object $record -Name "ownerFinalCloseDecision" -DefaultValue "")
$strictCloseValidatorCommand = [string](Get-PropertyOrDefault -Object $record -Name "strictCloseValidatorCommand" -DefaultValue "")
$runtimeSmokeExitCode = Get-PropertyOrDefault -Object $record -Name "runtimeSmokeExitCode" -DefaultValue $null

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "release-issue-final-close-decision") -Severity "blocker" -Detail "recordKind must be release-issue-final-close-decision.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Final close decision input must not publish, approve publication, or close the issue by itself.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-close-validator-command" -Passed ($strictCloseValidatorCommand.Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $strictCloseValidatorCommand.Contains("-FailOnNotCloseReady", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "strictCloseValidatorCommand must use Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady.")) | Out-Null

foreach ($pair in @(
  @("final-evidence-freeze-hash", "finalEvidenceFreezePath", "finalEvidenceFreezeSha256", "Final evidence freeze path and SHA256 must exist and match."),
  @("final-evidence-freeze-validation-hash", "finalEvidenceFreezeValidationPath", "finalEvidenceFreezeValidationSha256", "Final evidence freeze validation path and SHA256 must exist and match."),
  @("release-evidence-bundle-hash", "releaseEvidenceBundlePath", "releaseEvidenceBundleSha256", "Release evidence bundle path and SHA256 must exist and match."),
  @("post-publish-verification-validation-hash", "postPublishVerificationValidationPath", "postPublishVerificationValidationSha256", "Post-publish verification validation path and SHA256 must exist and match."),
  @("release-close-candidate-validation-hash", "releaseIssueCloseRecordCandidateValidationPath", "releaseIssueCloseRecordCandidateValidationSha256", "Release issue close candidate validation path and SHA256 must exist and match.")
)) {
  $items.Add((New-ValidationItem -Id $pair[0] -Passed (Test-FileHashMatches -Path (Get-PropertyOrDefault -Object $record -Name $pair[1] -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name $pair[2] -DefaultValue "")) -Severity "action-required" -Detail $pair[3])) | Out-Null
}

foreach ($field in @("ownerName", "ownerDecisionTimestampUtc", "releaseIssueId", "releaseIssueUrl", "rollbackOwner", "rollbackTrigger")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be real non-placeholder owner input.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "owner-final-close-decision-real" -Passed (-not (Test-IsPlaceholder -Value $ownerFinalCloseDecision) -and $ownerFinalCloseDecision.Equals("approved-to-close-after-real-proof", [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "ownerFinalCloseDecision must be approved-to-close-after-real-proof after real proof is validated.")) | Out-Null
$items.Add((New-ValidationItem -Id "rollback-plan-reviewed" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "rollbackPlanReviewed" -DefaultValue $false)) -Severity "action-required" -Detail "rollbackPlanReviewed must be true only after owner review.")) | Out-Null
$items.Add((New-ValidationItem -Id "real-post-publish-proof-confirmed" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "confirmsRealPostPublishProof" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must confirm real post-publish proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-package-source-confirmed" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "confirmsPublicPackageSource" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must confirm public package source.")) | Out-Null
$items.Add((New-ValidationItem -Id "clean-consumer-confirmed" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "confirmsCleanConsumerOutsideRepository" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must confirm clean consumer outside repository.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-project-reference-confirmed" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "confirmsNoProjectReference" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must confirm no ProjectReference.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-local-package-source-confirmed" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "confirmsNoLocalPackageSource" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must confirm no local package source.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-direct-nupkg-confirmed" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "confirmsNoDirectNupkgReference" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must confirm no direct .nupkg reference.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-smoke-passed-confirmed" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "confirmsRuntimeSmokePassed" -DefaultValue $false) -and $runtimeSmokeExitCode -eq 0) -Severity "action-required" -Detail "Owner must confirm runtime smoke passed with exit code 0.")) | Out-Null
$items.Add((New-ValidationItem -Id "logs-and-sha256-reviewed" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "confirmsLogsAndSha256Reviewed" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must confirm logs and SHA256 values were reviewed.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "owner-final-close-decision-ready-for-strict-close-validator"
}
else {
  "blocked-owner-final-close-decision-required"
}

$validation = [pscustomobject]@{
  recordKind = "release-issue-final-close-decision-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  isValidDecisionShape = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  ownerFinalCloseDecision = $ownerFinalCloseDecision
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates owner final close decision input only. It cannot close the release issue without the strict close record validator."
}

$jsonPath = Join-Path $OutputRoot "release-issue-final-close-decision-validation.json"
$markdownPath = Join-Path $OutputRoot "release-issue-final-close-decision-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Release Issue Final Close Decision Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| isValidDecisionShape | ``$($validation.isValidDecisionShape)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Safety Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release issue final close decision validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Release issue final close decision validation failed with $($failedBlockers.Count) blocker(s)."
}
