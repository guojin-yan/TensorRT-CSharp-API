[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-publish-result-import.json",
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

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-ExternalHttpsUrl {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  if ([string]::IsNullOrWhiteSpace($text) -or $text -like "<*>") { return $false }
  if ($text.StartsWith("file:", [StringComparison]::OrdinalIgnoreCase)) { return $false }
  if ([System.IO.Path]::IsPathRooted($text)) { return $false }
  return $text.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase)
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
  throw "Public publish result import not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$runtimeHashes = @((Get-PropertyOrDefault -Object $record -Name "runtimePackageSha256" -DefaultValue @()))
$githubRelease = Get-PropertyOrDefault -Object $record -Name "githubRelease" -DefaultValue $null
$managedPackage = Get-PropertyOrDefault -Object $record -Name "managedPackage" -DefaultValue $null
$runtimePackage = Get-PropertyOrDefault -Object $record -Name "runtimePackage" -DefaultValue $null
$ownerReview = Get-PropertyOrDefault -Object $record -Name "ownerReview" -DefaultValue $null
$rollbackReview = Get-PropertyOrDefault -Object $record -Name "rollbackReview" -DefaultValue $null
$finalCloseDecision = Get-PropertyOrDefault -Object $record -Name "finalCloseDecision" -DefaultValue $null

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-publish-result-import") -Severity "blocker" -Detail "recordKind must be public-publish-result-import.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Import validation must not execute publish or promote proof/close flags.")) | Out-Null

foreach ($field in @("ownerName", "ownerEmail", "publishedAtUtc", "selectedChannel", "nugetPackageSource", "packageId", "packageVersion", "managedNupkgPath", "publishCommandTranscriptPath")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be Owner-filled before the import can become ready.")) | Out-Null
}

foreach ($field in @("releaseUrl", "tagName", "managedAssetPath", "runtimeAssetPath")) {
  $items.Add((New-ValidationItem -Id "github-release-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $githubRelease -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "githubRelease.$field must be imported from Owner input.")) | Out-Null
}
foreach ($field in @("managedAssetSha256", "runtimeAssetSha256")) {
  $items.Add((New-ValidationItem -Id "github-release-$field" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $githubRelease -Name $field -DefaultValue "")) -Severity "action-required" -Detail "githubRelease.$field must be a 64-character SHA256.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "public-package-url" -Passed ((Test-ExternalHttpsUrl (Get-PropertyOrDefault -Object $record -Name "nugetPackageUrl" -DefaultValue "")) -or (Test-ExternalHttpsUrl (Get-PropertyOrDefault -Object $record -Name "githubPackageUrl" -DefaultValue ""))) -Severity "action-required" -Detail "A non-local HTTPS package URL is required.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-nupkg-sha256" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $record -Name "managedNupkgSha256" -DefaultValue "")) -Severity "action-required" -Detail "managedNupkgSha256 must be a 64-character SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-public-download-sha256" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $managedPackage -Name "publicDownloadSha256" -DefaultValue "")) -Severity "action-required" -Detail "managedPackage.publicDownloadSha256 must be a 64-character SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-public-download-sha256" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $runtimePackage -Name "publicDownloadSha256" -DefaultValue "")) -Severity "action-required" -Detail "runtimePackage.publicDownloadSha256 must be a 64-character SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-nupkg-sha256" -Passed ($runtimeHashes.Count -gt 0 -and @($runtimeHashes | Where-Object { Test-Sha256Format $_ }).Count -eq $runtimeHashes.Count) -Severity "action-required" -Detail "runtimePackageSha256 values must be 64-character SHA256 values.")) | Out-Null
$items.Add((New-ValidationItem -Id "transcript-sha256" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $record -Name "publishCommandTranscriptSha256" -DefaultValue "")) -Severity "action-required" -Detail "publishCommandTranscriptSha256 must be a 64-character SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "transcript-exists" -Passed (Test-Path -LiteralPath (Resolve-RepositoryPath -Path ([string](Get-PropertyOrDefault -Object $record -Name "publishCommandTranscriptPath" -DefaultValue ""))) -PathType Leaf) -Severity "action-required" -Detail "publishCommandTranscriptPath must exist.")) | Out-Null
foreach ($field in @("ownerReviewedPackageHash", "ownerReviewedPublicUrl", "rollbackPlanReviewed")) {
  $items.Add((New-ValidationItem -Id "owner-confirmation-$field" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name $field -DefaultValue $false)) -Severity "action-required" -Detail "$field must be true after Owner review.")) | Out-Null
}
foreach ($field in @("reviewer", "reviewedAtUtc", "approvalState")) {
  $items.Add((New-ValidationItem -Id "owner-review-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $ownerReview -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "ownerReview.$field must be imported from Owner input.")) | Out-Null
}
foreach ($field in @("reviewedBy", "reviewedAtUtc", "decision")) {
  $items.Add((New-ValidationItem -Id "rollback-review-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $rollbackReview -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "rollbackReview.$field must be imported from Owner input.")) | Out-Null
}
$items.Add((New-ValidationItem -Id "rollback-review-rollbackPlanSha256" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $rollbackReview -Name "rollbackPlanSha256" -DefaultValue "")) -Severity "action-required" -Detail "rollbackReview.rollbackPlanSha256 must be a 64-character SHA256.")) | Out-Null
foreach ($field in @("decision", "decidedAtUtc", "ownerReviewer", "releaseIssueUrl")) {
  $value = Get-PropertyOrDefault -Object $finalCloseDecision -Name $field -DefaultValue ""
  $passed = if ($field -eq "releaseIssueUrl") { Test-ExternalHttpsUrl $value } else { -not (Test-IsPlaceholder -Value $value) }
  $items.Add((New-ValidationItem -Id "final-close-decision-$field" -Passed $passed -Severity "action-required" -Detail "finalCloseDecision.$field must be imported from Owner input.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-public-publish-result-import"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-public-publish-result-import-required"
}
else {
  "public-publish-result-import-ready"
}

$validation = [pscustomobject]@{
  recordKind = "public-publish-result-import-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  notExecutedByAutomation = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Public publish result import validation only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "public-publish-result-import-validation.json"
$markdownPath = Join-Path $OutputRoot "public-publish-result-import-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Public Publish Result Import Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| notExecutedByAutomation | ``$($validation.notExecutedByAutomation)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public publish result import validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Public publish result import has blocker validation failures."
}
