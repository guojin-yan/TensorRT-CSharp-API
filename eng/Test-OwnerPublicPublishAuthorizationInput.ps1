[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-public-publish-authorization-input-import.json",
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
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
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

function Test-Sha256Format {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match "^[a-fA-F0-9]{64}$"
}

function Test-ExternalHttpsUrl {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return $text -match "^https://"
}

function Test-NonPlaceholder {
  param([AllowNull()][object]$Value)
  $text = ([string]$Value).Trim()
  if ([string]::IsNullOrWhiteSpace($text)) { return $false }
  if ($text.IndexOf("<owner-fill", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("template", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  return $true
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Import-OwnerPublicPublishAuthorizationInput.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-public-publish-authorization-input-import") -Severity "blocker" -Detail "recordKind must match authorization input import.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed (
      [bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)
    ) -Severity "blocker" -Detail "Authorization import must remain non-proof and side-effect free.")) | Out-Null

foreach ($field in @("authorizedBy", "authorizedAtUtc", "expectedCommit", "managedPackageId", "managedPackageVersion", "runtimePackageId", "runtimePackageVersion")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (Test-NonPlaceholder (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must be owner-filled.")) | Out-Null
}

foreach ($field in @("managedPackageSha256", "runtimePackageSha256", "publishCommandPlanSha256", "rollbackPlanSha256")) {
  $items.Add((New-ValidationItem -Id "sha256-$field" -Passed (Test-Sha256Format (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must be a 64-character SHA256.")) | Out-Null
}

foreach ($field in @("nugetPackageSource", "githubPackagesSource")) {
  $items.Add((New-ValidationItem -Id "url-$field" -Passed (Test-ExternalHttpsUrl (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must be an HTTPS source, not local feed.")) | Out-Null
}

foreach ($field in @("ownerAuthorizedPublicPublish", "authorizationPhraseMatches", "noLocalFeedConfirmation", "noProjectReferenceConfirmation", "noDirectNupkgConfirmation")) {
  $items.Add((New-ValidationItem -Id "confirmation-$field" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name $field -DefaultValue $false)) -Severity "action-required" -Detail "$field must be true before real publish result can be accepted.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-owner-public-publish-authorization-input" } elseif ($failedActionRequired.Count -eq 0) { "owner-public-publish-authorization-input-ready" } else { "blocked-owner-public-publish-authorization-input-required" }

$validation = [pscustomobject]@{
  recordKind = "owner-public-publish-authorization-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Authorization input validation is fail-closed and side-effect free. It does not publish or close release issues."
}

$jsonPath = Join-Path $OutputRoot "owner-public-publish-authorization-input-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-public-publish-authorization-input-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.passed)`` | $(ConvertTo-MarkdownCell $_.severity) | $(ConvertTo-MarkdownCell $_.detail) |"
}
$markdown = @"
# Owner Public Publish Authorization Input Validation

| Item | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Owner public publish authorization input validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Owner public publish authorization input validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

