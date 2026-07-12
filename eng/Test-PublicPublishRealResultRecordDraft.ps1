[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-publish-real-result-record-draft.json",
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

if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  throw "Public publish real result record draft not found: $InputPath"
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$fields = @((Get-PropertyOrDefault -Object $record -Name "ownerFields" -DefaultValue @()))
$blockedFields = @($fields | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$requiredNames = @("packageId", "packageVersion", "publicSource", "publicPackageUrl", "publicPackageSha256", "publishedAtUtc", "publishCommandTranscriptPath", "publishCommandTranscriptSha256", "packageOwnerAccount", "reviewer", "rollbackPlanReviewed", "forbiddenSubstituteScanResult")
$fieldNames = @($fields | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "") })

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-publish-real-result-record-draft") -Severity "blocker" -Detail "recordKind must be public-publish-real-result-record-draft.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "draftState" -DefaultValue "") -eq "blocked-public-publish-real-result-record-required") -Severity "blocker" -Detail "Draft must stay blocked until owner supplies real public publish result fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-fields-present" -Passed (@($requiredNames | Where-Object { $fieldNames -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Draft must expose all required public publish result fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-input-required" -Passed ($blockedFields.Count -eq 0) -Severity "action-required" -Detail "Owner must fill all real public publish result fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Draft must not publish, approve, promote proof, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-public-publish-real-result-record-draft" } else { "blocked-public-publish-real-result-record-required" }

$validation = [pscustomobject]@{
  recordKind = "public-publish-real-result-record-draft-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $InputPath
  validationState = $validationState
  requiredFieldCount = $fields.Count
  blockedRequiredFieldCount = $blockedFields.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  validationItems = @($items.ToArray())
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  boundary = "Validation checks the owner-filled public publish result draft shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "public-publish-real-result-record-draft-validation.json"
$markdownPath = Join-Path $OutputRoot "public-publish-real-result-record-draft-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Public Publish Real Result Record Draft Validation",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| validationState | ``$($validation.validationState)`` |",
  "| requiredFieldCount | ``$($validation.requiredFieldCount)`` |",
  "| blockedRequiredFieldCount | ``$($validation.blockedRequiredFieldCount)`` |",
  "| failedBlockerCount | ``$($validation.failedBlockerCount)`` |",
  "| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |",
  "| canPublishPublicly | ``$($validation.canPublishPublicly)`` |",
  "| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |",
  "",
  "## Boundary",
  "",
  $validation.boundary
)

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public publish real result record draft validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) RequiredFields=$($validation.requiredFieldCount) Blocked=$($validation.blockedRequiredFieldCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Public publish real result record draft validation failed with $($failedBlockers.Count) blocker(s)."
}
