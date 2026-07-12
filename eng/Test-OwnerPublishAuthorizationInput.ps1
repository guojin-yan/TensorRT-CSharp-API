[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-publish-authorization-input.template.json",
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
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Get-BoolPropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [bool]$DefaultValue)
  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue $DefaultValue
  if ($value -is [bool]) { return [bool]$value }
  $parsed = $false
  if ([bool]::TryParse(([string]$value).Trim(), [ref]$parsed)) { return $parsed }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
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
  if ([IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Test-FileHashMatchesIfExists {
  param([AllowNull()][object]$Path, [AllowNull()][object]$Sha256)
  $pathText = [string]$Path
  $shaText = [string]$Sha256
  if ((Test-IsPlaceholder -Value $pathText) -or -not (Test-Sha256Format -Value $shaText)) { return $false }
  $resolvedPath = Resolve-InputPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) { return $true }
  $actual = (Get-FileHash -LiteralPath $resolvedPath -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
}

function Test-DateTimeOffsetFormat {
  param([AllowNull()][object]$Value)
  if (Test-IsPlaceholder -Value $Value) { return $false }
  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse(([string]$Value).Trim(), [ref]$parsed)
}

function Test-BoolTrue {
  param([AllowNull()][object]$Value)
  if (Test-IsPlaceholder -Value $Value) { return $false }
  $parsed = $false
  return [bool]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed
}

function Test-NotDryRunArtifactPath {
  param([AllowNull()][object]$Path)
  $text = [string]$Path
  if (Test-IsPlaceholder -Value $text) { return $false }
  return -not $text.Contains("artifacts\github-actions-runs", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.Contains("artifacts/github-actions-runs", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.Contains("package-managed-dry-run", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.Contains("github-actions-runs", [StringComparison]::OrdinalIgnoreCase)
}

function Test-NoTokenLikeText {
  param([AllowNull()][object]$Object)
  $json = $Object | ConvertTo-Json -Depth 20
  $patterns = @(
    "ghp_[A-Za-z0-9_]{20,}",
    "github_pat_[A-Za-z0-9_]{20,}",
    "nuget_[A-Za-z0-9_]{20,}",
    "oy2[a-z0-9]{20,}",
    "NUGET_API_KEY=[^<\s]+",
    "GITHUB_TOKEN=[^<\s]+"
  )
  foreach ($pattern in $patterns) {
    if ($json -match $pattern) { return $false }
  }
  return $true
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner publish authorization input not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$authorizationDecision = [string](Get-PropertyOrDefault -Object $record -Name "authorizationDecision" -DefaultValue "")
$authorizedRoutes = @(Get-PropertyOrDefault -Object $record -Name "authorizedRoutes" -DefaultValue @())
$performsPublish = Get-BoolPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true
$usesPublishToken = Get-BoolPropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true
$canPublishPublicly = Get-BoolPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true
$canCloseReleaseIssue = Get-BoolPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true
$isPostPublishProof = Get-BoolPropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-publish-authorization-input") -Severity "blocker" -Detail "recordKind must be owner-publish-authorization-input.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $usesPublishToken -and -not $canPublishPublicly -and -not $canCloseReleaseIssue -and -not $isPostPublishProof) -Severity "blocker" -Detail "Authorization input must not publish, use token, claim post-publish proof, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-token-like-secret-persisted" -Passed (Test-NoTokenLikeText -Object $record) -Severity "blocker" -Detail "Authorization input must not persist token-like strings.")) | Out-Null
$items.Add((New-ValidationItem -Id "authorization-decision-approved-for-owner-run" -Passed ($authorizationDecision -eq "approved-for-owner-run") -Severity "action-required" -Detail "Only explicit approved-for-owner-run can make the input ready; automation still does not publish.")) | Out-Null
$items.Add((New-ValidationItem -Id "authorized-routes-present" -Passed ($authorizedRoutes.Count -gt 0) -Severity "action-required" -Detail "authorizedRoutes must list the owner-approved package routes.")) | Out-Null

foreach ($field in @("ownerName", "managedPackageVersion", "runtimePackageVersion")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be real owner input.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "owner-decision-timestamp-parseable" -Passed (Test-DateTimeOffsetFormat -Value (Get-PropertyOrDefault -Object $record -Name "ownerDecisionTimestampUtc" -DefaultValue "")) -Severity "action-required" -Detail "ownerDecisionTimestampUtc must be parseable.")) | Out-Null

foreach ($field in @("managedNupkgPath", "runtimeNupkgPath", "releaseNotesPath", "rollbackPlanPath")) {
  $items.Add((New-ValidationItem -Id "$field-not-dry-run-artifact" -Passed (Test-NotDryRunArtifactPath -Path (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must not point to package-managed-dry-run or github-actions-runs artifacts.")) | Out-Null
}

foreach ($pair in @(
  @{ Path = "managedNupkgPath"; Sha = "managedNupkgSha256" },
  @{ Path = "runtimeNupkgPath"; Sha = "runtimeNupkgSha256" },
  @{ Path = "releaseNotesPath"; Sha = "releaseNotesSha256" },
  @{ Path = "rollbackPlanPath"; Sha = "rollbackPlanSha256" }
)) {
  $shaValue = Get-PropertyOrDefault -Object $record -Name $pair.Sha -DefaultValue ""
  $items.Add((New-ValidationItem -Id "$($pair.Sha)-format" -Passed (Test-Sha256Format -Value $shaValue) -Severity "action-required" -Detail "$($pair.Sha) must be a 64-character SHA256.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$($pair.Sha)-hash-match-if-file-exists" -Passed (Test-FileHashMatchesIfExists -Path (Get-PropertyOrDefault -Object $record -Name $pair.Path -DefaultValue "") -Sha256 $shaValue) -Severity "action-required" -Detail "$($pair.Path), when present, must match $($pair.Sha).")) | Out-Null
}

foreach ($field in @("confirmsNoTokenPersisted", "confirmsNoDryRunArtifactSubstitution", "confirmsPackageHashesReviewed", "confirmsPublishCommandReviewed", "confirmsPublicPackageDownloadProofStillRequired")) {
  $items.Add((New-ValidationItem -Id "$field-true" -Passed (Test-BoolTrue -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must be true.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "owner-publish-authorization-input-ready-for-owner-run"
}
else {
  "blocked-owner-publish-authorization-required"
}

$validation = [pscustomobject]@{
  recordKind = "owner-publish-authorization-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  ownerPublishAuthorizationReady = ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Owner authorization validation only. A ready result means owner-run command review is complete; automation still must not publish and post-publish proof remains required."
}

$jsonPath = Join-Path $OutputRoot "owner-publish-authorization-input-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-publish-authorization-input-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Owner Publish Authorization Input Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| ownerPublishAuthorizationReady | ``$($validation.ownerPublishAuthorizationReady)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| usesPublishToken | ``$($validation.usesPublishToken)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| isPostPublishProof | ``$($validation.isPostPublishProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Safety Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Owner publish authorization input validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Owner publish authorization input validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount) PerformsPublish=False UsesPublishToken=False"
