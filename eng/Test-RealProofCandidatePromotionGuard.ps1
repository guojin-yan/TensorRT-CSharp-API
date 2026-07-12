[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-proof-candidate-promotion-guard.json",
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

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

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

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Real proof candidate promotion guard not found: $resolvedInputPath"
}

$guard = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$guardItems = @(Get-PropertyOrDefault -Object $guard -Name "guardItems" -DefaultValue @())

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $guard -Name "recordKind" -DefaultValue "") -eq "real-proof-candidate-promotion-guard") -Severity "blocker" -Detail "recordKind must be real-proof-candidate-promotion-guard.")) | Out-Null
$items.Add((New-ValidationItem -Id "guard-state" -Passed ([string](Get-PropertyOrDefault -Object $guard -Name "guardState" -DefaultValue "") -eq "blocked-real-proof-candidate-promotion-not-allowed") -Severity "blocker" -Detail "Guard must remain blocked by default.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-count" -Passed ([int](Get-PropertyOrDefault -Object $guard -Name "candidateCount" -DefaultValue 0) -eq 6) -Severity "blocker" -Detail "Promotion guard must cover 6 candidates.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-candidate-count" -Passed ([int](Get-PropertyOrDefault -Object $guard -Name "blockedCandidateCount" -DefaultValue 0) -eq 6) -Severity "blocker" -Detail "Default promotion guard must keep all candidates blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "promotion-allowed-count" -Passed ([int](Get-PropertyOrDefault -Object $guard -Name "promotionAllowedCandidateCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Default promotion guard must not allow candidate promotion.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $guard -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $guard -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $guard -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $guard -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Promotion guard must not promote proof or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $guard -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $guard -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Promotion guard must not publish or approve public publication.")) | Out-Null

foreach ($guardItem in $guardItems) {
  $candidateId = [string](Get-PropertyOrDefault -Object $guardItem -Name "candidateId" -DefaultValue "unknown-candidate")
  $requirements = @(Get-PropertyOrDefault -Object $guardItem -Name "requirements" -DefaultValue @())
  $blockedRequirementCount = @($requirements | Where-Object { -not [bool]$_.passed }).Count

  $items.Add((New-ValidationItem -Id "$candidateId-shape" -Passed ($requirements.Count -ge 4 -and [int](Get-PropertyOrDefault -Object $guardItem -Name "blockedRequirementCount" -DefaultValue -1) -eq $blockedRequirementCount) -Severity "blocker" -Detail "Each guard item must include blocked requirements.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$candidateId-promotion-blocked" -Passed (-not [bool](Get-PropertyOrDefault -Object $guardItem -Name "promotionAllowed" -DefaultValue $true)) -Severity "blocker" -Detail "Default guard item must not allow promotion.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$candidateId-owner-action-required" -Passed ($blockedRequirementCount -eq 0) -Severity "action-required" -Detail "Owner must satisfy every promotion guard requirement before candidate review.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-real-proof-candidate-promotion-guard"
}
else {
  "blocked-real-proof-candidate-promotion-not-allowed"
}

$validation = [pscustomobject]@{
  recordKind = "real-proof-candidate-promotion-guard-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  candidateCount = [int](Get-PropertyOrDefault -Object $guard -Name "candidateCount" -DefaultValue 0)
  promotionAllowedCandidateCount = [int](Get-PropertyOrDefault -Object $guard -Name "promotionAllowedCandidateCount" -DefaultValue 0)
  blockedCandidateCount = [int](Get-PropertyOrDefault -Object $guard -Name "blockedCandidateCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates candidate promotion guard shape only. It is not runtime proof, not publication approval, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "real-proof-candidate-promotion-guard-validation.json"
$markdownPath = Join-Path $OutputRoot "real-proof-candidate-promotion-guard-validation.md"
$validation | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace('|', '\|')) |"
}

$markdown = @"
# Real Proof Candidate Promotion Guard Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| candidateCount | ``$($validation.candidateCount)`` |
| promotionAllowedCandidateCount | ``$($validation.promotionAllowedCandidateCount)`` |
| blockedCandidateCount | ``$($validation.blockedCandidateCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canPromoteRuntimeProof | ``False`` |
| canCloseReleaseIssue | ``False`` |
| isRuntimeExecutionProof | ``False`` |
| isReleaseCloseProof | ``False`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join [Environment]::NewLine)

## Boundary

$($validation.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real proof candidate promotion guard validation written to $jsonPath"
Write-Host "Real proof candidate promotion guard validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState Candidates=$($validation.candidateCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Real proof candidate promotion guard validation failed with $($failedBlockers.Count) blocker(s)."
}
