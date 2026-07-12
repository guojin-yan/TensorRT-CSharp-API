[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\github-publish-and-ci-status-snapshot.json",
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

function Resolve-RepositoryPath { param([string]$Path) if ([System.IO.Path]::IsPathRooted($Path)) { return $Path } return Join-Path $RepositoryRoot $Path }
function Get-PropertyOrDefault { param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue) if ($null -eq $Object) { return $DefaultValue }; if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }; return $DefaultValue }
function New-ValidationItem { param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail) [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail } }
function ConvertTo-MarkdownCell { param([AllowNull()][object]$Value) if ($null -eq $Value) { return "" } return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ") }

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-GitHubPublishAndCiStatusSnapshot.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$statusItems = @((Get-PropertyOrDefault -Object $record -Name "statusItems" -DefaultValue @()))
$states = @($statusItems | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "state" -DefaultValue "") })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "github-publish-and-ci-status-snapshot") -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "github-actions-proof-missing" -Passed ($states -contains "missing-github-actions-proof") -Severity "action-required" -Detail "Snapshot must explicitly mark missing GitHub Actions proof until a real run URL/log/hash is imported.")) | Out-Null
$items.Add((New-ValidationItem -Id "package-publish-proof-blocked" -Passed ($states -contains "blocked-real-github-actions-package-publish-proof-required") -Severity "action-required" -Detail "Snapshot must explicitly block GitHub-hosted package publish proof without real run evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isGitHubActionsProof" -DefaultValue $true)) -Severity "blocker" -Detail "Snapshot must not claim publish, close, proof, or GitHub Actions proof.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-github-publish-and-ci-status-snapshot" } else { "blocked-github-actions-and-public-publish-proof-required" }
$validation = [pscustomobject]@{
  recordKind = "github-publish-and-ci-status-snapshot-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  statusItemCount = $statusItems.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  githubActionsProofState = "missing-github-actions-proof"
  packagePublishOnGitHubState = "blocked-real-github-actions-package-publish-proof-required"
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isGitHubActionsProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "GitHub/CI status validation is read-only and side-effect free; not runtime proof, not post-publish proof, not package publish proof, not GitHub Actions proof, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "github-publish-and-ci-status-snapshot-validation.json"
$markdownPath = Join-Path $OutputRoot "github-publish-and-ci-status-snapshot-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$rows = $validation.validationItems | ForEach-Object { "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.passed)`` | $(ConvertTo-MarkdownCell $_.severity) | $(ConvertTo-MarkdownCell $_.detail) |" }
$markdown = @"
# GitHub Publish And CI Status Snapshot Validation

| Item | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| statusItemCount | ``$($validation.statusItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| githubActionsProofState | ``$($validation.githubActionsProofState)`` |
| packagePublishOnGitHubState | ``$($validation.packagePublishOnGitHubState)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) { throw "GitHub publish and CI status snapshot validation failed with $($failedBlockers.Count) blocker(s)." }

Write-Host "GitHub publish and CI status snapshot validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"
