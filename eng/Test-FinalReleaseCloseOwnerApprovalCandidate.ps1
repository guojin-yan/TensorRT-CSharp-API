[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-release-close-owner-approval-candidate.json",
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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final release close Owner approval candidate not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$candidateItems = @((Get-PropertyOrDefault -Object $record -Name "closeCandidateItems" -DefaultValue @()))
$closeCandidateItemCount = [int](Get-PropertyOrDefault -Object $record -Name "closeCandidateItemCount" -DefaultValue 0)
$blockedCloseCandidateItemCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedCloseCandidateItemCount" -DefaultValue -1)

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-release-close-owner-approval-candidate") -Severity "blocker" -Detail "recordKind must be final-release-close-owner-approval-candidate.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "") -in @("blocked-no-final-release-close-owner-approval-candidate-ready", "final-release-close-owner-approval-candidate-created")) -Severity "blocker" -Detail "Candidate state must be explicit.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-counts-consistent" -Passed ($closeCandidateItemCount -eq $candidateItems.Count -and $blockedCloseCandidateItemCount -ge 0) -Severity "blocker" -Detail "Candidate counts must be internally consistent.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "Candidate must stay non-proof and non-publish.")) | Out-Null
$items.Add((New-ValidationItem -Id "why-not-release-close" -Passed ($raw.IndexOf("No real Owner final release close approval", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("No package push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("cannot close the release", [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Candidate must explain why it cannot close the release.")) | Out-Null

foreach ($candidate in $candidateItems) {
  $id = [string](Get-PropertyOrDefault -Object $candidate -Name "id" -DefaultValue "")
  $boundary = [string](Get-PropertyOrDefault -Object $candidate -Name "boundary" -DefaultValue "")
  $items.Add((New-ValidationItem -Id "$id-state" -Passed ([string](Get-PropertyOrDefault -Object $candidate -Name "candidateState" -DefaultValue "") -eq "final-release-close-owner-approval-candidate") -Severity "blocker" -Detail "$id must be explicit candidate state.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $candidate -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $candidate -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $candidate -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $candidate -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $candidate -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "$id must stay non-proof.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-boundary" -Passed ($boundary.IndexOf("not package push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("cannot close the release", [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "$id must state candidate non-proof boundary.")) | Out-Null
}

$failedItems = @($items | Where-Object { -not $_.passed })
$failedBlockers = @($failedItems | Where-Object { $_.severity -eq "blocker" }).Count
$validationState = if ($failedBlockers -eq 0) { [string](Get-PropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "blocked-no-final-release-close-owner-approval-candidate-ready") } else { "invalid-final-release-close-owner-approval-candidate" }

$validation = [ordered]@{
  recordKind = "final-release-close-owner-approval-candidate-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  candidateState = [string](Get-PropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "")
  closeCandidateItemCount = $closeCandidateItemCount
  blockedCloseCandidateItemCount = $blockedCloseCandidateItemCount
  failedBlockerCount = $failedBlockers
  failedActionRequiredCount = if ($closeCandidateItemCount -eq 0) { 4 } else { 0 }
  findingCount = $failedItems.Count
  performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $false)
  canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $false)
  canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $false)
  canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $false)
  isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $false)
  isPostPublishProof = [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $false)
  isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $false)
  validationItems = @($items.ToArray())
  failedItems = @($failedItems)
}

$jsonPath = Join-Path $OutputRoot "final-release-close-owner-approval-candidate-validation.json"
$markdownPath = Join-Path $OutputRoot "final-release-close-owner-approval-candidate-validation.md"
$validation | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Final Release Close Owner Approval Candidate Validation",
  "",
  "- validationState: ``$validationState``",
  "- failedBlockerCount: ``$failedBlockers``",
  "- failedActionRequiredCount: ``$($validation.failedActionRequiredCount)``",
  "- closeCandidateItemCount: ``$closeCandidateItemCount``",
  "- blockedCloseCandidateItemCount: ``$blockedCloseCandidateItemCount``"
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"

if ($Strict -and $failedBlockers -gt 0) {
  throw "Final release close Owner approval candidate validation failed with $failedBlockers blocker(s)."
}
