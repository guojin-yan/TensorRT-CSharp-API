[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-post-publish-audit-pack.json",
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
  throw "Final post-publish audit pack not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$lanes = @((Get-PropertyOrDefault -Object $record -Name "auditLanes" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-post-publish-audit-pack") -Severity "blocker" -Detail "recordKind must be final-post-publish-audit-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-count" -Passed ($lanes.Count -ge 7) -Severity "blocker" -Detail "Audit pack must include public package, post-publish, package consumer runtime, public bridge, final owner decision, and strict close lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-close-command" -Passed (([string](Get-PropertyOrDefault -Object $record -Name "strictCloseCommand" -DefaultValue "")).Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and ([string](Get-PropertyOrDefault -Object $record -Name "strictCloseCommand" -DefaultValue "")).Contains("-FailOnNotCloseReady", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "Audit pack must point at strict close validation.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Audit pack must not publish, promote proof, prove post-publish state, or close release issue.")) | Out-Null

foreach ($lane in $lanes) {
  $ready = [bool](Get-PropertyOrDefault -Object $lane -Name "ready" -DefaultValue $false)
  $laneId = [string](Get-PropertyOrDefault -Object $lane -Name "laneId" -DefaultValue "unknown-lane")
  $items.Add((New-ValidationItem -Id "$laneId-owner-action-required" -Passed $ready -Severity "action-required" -Detail "Lane $laneId must reach its required state before final post-publish audit pack is ready.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-final-post-publish-audit-pack"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-final-post-publish-audit-required"
}
else {
  "final-post-publish-audit-ready"
}

$validation = [pscustomobject]@{
  recordKind = "final-post-publish-audit-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  auditLaneCount = $lanes.Count
  blockedAuditLaneCount = @($lanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) }).Count
  readyAuditLaneCount = @($lanes | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) }).Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Final post-publish audit pack validation only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$jsonPath = Join-Path $OutputRoot "final-post-publish-audit-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "final-post-publish-audit-pack-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Final Post-Publish Audit Pack Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| auditLaneCount | ``$($validation.auditLaneCount)`` |
| blockedAuditLaneCount | ``$($validation.blockedAuditLaneCount)`` |
| readyAuditLaneCount | ``$($validation.readyAuditLaneCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final post-publish audit pack validation written to $jsonPath"
Write-Host "ValidationState=$validationState Lanes=$($validation.auditLaneCount) Blocked=$($validation.blockedAuditLaneCount) FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final post-publish audit pack has blocker validation failures."
}
