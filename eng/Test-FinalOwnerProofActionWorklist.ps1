[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-proof-action-worklist.json",
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

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
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
  throw "Final owner proof action worklist not found: $resolvedInputPath"
}

$worklist = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]

$actions = @(Get-PropertyOrDefault -Object $worklist -Name "actions" -DefaultValue @())
$actionIds = @($actions | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$actionRequiredIds = @($actions | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "actionRequiredId" -DefaultValue "") })
$requiredActionIds = @(
  "00-clean-external-package-consumer-owner-runbook",
  "00-post-publish-owner-verification-runbook",
  "01-real-model-runtime-owner-evidence",
  "02-package-consumer-runtime-clean-external-proof",
  "03-post-publish-verification-public-channel",
  "04-final-owner-real-input-template-pack",
  "05-owner-external-result-import-real-files",
  "06-owner-result-candidate-bridge-strict-promotion"
)
$requiredActionRequiredIds = @(
  "real-model-runtime-owner-proof-required",
  "package-consumer-runtime-owner-proof-required",
  "post-publish-verification-owner-proof-required",
  "final-owner-real-input-template-pack-owner-input-required",
  "owner-external-proof-result-import-owner-proof-required",
  "owner-result-candidate-bridge-real-proof-required"
)

$items.Add((New-ValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $worklist -Name "recordKind" -DefaultValue "") -eq "final-owner-proof-action-worklist") "blocker" "Worklist must use recordKind=final-owner-proof-action-worklist.")) | Out-Null
$items.Add((New-ValidationItem "state-blocked" ([string](Get-PropertyOrDefault -Object $worklist -Name "worklistState" -DefaultValue "") -eq "blocked-final-owner-proof-action-required") "blocker" "Worklist must stay blocked until real owner/public/post-publish proof exists.")) | Out-Null
$items.Add((New-ValidationItem "does-not-publish-or-promote" ((-not [bool](Get-PropertyOrDefault -Object $worklist -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $worklist -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $worklist -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $worklist -Name "canPromoteRuntimeProof" -DefaultValue $true))) "blocker" "Worklist must not publish, close, or promote proof.")) | Out-Null
$items.Add((New-ValidationItem "eight-actions-present" ($actions.Count -eq 8 -and [int](Get-PropertyOrDefault -Object $worklist -Name "actionCount" -DefaultValue -1) -eq 8 -and [int](Get-PropertyOrDefault -Object $worklist -Name "blockedActionCount" -DefaultValue -1) -eq 8) "blocker" "Worklist must cover two owner runbook preflight items plus all six final publish action-required items.")) | Out-Null
$items.Add((New-ValidationItem "no-missing-final-gate-actions" ([int](Get-PropertyOrDefault -Object $worklist -Name "missingActionRequiredIdCount" -DefaultValue -1) -eq 0) "blocker" "Worklist must map every final publish action-required id.")) | Out-Null
$items.Add((New-ValidationItem "final-gate-count-matches" ([int](Get-PropertyOrDefault -Object $worklist -Name "finalGateActionRequiredCount" -DefaultValue -1) -eq 6) "blocker" "Final publish gate currently exposes six action-required items; two runbook preflight items are added by this worklist.")) | Out-Null
$items.Add((New-ValidationItem "owner-proof-lane-count-matches" ([int](Get-PropertyOrDefault -Object $worklist -Name "releaseDashboardOwnerProofActionRequiredLaneCount" -DefaultValue -1) -eq 4) "blocker" "Release proof dashboard currently exposes four owner proof lanes.")) | Out-Null
$items.Add((New-ValidationItem "public-docs-clean-but-blocked" ([int](Get-PropertyOrDefault -Object $worklist -Name "publicDocsBlockedMatchCount" -DefaultValue -1) -eq 0 -and [string](Get-PropertyOrDefault -Object $worklist -Name "publicDocsGateState" -DefaultValue "") -eq "blocked-owner-public-postpublish-proof-required") "blocker" "Public docs gate must be clean but still blocked on real owner/public/post-publish proof.")) | Out-Null

foreach ($required in $requiredActionIds) {
  $items.Add((New-ValidationItem "action-$required-present" ($actionIds -contains $required) "blocker" "Action $required must be present.")) | Out-Null
}

foreach ($required in $requiredActionRequiredIds) {
  $items.Add((New-ValidationItem "gate-item-$required-mapped" ($actionRequiredIds -contains $required) "blocker" "Final gate action-required id $required must be mapped.")) | Out-Null
}

foreach ($action in $actions) {
  $id = [string](Get-PropertyOrDefault -Object $action -Name "id" -DefaultValue "")
  $requiredInputCount = [int](Get-PropertyOrDefault -Object $action -Name "requiredInputCount" -DefaultValue 0)
  $ownerCommands = @(Get-PropertyOrDefault -Object $action -Name "ownerCommands" -DefaultValue @())
  $validatorCommands = @(Get-PropertyOrDefault -Object $action -Name "validatorCommands" -DefaultValue @())
  $boundary = [string](Get-PropertyOrDefault -Object $action -Name "boundary" -DefaultValue "")
  $nonPromoting = (-not [bool](Get-PropertyOrDefault -Object $action -Name "performsPublish" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $action -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $action -Name "canPublishPublicly" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $action -Name "canCloseReleaseIssue" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $action -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $action -Name "isPostPublishProof" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $action -Name "isReleaseCloseProof" -DefaultValue $true))

  $items.Add((New-ValidationItem "action-$id-has-inputs-and-commands" ($requiredInputCount -ge 5 -and $ownerCommands.Count -ge 2 -and $validatorCommands.Count -ge 1) "blocker" "Each action must expose required inputs, owner commands, and validators.")) | Out-Null
  $items.Add((New-ValidationItem "action-$id-non-promoting" $nonPromoting "blocker" "Each action must remain non-publishing and non-promoting.")) | Out-Null
  $items.Add((New-ValidationItem "action-$id-boundary-non-proof" ($boundary.Contains("requires", [StringComparison]::OrdinalIgnoreCase) -or $boundary.Contains("not", [StringComparison]::OrdinalIgnoreCase) -or $boundary.Contains("cannot", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Each action must describe proof boundary.")) | Out-Null
}

foreach ($marker in @("local feed", "ProjectReference", "direct .nupkg", "build-only", "dry-run", "template", "candidate", "dashboard", "blocked-by-driver", "repository-external", "public package source URL", "stdoutPath", "stderrPath", "mergedTranscriptPath", "nonSubstituteConfirmations", "does not run dotnet nuget push")) {
  $items.Add((New-ValidationItem "forbidden-substitute-$($marker.Replace(' ', '-').Replace('.', 'dot'))-visible" ($raw.Contains($marker, [StringComparison]::OrdinalIgnoreCase)) "blocker" "Forbidden substitute marker '$marker' must remain visible.")) | Out-Null
}

$boundary = [string](Get-PropertyOrDefault -Object $worklist -Name "boundary" -DefaultValue "")
$items.Add((New-ValidationItem "top-level-boundary-non-proof" ($boundary.Contains("does not publish", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("cannot close", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("non-proof", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Top-level boundary must state this is a non-proof handoff.")) | Out-Null

$failedBlockerCount = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" }).Count
$validationState = if ($failedBlockerCount -eq 0) { "blocked-final-owner-proof-action-required" } else { "failed-final-owner-proof-action-worklist-validation" }

$report = [ordered]@{
  recordKind = "final-owner-proof-action-worklist-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  inputPath = $resolvedInputPath
  validationState = $validationState
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  actionCount = $actions.Count
  failedBlockerCount = [int]$failedBlockerCount
  validationItems = [object[]]@($items.ToArray())
  boundary = "Final owner proof action worklist validation checks handoff shape only. It does not publish, does not promote proof, and cannot replace real owner/public/post-publish evidence."
}

$jsonPath = Join-Path $OutputRoot "final-owner-proof-action-worklist-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-proof-action-worklist-validation.md"
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $items) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Final Owner Proof Action Worklist Validation

Generated at: ``$($report.generatedAtUtc)``

## Summary

- validationState: ``$($report.validationState)``
- actionCount: ``$($report.actionCount)``
- failedBlockerCount: ``$($report.failedBlockerCount)``
- performsPublish: ``False``
- canPromoteRuntimeProof: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Validation Items

| Id | Passed | Severity | Detail |
| --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($report.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Final owner proof action worklist validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$($report.validationState) Actions=$($report.actionCount) FailedBlockers=$($report.failedBlockerCount)"

if ($Strict -and $failedBlockerCount -gt 0) {
  throw "Final owner proof action worklist validation failed with $failedBlockerCount blocker(s)."
}
