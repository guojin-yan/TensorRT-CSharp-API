[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\remote-ci-and-public-publish-proof-backfill-gate.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-RemoteCiAndPublicPublishProofBackfillGate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @((Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @()))
$laneIds = @($lanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$requiredLaneIds = @(
  "github-source-head-status",
  "github-actions-run-proof",
  "owner-public-publish-result",
  "public-package-download-proof",
  "post-publish-clean-consumer-proof",
  "final-prepublish-freeze"
)
$missingLaneIds = @($requiredLaneIds | Where-Object { $laneIds -notcontains $_ })
$githubActionsRunProofRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "githubActionsRunProofRequiredFieldCount" -DefaultValue 0)
$githubActionsRunProofRejectedStateCount = [int](Get-PropertyOrDefault -Object $record -Name "githubActionsRunProofRejectedStateCount" -DefaultValue 0)
$githubActionsRunProofRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "githubActionsRunProofRejectedSubstituteCount" -DefaultValue 0)
$postPublishLane = $lanes | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "post-publish-clean-consumer-proof" } | Select-Object -First 1
$postPublishLaneRequiresRealProof = $null -ne $postPublishLane -and
  [bool](Get-PropertyOrDefault -Object $postPublishLane -Name "requireProofReady" -DefaultValue $false) -and
  [string](Get-PropertyOrDefault -Object $postPublishLane -Name "proofReadyProperty" -DefaultValue "") -eq "proofCandidateReady" -and
  [bool](Get-PropertyOrDefault -Object $postPublishLane -Name "stateReady" -DefaultValue $false) -and
  -not [bool](Get-PropertyOrDefault -Object $postPublishLane -Name "proofReady" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $postPublishLane -Name "ready" -DefaultValue $true)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "remote-ci-and-public-publish-proof-backfill-gate") -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "state" -Passed (@("blocked-remote-ci-and-public-publish-proof-backfill-required", "remote-ci-and-public-publish-proof-backfill-ready-for-owner-review") -contains [string](Get-PropertyOrDefault -Object $record -Name "gateState" -DefaultValue "")) -Severity "blocker" -Detail "gateState must be blocked or owner-review ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-lanes" -Passed ($missingLaneIds.Count -eq 0) -Severity "blocker" -Detail ("Missing lanes: " + ($missingLaneIds -join ", ")))) | Out-Null
$items.Add((New-ValidationItem -Id "github-actions-run-proof-contract-counts" -Passed ($githubActionsRunProofRequiredFieldCount -eq 13 -and $githubActionsRunProofRejectedStateCount -eq 12 -and $githubActionsRunProofRejectedSubstituteCount -eq 10) -Severity "blocker" -Detail "Remote proof backfill gate must carry GitHub Actions run proof contract counts from Test-GitHubActionsRunEvidenceImport.ps1.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-proof-lane-requires-real-proof" -Passed $postPublishLaneRequiresRealProof -Severity "blocker" -Detail "Post-publish clean consumer lane must not become ready from validation-ready alone; proofCandidateReady must be true.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-until-real-proof" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "readyForOwnerReview" -DefaultValue $true)) -Severity "action-required" -Detail "Gate must remain blocked until real GitHub Actions, publish result, public download, and post-publish clean consumer proof exist.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isGitHubActionsProof" -DefaultValue $true)) -Severity "blocker" -Detail "Gate must not publish, close, promote proof, or claim GitHub Actions proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary-failures" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "boundaryFailureCount" -DefaultValue 999) -eq 0) -Severity "blocker" -Detail "All lanes must keep non-proof/no-side-effect flags false.")) | Out-Null

foreach ($marker in @("local feed", "ProjectReference", "direct .nupkg", "dashboard", "dry-run", "queued workflow", "missing runner", "TensorRtExec report", "local package consumer")) {
  $raw = $record | ConvertTo-Json -Depth 16
  $items.Add((New-ValidationItem -Id "forbidden-substitute-$($marker.Replace(' ', '-').Replace('.', 'dot'))" -Passed ($raw.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Forbidden substitute marker must be documented: $marker")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-remote-ci-and-public-publish-proof-backfill-gate" } elseif ($failedActionRequired.Count -gt 0) { "blocked-remote-ci-and-public-publish-proof-backfill-required" } else { [string](Get-PropertyOrDefault -Object $record -Name "gateState" -DefaultValue "blocked-remote-ci-and-public-publish-proof-backfill-required") }

$validation = [pscustomobject]@{
  recordKind = "remote-ci-and-public-publish-proof-backfill-gate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  laneCount = $lanes.Count
  blockedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0)
  boundaryFailureCount = [int](Get-PropertyOrDefault -Object $record -Name "boundaryFailureCount" -DefaultValue 0)
  githubActionsRunProofRequiredFieldCount = $githubActionsRunProofRequiredFieldCount
  githubActionsRunProofRejectedStateCount = $githubActionsRunProofRejectedStateCount
  githubActionsRunProofRejectedSubstituteCount = $githubActionsRunProofRejectedSubstituteCount
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  sourceHeadPresentOnRemote = [bool](Get-PropertyOrDefault -Object $record -Name "sourceHeadPresentOnRemote" -DefaultValue $false)
  remoteHeadMatchesCurrentHead = [bool](Get-PropertyOrDefault -Object $record -Name "remoteHeadMatchesCurrentHead" -DefaultValue $false)
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
  safetyBoundary = "Remote CI/public publish proof backfill validation is read-only and side-effect free; not runtime proof, not post-publish proof, not package publish proof, not GitHub Actions proof, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "remote-ci-and-public-publish-proof-backfill-gate-validation.json"
$markdownPath = Join-Path $OutputRoot "remote-ci-and-public-publish-proof-backfill-gate-validation.md"
$validation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$rows = $validation.validationItems | ForEach-Object { "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.passed)`` | $(ConvertTo-MarkdownCell $_.severity) | $(ConvertTo-MarkdownCell $_.detail) |" }
$markdown = @"
# Remote CI And Public Publish Proof Backfill Gate Validation

| Item | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| laneCount | ``$($validation.laneCount)`` |
| blockedLaneCount | ``$($validation.blockedLaneCount)`` |
| boundaryFailureCount | ``$($validation.boundaryFailureCount)`` |
| githubActionsRunProofRequiredFieldCount | ``$($validation.githubActionsRunProofRequiredFieldCount)`` |
| githubActionsRunProofRejectedStateCount | ``$($validation.githubActionsRunProofRejectedStateCount)`` |
| githubActionsRunProofRejectedSubstituteCount | ``$($validation.githubActionsRunProofRejectedSubstituteCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| sourceHeadPresentOnRemote | ``$($validation.sourceHeadPresentOnRemote)`` |
| remoteHeadMatchesCurrentHead | ``$($validation.remoteHeadMatchesCurrentHead)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) { throw "Remote CI and public publish proof backfill gate validation failed with $($failedBlockers.Count) blocker(s)." }

Write-Host "Remote CI and public publish proof backfill gate validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"
