[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
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

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-BackfillLane {
  param(
    [string]$Id,
    [string]$Artifact,
    [AllowNull()][object]$Record,
    [string]$StateProperty,
    [string]$DefaultState,
    [string[]]$ReadyStates,
    [string]$RequiredEvidence,
    [bool]$RequireProofReady,
    [string]$ProofReadyProperty = "proofCandidateReady"
  )

  $state = [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue $DefaultState)
  $present = $null -ne $Record
  $stateReady = $present -and ($ReadyStates -contains $state)
  $proofReady = [bool](Get-PropertyOrDefault -Object $Record -Name $ProofReadyProperty -DefaultValue $false)
  $isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isRuntimeExecutionProof" -DefaultValue $false)
  $isPostPublishProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isPostPublishProof" -DefaultValue $false)
  $isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isReleaseCloseProof" -DefaultValue $false)
  $isGitHubActionsProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isGitHubActionsProof" -DefaultValue $false)
  $ready = if ($RequireProofReady) { $stateReady -and $proofReady } else { $stateReady }
  $failedBlockerCount = [int](Get-PropertyOrDefault -Object $Record -Name "failedBlockerCount" -DefaultValue 0)
  $failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $Record -Name "failedActionRequiredCount" -DefaultValue 0)
  $performsPublish = [bool](Get-PropertyOrDefault -Object $Record -Name "performsPublish" -DefaultValue $false)
  $canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $Record -Name "canPromoteRuntimeProof" -DefaultValue $false)
  $canPublishPublicly = [bool](Get-PropertyOrDefault -Object $Record -Name "canPublishPublicly" -DefaultValue $false)
  $canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false)

  [pscustomobject]@{
    id = $Id
    artifact = $Artifact
    present = $present
    state = $state
    readyStates = @($ReadyStates)
    stateReady = $stateReady
    requireProofReady = $RequireProofReady
    proofReadyProperty = $ProofReadyProperty
    proofReady = $proofReady
    ready = $ready
    blocked = -not $ready
    failedBlockerCount = $failedBlockerCount
    failedActionRequiredCount = $failedActionRequiredCount
    requiredEvidence = $RequiredEvidence
    performsPublish = $performsPublish
    canPromoteRuntimeProof = $canPromoteRuntimeProof
    canPublishPublicly = $canPublishPublicly
    canCloseReleaseIssue = $canCloseReleaseIssue
    isRuntimeExecutionProof = $isRuntimeExecutionProof
    isPostPublishProof = $isPostPublishProof
    isReleaseCloseProof = $isReleaseCloseProof
    isGitHubActionsProof = $isGitHubActionsProof
    boundaryOk = (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue -and -not $canPromoteRuntimeProof -and -not $isRuntimeExecutionProof -and -not $isPostPublishProof -and -not $isReleaseCloseProof -and -not $isGitHubActionsProof)
    boundary = "Remote CI and public publish proof backfill lane only; not runtime proof, not post-publish proof, not publish approval, not release close approval, not GitHub Actions proof, and not package push."
  }
}

$githubStatus = Read-JsonOrNull "artifacts\final-release\github-publish-and-ci-status-snapshot.json"
$githubStatusValidation = Read-JsonOrNull "artifacts\final-release\github-publish-and-ci-status-snapshot-validation.json"
$githubActionsRunEvidenceValidation = Read-JsonOrNull "artifacts\final-release\github-actions-run-evidence-import-validation.json"
$ownerPublishResultCandidate = Read-JsonOrNull "artifacts\final-release\owner-public-publish-execution-result-candidate.json"
$ownerPublishResultCandidateValidation = Read-JsonOrNull "artifacts\final-release\owner-public-publish-execution-result-candidate-validation.json"
$publicPackageDownloadProofCandidateValidation = Read-JsonOrNull "artifacts\final-release\public-package-download-proof-candidate-validation.json"
$postPublishCleanConsumerProofResultCandidate = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-candidate.json"
$postPublishCleanConsumerProofResultValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-validation.json"
$finalPrepublishQualityFreezeValidation = Read-JsonOrNull "artifacts\final-release\final-prepublish-quality-freeze-dashboard-validation.json"

$lanes = @(
  New-BackfillLane -Id "github-source-head-status" -Artifact "artifacts/final-release/github-publish-and-ci-status-snapshot-validation.json" -Record $githubStatusValidation -StateProperty "validationState" -DefaultState "missing-github-publish-and-ci-status-snapshot-validation" -ReadyStates @("blocked-github-actions-and-public-publish-proof-required") -RequiredEvidence "Read-only remote HEAD containment snapshot plus explicit missing CI/publish proof classification." -RequireProofReady $false
  New-BackfillLane -Id "github-actions-run-proof" -Artifact "artifacts/final-release/github-actions-run-evidence-import-validation.json" -Record $githubActionsRunEvidenceValidation -StateProperty "validationState" -DefaultState "missing-github-actions-run-evidence-import-validation" -ReadyStates @("github-actions-run-evidence-ready") -RequiredEvidence "Real GitHub Actions workflow run URL, run id, head SHA, conclusion, log hash, and artifact hash validated by Test-GitHubActionsRunEvidenceImport.ps1." -RequireProofReady $false
  New-BackfillLane -Id "owner-public-publish-result" -Artifact "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json" -Record $ownerPublishResultCandidateValidation -StateProperty "validationState" -DefaultState "missing-owner-public-publish-execution-result-candidate-validation" -ReadyStates @("owner-public-publish-execution-result-candidate-ready") -RequiredEvidence "Owner-supplied public publish result with package identity, public URLs, hashes, transcript hashes, reviewer, and authorization linkage." -RequireProofReady $false
  New-BackfillLane -Id "public-package-download-proof" -Artifact "artifacts/final-release/public-package-download-proof-candidate-validation.json" -Record $publicPackageDownloadProofCandidateValidation -StateProperty "validationState" -DefaultState "missing-public-package-download-proof-candidate-validation" -ReadyStates @("public-package-download-proof-candidate-ready") -RequiredEvidence "Validated public package download candidate from Test-PublicPackageDownloadProofCandidate.ps1 with URL/source, SHA256, package identity, and non-local source proof." -RequireProofReady $false
  New-BackfillLane -Id "post-publish-clean-consumer-proof" -Artifact "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json" -Record $postPublishCleanConsumerProofResultValidation -StateProperty "validationState" -DefaultState "missing-post-publish-clean-consumer-proof-result-validation" -ReadyStates @("post-publish-clean-consumer-proof-result-validation-ready") -RequiredEvidence "Repository-external clean consumer restore/build/run logs, host metadata, package source, exit code, and SHA256 values after public publication. Validation-ready alone is not enough; proofCandidateReady must be true." -RequireProofReady $true -ProofReadyProperty "proofCandidateReady"
  New-BackfillLane -Id "final-prepublish-freeze" -Artifact "artifacts/final-release/final-prepublish-quality-freeze-dashboard-validation.json" -Record $finalPrepublishQualityFreezeValidation -StateProperty "validationState" -DefaultState "missing-final-prepublish-quality-freeze-dashboard-validation" -ReadyStates @("blocked-final-prepublish-quality-freeze-owner-action-required") -RequiredEvidence "Local quality freeze remains structurally valid and blocked until Owner execution evidence exists." -RequireProofReady $false
)

$blockedLanes = @($lanes | Where-Object { -not $_.ready })
$boundaryFailures = @($lanes | Where-Object { -not $_.boundaryOk })
$failedBlockers = @($lanes | Where-Object { $_.failedBlockerCount -gt 0 })
$failedActionRequiredCount = ($lanes | Measure-Object -Property failedActionRequiredCount -Sum).Sum
if ($null -eq $failedActionRequiredCount) { $failedActionRequiredCount = 0 }
$sourceHeadPresentOnRemote = [bool](Get-PropertyOrDefault -Object $githubStatus -Name "sourceHeadPresentOnRemote" -DefaultValue $false)
$remoteHeadMatchesCurrentHead = [bool](Get-PropertyOrDefault -Object $githubStatus -Name "remoteHeadMatchesCurrentHead" -DefaultValue $false)

$gateState = if ($blockedLanes.Count -eq 0 -and $boundaryFailures.Count -eq 0 -and $failedBlockers.Count -eq 0) {
  "remote-ci-and-public-publish-proof-backfill-ready-for-owner-review"
} else {
  "blocked-remote-ci-and-public-publish-proof-backfill-required"
}

$record = [pscustomobject]@{
  recordKind = "remote-ci-and-public-publish-proof-backfill-gate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  gateState = $gateState
  laneCount = $lanes.Count
  readyLaneCount = @($lanes | Where-Object { $_.ready }).Count
  blockedLaneCount = $blockedLanes.Count
  boundaryFailureCount = $boundaryFailures.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = [int]$failedActionRequiredCount
  sourceHeadPresentOnRemote = $sourceHeadPresentOnRemote
  remoteHeadMatchesCurrentHead = $remoteHeadMatchesCurrentHead
  readyForOwnerReview = $gateState -eq "remote-ci-and-public-publish-proof-backfill-ready-for-owner-review"
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isGitHubActionsProof = $false
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  lanes = @($lanes)
  sourceArtifacts = @($lanes | ForEach-Object { $_.artifact })
  forbiddenSubstitutes = @("local feed", "ProjectReference", "direct .nupkg", "dashboard", "dry-run", "manual approval", "queued workflow", "missing runner", "sidecar-only", "TensorRtExec report", "local dotnet test", "local package consumer")
  safetyBoundary = "Remote CI and public publish proof backfill gate is a local read-only aggregator only. It does not trigger GitHub Actions, does not run publish workflows, does not push packages, does not close release issues, and is not runtime proof, post-publish proof, release close approval, package publish proof, or GitHub Actions proof."
}

$jsonPath = Join-Path $OutputRoot "remote-ci-and-public-publish-proof-backfill-gate.json"
$markdownPath = Join-Path $OutputRoot "remote-ci-and-public-publish-proof-backfill-gate.md"
$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $lanes | ForEach-Object { "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.state) | $(ConvertTo-MarkdownCell $_.requiredEvidence) |" }
$markdown = @"
# Remote CI And Public Publish Proof Backfill Gate

| Item | Value |
|---|---|
| gateState | ``$($record.gateState)`` |
| laneCount | ``$($record.laneCount)`` |
| readyLaneCount | ``$($record.readyLaneCount)`` |
| blockedLaneCount | ``$($record.blockedLaneCount)`` |
| boundaryFailureCount | ``$($record.boundaryFailureCount)`` |
| failedBlockerCount | ``$($record.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($record.failedActionRequiredCount)`` |
| sourceHeadPresentOnRemote | ``$($record.sourceHeadPresentOnRemote)`` |
| remoteHeadMatchesCurrentHead | ``$($record.remoteHeadMatchesCurrentHead)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Lanes

| ID | Ready | State | Required Evidence |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Remote CI and public publish proof backfill gate written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "GateState=$($record.gateState) Ready=$($record.readyLaneCount) Blocked=$($record.blockedLaneCount)"
