[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

function New-NonProofSignal {
  param(
    [int]$Order,
    [string]$Id,
    [string]$SignalText,
    [string]$WhyNotProof,
    [string]$RequiredRealProof,
    [string]$OwnerAction
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    signalText = $SignalText
    whyNotProof = $WhyNotProof
    requiredRealProof = $RequiredRealProof
    ownerAction = $OwnerAction
    ownerActionRequired = $true
    blocked = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Non-proof runner signal only. It cannot substitute public package publish, external clean consumer, runtime proof, PostPublish proof, or release close proof."
  }
}

$signals = @(
  New-NonProofSignal 1 "queued-workflow" "queued workflow" "A queued workflow proves only that work was requested; it does not prove that public packages were published or consumed." "Real NuGet/GitHub Packages package page/download URLs, hashes, and publish transcript." "Wait for completed public publish execution and import real Owner evidence."
  New-NonProofSignal 2 "manual-approval-only" "manual approval" "Manual approval without public package evidence is an authorization signal, not proof of publication or runtime behavior." "Owner authorization plus real command transcript and public package artifacts." "Capture approval separately, then execute/import real evidence after authorization."
  New-NonProofSignal 3 "missing-self-hosted-runner" "missing self-hosted runner" "A missing runner explains why a workflow did not execute; it is not proof of success." "Completed runner transcript with public publish and external clean consumer evidence." "Keep release blocked until a compatible runner produces real logs."
  New-NonProofSignal 4 "sidecar-only-report" "sidecar-only report" "A sidecar report can support diagnostics, but cannot replace public package or runtime smoke evidence." "PostPublish owner input, strict cross-check, and runtime smoke records." "Bind sidecar reports only as supporting context."
  New-NonProofSignal 5 "tensorrtexec-report-only" "TensorRtExec report" "TensorRtExec output is not a managed package consumer proof and cannot replace runtime smoke proof." "Repository-external clean consumer restore/build/test and runtime smoke evidence." "Keep TensorRtExec as optional sidecar evidence."
  New-NonProofSignal 6 "dry-run-workflow" "dry-run workflow" "Dry-run validates orchestration shape only; it does not prove a public feed changed." "Public package source evidence and downloaded package hashes." "Treat dry-run outputs as planning artifacts only."
  New-NonProofSignal 7 "dashboard-artifact" "dashboard artifact" "A dashboard summarizes status but cannot become the status proof itself." "Owner evidence contract fields populated from real external records." "Use dashboard only to find the next real evidence action."
  New-NonProofSignal 8 "local-feed-consumer" "local feed" "Local feeds, ProjectReference, and direct nupkg references are substitute paths and cannot prove public package consumption." "Clean consumer outside the repository using only public package sources." "Run external clean consumer after real publish."
  New-NonProofSignal 9 "skipped-or-cancelled-job" "skipped or cancelled job" "Skipped/cancelled jobs prove no successful execution path." "Successful completed job logs plus public artifact evidence." "Re-run only after Owner authorization and runner readiness."
)

$artifactRoot = Join-Path $RepositoryRoot "artifacts\github-actions-runs"
$detectedFiles = @()
if (Test-Path -LiteralPath $artifactRoot -PathType Container) {
  $detectedFiles = @(Get-ChildItem -LiteralPath $artifactRoot -Recurse -File -ErrorAction SilentlyContinue)
}

$record = [pscustomobject]@{
  recordKind = "github-actions-runner-non-proof-guard-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  guardState = "github-actions-runner-non-proof-guard-ready-non-proof"
  detectedArtifactRoot = "artifacts/github-actions-runs"
  detectedArtifactFileCount = $detectedFiles.Count
  signalCount = $signals.Count
  nonProofSignals = @($signals)
  ownerActionRequired = $true
  blocked = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "GitHub Actions runner non-proof guard only. It never dispatches workflows, publishes packages/articles, deletes/delists/withdraws/deprecates packages, promotes proof, or closes release issues."
}

$jsonPath = Join-Path $OutputRoot "github-actions-runner-non-proof-guard-pack.json"
$mdPath = Join-Path $OutputRoot "github-actions-runner-non-proof-guard-pack.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# GitHub Actions Runner Non-Proof Guard Pack") | Out-Null
$md.Add("") | Out-Null
$md.Add("- guardState: ``$($record.guardState)``") | Out-Null
$md.Add("- signalCount: ``$($record.signalCount)``") | Out-Null
$md.Add("- detectedArtifactFileCount: ``$($record.detectedArtifactFileCount)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| # | Signal | Why Not Proof | Required Real Proof |") | Out-Null
$md.Add("| ---: | --- | --- | --- |") | Out-Null
foreach ($signal in $signals) {
  $md.Add("| $($signal.order) | ``$($signal.id)`` | $(ConvertTo-MarkdownCell $signal.whyNotProof) | $(ConvertTo-MarkdownCell $signal.requiredRealProof) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "GitHubActionsRunnerNonProofGuardPackState=$($record.guardState) Signals=$($record.signalCount) DetectedFiles=$($record.detectedArtifactFileCount)"
