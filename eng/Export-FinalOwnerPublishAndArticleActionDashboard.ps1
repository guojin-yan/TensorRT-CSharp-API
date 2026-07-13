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

function New-ActionGate {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string]$Artifact,
    [string]$CurrentState,
    [string]$OwnerNextAction,
    [string]$ValidatorCommand
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    artifact = $Artifact
    currentState = $CurrentState
    gateState = "blocked-owner-action-required"
    ownerNextAction = $OwnerNextAction
    validatorCommand = $ValidatorCommand
    ownerActionRequired = $true
    blocked = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Action dashboard gate only. It points to the next Owner action and never substitutes proof, publish approval, or release close approval."
  }
}

function Get-State {
  param([string]$RelativePath, [string]$PropertyName, [string]$DefaultState)
  $record = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $RelativePath
  return [string](Get-PropertyOrDefault -Object $record -Name $PropertyName -DefaultValue $DefaultState)
}

$gates = @(
  New-ActionGate 1 "final-readonly-audit" "Final readonly publish audit" "artifacts/final-release/final-readonly-publish-audit-pack-validation.json" (Get-State "artifacts\final-release\final-readonly-publish-audit-pack-validation.json" "validationState" "missing-final-readonly-publish-audit-pack-validation") "Keep the readonly audit green before any Owner execution; do not treat it as proof." "Export-FinalReadonlyPublishAuditPack.ps1; Test-FinalReadonlyPublishAuditPack.ps1 -Strict"
  New-ActionGate 2 "owner-one-screen-manual" "Owner one-screen execution manual" "artifacts/final-release/final-owner-one-screen-execution-manual-validation.json" (Get-State "artifacts\final-release\final-owner-one-screen-execution-manual-validation.json" "validationState" "missing-final-owner-one-screen-execution-manual-validation") "Use the manual as Owner-only handoff; do not execute publish without explicit Owner authorization." "Export-FinalOwnerOneScreenExecutionManual.ps1; Test-FinalOwnerOneScreenExecutionManual.ps1 -Strict"
  New-ActionGate 3 "publish-replay-checklist" "Owner publish replay checklist" "artifacts/final-release/final-owner-publish-execution-replay-checklist-pack-validation.json" (Get-State "artifacts\final-release\final-owner-publish-execution-replay-checklist-pack-validation.json" "validationState" "missing-final-owner-publish-execution-replay-checklist-pack-validation") "Follow the six replay steps and capture command/transcript/hash evidence after Owner authorization." "Export-FinalOwnerPublishExecutionReplayChecklistPack.ps1; Test-FinalOwnerPublishExecutionReplayChecklistPack.ps1 -Strict"
  New-ActionGate 4 "evidence-import-runbook" "Owner publish evidence import runbook" "artifacts/final-release/final-owner-publish-evidence-import-runbook-validation.json" (Get-State "artifacts\final-release\final-owner-publish-evidence-import-runbook-validation.json" "validationState" "missing-final-owner-publish-evidence-import-runbook-validation") "Fill real 178-field Owner public publish evidence in source order; do not invent fields." "Export-FinalOwnerPublishEvidenceImportRunbook.ps1; Test-FinalOwnerPublishEvidenceImportRunbook.ps1 -Strict"
  New-ActionGate 5 "owner-intake-dry-run" "Owner real publish evidence intake dry-run" "artifacts/final-release/owner-real-publish-evidence-intake-dry-run-pack-validation.json" (Get-State "artifacts\final-release\owner-real-publish-evidence-intake-dry-run-pack-validation.json" "validationState" "missing-owner-real-publish-evidence-intake-dry-run-pack-validation") "Use the dry-run intake groups only as required evidence checklist; Owner must still supply real evidence." "Export-OwnerRealPublishEvidenceIntakeDryRunPack.ps1; Test-OwnerRealPublishEvidenceIntakeDryRunPack.ps1 -Strict"
  New-ActionGate 6 "post-publish-cross-check" "PostPublish strict cross-check" "artifacts/final-release/post-publish-strict-cross-check-pack-validation.json" (Get-State "artifacts\final-release\post-publish-strict-cross-check-pack-validation.json" "validationState" "missing-post-publish-strict-cross-check-pack-validation") "Replace placeholders with public package URLs, hashes, and repository-external clean consumer evidence." "Export-PostPublishStrictCrossCheckPack.ps1; Test-PostPublishStrictCrossCheckPack.ps1 -Strict"
  New-ActionGate 7 "article-readiness-matrix" "Public article readiness matrix" "artifacts/final-release/public-article-readiness-matrix-validation.json" (Get-State "artifacts\final-release\public-article-readiness-matrix-validation.json" "validationState" "missing-public-article-readiness-matrix-validation") "Keep public articles blocked from publish/proof claims until Owner proof is accepted." "Export-PublicArticleReadinessMatrix.ps1; Test-PublicArticleReadinessMatrix.ps1 -Strict"
  New-ActionGate 8 "post-publish-article-proof-gate" "PostPublish article proof gate" "artifacts/final-release/post-publish-article-proof-gate-validation.json" (Get-State "artifacts\final-release\post-publish-article-proof-gate-validation.json" "validationState" "missing-post-publish-article-proof-gate-validation") "Block NuGet/GitHub/clean-consumer/release-closed claims until strict post-publish proof exists." "Export-PostPublishArticleProofGate.ps1; Test-PostPublishArticleProofGate.ps1 -Strict"
  New-ActionGate 9 "release-close-strict-closure" "Release close strict evidence closure" "artifacts/final-release/release-close-strict-evidence-closure-validation.json" (Get-State "artifacts\final-release\release-close-strict-evidence-closure-validation.json" "validationState" "missing-release-close-strict-evidence-closure-validation") "Close release only after real public publish, PostPublish, rollback review, close decision, and strict acceptance pass." "Export-ReleaseCloseStrictEvidenceClosure.ps1; Test-ReleaseCloseStrictEvidenceClosure.ps1 -Strict"
)

$forbidden = @("local feed", "ProjectReference", "direct .nupkg", "dashboard", "dry-run", "manual approval", "queued GitHub Actions run", "missing self-hosted runner", "sidecar-only", "TensorRtExec report")
$record = [pscustomobject]@{
  recordKind = "final-owner-publish-and-article-action-dashboard"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  dashboardState = "blocked-final-owner-publish-and-article-action-dashboard-owner-action-required"
  gateCount = $gates.Count
  blockedGateCount = @($gates | Where-Object { [bool]$_.blocked }).Count
  gates = @($gates)
  forbiddenNonProofSubstitutes = @($forbidden)
  ownerActionRequired = $true
  performsPublish = $false
  performsNuGetPublish = $false
  performsGitHubPackagesPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner publish and article action dashboard only. It is a blocked Owner handoff view and never publishes packages or articles, never triggers workflows, never closes release issues, and never accepts local feed, ProjectReference, direct nupkg, dashboard, dry-run, manual approval, queued workflow, missing runner, sidecar-only, or TensorRtExec report as proof."
}

$jsonPath = Join-Path $OutputRoot "final-owner-publish-and-article-action-dashboard.json"
$mdPath = Join-Path $OutputRoot "final-owner-publish-and-article-action-dashboard.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Final Owner Publish And Article Action Dashboard") | Out-Null
$md.Add("") | Out-Null
$md.Add("- dashboardState: ``$($record.dashboardState)``") | Out-Null
$md.Add("- gateCount: ``$($record.gateCount)``") | Out-Null
$md.Add("- blockedGateCount: ``$($record.blockedGateCount)``") | Out-Null
$md.Add("- performsPublish: ``False``") | Out-Null
$md.Add("- canCloseReleaseIssue: ``False``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| # | Gate | Current State | Owner Next Action |") | Out-Null
$md.Add("| --- | --- | --- | --- |") | Out-Null
foreach ($gate in $gates) {
  $md.Add("| $($gate.order) | ``$($gate.id)`` | ``$(ConvertTo-MarkdownCell $gate.currentState)`` | $(ConvertTo-MarkdownCell $gate.ownerNextAction) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "FinalOwnerPublishAndArticleActionDashboardState=$($record.dashboardState) Gates=$($record.gateCount) Blocked=$($record.blockedGateCount)"
