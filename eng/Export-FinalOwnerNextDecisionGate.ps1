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

function Ensure-Artifact {
  param([string]$RelativePath, [string]$ExportScript)
  $record = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $RelativePath
  if ($null -eq $record -and -not [string]::IsNullOrWhiteSpace($ExportScript)) {
    & (Join-Path $RepositoryRoot "eng\$ExportScript") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
    $record = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $RelativePath
  }
  return $record
}

function New-DecisionOption {
  param(
    [string]$Id,
    [string]$Title,
    [string]$RequiredOwnerAuthorization,
    [string[]]$AllowedCommands,
    [string[]]$StillBlockedCommands,
    [string[]]$RequiredEvidence,
    [string[]]$NextValidators
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    requiredOwnerAuthorization = $RequiredOwnerAuthorization
    allowedCommands = @($AllowedCommands)
    stillBlockedCommands = @($StillBlockedCommands)
    requiredEvidence = @($RequiredEvidence)
    nextValidators = @($NextValidators)
    ownerActionRequired = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

$handoff = Ensure-Artifact "artifacts\final-release\final-owner-handoff-index.json" "Export-FinalOwnerHandoffIndex.ps1"
$summary = Ensure-Artifact "artifacts\final-release\owner-final-authorization-request-summary.json" "Export-OwnerFinalAuthorizationRequestSummary.ps1"
$commandGuard = Ensure-Artifact "artifacts\final-release\owner-authorization-command-guard-pack.json" "Export-OwnerAuthorizationCommandGuardPack.ps1"
$articleReadiness = Ensure-Artifact "artifacts\final-release\public-article-source-patch-apply-readiness-pack.json" "Export-PublicArticleSourcePatchApplyReadinessPack.ps1"
$runnerGuard = Ensure-Artifact "artifacts\final-release\github-actions-runner-non-proof-guard-pack.json" "Export-GitHubActionsRunnerNonProofGuardPack.ps1"

$blockedCommands = @(Get-PropertyOrDefault -Object $summary -Name "nowBlockedCommands" -DefaultValue @())
$readonlyCommands = @(Get-PropertyOrDefault -Object $summary -Name "allowedReadonlyNonProofCommands" -DefaultValue @())
$handoffSections = @(Get-PropertyOrDefault -Object $handoff -Name "sections" -DefaultValue @())
$articleReadinessCount = [int](Get-PropertyOrDefault -Object $articleReadiness -Name "readinessItemCount" -DefaultValue 0)
$blockedSectionCount = [int](Get-PropertyOrDefault -Object $handoff -Name "blockedSectionCount" -DefaultValue 0)
$runnerSignals = @(Get-PropertyOrDefault -Object $runnerGuard -Name "nonProofSignals" -DefaultValue @())

$decisionOptions = @(
  New-DecisionOption `
    -Id "authorize-real-public-publish" `
    -Title "Owner authorizes real public package publish and post-publish proof collection" `
    -RequiredOwnerAuthorization "Explicitly authorize NuGet.org/GitHub Packages publish target, package versions, credentials scope, transcript capture path, rollback/no-rollback policy, and external clean-consumer proof collection." `
    -AllowedCommands @("dotnet-nuget-push-after-owner-authorization", "github-packages-push-after-owner-authorization", "post-publish-clean-consumer-proof-collection", "strict-release-close-after-real-proof") `
    -StillBlockedCommands @("destructive-package-delete", "delist-without-owner-approval", "article-publish-before-proof-gates") `
    -RequiredEvidence @("public package URL", "public package hash", "publish transcript", "repository-external public-source consumer result", "PostPublish strict cross-check", "release close strict evidence closure") `
    -NextValidators @("Test-OwnerAuthorizationCommandGuardPack.ps1 -Strict", "Test-PostPublishStrictCrossCheckPack.ps1 -Strict", "Test-PostPublishArticleProofGate.ps1 -Strict", "Test-ReleaseCloseStrictEvidenceClosure.ps1 -Strict")

  New-DecisionOption `
    -Id "authorize-article-source-patch-only" `
    -Title "Owner authorizes article source claim downgrades without package publish" `
    -RequiredOwnerAuthorization "Explicitly authorize applying the mapped public-article source patch proposals while keeping package publish, workflow dispatch, proof promotion, and release close blocked." `
    -AllowedCommands @("apply-public-article-source-patch-proposals-after-owner-approval", "article-claim-boundary-rescan", "readonly-dashboard-validation") `
    -StillBlockedCommands @("dotnet-nuget-push", "github-packages-push", "workflow-dispatch-publish", "article-publish", "release-close") `
    -RequiredEvidence @("Owner article patch approval", "public article patch readiness mapping", "post-patch blocked-claim scan", "no public publish claim without real proof") `
    -NextValidators @("Test-PublicArticleSourcePatchApplyReadinessPack.ps1 -Strict", "Test-PublicArticleBlockedClaimOwnerReviewList.ps1 -Strict", "Test-OwnerAuthorizationCommandGuardPack.ps1 -Strict")

  New-DecisionOption `
    -Id "keep-blocked-wait-for-owner" `
    -Title "No authorization yet; keep final release blocked" `
    -RequiredOwnerAuthorization "No new authorization. Keep using only readonly non-proof dashboards and guard validations." `
    -AllowedCommands @("Export-FinalOwnerNextDecisionGate.ps1", "Test-FinalOwnerNextDecisionGate.ps1 -Strict", "Export-FinalOwnerHandoffIndex.ps1", "Test-FinalOwnerHandoffIndex.ps1 -Strict") `
    -StillBlockedCommands @("dotnet-nuget-push", "github-packages-push", "workflow-dispatch-publish", "article-publish", "release-close", "proof-promotion") `
    -RequiredEvidence @("none-new; Owner decision is still required") `
    -NextValidators @("Test-FinalOwnerNextDecisionGate.ps1 -Strict", "Test-FinalOwnerHandoffIndex.ps1 -Strict", "Test-OwnerFinalAuthorizationRequestSummary.ps1 -Strict")
)

$record = [pscustomobject]@{
  recordKind = "final-owner-next-decision-gate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  gateState = "blocked-final-owner-next-decision-required"
  recommendedDefault = "keep-blocked-wait-for-owner"
  ownerMustChooseOneOf = @($decisionOptions | ForEach-Object { $_.id })
  decisionOptionCount = [int]$decisionOptions.Count
  blockedCommandCount = [int]$blockedCommands.Count
  allowedReadonlyCommandCount = [int]$readonlyCommands.Count
  handoffSectionCount = [int]$handoffSections.Count
  blockedHandoffSectionCount = $blockedSectionCount
  articleReadinessItemCount = $articleReadinessCount
  githubActionsRunnerNonProofSignalCount = [int]$runnerSignals.Count
  decisionOptions = @($decisionOptions)
  forbiddenSubstitutes = @("local feed", "ProjectReference", "direct .nupkg", "dashboard", "dry-run", "manual approval only", "queued workflow", "missing self-hosted runner", "sidecar-only", "TensorRtExec report")
  ownerActionRequired = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner next decision gate only. It summarizes allowed next paths and never publishes packages/articles, dispatches workflows, applies article source patches, promotes proof, or closes release issues."
}

$jsonPath = Join-Path $OutputRoot "final-owner-next-decision-gate.json"
$mdPath = Join-Path $OutputRoot "final-owner-next-decision-gate.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Final Owner Next Decision Gate") | Out-Null
$md.Add("") | Out-Null
$md.Add("- gateState: ``$($record.gateState)``") | Out-Null
$md.Add("- recommendedDefault: ``$($record.recommendedDefault)``") | Out-Null
$md.Add("- decisionOptionCount: ``$($record.decisionOptionCount)``") | Out-Null
$md.Add("- blockedCommandCount: ``$($record.blockedCommandCount)``") | Out-Null
$md.Add("- blockedHandoffSectionCount: ``$($record.blockedHandoffSectionCount)``") | Out-Null
$md.Add("- articleReadinessItemCount: ``$($record.articleReadinessItemCount)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("## Owner Decision Options") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Option | Meaning | Still Blocked | Next Validators |") | Out-Null
$md.Add("| --- | --- | --- | --- |") | Out-Null
foreach ($option in $decisionOptions) {
  $md.Add("| ``$($option.id)`` | $(ConvertTo-MarkdownCell $option.title) | $(ConvertTo-MarkdownCell (($option.stillBlockedCommands) -join ', ')) | $(ConvertTo-MarkdownCell (($option.nextValidators) -join '; ')) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Forbidden Proof Substitutes") | Out-Null
$md.Add("") | Out-Null
foreach ($item in $record.forbiddenSubstitutes) {
  $md.Add("- ``$item``") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md

Write-Host "FinalOwnerNextDecisionGateState=$($record.gateState) Options=$($record.decisionOptionCount) BlockedCommands=$($record.blockedCommandCount) BlockedSections=$($record.blockedHandoffSectionCount)"
