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

function Get-StateValue {
  param([object]$Record, [string[]]$PropertyNames, [string]$DefaultState)
  foreach ($name in $PropertyNames) {
    $value = Get-PropertyOrDefault -Object $Record -Name $name -DefaultValue $null
    if ($null -ne $value -and -not [string]::IsNullOrWhiteSpace([string]$value)) {
      return [string]$value
    }
  }
  return $DefaultState
}

function New-HandoffSection {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string]$Artifact,
    [string]$CurrentState,
    [string]$Purpose,
    [string]$OwnerNextAction,
    [string]$ValidatorCommand
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    artifact = $Artifact
    currentState = $CurrentState
    purpose = $Purpose
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
    boundary = "Final owner handoff index section only. It links to evidence artifacts and never substitutes real proof."
  }
}

function Ensure-Artifact {
  param([string]$RelativePath, [string]$ExportScript)
  $record = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $RelativePath
  if ($null -eq $record -and -not [string]::IsNullOrWhiteSpace($ExportScript)) {
    & (Join-Path $RepositoryRoot "eng\$ExportScript") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
    $record = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $RelativePath
  }
  return $record
}

$dashboard = Ensure-Artifact "artifacts\final-release\final-owner-publish-and-article-action-dashboard.json" "Export-FinalOwnerPublishAndArticleActionDashboard.ps1"
$commandGuard = Ensure-Artifact "artifacts\final-release\owner-authorization-command-guard-pack.json" "Export-OwnerAuthorizationCommandGuardPack.ps1"
$authorizationSummary = Ensure-Artifact "artifacts\final-release\owner-final-authorization-request-summary.json" "Export-OwnerFinalAuthorizationRequestSummary.ps1"
$repairPack = Ensure-Artifact "artifacts\final-release\owner-missing-real-publish-evidence-repair-pack.json" "Export-OwnerMissingRealPublishEvidenceRepairPack.ps1"
$articleReadiness = Ensure-Artifact "artifacts\final-release\public-article-source-patch-apply-readiness-pack.json" "Export-PublicArticleSourcePatchApplyReadinessPack.ps1"
$crossCheck = Ensure-Artifact "artifacts\final-release\post-publish-strict-cross-check-pack.json" "Export-PostPublishStrictCrossCheckPack.ps1"
$articleProofGate = Ensure-Artifact "artifacts\final-release\post-publish-article-proof-gate.json" "Export-PostPublishArticleProofGate.ps1"
$runnerGuard = Ensure-Artifact "artifacts\final-release\github-actions-runner-non-proof-guard-pack.json" "Export-GitHubActionsRunnerNonProofGuardPack.ps1"
$releaseClose = Ensure-Artifact "artifacts\final-release\release-close-strict-evidence-closure.json" "Export-ReleaseCloseStrictEvidenceClosure.ps1"

$sections = @(
  New-HandoffSection 1 "owner-final-authorization-request-summary" "Owner final authorization request summary" "artifacts/final-release/owner-final-authorization-request-summary.json" (Get-StateValue $authorizationSummary @("requestState") "missing-owner-final-authorization-request-summary") "One-screen Owner decision point for real publish vs continued block." "Authorize real public publish or keep publish/article/release-close commands blocked." "Export-OwnerFinalAuthorizationRequestSummary.ps1; Test-OwnerFinalAuthorizationRequestSummary.ps1 -Strict"
  New-HandoffSection 2 "owner-authorization-command-guard" "Owner authorization command guard" "artifacts/final-release/owner-authorization-command-guard-pack.json" (Get-StateValue $commandGuard @("guardState") "missing-owner-authorization-command-guard-pack") "Classifies blocked publish/workflow/release/article commands and readonly non-proof commands." "Use it before running any publish, workflow dispatch, article publish, or release close command." "Export-OwnerAuthorizationCommandGuardPack.ps1; Test-OwnerAuthorizationCommandGuardPack.ps1 -Strict"
  New-HandoffSection 3 "final-owner-action-dashboard" "Final Owner action dashboard" "artifacts/final-release/final-owner-publish-and-article-action-dashboard.json" (Get-StateValue $dashboard @("dashboardState") "missing-final-owner-action-dashboard") "Aggregates current final release/action gates." "Use dashboard to navigate remaining Owner actions; do not treat it as proof." "Export-FinalOwnerPublishAndArticleActionDashboard.ps1; Test-FinalOwnerPublishAndArticleActionDashboard.ps1 -Strict"
  New-HandoffSection 4 "owner-missing-real-evidence-repair-pack" "Owner missing real evidence repair pack" "artifacts/final-release/owner-missing-real-publish-evidence-repair-pack.json" (Get-StateValue $repairPack @("repairState") "missing-owner-missing-real-evidence-repair-pack") "Lists missing Owner public publish/PostPublish/release close evidence." "Fill missing fields from real public publish and external clean consumer records." "Export-OwnerMissingRealPublishEvidenceRepairPack.ps1; Test-OwnerMissingRealPublishEvidenceRepairPack.ps1 -Strict"
  New-HandoffSection 5 "article-source-patch-readiness" "Article source patch apply readiness" "artifacts/final-release/public-article-source-patch-apply-readiness-pack.json" (Get-StateValue $articleReadiness @("readinessPackState") "missing-public-article-source-patch-apply-readiness-pack") "Shows whether article patch proposals still map to current source lines." "Only apply article patches after explicit Owner approval for source edits." "Export-PublicArticleSourcePatchApplyReadinessPack.ps1; Test-PublicArticleSourcePatchApplyReadinessPack.ps1 -Strict"
  New-HandoffSection 6 "post-publish-strict-cross-check" "PostPublish strict cross-check" "artifacts/final-release/post-publish-strict-cross-check-pack.json" (Get-StateValue $crossCheck @("crossCheckState","packState","recordState") "missing-post-publish-strict-cross-check-pack") "Validates public URLs, hashes, and no-substitute clean consumer evidence." "Run after real Owner evidence import; failures remain blockers." "Export-PostPublishStrictCrossCheckPack.ps1; Test-PostPublishStrictCrossCheckPack.ps1 -Strict"
  New-HandoffSection 7 "post-publish-article-proof-gate" "PostPublish article proof gate" "artifacts/final-release/post-publish-article-proof-gate.json" (Get-StateValue $articleProofGate @("gateState","proofGateState","recordState") "missing-post-publish-article-proof-gate") "Blocks article claims until proof gates clear." "Publish only claims backed by real proof." "Export-PostPublishArticleProofGate.ps1; Test-PostPublishArticleProofGate.ps1 -Strict"
  New-HandoffSection 8 "github-actions-runner-non-proof-guard" "GitHub Actions runner non-proof guard" "artifacts/final-release/github-actions-runner-non-proof-guard-pack.json" (Get-StateValue $runnerGuard @("guardState") "missing-github-actions-runner-non-proof-guard-pack") "Prevents queued workflow/manual approval/missing runner/sidecar report from being promoted as proof." "Use workflow artifacts only as context unless real public evidence is present." "Export-GitHubActionsRunnerNonProofGuardPack.ps1; Test-GitHubActionsRunnerNonProofGuardPack.ps1 -Strict"
  New-HandoffSection 9 "release-close-strict-closure" "Release close strict closure" "artifacts/final-release/release-close-strict-evidence-closure.json" (Get-StateValue $releaseClose @("closureState","recordState","releaseCloseState") "missing-release-close-strict-evidence-closure") "Final release close gate for real publish, PostPublish, rollback, and no-substitute evidence." "Close release only after strict closure accepts real proof." "Export-ReleaseCloseStrictEvidenceClosure.ps1; Test-ReleaseCloseStrictEvidenceClosure.ps1 -Strict"
)

$record = [pscustomobject]@{
  recordKind = "final-owner-handoff-index"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  handoffState = "blocked-final-owner-handoff-index-awaiting-owner-authorization"
  sectionCount = $sections.Count
  blockedSectionCount = @($sections | Where-Object { [bool]$_.blocked }).Count
  sections = @($sections)
  ownerActionRequired = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner handoff index only. It links guard, evidence, article, PostPublish, runner, and release-close artifacts and never publishes packages/articles, dispatches workflows, promotes proof, or closes release issues."
}

$jsonPath = Join-Path $OutputRoot "final-owner-handoff-index.json"
$mdPath = Join-Path $OutputRoot "final-owner-handoff-index.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Final Owner Handoff Index") | Out-Null
$md.Add("") | Out-Null
$md.Add("- handoffState: ``$($record.handoffState)``") | Out-Null
$md.Add("- sectionCount: ``$($record.sectionCount)``") | Out-Null
$md.Add("- blockedSectionCount: ``$($record.blockedSectionCount)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| # | Section | State | Artifact | Owner Next Action |") | Out-Null
$md.Add("| ---: | --- | --- | --- | --- |") | Out-Null
foreach ($section in $sections) {
  $md.Add("| $($section.order) | ``$($section.id)`` | ``$(ConvertTo-MarkdownCell $section.currentState)`` | ``$($section.artifact)`` | $(ConvertTo-MarkdownCell $section.ownerNextAction) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "FinalOwnerHandoffIndexState=$($record.handoffState) Sections=$($record.sectionCount) Blocked=$($record.blockedSectionCount)"
