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

function Add-MissingAction {
  param(
    [System.Collections.Generic.List[object]]$List,
    [int]$Order,
    [string]$Id,
    [string]$Lane,
    [string]$OwnerAction,
    [string]$ValidatorCommand,
    [string]$RequiredEvidence,
    [string]$ForbiddenSubstitutes
  )
  $List.Add([pscustomobject]@{
    order = $Order
    id = $Id
    lane = $Lane
    ownerAction = $OwnerAction
    validatorCommand = $ValidatorCommand
    requiredEvidence = $RequiredEvidence
    forbiddenSubstitutes = $ForbiddenSubstitutes
    ownerActionRequired = $true
    blocked = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }) | Out-Null
}

$dashboard = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-owner-publish-and-article-action-dashboard.json"
$repair = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-missing-real-publish-evidence-repair-pack.json"
$crossCheck = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-strict-cross-check-pack.json"
$articleGate = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-article-proof-gate.json"
$review = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\public-article-blocked-claim-owner-review-list.json"

if ($null -eq $dashboard) { & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerPublishAndArticleActionDashboard.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot; $dashboard = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-owner-publish-and-article-action-dashboard.json" }
if ($null -eq $repair) { & (Join-Path $RepositoryRoot "eng\Export-OwnerMissingRealPublishEvidenceRepairPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot; $repair = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-missing-real-publish-evidence-repair-pack.json" }
if ($null -eq $crossCheck) { & (Join-Path $RepositoryRoot "eng\Export-PostPublishStrictCrossCheckPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot; $crossCheck = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-strict-cross-check-pack.json" }
if ($null -eq $articleGate) { & (Join-Path $RepositoryRoot "eng\Export-PostPublishArticleProofGate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot; $articleGate = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-article-proof-gate.json" }
if ($null -eq $review) { & (Join-Path $RepositoryRoot "eng\Export-PublicArticleBlockedClaimOwnerReviewList.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot; $review = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\public-article-blocked-claim-owner-review-list.json" }

$actions = New-Object System.Collections.Generic.List[object]
Add-MissingAction $actions 1 "owner-authorize-real-publish" "owner-decision" "Owner must explicitly authorize public NuGet.org/GitHub Packages publish scope before any publish command runs." "Export-FinalOwnerOneScreenExecutionManual.ps1; Test-FinalOwnerOneScreenExecutionManual.ps1 -Strict" "Owner authorization id, scope, timestamp, reviewer, transcript hash." "manual approval only; queued workflow; dry-run; dashboard"
Add-MissingAction $actions 2 "run-real-public-publish" "public-publish" "After authorization, execute real public publish and capture package page/download URLs, command transcript, package hashes, and source commit." "Export-OwnerPublicPublishExecutionResultInputTemplate.ps1; Test-OwnerPublicPublishExecutionResultPreflight.ps1 -Strict" "NuGet.org/GitHub Packages page URLs, public download URLs, SHA256 hashes, command transcript." "local feed; direct nupkg; ProjectReference; build-only"
Add-MissingAction $actions 3 "run-external-clean-consumer" "post-publish-clean-consumer" "Create a repository-external clean consumer and restore/build/test only from public package sources." "Export-PostPublishVerificationOwnerInputTemplate.ps1; Test-PostPublishVerificationOwnerInput.ps1 -Strict" "External consumer root, restore/build/test logs, no local feed/direct nupkg/ProjectReference confirmations." "repository-local consumer; local package source; direct nupkg"
Add-MissingAction $actions 4 "import-real-evidence" "evidence-import" "Import the 178-field Owner public publish result and PostPublish owner input from real external evidence." "Export-FinalOwnerPublishEvidenceImportRunbook.ps1; Test-FinalOwnerPublishEvidenceImportRunbook.ps1 -Strict" "Complete Owner public publish result and PostPublish owner input records." "template placeholders; generated dashboard; dry-run pack"
Add-MissingAction $actions 5 "pass-post-publish-cross-check" "post-publish-cross-check" "Align Owner input, projected record, URLs, hashes, and no-substitute flags until strict cross-check passes." "Export-PostPublishStrictCrossCheckPack.ps1; Test-PostPublishStrictCrossCheckPack.ps1 -Strict" "Matching package URL/hash fields and external clean consumer proof." "inconsistent URLs; missing hashes; local source"
Add-MissingAction $actions 6 "clear-article-proof-gate" "article-proof" "Only unlock public article claims whose NuGet, clean consumer, runtime, and release-close proof gates have passed." "Export-PostPublishArticleProofGate.ps1; Test-PostPublishArticleProofGate.ps1 -Strict" "Passed article proof gates plus reviewed blocked-claim list." "draft claim; roadmap statement; TensorRtExec sidecar-only"
Add-MissingAction $actions 7 "close-release-strictly" "release-close" "Close the release only after strict closure accepts real publish, PostPublish, rollback, approval, and forbidden-substitute evidence." "Export-ReleaseCloseStrictEvidenceClosure.ps1; Test-ReleaseCloseStrictEvidenceClosure.ps1 -Strict" "Owner close decision, rollback/no-rollback approval, final acceptance hashes." "manual close only; release note only; missing self-hosted runner"

$sourceStates = [pscustomobject]@{
  dashboardState = [string](Get-PropertyOrDefault -Object $dashboard -Name "dashboardState" -DefaultValue "missing-dashboard")
  dashboardBlockedGateCount = [int](Get-PropertyOrDefault -Object $dashboard -Name "blockedGateCount" -DefaultValue 0)
  repairState = [string](Get-PropertyOrDefault -Object $repair -Name "repairState" -DefaultValue "missing-repair")
  ownerPublishMissingFieldCount = [int](Get-PropertyOrDefault -Object $repair -Name "ownerPublishMissingFieldCount" -DefaultValue 0)
  postPublishMissingFieldCount = [int](Get-PropertyOrDefault -Object $repair -Name "postPublishMissingFieldCount" -DefaultValue 0)
  failedPostPublishCrossCheckCount = [int](Get-PropertyOrDefault -Object $crossCheck -Name "failedCrossCheckCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $repair -Name "failedPostPublishCrossCheckCount" -DefaultValue 0)))
  articleProofGateState = [string](Get-PropertyOrDefault -Object $articleGate -Name "gateState" -DefaultValue ([string](Get-PropertyOrDefault -Object $articleGate -Name "proofGateState" -DefaultValue "missing-article-proof-gate")))
  blockedArticleClaimCount = [int](Get-PropertyOrDefault -Object $review -Name "blockedClaimCount" -DefaultValue 0)
}

$record = [pscustomobject]@{
  recordKind = "owner-final-missing-action-one-screen-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  oneScreenState = "blocked-owner-final-missing-actions-required"
  sourceStates = $sourceStates
  missingActionCount = $actions.Count
  missingActions = @($actions.ToArray())
  ownerActionRequired = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner final missing-action one-screen pack only. It summarizes remaining human actions and never publishes packages/articles, imports proof by itself, promotes proof, or closes release issues."
}

$jsonPath = Join-Path $OutputRoot "owner-final-missing-action-one-screen-pack.json"
$mdPath = Join-Path $OutputRoot "owner-final-missing-action-one-screen-pack.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Owner Final Missing Action One-Screen Pack") | Out-Null
$md.Add("") | Out-Null
$md.Add("- oneScreenState: ``$($record.oneScreenState)``") | Out-Null
$md.Add("- missingActionCount: ``$($record.missingActionCount)``") | Out-Null
$md.Add("- dashboardBlockedGateCount: ``$($sourceStates.dashboardBlockedGateCount)``") | Out-Null
$md.Add("- ownerPublishMissingFieldCount: ``$($sourceStates.ownerPublishMissingFieldCount)``") | Out-Null
$md.Add("- postPublishMissingFieldCount: ``$($sourceStates.postPublishMissingFieldCount)``") | Out-Null
$md.Add("- blockedArticleClaimCount: ``$($sourceStates.blockedArticleClaimCount)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| # | Missing Action | Lane | Validator | Forbidden Substitutes |") | Out-Null
$md.Add("| ---: | --- | --- | --- | --- |") | Out-Null
foreach ($action in $actions) {
  $md.Add("| $($action.order) | $(ConvertTo-MarkdownCell $action.ownerAction) | ``$($action.lane)`` | ``$(ConvertTo-MarkdownCell $action.validatorCommand)`` | $(ConvertTo-MarkdownCell $action.forbiddenSubstitutes) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "OwnerFinalMissingActionOneScreenPackState=$($record.oneScreenState) MissingActions=$($record.missingActionCount)"
