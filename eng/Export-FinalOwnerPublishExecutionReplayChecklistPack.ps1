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

function New-ReplayStep {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string]$OwnerAction,
    [string[]]$RequiredEvidence,
    [string[]]$ValidatorCommands
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    ownerAction = $OwnerAction
    requiredEvidence = @($RequiredEvidence)
    requiredEvidenceCount = @($RequiredEvidence).Count
    validatorCommands = @($ValidatorCommands)
    ownerActionRequired = $true
    blocked = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isProof = $false
    boundary = "Replay checklist step only. It prepares Owner execution evidence capture and never runs publish, uploads packages, or promotes proof."
  }
}

$manual = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-owner-one-screen-execution-manual-validation.json"
$intake = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-real-publish-evidence-intake-dry-run-pack-validation.json"
$postPublish = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-strict-cross-check-pack-validation.json"
$closure = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\release-close-strict-evidence-closure-validation.json"

$manualState = [string](Get-PropertyOrDefault -Object $manual -Name "validationState" -DefaultValue "missing-final-owner-one-screen-execution-manual-validation")
$intakeState = [string](Get-PropertyOrDefault -Object $intake -Name "validationState" -DefaultValue "missing-owner-real-publish-evidence-intake-dry-run-pack-validation")
$postPublishState = [string](Get-PropertyOrDefault -Object $postPublish -Name "validationState" -DefaultValue "missing-post-publish-strict-cross-check-pack-validation")
$closureState = [string](Get-PropertyOrDefault -Object $closure -Name "validationState" -DefaultValue "missing-release-close-strict-evidence-closure-validation")

$commonForbidden = @("local feed", "ProjectReference", "direct .nupkg", "dashboard", "dry-run", "manual approval", "queued GitHub Actions run", "missing self-hosted runner", "sidecar-only", "TensorRtExec report")
$steps = @(
  New-ReplayStep 1 "readonly-freeze" "Refresh final readonly audit and Owner manual" "Run readonly validators and confirm automation remains side-effect free before any Owner action." @("final-readonly-publish-audit-pack-validation.json", "final-owner-one-screen-execution-manual-validation.json") @("Export-FinalReadonlyPublishAuditPack.ps1", "Test-FinalReadonlyPublishAuditPack.ps1 -Strict", "Export-FinalOwnerOneScreenExecutionManual.ps1", "Test-FinalOwnerOneScreenExecutionManual.ps1 -Strict")
  New-ReplayStep 2 "owner-authorization-capture" "Capture explicit Owner authorization" "Owner records the exact package ids, versions, target sources, timestamp, scope, and authorization transcript hash." @("ownerAuthorizationId", "ownerAuthorizationScope", "ownerAuthorizationTimestampUtc", "ownerApprovalTranscriptPath", "ownerApprovalTranscriptSha256") @("Export-OwnerPublicPublishExecutionResultInputTemplate.ps1", "Test-OwnerPublicPublishExecutionResultPreflight.ps1 -Strict")
  New-ReplayStep 3 "publish-command-capture" "Capture public publish command and transcript" "Owner executes the approved command only after authorization and records command plan, stdout, stderr, merged transcript, exit code, and SHA256 values." @("publishCommandPlanPath", "publishCommandPlanSha256", "managedPublishCommandSha256", "runtimePublishCommandSha256", "nugetPushTranscriptPath", "nugetPushTranscriptSha256", "nugetPushExitCode") @("Import-OwnerPublicPublishExecutionResultCandidate.ps1", "Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict")
  New-ReplayStep 4 "public-package-identity" "Capture public package URL/download/hash" "Owner records package page URL, managed/runtime download URLs, public source channel, downloaded package path, and SHA256 values from the public source." @("publicPackageUrl", "publicPackageDownloadedPath", "publicPackageSha256", "managedPackageUrl", "managedPackageSha256", "runtimePackageUrl", "runtimePackageSha256") @("Export-OwnerRealPublishEvidenceIntakeDryRunPack.ps1", "Test-OwnerRealPublishEvidenceIntakeDryRunPack.ps1 -Strict")
  New-ReplayStep 5 "post-publish-clean-consumer" "Run clean consumer outside repository" "Owner restores/builds/runs an external consumer from the public channel with no local feed, no ProjectReference, and no direct nupkg reference." @("cleanConsumerRestoreRoot", "cleanConsumerRestoreNoLocalFeedEvidence", "cleanConsumerRuntimeSmokeReportPath", "cleanConsumerRuntimeSmokeReportSha256", "post-publish-verification-record.json") @("Export-PostPublishVerificationRecordFromOwnerInput.ps1", "Test-PostPublishVerificationRecord.ps1 -Strict", "Export-PostPublishStrictCrossCheckPack.ps1", "Test-PostPublishStrictCrossCheckPack.ps1 -Strict")
  New-ReplayStep 6 "rollback-and-close-readiness" "Capture rollback review and close decision" "Owner records rollback review and final close decision only after real public publish and PostPublish evidence pass strict validators." @("final-owner-rollback-review-validation.json", "final-owner-close-decision-validation.json", "release-close-strict-evidence-closure-validation.json") @("Import-FinalOwnerRollbackReview.ps1", "Test-FinalOwnerRollbackReview.ps1 -Strict", "Import-FinalOwnerCloseDecision.ps1", "Test-FinalOwnerCloseDecision.ps1 -Strict", "Export-ReleaseCloseStrictEvidenceClosure.ps1", "Test-ReleaseCloseStrictEvidenceClosure.ps1 -Strict")
)

$record = [pscustomobject]@{
  recordKind = "final-owner-publish-execution-replay-checklist-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  checklistState = "blocked-final-owner-publish-execution-replay-owner-action-required"
  stepCount = $steps.Count
  blockedStepCount = @($steps | Where-Object { [bool]$_.blocked }).Count
  steps = @($steps)
  sourceStates = [pscustomobject]@{
    ownerManualValidationState = $manualState
    ownerPublishEvidenceIntakeValidationState = $intakeState
    postPublishStrictCrossCheckValidationState = $postPublishState
    releaseCloseStrictEvidenceClosureValidationState = $closureState
  }
  forbiddenNonProofSubstitutes = @($commonForbidden)
  ownerActionRequired = $true
  blocked = $true
  dryRunOnly = $true
  performsPublish = $false
  performsNuGetPublish = $false
  performsGitHubPackagesPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner publish execution replay checklist pack only. It prepares evidence capture for an Owner-executed public publish, but never runs dotnet nuget push, never publishes GitHub Packages, never triggers workflows, never deletes/delists/withdraws/deprecates, and never accepts local feed, ProjectReference, direct nupkg, dashboard, dry-run, manual approval, queued workflow, missing runner, sidecar-only, or TensorRtExec report as proof."
}

$jsonPath = Join-Path $OutputRoot "final-owner-publish-execution-replay-checklist-pack.json"
$mdPath = Join-Path $OutputRoot "final-owner-publish-execution-replay-checklist-pack.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Final Owner Publish Execution Replay Checklist Pack") | Out-Null
$md.Add("") | Out-Null
$md.Add("- checklistState: ``$($record.checklistState)``") | Out-Null
$md.Add("- stepCount: ``$($record.stepCount)``") | Out-Null
$md.Add("- blockedStepCount: ``$($record.blockedStepCount)``") | Out-Null
$md.Add("- performsPublish: ``False``") | Out-Null
$md.Add("- canCloseReleaseIssue: ``False``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| # | Step | Required Evidence | Validators |") | Out-Null
$md.Add("| --- | --- | --- | --- |") | Out-Null
foreach ($step in $steps) {
  $md.Add("| $($step.order) | $($step.id) | $(ConvertTo-MarkdownCell ($step.requiredEvidence -join '; ')) | ``$(ConvertTo-MarkdownCell ($step.validatorCommands -join '; '))`` |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "FinalOwnerPublishExecutionReplayChecklistPackState=$($record.checklistState) Steps=$($record.stepCount) Blocked=$($record.blockedStepCount)"
