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

function New-ManualStep {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string]$OwnerCommandOrAction,
    [string]$RequiredArtifact,
    [string]$RequiredHash,
    [string]$ValidatorCommand,
    [string[]]$ForbiddenSubstitutes
  )
  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    ownerCommandOrAction = $OwnerCommandOrAction
    requiredArtifact = $RequiredArtifact
    requiredHash = $RequiredHash
    validatorCommand = $ValidatorCommand
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    notExecutedByAutomation = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isProofByItself = $false
    boundary = "One-screen manual step only. Owner execution or imported evidence is required; this manual never performs the command and never substitutes proof."
  }
}

$oneScreenPackRelativePath = "artifacts\final-release\final-owner-execution-one-screen-pack.json"
$oneScreenPackValidationRelativePath = "artifacts\final-release\final-owner-execution-one-screen-pack-validation.json"
$oneScreenPackPath = Join-Path $RepositoryRoot $oneScreenPackRelativePath
$oneScreenPackValidationPath = Join-Path $RepositoryRoot $oneScreenPackValidationRelativePath
if (-not (Test-Path -LiteralPath $oneScreenPackPath -PathType Leaf) -or -not (Test-Path -LiteralPath $oneScreenPackValidationPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerExecutionOneScreenPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
  & (Join-Path $RepositoryRoot "eng\Test-FinalOwnerExecutionOneScreenPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot -Strict
}

$oneScreenPack = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $oneScreenPackRelativePath
$oneScreenPackValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $oneScreenPackValidationRelativePath

function Get-OneScreenMetric {
  param([string]$Name, [AllowNull()][object]$DefaultValue)
  $value = Get-PropertyOrDefault -Object $oneScreenPackValidation -Name $Name -DefaultValue $null
  if ($null -ne $value) { return $value }
  return Get-PropertyOrDefault -Object $oneScreenPack -Name $Name -DefaultValue $DefaultValue
}

$releaseCloseRealInputChain = @(Convert-ToArray -Value (Get-PropertyOrDefault -Object $oneScreenPack -Name "releaseCloseRealInputChain" -DefaultValue @()))
$releaseCloseRealInputChainForManual = foreach ($step in $releaseCloseRealInputChain) {
  [pscustomobject]@{
    order = [int](Get-PropertyOrDefault -Object $step -Name "order" -DefaultValue 0)
    id = [string](Get-PropertyOrDefault -Object $step -Name "id" -DefaultValue "")
    title = [string](Get-PropertyOrDefault -Object $step -Name "title" -DefaultValue "")
    currentState = [string](Get-PropertyOrDefault -Object $step -Name "currentState" -DefaultValue "")
    requiredReadyState = [string](Get-PropertyOrDefault -Object $step -Name "requiredReadyState" -DefaultValue "")
    requiredFieldCount = [int](Get-PropertyOrDefault -Object $step -Name "requiredFieldCount" -DefaultValue 0)
    rejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $step -Name "rejectedSubstituteCount" -DefaultValue 0)
    sourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $step -Name "sourceReadinessSignalCount" -DefaultValue 0)
    blockedRealInputCount = [int](Get-PropertyOrDefault -Object $step -Name "blockedRealInputCount" -DefaultValue 0)
    proofCandidateReady = [bool](Get-PropertyOrDefault -Object $step -Name "proofCandidateReady" -DefaultValue $false)
    sourceLinkageReady = [bool](Get-PropertyOrDefault -Object $step -Name "sourceLinkageReady" -DefaultValue $false)
    ownerAction = [string](Get-PropertyOrDefault -Object $step -Name "ownerAction" -DefaultValue "")
    strictValidator = [string](Get-PropertyOrDefault -Object $step -Name "strictValidator" -DefaultValue "")
    blockedReason = [string](Get-PropertyOrDefault -Object $step -Name "blockedReason" -DefaultValue "")
    forbiddenSubstitutes = @(Convert-ToArray -Value (Get-PropertyOrDefault -Object $step -Name "forbiddenSubstitutes" -DefaultValue @()))
    boundary = [string](Get-PropertyOrDefault -Object $step -Name "boundary" -DefaultValue "")
  }
}

$releaseCloseRealInputChainCount = [int](Get-OneScreenMetric -Name "releaseCloseRealInputChainCount" -DefaultValue 0)
$releaseCloseRealInputChainRequiredFieldCount = [int](Get-OneScreenMetric -Name "releaseCloseRealInputChainRequiredFieldCount" -DefaultValue 0)
$releaseCloseRealInputChainRejectedSubstituteCount = [int](Get-OneScreenMetric -Name "releaseCloseRealInputChainRejectedSubstituteCount" -DefaultValue 0)
$releaseCloseRealInputChainSourceReadinessSignalCount = [int](Get-OneScreenMetric -Name "releaseCloseRealInputChainSourceReadinessSignalCount" -DefaultValue 0)
$releaseCloseRealInputChainBlockedRealInputCount = [int](Get-OneScreenMetric -Name "releaseCloseRealInputChainBlockedRealInputCount" -DefaultValue 0)
$publicPackageDownloadProofRequiredFieldCount = [int](Get-OneScreenMetric -Name "publicPackageDownloadProofRequiredFieldCount" -DefaultValue 0)
$publicPackageDownloadProofRejectedSubstituteCount = [int](Get-OneScreenMetric -Name "publicPackageDownloadProofRejectedSubstituteCount" -DefaultValue 0)
$publicPackageDownloadProofSourceReadinessSignalCount = [int](Get-OneScreenMetric -Name "publicPackageDownloadProofSourceReadinessSignalCount" -DefaultValue 0)
$publicPackageDownloadProofCandidateReady = [bool](Get-OneScreenMetric -Name "publicPackageDownloadProofCandidateReady" -DefaultValue $false)
$publicPackageDownloadProofSourceGitHubActionsRunEvidenceReady = [bool](Get-OneScreenMetric -Name "publicPackageDownloadProofSourceGitHubActionsRunEvidenceReady" -DefaultValue $false)
$publicPackageDownloadProofSourceOwnerPublicPublishResultReady = [bool](Get-OneScreenMetric -Name "publicPackageDownloadProofSourceOwnerPublicPublishResultReady" -DefaultValue $false)
$postPublishCleanConsumerProofRequiredFieldCount = [int](Get-OneScreenMetric -Name "postPublishCleanConsumerProofRequiredFieldCount" -DefaultValue 0)
$postPublishCleanConsumerProofRejectedSubstituteCount = [int](Get-OneScreenMetric -Name "postPublishCleanConsumerProofRejectedSubstituteCount" -DefaultValue 0)
$postPublishCleanConsumerProofBlockedRealInputCount = [int](Get-OneScreenMetric -Name "postPublishCleanConsumerProofBlockedRealInputCount" -DefaultValue 0)
$postPublishCleanConsumerProofSourceReadinessSignalCount = [int](Get-OneScreenMetric -Name "postPublishCleanConsumerProofSourceReadinessSignalCount" -DefaultValue 0)
$postPublishCleanConsumerProofCandidateReady = [bool](Get-OneScreenMetric -Name "postPublishCleanConsumerProofCandidateReady" -DefaultValue $false)
$postPublishCleanConsumerProofSourceProofLinkageReady = [bool](Get-OneScreenMetric -Name "postPublishCleanConsumerProofSourceProofLinkageReady" -DefaultValue $false)
$postPublishCleanConsumerProofSourceGitHubActionsRunEvidenceReady = [bool](Get-OneScreenMetric -Name "postPublishCleanConsumerProofSourceGitHubActionsRunEvidenceReady" -DefaultValue $false)
$postPublishCleanConsumerProofSourceOwnerPublicPublishResultReady = [bool](Get-OneScreenMetric -Name "postPublishCleanConsumerProofSourceOwnerPublicPublishResultReady" -DefaultValue $false)
$postPublishCleanConsumerProofSourcePublicPackageDownloadProofReady = [bool](Get-OneScreenMetric -Name "postPublishCleanConsumerProofSourcePublicPackageDownloadProofReady" -DefaultValue $false)
$releaseEvidenceBundleSha256 = [string](Get-OneScreenMetric -Name "releaseEvidenceBundleSha256" -DefaultValue "")
$finalCloseStrictValidatorOutputState = [string](Get-OneScreenMetric -Name "finalCloseStrictValidatorOutputState" -DefaultValue "missing-final-close-strict-validator-output-state")

$oneScreenSourceArtifacts = @(
  "artifacts/final-release/final-owner-execution-one-screen-pack.json",
  "artifacts/final-release/final-owner-execution-one-screen-pack-validation.json"
)

$commonForbidden = @("local feed", "ProjectReference", "direct .nupkg", "dashboard", "dry-run", "manual approval", "queued GitHub Actions run", "missing self-hosted runner", "sidecar-only", "TensorRtExec report", "public package download proof alone", "post-publish validation-ready without proofCandidateReady", "release evidence bundle hash only", "strict close validator output without real proof")
$steps = @(
  New-ManualStep 1 "preflight-freeze" "Refresh pre-execution freeze and strict closure" "Run readonly preflight validators; do not publish." "final-public-publish-pre-execution-freeze-validation.json; release-close-strict-evidence-closure-validation.json" "N/A - readonly validation output" "Export-FinalPublicPublishPreExecutionFreeze.ps1; Test-FinalPublicPublishPreExecutionFreeze.ps1 -Strict; Export-ReleaseCloseStrictEvidenceClosure.ps1; Test-ReleaseCloseStrictEvidenceClosure.ps1 -Strict" $commonForbidden
  New-ManualStep 2 "owner-authorization" "Record Owner authorization scope" "Owner records approval id/scope/timestamp for the exact package ids and versions." "owner-public-publish-execution-result input fields" "authorization record/hash entered by Owner" "Export-OwnerPublicPublishExecutionResultInputTemplate.ps1; Test-OwnerPublicPublishExecutionResultPreflight.ps1 -Strict" $commonForbidden
  New-ManualStep 3 "public-publish-command" "Owner executes public publish command manually" "Owner runs approved dotnet nuget push/GitHub Packages command outside this automation and records command hash." "public publish result input" "publish command plan hash; managed/runtime command hash" "Import-OwnerPublicPublishExecutionResultCandidate.ps1; Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict" $commonForbidden
  New-ManualStep 4 "package-page-and-download" "Capture public package page/download URLs" "Owner records package page URL, managed/runtime download URLs, versions, timestamps." "post-publish-verification-owner-input.template.json" "downloaded managed/runtime nupkg SHA256" "Test-PostPublishVerificationOwnerInput.ps1 -Strict" $commonForbidden
  New-ManualStep 5 "post-publish-clean-consumer" "Run clean consumer restore/build/smoke from public package" "Owner runs clean external consumer using public package source, no ProjectReference/local feed/direct nupkg." "post-publish-verification-record.json" "restore/build/smoke/native asset listing/dependency probe SHA256" "Export-PostPublishVerificationRecordFromOwnerInput.ps1; Test-PostPublishVerificationRecord.ps1 -Strict" $commonForbidden
  New-ManualStep 6 "rollback-review" "Complete rollback review without executing rollback" "Owner records rollback plan and confirms delete/delist/withdraw/deprecate remain forbidden unless separately authorized." "final-owner-rollback-review-import.json" "rollback review artifact SHA256" "Import-FinalOwnerRollbackReview.ps1; Test-FinalOwnerRollbackReview.ps1 -Strict" $commonForbidden
  New-ManualStep 7 "close-decision" "Owner final close decision after real proof passes" "Owner records close decision only after public publish, PostPublish, rollback review, and strict closure pass." "final-owner-close-decision-import.json" "release evidence bundle SHA256" "Import-FinalOwnerCloseDecision.ps1; Test-FinalOwnerCloseDecision.ps1 -Strict" $commonForbidden
  New-ManualStep 8 "strict-final-verification" "Run strict closure, dashboard, classification audit, and acceptance gate" "Run final validators; close issue only if all strict gates accept real evidence." "release-close-strict-evidence-closure.json; final-public-publish-acceptance-gate.json" "evidence bundle/classification/forbidden scan hashes" "Export-ReleaseEvidenceBundle.ps1; Test-ReleaseEvidenceClassificationAudit.ps1 -Strict; Export-ReleaseCloseStrictEvidenceClosure.ps1; Test-ReleaseCloseStrictEvidenceClosure.ps1 -Strict; Test-FinalPublicPublishAcceptanceGate.ps1 -Strict" $commonForbidden
)

$record = [pscustomobject]@{
  recordKind = "final-owner-one-screen-execution-manual"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  manualState = "blocked-final-owner-one-screen-execution-manual-owner-action-required"
  stepCount = $steps.Count
  steps = @($steps)
  sourceArtifacts = @($oneScreenSourceArtifacts)
  releaseCloseRealInputChainCount = $releaseCloseRealInputChainCount
  releaseCloseRealInputChainRequiredFieldCount = $releaseCloseRealInputChainRequiredFieldCount
  releaseCloseRealInputChainRejectedSubstituteCount = $releaseCloseRealInputChainRejectedSubstituteCount
  releaseCloseRealInputChainSourceReadinessSignalCount = $releaseCloseRealInputChainSourceReadinessSignalCount
  releaseCloseRealInputChainBlockedRealInputCount = $releaseCloseRealInputChainBlockedRealInputCount
  releaseCloseRealInputChain = @($releaseCloseRealInputChainForManual)
  publicPackageDownloadProofRequiredFieldCount = $publicPackageDownloadProofRequiredFieldCount
  publicPackageDownloadProofRejectedSubstituteCount = $publicPackageDownloadProofRejectedSubstituteCount
  publicPackageDownloadProofSourceReadinessSignalCount = $publicPackageDownloadProofSourceReadinessSignalCount
  publicPackageDownloadProofCandidateReady = $publicPackageDownloadProofCandidateReady
  publicPackageDownloadProofSourceGitHubActionsRunEvidenceReady = $publicPackageDownloadProofSourceGitHubActionsRunEvidenceReady
  publicPackageDownloadProofSourceOwnerPublicPublishResultReady = $publicPackageDownloadProofSourceOwnerPublicPublishResultReady
  postPublishCleanConsumerProofRequiredFieldCount = $postPublishCleanConsumerProofRequiredFieldCount
  postPublishCleanConsumerProofRejectedSubstituteCount = $postPublishCleanConsumerProofRejectedSubstituteCount
  postPublishCleanConsumerProofBlockedRealInputCount = $postPublishCleanConsumerProofBlockedRealInputCount
  postPublishCleanConsumerProofSourceReadinessSignalCount = $postPublishCleanConsumerProofSourceReadinessSignalCount
  postPublishCleanConsumerProofCandidateReady = $postPublishCleanConsumerProofCandidateReady
  postPublishCleanConsumerProofSourceProofLinkageReady = $postPublishCleanConsumerProofSourceProofLinkageReady
  postPublishCleanConsumerProofSourceGitHubActionsRunEvidenceReady = $postPublishCleanConsumerProofSourceGitHubActionsRunEvidenceReady
  postPublishCleanConsumerProofSourceOwnerPublicPublishResultReady = $postPublishCleanConsumerProofSourceOwnerPublicPublishResultReady
  postPublishCleanConsumerProofSourcePublicPackageDownloadProofReady = $postPublishCleanConsumerProofSourcePublicPackageDownloadProofReady
  releaseEvidenceBundleSha256 = $releaseEvidenceBundleSha256
  finalCloseStrictValidatorOutputState = $finalCloseStrictValidatorOutputState
  forbiddenNonProofSubstitutes = @($commonForbidden)
  forbiddenReleaseActions = @("delete", "delist", "withdraw", "deprecate", "dotnet nuget delete", "nuget delete")
  ownerActionRequired = $true
  notExecutedByAutomation = $true
  passed = $false
  performsPublish = $false
  performsGitHubPackagesPublish = $false
  performsNuGetPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  manualIsProof = $false
  boundary = "Owner one-screen execution manual is human-readable guidance only. It may show dotnet nuget push as an Owner-only command placeholder, but it never executes publish, never uploads NuGet or GitHub Packages, never deletes/delists/withdraws/deprecates, and never makes dashboard/dry-run/manual approval/queued workflow/missing runner/local feed/ProjectReference/direct nupkg proof."
}

$jsonPath = Join-Path $OutputRoot "final-owner-one-screen-execution-manual.json"
$mdPath = Join-Path $OutputRoot "final-owner-one-screen-execution-manual.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Final Owner One-Screen Execution Manual") | Out-Null
$md.Add("") | Out-Null
$md.Add("- manualState: ``$($record.manualState)``") | Out-Null
$md.Add("- notExecutedByAutomation: ``True``") | Out-Null
$md.Add("- manualIsProof: ``False``") | Out-Null
$md.Add("- releaseCloseRealInputChainCount: ``$($record.releaseCloseRealInputChainCount)``") | Out-Null
$md.Add("- releaseCloseRealInputChainRequiredFieldCount: ``$($record.releaseCloseRealInputChainRequiredFieldCount)``") | Out-Null
$md.Add("- releaseCloseRealInputChainRejectedSubstituteCount: ``$($record.releaseCloseRealInputChainRejectedSubstituteCount)``") | Out-Null
$md.Add("- releaseCloseRealInputChainSourceReadinessSignalCount: ``$($record.releaseCloseRealInputChainSourceReadinessSignalCount)``") | Out-Null
$md.Add("- releaseCloseRealInputChainBlockedRealInputCount: ``$($record.releaseCloseRealInputChainBlockedRealInputCount)``") | Out-Null
$md.Add("- publicPackageDownloadProofRequiredFieldCount: ``$($record.publicPackageDownloadProofRequiredFieldCount)``") | Out-Null
$md.Add("- publicPackageDownloadProofCandidateReady: ``$($record.publicPackageDownloadProofCandidateReady)``") | Out-Null
$md.Add("- postPublishCleanConsumerProofRequiredFieldCount: ``$($record.postPublishCleanConsumerProofRequiredFieldCount)``") | Out-Null
$md.Add("- postPublishCleanConsumerProofCandidateReady: ``$($record.postPublishCleanConsumerProofCandidateReady)``") | Out-Null
$md.Add("- postPublishCleanConsumerProofSourceProofLinkageReady: ``$($record.postPublishCleanConsumerProofSourceProofLinkageReady)``") | Out-Null
$md.Add("- releaseEvidenceBundleSha256: ``$($record.releaseEvidenceBundleSha256)``") | Out-Null
$md.Add("- finalCloseStrictValidatorOutputState: ``$($record.finalCloseStrictValidatorOutputState)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| # | Step | Owner Command / Action | Required Artifact | Required Hash | Validator |") | Out-Null
$md.Add("| --- | --- | --- | --- | --- | --- |") | Out-Null
foreach ($step in $steps) {
  $md.Add("| $($step.order) | $($step.id) | $(ConvertTo-MarkdownCell $step.ownerCommandOrAction) | $(ConvertTo-MarkdownCell $step.requiredArtifact) | $(ConvertTo-MarkdownCell $step.requiredHash) | ``$(ConvertTo-MarkdownCell $step.validatorCommand)`` |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Release Close Real Input Chain") | Out-Null
$md.Add("") | Out-Null
$md.Add("| # | ID | State | Required Fields | Rejected Substitutes | Source Signals | Blocked Inputs | Proof Ready | Linkage Ready |") | Out-Null
$md.Add("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |") | Out-Null
foreach ($step in $releaseCloseRealInputChainForManual) {
  $md.Add("| $($step.order) | ``$(ConvertTo-MarkdownCell $step.id)`` | ``$(ConvertTo-MarkdownCell $step.currentState)`` | $($step.requiredFieldCount) | $($step.rejectedSubstituteCount) | $($step.sourceReadinessSignalCount) | $($step.blockedRealInputCount) | ``$($step.proofCandidateReady)`` | ``$($step.sourceLinkageReady)`` |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Forbidden Substitutes") | Out-Null
$md.Add("") | Out-Null
foreach ($item in $commonForbidden) { $md.Add("- ``$item``") | Out-Null }
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md

Write-Host "FinalOwnerOneScreenExecutionManualState=$($record.manualState) Steps=$($steps.Count) NotExecutedByAutomation=True"
