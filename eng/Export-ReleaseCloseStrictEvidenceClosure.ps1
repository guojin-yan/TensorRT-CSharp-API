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

function Get-ArtifactPath {
  param([string]$RelativePath)
  if ([System.IO.Path]::IsPathRooted($RelativePath)) { return $RelativePath }
  return Join-Path $RepositoryRoot $RelativePath
}

function New-ClosureLane {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Artifact,
    [AllowNull()][object]$Record,
    [string]$StateProperty,
    [string]$ExpectedReadyState,
    [string]$RequiredEvidence,
    [string]$OwnerNextAction,
    [string]$ValidatorCommand,
    [bool]$ReadyWhenExpectedState = $true
  )

  $exists = $null -ne $Record
  $state = if ($exists) { [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue "missing-$StateProperty") } else { "missing-artifact" }
  $failedBlockerCount = if ($exists) { [int](Get-PropertyOrDefault -Object $Record -Name "failedBlockerCount" -DefaultValue -1) } else { -1 }
  $failedActionRequiredCount = if ($exists) { [int](Get-PropertyOrDefault -Object $Record -Name "failedActionRequiredCount" -DefaultValue -1) } else { -1 }
  $canClose = if ($exists) { [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false) } else { $false }
  $isReleaseCloseProof = if ($exists) { [bool](Get-PropertyOrDefault -Object $Record -Name "isReleaseCloseProof" -DefaultValue $false) } else { $false }
  $isPostPublishProof = if ($exists) { [bool](Get-PropertyOrDefault -Object $Record -Name "isPostPublishProof" -DefaultValue $false) } else { $false }
  $ready = $exists -and [string]::Equals($state, $ExpectedReadyState, [StringComparison]::OrdinalIgnoreCase)
  if (-not $ReadyWhenExpectedState) { $ready = $exists -and $failedBlockerCount -eq 0 }

  [pscustomobject]@{
    id = $Id
    title = $Title
    artifact = $Artifact
    artifactExists = $exists
    state = $state
    expectedReadyState = $ExpectedReadyState
    ready = $ready
    failedBlockerCount = $failedBlockerCount
    failedActionRequiredCount = $failedActionRequiredCount
    canCloseReleaseIssue = $canClose
    isReleaseCloseProof = $isReleaseCloseProof
    isPostPublishProof = $isPostPublishProof
    requiredEvidence = $RequiredEvidence
    ownerNextAction = $OwnerNextAction
    validatorCommand = $ValidatorCommand
    boundary = "This lane is evidence-gate input only. It cannot execute publish, delete, delist, withdraw, deprecate, close an issue, or substitute real public package and post-publish proof."
  }
}

function New-CrossCheck {
  param([string]$Id, [bool]$Passed, [string]$Detail, [string]$OwnerNextAction)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    detail = $Detail
    ownerActionRequired = -not $Passed
    ownerNextAction = $OwnerNextAction
  }
}

function Get-TextOrEmpty {
  param([AllowNull()][object]$Record, [string]$Name)
  return [string](Get-PropertyOrDefault -Object $Record -Name $Name -DefaultValue "")
}

function Test-RealText {
  param([AllowNull()][object]$Value)
  return -not (Test-OwnerPlaceholder $Value)
}

$publishCandidate = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-candidate.json"
$publishValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-candidate-validation.json"
$postPublishOwnerInput = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-verification-owner-input.template.json"
$postPublishOwnerInputValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-verification-owner-input-validation.json"
$postPublishRecord = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-verification-record.json"
$postPublishValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-verification-validation.json"
$rollbackImport = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-owner-rollback-review-import.json"
$rollbackValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-owner-rollback-review-validation.json"
$closeImport = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-owner-close-decision-import.json"
$closeValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-owner-close-decision-validation.json"
$evidenceBundle = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\release-evidence-bundle.json"
$classificationAudit = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\release-evidence-classification-audit.json"
$acceptanceGate = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-public-publish-acceptance-gate.json"
$forbiddenScan = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\public-publish-forbidden-substitute-scan.json"

$lanes = @(
  New-ClosureLane -Id "owner-public-publish-result" -Title "Owner public publish result candidate validation" -Artifact "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json" -Record $publishValidation -StateProperty "validationState" -ExpectedReadyState "owner-public-publish-execution-result-ready-with-real-public-package-proof" -RequiredEvidence "Owner-executed public publish result with public package URLs, package versions, SHA256 values, command plan hash, runner/source proof, and forbidden substitute scan hash." -OwnerNextAction "Owner must execute the approved public publish command outside automation and import the real result." -ValidatorCommand "Import-OwnerPublicPublishExecutionResultCandidate.ps1; Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict"
  New-ClosureLane -Id "post-publish-owner-input" -Title "PostPublish owner input validation" -Artifact "artifacts/final-release/post-publish-verification-owner-input-validation.json" -Record $postPublishOwnerInputValidation -StateProperty "validationState" -ExpectedReadyState "post-publish-verification-owner-input-ready" -RequiredEvidence "Owner-filled post-publish input from public package channel, downloaded package hashes, clean consumer logs, rollback review hash, and forbidden substitute scan hash." -OwnerNextAction "Fill post-publish owner input from public package download and rerun validation." -ValidatorCommand "Test-PostPublishVerificationOwnerInput.ps1 -Strict"
  New-ClosureLane -Id "post-publish-record" -Title "PostPublish verification record validation" -Artifact "artifacts/final-release/post-publish-verification-validation.json" -Record $postPublishValidation -StateProperty "validationState" -ExpectedReadyState "post-publish-verification-proof" -RequiredEvidence "Promotable post-publish verification record with clean consumer runtime smoke, no ProjectReference/local feed/direct nupkg, and matching log hashes." -OwnerNextAction "Export post-publish record from accepted owner input and validate the real proof record." -ValidatorCommand "Export-PostPublishVerificationRecordFromOwnerInput.ps1; Test-PostPublishVerificationRecord.ps1 -Strict"
  New-ClosureLane -Id "rollback-review" -Title "Final owner rollback review validation" -Artifact "artifacts/final-release/final-owner-rollback-review-validation.json" -Record $rollbackValidation -StateProperty "validationState" -ExpectedReadyState "final-owner-rollback-review-validation-ready-non-proof" -RequiredEvidence "Rollback plan governance must exist and remain non-proof; delete/delist/withdraw/deprecate actions remain forbidden unless separately authorized." -OwnerNextAction "Complete rollback review inputs, but do not execute rollback/delete/delist/withdraw/deprecate from this gate." -ValidatorCommand "Import-FinalOwnerRollbackReview.ps1; Test-FinalOwnerRollbackReview.ps1 -Strict"
  New-ClosureLane -Id "close-decision" -Title "Final owner close decision validation" -Artifact "artifacts/final-release/final-owner-close-decision-validation.json" -Record $closeValidation -StateProperty "validationState" -ExpectedReadyState "final-owner-close-decision-ready-with-real-public-proof-chain" -RequiredEvidence "Owner final close decision that references the accepted evidence bundle, public package proof, post-publish proof, and rollback review." -OwnerNextAction "Owner must provide final close decision only after every real proof lane is accepted." -ValidatorCommand "Import-FinalOwnerCloseDecision.ps1; Test-FinalOwnerCloseDecision.ps1 -Strict"
  New-ClosureLane -Id "release-evidence-bundle" -Title "Release evidence bundle" -Artifact "artifacts/final-release/release-evidence-bundle.json" -Record $evidenceBundle -StateProperty "bundleState" -ExpectedReadyState "release-evidence-bundle-ready-with-real-public-postpublish-close-proof" -RequiredEvidence "Evidence bundle must carry strict closure validator outputs as non-proof until all real proof lanes pass." -OwnerNextAction "Regenerate release evidence bundle after strict closure and all owner inputs are accepted." -ValidatorCommand "Export-ReleaseEvidenceBundle.ps1"
  New-ClosureLane -Id "classification-audit" -Title "Release evidence classification audit" -Artifact "artifacts/final-release/release-evidence-classification-audit.json" -Record $classificationAudit -StateProperty "auditState" -ExpectedReadyState "classification-audit-passed-non-proof-boundaries-intact" -RequiredEvidence "Classification audit must keep dashboards, dry-runs, queued workflows, manual approvals, sidecars, and TensorRtExec reports non-proof." -OwnerNextAction "Keep non-proof boundaries intact and rerun classification audit after bundle regeneration." -ValidatorCommand "Test-ReleaseEvidenceClassificationAudit.ps1 -Strict"
  New-ClosureLane -Id "final-public-publish-acceptance-gate" -Title "Final public publish acceptance gate" -Artifact "artifacts/final-release/final-public-publish-acceptance-gate.json" -Record $acceptanceGate -StateProperty "gateState" -ExpectedReadyState "final-public-publish-acceptance-ready-with-owner-evidence" -RequiredEvidence "Final gate must see owner public publish result, post-publish proof, close decision, and owner convergence as accepted real evidence." -OwnerNextAction "Complete blocked owner evidence lanes; do not treat failedBlockerCount=0 as proof." -ValidatorCommand "Test-FinalPublicPublishAcceptanceGate.ps1 -Strict"
)

$publishManagedVersion = Get-TextOrEmpty -Record $postPublishValidation -Name "managedPackageVersion"
$publishRuntimeVersion = Get-TextOrEmpty -Record $postPublishValidation -Name "runtimePackageVersion"
$postPublishVersion = Get-TextOrEmpty -Record $postPublishRecord -Name "publishedVersion"
$closeDecisionVersion = Get-TextOrEmpty -Record $closeImport -Name "packageVersion"
$postPublishPackagePage = Get-TextOrEmpty -Record $postPublishRecord -Name "packagePageUrl"
$ownerInputPackagePage = Get-TextOrEmpty -Record $postPublishOwnerInput -Name "packagePageUrl"
$managedUrl = Get-TextOrEmpty -Record $postPublishValidation -Name "managedPackageUrl"
$runtimeUrl = Get-TextOrEmpty -Record $postPublishValidation -Name "runtimePackageUrl"
$ownerInputRollbackPath = Get-TextOrEmpty -Record $postPublishOwnerInput -Name "rollbackReviewPath"
$recordRollbackPath = Get-TextOrEmpty -Record $postPublishRecord -Name "rollbackReviewPath"
$ownerInputRollbackHash = Get-TextOrEmpty -Record $postPublishOwnerInput -Name "rollbackReviewSha256"
$recordRollbackHash = Get-TextOrEmpty -Record $postPublishRecord -Name "rollbackReviewSha256"
$ownerInputForbiddenPath = Get-TextOrEmpty -Record $postPublishOwnerInput -Name "forbiddenSubstituteScanPath"
$recordForbiddenPath = Get-TextOrEmpty -Record $postPublishRecord -Name "forbiddenSubstituteScanPath"
$ownerInputForbiddenHash = Get-TextOrEmpty -Record $postPublishOwnerInput -Name "forbiddenSubstituteScanSha256"
$recordForbiddenHash = Get-TextOrEmpty -Record $postPublishRecord -Name "forbiddenSubstituteScanSha256"

$crossChecks = @(
  New-CrossCheck -Id "public-package-url-present-and-public" -Passed ((Test-RealText $managedUrl) -and (Test-RealText $runtimeUrl) -and ($managedUrl -match '^https?://' -or $runtimeUrl -match '^https?://')) -Detail "managedPackageUrl=$managedUrl; runtimePackageUrl=$runtimeUrl" -OwnerNextAction "Import public package download URLs from the real public feed, not local files or dashboards."
  New-CrossCheck -Id "public-package-version-consistency" -Passed ((Test-RealText $publishManagedVersion) -and (Test-RealText $publishRuntimeVersion) -and (($postPublishVersion -eq "" -or $postPublishVersion -eq $publishManagedVersion -or $postPublishVersion -eq $publishRuntimeVersion)) -and ($closeDecisionVersion -eq "" -or $closeDecisionVersion -eq $publishManagedVersion -or $closeDecisionVersion -eq $publishRuntimeVersion)) -Detail "managed=$publishManagedVersion; runtime=$publishRuntimeVersion; postPublish=$postPublishVersion; closeDecision=$closeDecisionVersion" -OwnerNextAction "Align publish result, PostPublish, and close decision package versions from the same public release."
  New-CrossCheck -Id "package-page-url-consistency" -Passed ((Test-RealText $postPublishPackagePage) -and (Test-RealText $ownerInputPackagePage) -and $postPublishPackagePage -eq $ownerInputPackagePage -and $postPublishPackagePage -match '^https?://') -Detail "ownerInput=$ownerInputPackagePage; postPublish=$postPublishPackagePage" -OwnerNextAction "Provide the public package page URL in owner input and projected PostPublish record."
  New-CrossCheck -Id "rollback-review-path-hash-consistency" -Passed ((Test-RealText $ownerInputRollbackPath) -and (Test-RealText $recordRollbackPath) -and $ownerInputRollbackPath -eq $recordRollbackPath -and (Test-Sha256Text $ownerInputRollbackHash) -and $ownerInputRollbackHash -eq $recordRollbackHash) -Detail "ownerInputPath=$ownerInputRollbackPath; recordPath=$recordRollbackPath; ownerInputHash=$ownerInputRollbackHash; recordHash=$recordRollbackHash" -OwnerNextAction "Reference the same rollback review artifact and SHA256 from PostPublish input and close decision material."
  New-CrossCheck -Id "forbidden-substitute-scan-path-hash-consistency" -Passed ((Test-RealText $ownerInputForbiddenPath) -and (Test-RealText $recordForbiddenPath) -and $ownerInputForbiddenPath -eq $recordForbiddenPath -and (Test-Sha256Text $ownerInputForbiddenHash) -and $ownerInputForbiddenHash -eq $recordForbiddenHash -and $null -ne $forbiddenScan) -Detail "ownerInputPath=$ownerInputForbiddenPath; recordPath=$recordForbiddenPath; ownerInputHash=$ownerInputForbiddenHash; recordHash=$recordForbiddenHash" -OwnerNextAction "Reference the public-publish forbidden substitute scan and its SHA256 from PostPublish and final close validation."
  New-CrossCheck -Id "failed-blocker-count-zero-is-not-proof" -Passed (($lanes | Where-Object { [int]$_.failedBlockerCount -eq 0 -and -not [bool]$_.ready }).Count -gt 0) -Detail "At least one lane has failedBlockerCount=0 while still not ready; the strict closure treats that as blocked." -OwnerNextAction "Do not promote failedBlockerCount=0 without real proof state and cross-checks."
)

$fakeReadySubstituteCases = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "dashboard",
  "dry-run",
  "manual approval",
  "queued GitHub Actions run",
  "missing self-hosted runner",
  "sidecar-only",
  "TensorRtExec report"
) | ForEach-Object {
  [pscustomobject]@{
    id = ("fake-ready-" + ($_ -replace '[^A-Za-z0-9]+', '-').Trim('-').ToLowerInvariant())
    substitute = $_
    fakeReadyShapeBlocked = $true
    canCloseReleaseIssue = $false
    canPublishPublicly = $false
    isReleaseCloseProof = $false
    ownerActionRequired = $true
    blockingReason = "$_ can look complete but is a forbidden non-proof substitute for public package/post-publish/release-close evidence."
  }
}

$blockedLanes = @($lanes | Where-Object { -not [bool]$_.ready })
$failedCrossChecks = @($crossChecks | Where-Object { -not [bool]$_.passed })
$blockedFakeReady = @($fakeReadySubstituteCases | Where-Object { [bool]$_.fakeReadyShapeBlocked })
$ready = $blockedLanes.Count -eq 0 -and $failedCrossChecks.Count -eq 0
$state = if ($ready) { "release-close-strict-evidence-closure-ready" } else { "blocked-release-close-strict-evidence-closure-owner-evidence-required" }

$record = [pscustomobject]@{
  recordKind = "release-close-strict-evidence-closure"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  closureState = $state
  strictEvidenceClosureReady = $ready
  laneCount = $lanes.Count
  blockedLaneCount = $blockedLanes.Count
  readyLaneCount = $lanes.Count - $blockedLanes.Count
  crossCheckCount = $crossChecks.Count
  failedCrossCheckCount = $failedCrossChecks.Count
  fakeReadySubstituteCaseCount = @($fakeReadySubstituteCases).Count
  blockedFakeReadySubstituteCaseCount = $blockedFakeReady.Count
  lanes = @($lanes)
  crossChecks = @($crossChecks)
  fakeReadySubstituteCases = @($fakeReadySubstituteCases)
  forbiddenNonProofSubstitutes = @($fakeReadySubstituteCases | ForEach-Object { $_.substitute })
  ownerActionRequired = -not $ready
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $ready
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $ready
  failedBlockerCountIsNotProof = $true
  dashboardIsProof = $false
  dryRunIsProof = $false
  manualApprovalIsProof = $false
  queuedWorkflowIsProof = $false
  missingRunnerIsProof = $false
  boundary = "Strict closure aggregates real evidence validators only. It never executes dotnet nuget push, GitHub Packages publish, delete, delist, withdraw, or deprecate. failedBlockerCount=0, dashboards, dry-runs, manual approvals, queued workflows, missing runners, sidecar-only evidence, TensorRtExec reports, local feeds, ProjectReference, and direct .nupkg references are non-proof."
}

$jsonPath = Join-Path $OutputRoot "release-close-strict-evidence-closure.json"
$mdPath = Join-Path $OutputRoot "release-close-strict-evidence-closure.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# ReleaseClose Strict Evidence Closure") | Out-Null
$md.Add("") | Out-Null
$md.Add("- closureState: ``$state``") | Out-Null
$md.Add("- strictEvidenceClosureReady: ``$ready``") | Out-Null
$md.Add("- blockedLaneCount: ``$($blockedLanes.Count)``") | Out-Null
$md.Add("- failedCrossCheckCount: ``$($failedCrossChecks.Count)``") | Out-Null
$md.Add("- blockedFakeReadySubstituteCaseCount: ``$($blockedFakeReady.Count)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("## Lanes") | Out-Null
$md.Add("") | Out-Null
$md.Add("| ID | Ready | State | Owner Next Action | Validator |") | Out-Null
$md.Add("| --- | --- | --- | --- | --- |") | Out-Null
foreach ($lane in $lanes) {
  $md.Add("| $($lane.id) | $($lane.ready) | $(ConvertTo-MarkdownCell $lane.state) | $(ConvertTo-MarkdownCell $lane.ownerNextAction) | ``$(ConvertTo-MarkdownCell $lane.validatorCommand)`` |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Cross Checks") | Out-Null
$md.Add("") | Out-Null
$md.Add("| ID | Passed | Detail | Owner Next Action |") | Out-Null
$md.Add("| --- | --- | --- | --- |") | Out-Null
foreach ($check in $crossChecks) {
  $md.Add("| $($check.id) | $($check.passed) | $(ConvertTo-MarkdownCell $check.detail) | $(ConvertTo-MarkdownCell $check.ownerNextAction) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Fake-Ready Substitutes") | Out-Null
$md.Add("") | Out-Null
$md.Add(($fakeReadySubstituteCases | ForEach-Object { "- $($_.substitute): blocked, non-proof" })) | Out-Null
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md

Write-Host "ReleaseCloseStrictEvidenceClosureState=$state BlockedLanes=$($blockedLanes.Count) FailedCrossChecks=$($failedCrossChecks.Count) FakeReadyBlocked=$($blockedFakeReady.Count)"
