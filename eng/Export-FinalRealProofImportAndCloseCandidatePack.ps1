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

function Get-ArtifactHashOrEmpty {
  param([string]$RelativePath)
  $path = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return "" }
  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-StateFromRecord {
  param([AllowNull()][object]$Record, [string]$Name, [string]$MissingState)
  return [string](Get-PropertyOrDefault -Object $Record -Name $Name -DefaultValue $MissingState)
}

function Test-AcceptedState {
  param([AllowNull()][string]$State)
  if ([string]::IsNullOrWhiteSpace($State)) { return $false }
  foreach ($blocked in @("missing", "blocked", "invalid", "failed", "required", "template", "draft", "dry-run", "dryrun", "non-proof", "no-proof", "owner-action")) {
    if ($State.IndexOf($blocked, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
      return $false
    }
  }
  foreach ($accepted in @("accepted", "ready", "passed", "proof-ready", "real-proof")) {
    if ($State.IndexOf($accepted, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
      return $true
    }
  }
  return $false
}

function Get-PhaseById {
  param([AllowNull()][object[]]$Phases, [string]$Id)
  foreach ($phase in @($Phases)) {
    if ([string](Get-PropertyOrDefault -Object $phase -Name "id" -DefaultValue "") -eq $Id) {
      return $phase
    }
  }
  return $null
}

function New-PhaseImportSpec {
  param([string]$Id, [string[]]$ValidationPaths, [string[]]$RequiredProofFields)
  [pscustomobject]@{
    id = $Id
    validationPaths = @($ValidationPaths)
    requiredProofFields = @($RequiredProofFields)
  }
}

if (-not (Test-Path -LiteralPath (Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path "artifacts/final-release/final-real-proof-input-availability-sweep-validation.json") -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-FinalRealProofInputAvailabilitySweep.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
  & (Join-Path $RepositoryRoot "eng\Test-FinalRealProofInputAvailabilitySweep.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot -Strict
}

$sweep = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/final-real-proof-input-availability-sweep.json"
$sweepValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/final-real-proof-input-availability-sweep-validation.json"
$finalIntakeValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/owner-public-publish-execution-final-intake-pack-validation.json"
$ownerReadinessValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/owner-real-publish-evidence-import-readiness-dashboard-validation.json"
$closeCandidateAuditValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/release-close-final-candidate-audit-pack-validation.json"
$releaseEvidenceBundle = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/release-evidence-bundle.json"
$classificationAudit = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/release-evidence-classification-audit.json"
$finalClosureBridgeValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/final-public-release-closure-bridge-validation.json"
$closeDecisionValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/release-issue-close-owner-decision-input-validation.json"

$sweepPhases = @(Convert-ToArray (Get-PropertyOrDefault -Object $sweep -Name "phases" -DefaultValue @()))
$phaseSpecs = @(
  New-PhaseImportSpec -Id "claim-boundary-preflight" -ValidationPaths @("artifacts/final-release/public-publish-final-owner-execution-pack-validation.json", "artifacts/final-release/release-docs-and-nuget-metadata-audit-validation.json") -RequiredProofFields @("docsHash", "readmeHash", "metadataReview")
  New-PhaseImportSpec -Id "owner-public-publish-execution" -ValidationPaths @("artifacts/final-release/public-publish-result-import-validation.json", "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json") -RequiredProofFields @("publicPackagePageUrl", "publicPackageDownloadUrl", "publishTranscriptSha256")
  New-PhaseImportSpec -Id "github-actions-run-proof" -ValidationPaths @("artifacts/final-release/github-publish-and-ci-status-snapshot-validation.json", "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.json") -RequiredProofFields @("runId", "runUrl", "headSha", "logSha256")
  New-PhaseImportSpec -Id "public-managed-package-download-proof" -ValidationPaths @("artifacts/final-release/public-package-download-proof-input-validation.json", "artifacts/final-release/public-package-download-proof-candidate-validation.json") -RequiredProofFields @("managedPublicDownloadUrl", "managedNupkgSha256", "managedNupkgSizeBytes")
  New-PhaseImportSpec -Id "public-runtime-package-download-proof" -ValidationPaths @("artifacts/final-release/public-package-download-proof-input-validation.json", "artifacts/final-release/public-package-download-proof-candidate-validation.json") -RequiredProofFields @("runtimePublicDownloadUrl", "runtimeNupkgSha256", "runtimeNupkgSizeBytes")
  New-PhaseImportSpec -Id "repository-external-clean-consumer" -ValidationPaths @("artifacts/final-release/external-clean-consumer-execution-result-validation.json", "artifacts/final-release/clean-consumer-runtime-proof-cross-check-gate-validation.json") -RequiredProofFields @("externalWorkspaceRoot", "restoreLogSha256", "buildLogSha256", "smokeLogSha256")
  New-PhaseImportSpec -Id "post-publish-clean-consumer-proof" -ValidationPaths @("artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json", "artifacts/final-release/post-publish-clean-consumer-real-proof-from-owner-result-validation.json") -RequiredProofFields @("postPublishRestoreLogSha256", "postPublishBuildLogSha256", "postPublishSmokeLogSha256")
  New-PhaseImportSpec -Id "post-publish-user-verification" -ValidationPaths @("artifacts/final-release/post-publish-user-verification-pack-validation.json", "artifacts/final-release/post-publish-verification-record-validation.json") -RequiredProofFields @("consumerProjectIdentity", "stdoutSummary", "stderrSummary")
  New-PhaseImportSpec -Id "strict-close-convergence" -ValidationPaths @("artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json", "artifacts/final-release/final-public-release-closure-bridge-validation.json", "artifacts/final-release/release-close-final-candidate-audit-pack-validation.json") -RequiredProofFields @("strictCloseReadyState", "finalClosureBridgeState", "acceptedProofLaneCount")
  New-PhaseImportSpec -Id "release-issue-close-owner-decision" -ValidationPaths @("artifacts/final-release/release-issue-close-owner-decision-input-validation.json", "artifacts/final-release/final-release-close-approval-real-input-from-owner-result-validation.json") -RequiredProofFields @("releaseIssueUrl", "ownerCloseDecision", "approvedPostPublishProofHash")
  New-PhaseImportSpec -Id "release-issue-close-record" -ValidationPaths @("artifacts/final-release/release-issue-close-record-validation.json", "artifacts/final-release/release-issue-close-record-real-input-map-validation.json") -RequiredProofFields @("closeRecordId", "releaseEvidenceBundleSha256", "classificationAuditSha256")
  New-PhaseImportSpec -Id "final-bundle-classification-lock" -ValidationPaths @("artifacts/final-release/release-evidence-bundle.json", "artifacts/final-release/release-evidence-classification-audit.json") -RequiredProofFields @("releaseEvidenceBundleSha256", "classificationAuditSha256", "classificationAuditState")
)

$phaseRecords = New-Object System.Collections.Generic.List[object]
foreach ($spec in $phaseSpecs) {
  $sweepPhase = Get-PhaseById -Phases $sweepPhases -Id $spec.id
  $sweepPhaseState = [string](Get-PropertyOrDefault -Object $sweepPhase -Name "phaseState" -DefaultValue "missing-sweep-phase")
  $sweepProofReady = [bool](Get-PropertyOrDefault -Object $sweepPhase -Name "proofReady" -DefaultValue $false)
  $sweepAvailable = [int](Get-PropertyOrDefault -Object $sweepPhase -Name "availableRealInputFileCount" -DefaultValue 0)
  $sweepInvalid = [int](Get-PropertyOrDefault -Object $sweepPhase -Name "invalidRealInputFileCount" -DefaultValue 0)

  $validatorStates = @(
    foreach ($path in @($spec.validationPaths)) {
      $record = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $path
      $stateName = if ($path.EndsWith("release-evidence-bundle.json", [System.StringComparison]::OrdinalIgnoreCase)) { "bundleState" } elseif ($path.EndsWith("release-evidence-classification-audit.json", [System.StringComparison]::OrdinalIgnoreCase)) { "auditState" } else { "validationState" }
      [pscustomobject]@{
        path = $path
        state = Get-StateFromRecord -Record $record -Name $stateName -MissingState "missing-$([System.IO.Path]::GetFileNameWithoutExtension($path))"
        accepted = Test-AcceptedState -State (Get-StateFromRecord -Record $record -Name $stateName -MissingState "missing-$([System.IO.Path]::GetFileNameWithoutExtension($path))")
      }
    }
  )

  $validatorsAccepted = $true
  foreach ($item in $validatorStates) {
    if (-not [bool]$item.accepted) {
      $validatorsAccepted = $false
    }
  }

  $accepted = $sweepProofReady -and $validatorsAccepted -and $sweepInvalid -eq 0
  $phaseState = if ($accepted) {
    "accepted-real-proof-phase"
  }
  elseif ($sweepInvalid -gt 0) {
    "invalid-real-proof-input"
  }
  elseif ($sweepAvailable -eq 0) {
    "missing-real-proof-input"
  }
  else {
    "blocked-real-proof-validator-chain"
  }

  $blockedReasons = New-Object System.Collections.Generic.List[string]
  if ($sweepAvailable -eq 0) { $blockedReasons.Add("explicit-owner-real-input-file-missing") | Out-Null }
  if ($sweepInvalid -gt 0) { $blockedReasons.Add("explicit-owner-real-input-file-invalid") | Out-Null }
  if (-not $validatorsAccepted) { $blockedReasons.Add("strict-validator-chain-not-accepted") | Out-Null }
  if ($spec.id -eq "release-issue-close-owner-decision") {
    $finalBridgeState = Get-StateFromRecord -Record $finalClosureBridgeValidation -Name "validationState" -MissingState "missing-final-public-release-closure-bridge-validation"
    if (-not (Test-AcceptedState -State $finalBridgeState)) {
      $blockedReasons.Add("approved-close-decision-rejected-while-final-bridge-blocked") | Out-Null
    }
  }

  $phaseRecords.Add([pscustomobject]@{
      id = $spec.id
      phaseState = $phaseState
      sweepPhaseState = $sweepPhaseState
      acceptedProof = $accepted
      missingProof = $sweepAvailable -eq 0
      invalidProof = $sweepInvalid -gt 0
      availableRealInputFileCount = $sweepAvailable
      requiredProofFields = @($spec.requiredProofFields)
      requiredProofFieldCount = @($spec.requiredProofFields).Count
      validatorStates = @($validatorStates)
      validatorsAccepted = $validatorsAccepted
      blockedReasons = @($blockedReasons.ToArray())
      blockedReasonCount = $blockedReasons.Count
      ownerActionRequired = -not $accepted
      boundary = "Final real proof import phase aggregation only; it cannot substitute templates, dashboards, local feeds, ProjectReference, direct .nupkg, queued workflows, missing runners, dry-runs, or close the release issue."
    }) | Out-Null
}

$phaseArray = @($phaseRecords.ToArray())
$acceptedProofPhaseCount = @($phaseArray | Where-Object { [bool]$_.acceptedProof }).Count
$missingProofPhaseCount = @($phaseArray | Where-Object { [bool]$_.missingProof }).Count
$invalidProofPhaseCount = @($phaseArray | Where-Object { [bool]$_.invalidProof }).Count
$blockedProofPhaseCount = @($phaseArray | Where-Object { -not [bool]$_.acceptedProof }).Count
$closeCandidateReady = $acceptedProofPhaseCount -eq $phaseArray.Count -and $missingProofPhaseCount -eq 0 -and $invalidProofPhaseCount -eq 0
$closeDecisionState = Get-StateFromRecord -Record $closeDecisionValidation -Name "validationState" -MissingState "missing-release-issue-close-owner-decision-input-validation"
$finalBridgeState = Get-StateFromRecord -Record $finalClosureBridgeValidation -Name "validationState" -MissingState "missing-final-public-release-closure-bridge-validation"
$closeDecisionBlockedByFinalBridge = -not (Test-AcceptedState -State $finalBridgeState)

$record = [pscustomobject]@{
  recordKind = "final-real-proof-import-and-close-candidate-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  candidateState = if ($closeCandidateReady) { "final-real-proof-import-close-candidate-ready-for-owner-review" } else { "blocked-final-real-proof-import-and-close-candidate-owner-evidence-required" }
  phaseCount = $phaseArray.Count
  acceptedProofPhaseCount = $acceptedProofPhaseCount
  missingProofPhaseCount = $missingProofPhaseCount
  invalidProofPhaseCount = $invalidProofPhaseCount
  blockedProofPhaseCount = $blockedProofPhaseCount
  closeCandidateReady = $closeCandidateReady
  closeDecisionValidationState = $closeDecisionState
  finalPublicReleaseClosureBridgeValidationState = $finalBridgeState
  closeDecisionBlockedByFinalBridge = $closeDecisionBlockedByFinalBridge
  finalRealProofInputAvailabilitySweepValidationState = Get-StateFromRecord -Record $sweepValidation -Name "validationState" -MissingState "missing-final-real-proof-input-availability-sweep-validation"
  ownerPublicPublishExecutionFinalIntakePackValidationState = Get-StateFromRecord -Record $finalIntakeValidation -Name "validationState" -MissingState "missing-owner-public-publish-execution-final-intake-pack-validation"
  ownerRealPublishEvidenceImportReadinessDashboardValidationState = Get-StateFromRecord -Record $ownerReadinessValidation -Name "validationState" -MissingState "missing-owner-real-publish-evidence-import-readiness-dashboard-validation"
  releaseCloseFinalCandidateAuditPackValidationState = Get-StateFromRecord -Record $closeCandidateAuditValidation -Name "validationState" -MissingState "missing-release-close-final-candidate-audit-pack-validation"
  releaseEvidenceBundleState = Get-StateFromRecord -Record $releaseEvidenceBundle -Name "bundleState" -MissingState "missing-release-evidence-bundle"
  releaseEvidenceClassificationAuditState = Get-StateFromRecord -Record $classificationAudit -Name "auditState" -MissingState "missing-release-evidence-classification-audit"
  releaseEvidenceBundleSha256 = Get-ArtifactHashOrEmpty -RelativePath "artifacts/final-release/release-evidence-bundle.json"
  releaseEvidenceClassificationAuditSha256 = Get-ArtifactHashOrEmpty -RelativePath "artifacts/final-release/release-evidence-classification-audit.json"
  finalRealProofInputAvailabilitySweepSha256 = Get-ArtifactHashOrEmpty -RelativePath "artifacts/final-release/final-real-proof-input-availability-sweep.json"
  finalRealProofInputAvailabilitySweepValidationSha256 = Get-ArtifactHashOrEmpty -RelativePath "artifacts/final-release/final-real-proof-input-availability-sweep-validation.json"
  forbiddenSubstituteRejectionCount = [int](Get-PropertyOrDefault -Object $sweep -Name "forbiddenSubstituteCount" -DefaultValue 0)
  forbiddenSubstituteKinds = @(Get-PropertyOrDefault -Object $sweep -Name "forbiddenSubstituteKinds" -DefaultValue @("local feed", "ProjectReference", "direct .nupkg", "queued workflow", "missing runner", "dry-run", "template", "dashboard", "audit", "bundle"))
  phases = @($phaseArray)
  ownerActionRequired = -not $closeCandidateReady
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isReleaseCloseRecordProof = $false
  boundary = "Final real proof import and close candidate pack aggregates accepted real proof status only. It does not publish packages, does not download packages, does not run clean consumers, does not approve publication, does not close release issues, and cannot turn templates, dashboards, local feeds, ProjectReference, direct .nupkg, queued workflows, missing runners, or dry-runs into proof."
}

$jsonPath = Join-Path $OutputRoot "final-real-proof-import-and-close-candidate-pack.json"
$mdPath = Join-Path $OutputRoot "final-real-proof-import-and-close-candidate-pack.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 18)

$rows = foreach ($phase in $phaseArray) {
  "| ``$($phase.id)`` | ``$($phase.phaseState)`` | ``$($phase.acceptedProof)`` | ``$($phase.blockedReasonCount)`` |"
}

Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Final Real Proof Import And Close Candidate Pack",
  "",
  "- candidateState: ``$($record.candidateState)``",
  "- phaseCount: ``$($record.phaseCount)``",
  "- acceptedProofPhaseCount: ``$($record.acceptedProofPhaseCount)``",
  "- missingProofPhaseCount: ``$($record.missingProofPhaseCount)``",
  "- invalidProofPhaseCount: ``$($record.invalidProofPhaseCount)``",
  "- closeCandidateReady: ``$($record.closeCandidateReady)``",
  "- closeDecisionBlockedByFinalBridge: ``$($record.closeDecisionBlockedByFinalBridge)``",
  "- forbiddenSubstituteRejectionCount: ``$($record.forbiddenSubstituteRejectionCount)``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "| Phase | State | Accepted Proof | Blocked Reasons |",
  "| --- | --- | ---: | ---: |",
  @($rows),
  "",
  "## Boundary",
  "",
  $record.boundary
)

Write-Host "FinalRealProofImportAndCloseCandidatePackState=$($record.candidateState) Accepted=$acceptedProofPhaseCount Missing=$missingProofPhaseCount Invalid=$invalidProofPhaseCount CloseCandidateReady=$closeCandidateReady"
