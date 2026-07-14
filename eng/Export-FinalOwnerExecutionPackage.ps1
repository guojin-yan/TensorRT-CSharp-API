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

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function ConvertTo-Array {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return @()
  }

  if ($Value -is [System.Array]) {
    return @($Value)
  }

  return @($Value)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-ArtifactSha256OrEmpty {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return ""
  }

  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-RecordState {
  param([AllowNull()][object]$Record)

  foreach ($name in @("validationState", "packageState", "packState", "candidateState", "convergenceState", "bundleState", "worklistState", "alignmentState")) {
    $value = [string](Get-PropertyOrDefault -Object $Record -Name $name -DefaultValue "")
    if (-not [string]::IsNullOrWhiteSpace($value)) {
      return $value
    }
  }

  return "missing-state"
}

function New-SourceArtifactEvidence {
  param([string]$Id, [string]$RelativePath, [AllowNull()][object]$Record)

  $path = Join-Path $RepositoryRoot $RelativePath
  $exists = Test-Path -LiteralPath $path -PathType Leaf

  [pscustomobject]@{
    id = $Id
    artifactPath = $RelativePath
    exists = $exists
    sha256 = if ($exists) { Get-ArtifactSha256OrEmpty -RelativePath $RelativePath } else { "" }
    recordKind = [string](Get-PropertyOrDefault -Object $Record -Name "recordKind" -DefaultValue "missing")
    state = Get-RecordState -Record $Record
    proofCandidateReady = [bool](Get-PropertyOrDefault -Object $Record -Name "proofCandidateReady" -DefaultValue $false)
    sourceProofLinkageReady = [bool](Get-PropertyOrDefault -Object $Record -Name "sourceProofLinkageReady" -DefaultValue $false)
    ownerActionRequired = [bool](Get-PropertyOrDefault -Object $Record -Name "ownerActionRequired" -DefaultValue $true)
    failedBlockerCount = [int](Get-PropertyOrDefault -Object $Record -Name "failedBlockerCount" -DefaultValue 0)
    failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $Record -Name "failedActionRequiredCount" -DefaultValue 0)
    performsPublish = [bool](Get-PropertyOrDefault -Object $Record -Name "performsPublish" -DefaultValue $false)
    canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $Record -Name "canPromoteRuntimeProof" -DefaultValue $false)
    canPublishPublicly = [bool](Get-PropertyOrDefault -Object $Record -Name "canPublishPublicly" -DefaultValue $false)
    canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false)
    isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isRuntimeExecutionProof" -DefaultValue $false)
    isPostPublishProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isPostPublishProof" -DefaultValue $false)
    isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isReleaseCloseProof" -DefaultValue $false)
    boundary = "Source artifact evidence is local path/hash/state traceability only. Hash presence cannot substitute real Owner proof, post-publish CleanConsumer proof, publish approval, final close decision, or release issue close."
  }
}

function Get-ResultArtifactForAction {
  param([string]$ActionId)

  switch ($ActionId) {
    "00-clean-external-package-consumer-owner-runbook" {
      return @(
        "artifacts/final-release/clean-external-package-consumer-owner-runbook.json",
        "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json",
        "artifacts/final-release/owner-external-proof-execution-result.input.json"
      )
    }
    "00-post-publish-owner-verification-runbook" {
      return @(
        "artifacts/final-release/post-publish-owner-verification-runbook.json",
        "artifacts/final-release/post-publish-owner-verification-runbook-validation.json",
        "artifacts/final-release/owner-external-proof-execution-result.input.json"
      )
    }
    "01-real-model-runtime-owner-evidence" {
      return @(
        "artifacts/user-acceptance/sample-run-evidence-record.json",
        "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
        "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json"
      )
    }
    "02-package-consumer-runtime-clean-external-proof" {
      return @(
        "artifacts/final-release/package-consumer-runtime-proof-record.json",
        "artifacts/final-release/package-consumer-runtime-proof-record-validation.json",
        "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json"
      )
    }
    "03-post-publish-verification-public-channel" {
      return @(
        "artifacts/final-release/post-publish-verification-record.json",
        "artifacts/final-release/post-publish-verification-validation.json",
        "artifacts/final-release/post-publish-verification-owner-input-validation.json"
      )
    }
    "04-final-owner-real-input-template-pack" {
      return @(
        "artifacts/final-release/final-owner-real-input-template-pack.json",
        "artifacts/final-release/final-owner-real-input-template-pack-validation.json",
        "artifacts/final-release/owner-real-inputs"
      )
    }
    "05-owner-external-result-import-real-files" {
      return @(
        "artifacts/final-release/owner-external-proof-execution-result-import.json",
        "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
        "artifacts/final-release/real-external-proof-record-import-validator.json"
      )
    }
    "06-owner-result-candidate-bridge-strict-promotion" {
      return @(
        "artifacts/final-release/real-proof-record-candidate-from-owner-result-import.json",
        "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json",
        "artifacts/final-release/final-publish-proof-gate-report.json"
      )
    }
    default {
      return @()
    }
  }
}

function Get-InputContractForAction {
  param([object]$Action)

  $sourceArtifact = [string](Get-PropertyOrDefault -Object $Action -Name "sourceArtifact" -DefaultValue "")
  $resultArtifacts = @(Get-ResultArtifactForAction -ActionId ([string](Get-PropertyOrDefault -Object $Action -Name "id" -DefaultValue "")))

  [pscustomobject]@{
    sourceArtifact = $sourceArtifact
    expectedResultArtifacts = @($resultArtifacts)
    requiredFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $Action -Name "requiredInputs" -DefaultValue @())
    ownerMustProvideRealFiles = $true
    requiresExistingLog = $true
    requiresSha256 = $true
    requiresExitCodeZeroWhenRuntimeProof = $true
    requiresOwnerReview = $true
    acceptsTemplateOnly = $false
    acceptsCandidateAsProof = $false
    acceptsDashboardAsProof = $false
  }
}

function New-ExecutionStepFromAction {
  param([object]$Action, [int]$Order)

  $id = [string](Get-PropertyOrDefault -Object $Action -Name "id" -DefaultValue "")
  $laneId = [string](Get-PropertyOrDefault -Object $Action -Name "laneId" -DefaultValue "")
  $ownerCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $Action -Name "ownerCommands" -DefaultValue @())
  $validatorCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $Action -Name "validatorCommands" -DefaultValue @())
  $forbiddenSubstitutes = Convert-ToStringArray (Get-PropertyOrDefault -Object $Action -Name "forbiddenSubstitutes" -DefaultValue @())
  $inputContract = Get-InputContractForAction -Action $Action

  [pscustomobject]@{
    order = $Order
    id = $id
    laneId = $laneId
    phase = [string](Get-PropertyOrDefault -Object $Action -Name "phase" -DefaultValue $laneId)
    state = [string](Get-PropertyOrDefault -Object $Action -Name "state" -DefaultValue "blocked-owner-action-required")
    blocked = $true
    actionRequiredId = [string](Get-PropertyOrDefault -Object $Action -Name "actionRequiredId" -DefaultValue "")
    detail = [string](Get-PropertyOrDefault -Object $Action -Name "detail" -DefaultValue "")
    inputContract = $inputContract
    requiredInputs = Convert-ToStringArray (Get-PropertyOrDefault -Object $Action -Name "requiredInputs" -DefaultValue @())
    requiredInputCount = [int](Get-PropertyOrDefault -Object $Action -Name "requiredInputCount" -DefaultValue 0)
    ownerCommands = @($ownerCommands)
    ownerCommandCount = @($ownerCommands).Count
    validatorCommands = @($validatorCommands)
    validatorCommandCount = @($validatorCommands).Count
    expectedResultArtifacts = @($inputContract.expectedResultArtifacts)
    expectedResultArtifactCount = @($inputContract.expectedResultArtifacts).Count
    forbiddenSubstitutes = @($forbiddenSubstitutes)
    forbiddenSubstituteCount = @($forbiddenSubstitutes).Count
    promotionBoundary = [string](Get-PropertyOrDefault -Object $Action -Name "boundary" -DefaultValue "")
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner execution step only. It prepares real owner inputs, result artifacts, and validators; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

function New-OwnerReleaseCloseHardGate {
  param([object]$Step)

  $id = [string](Get-PropertyOrDefault -Object $Step -Name "id" -DefaultValue "")
  $blockedReason = [string](Get-PropertyOrDefault -Object $Step -Name "blockedReason" -DefaultValue "")

  [pscustomobject]@{
    order = [int](Get-PropertyOrDefault -Object $Step -Name "order" -DefaultValue 0)
    id = $id
    title = [string](Get-PropertyOrDefault -Object $Step -Name "title" -DefaultValue $id)
    sourceArtifact = [string](Get-PropertyOrDefault -Object $Step -Name "sourceArtifact" -DefaultValue "")
    currentState = [string](Get-PropertyOrDefault -Object $Step -Name "currentState" -DefaultValue "")
    requiredReadyState = [string](Get-PropertyOrDefault -Object $Step -Name "requiredReadyState" -DefaultValue "")
    requiredFieldCount = [int](Get-PropertyOrDefault -Object $Step -Name "requiredFieldCount" -DefaultValue 0)
    rejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $Step -Name "rejectedSubstituteCount" -DefaultValue 0)
    sourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $Step -Name "sourceReadinessSignalCount" -DefaultValue 0)
    blockedRealInputCount = [int](Get-PropertyOrDefault -Object $Step -Name "blockedRealInputCount" -DefaultValue 0)
    proofCandidateReady = [bool](Get-PropertyOrDefault -Object $Step -Name "proofCandidateReady" -DefaultValue $false)
    sourceLinkageReady = [bool](Get-PropertyOrDefault -Object $Step -Name "sourceLinkageReady" -DefaultValue $false)
    artifactSha256 = [string](Get-PropertyOrDefault -Object $Step -Name "artifactSha256" -DefaultValue "")
    ownerAction = [string](Get-PropertyOrDefault -Object $Step -Name "ownerAction" -DefaultValue "")
    strictValidator = [string](Get-PropertyOrDefault -Object $Step -Name "strictValidator" -DefaultValue "")
    blockedReason = $blockedReason
    blocked = $true
    ownerActionRequired = $true
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Release-close hard gate only. It records the required Owner input and strict validator; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. $blockedReason"
  }
}

$worklist = Read-JsonOrNull "artifacts\final-release\final-owner-proof-action-worklist.json"
$worklistValidation = Read-JsonOrNull "artifacts\final-release\final-owner-proof-action-worklist-validation.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$readinessPack = Read-JsonOrNull "artifacts\final-release\release-publish-readiness-evidence-pack.json"
$ownerRuntimeSmokeFieldAlignment = Read-JsonOrNull "artifacts\final-release\package-consumer-owner-runtime-smoke-field-alignment.json"
$ownerRuntimeSmokeFieldAlignmentValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-owner-runtime-smoke-field-alignment-validation.json"
$finalOwnerExecutionOneScreenPack = Read-JsonOrNull "artifacts\final-release\final-owner-execution-one-screen-pack.json"
$finalOwnerExecutionOneScreenPackValidation = Read-JsonOrNull "artifacts\final-release\final-owner-execution-one-screen-pack-validation.json"
$finalOwnerStrictCloseExecutionOrderValidation = Read-JsonOrNull "artifacts\final-release\final-owner-strict-close-execution-order-validation.json"
$postPublishCleanConsumerProofResultValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-validation.json"
$publicPackageDownloadProofCandidateValidation = Read-JsonOrNull "artifacts\final-release\public-package-download-proof-candidate-validation.json"
$ownerPublicPublishExecutionResultCandidateValidation = Read-JsonOrNull "artifacts\final-release\owner-public-publish-execution-result-candidate-validation.json"
$releaseIssueCloseOwnerDecisionInputValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-owner-decision-input-validation.json"
$finalCloseGateConvergenceValidation = Read-JsonOrNull "artifacts\final-release\final-close-gate-convergence-validation.json"
$ownerRuntimeSmokeFieldAlignmentState = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "alignmentState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment")
$ownerRuntimeSmokeFieldAlignmentValidationState = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignmentValidation -Name "validationState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment-validation")
$ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "runtimeSmokeStatus" -DefaultValue "Smoke=missing")
$ownerRuntimeSmokeFieldAlignmentFieldCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "fieldCount" -DefaultValue 0)
$ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "missingRequiredFieldCount" -DefaultValue -1)
$ownerRuntimeSmokeFieldAlignmentFailedBlockerCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignmentValidation -Name "failedBlockerCount" -DefaultValue -1)

$actions = @()
if ($null -ne $worklist) {
  $actions = @((Get-PropertyOrDefault -Object $worklist -Name "actions" -DefaultValue @()))
}

$executionSteps = @()
for ($i = 0; $i -lt $actions.Count; $i++) {
  $executionSteps += New-ExecutionStepFromAction -Action $actions[$i] -Order ($i + 1)
}

$allForbiddenSubstitutes = @($executionSteps | ForEach-Object { $_.forbiddenSubstitutes } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ } | Select-Object -Unique)
$allValidatorCommands = @($executionSteps | ForEach-Object { $_.validatorCommands } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ } | Select-Object -Unique)
$allOwnerCommands = @($executionSteps | ForEach-Object { $_.ownerCommands } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ } | Select-Object -Unique)
$allResultArtifacts = @($executionSteps | ForEach-Object { $_.expectedResultArtifacts } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ } | Select-Object -Unique)

$releaseCloseRealInputChain = @(ConvertTo-Array (Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPack -Name "releaseCloseRealInputChain" -DefaultValue @()))
$ownerReleaseCloseHardGates = @($releaseCloseRealInputChain | ForEach-Object { New-OwnerReleaseCloseHardGate -Step $_ })
$releaseCloseRealInputChainRequiredFieldCount = [int](Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name "releaseCloseRealInputChainRequiredFieldCount" -DefaultValue (Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPack -Name "releaseCloseRealInputChainRequiredFieldCount" -DefaultValue 0))
$releaseCloseRealInputChainRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name "releaseCloseRealInputChainRejectedSubstituteCount" -DefaultValue (Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPack -Name "releaseCloseRealInputChainRejectedSubstituteCount" -DefaultValue 0))
$releaseCloseRealInputChainSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name "releaseCloseRealInputChainSourceReadinessSignalCount" -DefaultValue (Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPack -Name "releaseCloseRealInputChainSourceReadinessSignalCount" -DefaultValue 0))
$releaseCloseRealInputChainBlockedRealInputCount = [int](Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name "releaseCloseRealInputChainBlockedRealInputCount" -DefaultValue (Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPack -Name "releaseCloseRealInputChainBlockedRealInputCount" -DefaultValue 0))
$releaseEvidenceBundleSha256 = [string](Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name "releaseEvidenceBundleSha256" -DefaultValue (Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPack -Name "releaseEvidenceBundleSha256" -DefaultValue ""))
$finalCloseStrictValidatorOutputState = [string](Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name "finalCloseStrictValidatorOutputState" -DefaultValue (Get-PropertyOrDefault -Object $finalCloseGateConvergenceValidation -Name "validationState" -DefaultValue "missing-final-close-gate-convergence-validation"))
$publicPackageDownloadProofCandidateReady = [bool](Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name "publicPackageDownloadProofCandidateReady" -DefaultValue (Get-PropertyOrDefault -Object $publicPackageDownloadProofCandidateValidation -Name "proofCandidateReady" -DefaultValue $false))
$postPublishProofCandidateReady = [bool](Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name "postPublishCleanConsumerProofCandidateReady" -DefaultValue (Get-PropertyOrDefault -Object $postPublishCleanConsumerProofResultValidation -Name "proofCandidateReady" -DefaultValue $false))
$postPublishProofSourceLinkageReady = [bool](Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name "postPublishCleanConsumerProofSourceProofLinkageReady" -DefaultValue (Get-PropertyOrDefault -Object $postPublishCleanConsumerProofResultValidation -Name "sourceProofLinkageReady" -DefaultValue $false))
$releaseIssueCloseOwnerDecisionValidationState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseOwnerDecisionInputValidation -Name "validationState" -DefaultValue "missing-release-issue-close-owner-decision-input-validation")
$ownerPublicPublishResultCandidateValidationState = [string](Get-PropertyOrDefault -Object $ownerPublicPublishExecutionResultCandidateValidation -Name "validationState" -DefaultValue "missing-owner-public-publish-execution-result-candidate-validation")
$publicPackageDownloadProofCandidateValidationState = [string](Get-PropertyOrDefault -Object $publicPackageDownloadProofCandidateValidation -Name "validationState" -DefaultValue "missing-public-package-download-proof-candidate-validation")
$postPublishCleanConsumerProofResultValidationState = [string](Get-PropertyOrDefault -Object $postPublishCleanConsumerProofResultValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-result-validation")
$finalOwnerExecutionOneScreenPackValidationState = [string](Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name "validationState" -DefaultValue "missing-final-owner-execution-one-screen-pack-validation")
$finalOwnerStrictCloseExecutionOrderValidationState = [string](Get-PropertyOrDefault -Object $finalOwnerStrictCloseExecutionOrderValidation -Name "validationState" -DefaultValue "missing-final-owner-strict-close-execution-order-validation")
$finalCloseProofAdmissionRequiredFieldCount = [int](Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name "finalCloseProofAdmissionRequiredFieldCount" -DefaultValue (Get-PropertyOrDefault -Object $finalCloseGateConvergenceValidation -Name "finalCloseProofAdmissionRequiredFieldCount" -DefaultValue 0))
$finalCloseRejectedNonProofStateCount = [int](Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name "finalCloseRejectedNonProofStateCount" -DefaultValue (Get-PropertyOrDefault -Object $finalCloseGateConvergenceValidation -Name "finalCloseRejectedNonProofStateCount" -DefaultValue 0))
$hardGateForbiddenSubstitutes = @(
  "public package download proof alone",
  "post-publish validation-ready without proofCandidateReady",
  "release evidence bundle hash only",
  "strict close validator output without real proof",
  "strict close validator output alone",
  "bundle hash without Owner final close decision"
)
$allForbiddenSubstitutes = @($allForbiddenSubstitutes + $hardGateForbiddenSubstitutes | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Select-Object -Unique)

$sourceArtifacts = @(
  "artifacts/final-release/final-owner-proof-action-worklist.json",
  "artifacts/final-release/final-owner-proof-action-worklist-validation.json",
  "artifacts/final-release/clean-external-package-consumer-owner-runbook.json",
  "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json",
  "artifacts/final-release/post-publish-owner-verification-runbook.json",
  "artifacts/final-release/post-publish-owner-verification-runbook-validation.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/release-publish-readiness-evidence-pack.json",
  "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json",
  "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json",
  "artifacts/final-release/final-owner-execution-one-screen-pack.json",
  "artifacts/final-release/final-owner-execution-one-screen-pack-validation.json",
  "artifacts/final-release/final-owner-strict-close-execution-order-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json",
  "artifacts/final-release/public-package-download-proof-candidate-validation.json",
  "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json",
  "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
  "artifacts/final-release/final-close-gate-convergence-validation.json"
)

$sourceArtifactEvidence = @(
  New-SourceArtifactEvidence -Id "final-owner-proof-action-worklist" -RelativePath "artifacts/final-release/final-owner-proof-action-worklist.json" -Record $worklist
  New-SourceArtifactEvidence -Id "final-owner-proof-action-worklist-validation" -RelativePath "artifacts/final-release/final-owner-proof-action-worklist-validation.json" -Record $worklistValidation
  New-SourceArtifactEvidence -Id "final-owner-execution-one-screen-pack" -RelativePath "artifacts/final-release/final-owner-execution-one-screen-pack.json" -Record $finalOwnerExecutionOneScreenPack
  New-SourceArtifactEvidence -Id "final-owner-execution-one-screen-pack-validation" -RelativePath "artifacts/final-release/final-owner-execution-one-screen-pack-validation.json" -Record $finalOwnerExecutionOneScreenPackValidation
  New-SourceArtifactEvidence -Id "final-owner-strict-close-execution-order-validation" -RelativePath "artifacts/final-release/final-owner-strict-close-execution-order-validation.json" -Record $finalOwnerStrictCloseExecutionOrderValidation
  New-SourceArtifactEvidence -Id "owner-public-publish-execution-result-candidate-validation" -RelativePath "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json" -Record $ownerPublicPublishExecutionResultCandidateValidation
  New-SourceArtifactEvidence -Id "public-package-download-proof-candidate-validation" -RelativePath "artifacts/final-release/public-package-download-proof-candidate-validation.json" -Record $publicPackageDownloadProofCandidateValidation
  New-SourceArtifactEvidence -Id "post-publish-clean-consumer-proof-result-validation" -RelativePath "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json" -Record $postPublishCleanConsumerProofResultValidation
  New-SourceArtifactEvidence -Id "release-issue-close-owner-decision-input-validation" -RelativePath "artifacts/final-release/release-issue-close-owner-decision-input-validation.json" -Record $releaseIssueCloseOwnerDecisionInputValidation
  New-SourceArtifactEvidence -Id "final-close-gate-convergence-validation" -RelativePath "artifacts/final-release/final-close-gate-convergence-validation.json" -Record $finalCloseGateConvergenceValidation
  New-SourceArtifactEvidence -Id "release-evidence-bundle" -RelativePath "artifacts/final-release/release-evidence-bundle.json" -Record $releaseEvidenceBundle
  New-SourceArtifactEvidence -Id "release-publish-readiness-evidence-pack" -RelativePath "artifacts/final-release/release-publish-readiness-evidence-pack.json" -Record $readinessPack
  New-SourceArtifactEvidence -Id "package-consumer-owner-runtime-smoke-field-alignment" -RelativePath "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json" -Record $ownerRuntimeSmokeFieldAlignment
  New-SourceArtifactEvidence -Id "package-consumer-owner-runtime-smoke-field-alignment-validation" -RelativePath "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json" -Record $ownerRuntimeSmokeFieldAlignmentValidation
)
$sourceArtifactEvidenceMissingCount = @($sourceArtifactEvidence | Where-Object { -not $_.exists }).Count
$sourceArtifactEvidenceSha256Count = @($sourceArtifactEvidence | Where-Object { [System.Text.RegularExpressions.Regex]::IsMatch([string]$_.sha256, "^[0-9a-f]{64}$") }).Count
$sourceArtifactEvidenceNonProofBoundaryCount = @($sourceArtifactEvidence | Where-Object {
    -not $_.performsPublish -and
    -not $_.canPromoteRuntimeProof -and
    -not $_.canPublishPublicly -and
    -not $_.canCloseReleaseIssue -and
    -not $_.isRuntimeExecutionProof -and
    -not $_.isPostPublishProof -and
    -not $_.isReleaseCloseProof
  }).Count

$record = [ordered]@{
  recordKind = "final-owner-execution-package"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  packageState = "blocked-final-owner-execution-required"
  sourceWorklistState = [string](Get-PropertyOrDefault -Object $worklist -Name "worklistState" -DefaultValue "missing-final-owner-proof-action-worklist")
  sourceWorklistValidationState = [string](Get-PropertyOrDefault -Object $worklistValidation -Name "validationState" -DefaultValue "missing-final-owner-proof-action-worklist-validation")
  releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  releasePublishReadinessEvidencePackState = [string](Get-PropertyOrDefault -Object $readinessPack -Name "packState" -DefaultValue "missing-release-publish-readiness-evidence-pack")
  packageConsumerOwnerRuntimeSmokeFieldAlignmentState = $ownerRuntimeSmokeFieldAlignmentState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState = $ownerRuntimeSmokeFieldAlignmentValidationState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = $ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount = $ownerRuntimeSmokeFieldAlignmentFieldCount
  packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = $ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount = $ownerRuntimeSmokeFieldAlignmentFailedBlockerCount
  finalOwnerExecutionOneScreenPackValidationState = $finalOwnerExecutionOneScreenPackValidationState
  finalOwnerStrictCloseExecutionOrderValidationState = $finalOwnerStrictCloseExecutionOrderValidationState
  ownerPublicPublishResultCandidateValidationState = $ownerPublicPublishResultCandidateValidationState
  publicPackageDownloadProofCandidateValidationState = $publicPackageDownloadProofCandidateValidationState
  postPublishCleanConsumerProofResultValidationState = $postPublishCleanConsumerProofResultValidationState
  releaseIssueCloseOwnerDecisionValidationState = $releaseIssueCloseOwnerDecisionValidationState
  finalCloseStrictValidatorOutputState = $finalCloseStrictValidatorOutputState
  releaseEvidenceBundleSha256 = $releaseEvidenceBundleSha256
  releaseCloseRealInputChainCount = $releaseCloseRealInputChain.Count
  blockedReleaseCloseRealInputChainCount = @($ownerReleaseCloseHardGates | Where-Object { $_.blocked }).Count
  releaseCloseRealInputChainRequiredFieldCount = $releaseCloseRealInputChainRequiredFieldCount
  releaseCloseRealInputChainRejectedSubstituteCount = $releaseCloseRealInputChainRejectedSubstituteCount
  releaseCloseRealInputChainSourceReadinessSignalCount = $releaseCloseRealInputChainSourceReadinessSignalCount
  releaseCloseRealInputChainBlockedRealInputCount = $releaseCloseRealInputChainBlockedRealInputCount
  releaseCloseRealInputChain = @($releaseCloseRealInputChain)
  ownerReleaseCloseHardGateCount = $ownerReleaseCloseHardGates.Count
  blockedOwnerReleaseCloseHardGateCount = @($ownerReleaseCloseHardGates | Where-Object { $_.blocked }).Count
  ownerReleaseCloseHardGates = @($ownerReleaseCloseHardGates)
  publicPackageDownloadProofCandidateReady = $publicPackageDownloadProofCandidateReady
  postPublishProofCandidateReady = $postPublishProofCandidateReady
  postPublishProofSourceLinkageReady = $postPublishProofSourceLinkageReady
  finalCloseProofAdmissionRequiredFieldCount = $finalCloseProofAdmissionRequiredFieldCount
  finalCloseRejectedNonProofStateCount = $finalCloseRejectedNonProofStateCount
  actionCount = $actions.Count
  executionStepCount = $executionSteps.Count
  blockedExecutionStepCount = @($executionSteps | Where-Object { $_.blocked }).Count
  actionRequiredIds = @($executionSteps | ForEach-Object { $_.actionRequiredId })
  executionSteps = @($executionSteps)
  ownerCommandSequence = @($allOwnerCommands)
  validatorCommands = @($allValidatorCommands)
  expectedResultArtifacts = @($allResultArtifacts)
  forbiddenSubstitutes = @($allForbiddenSubstitutes)
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isPackagePush = $false
  requiresOwnerManualPublish = $true
  requiresRealExternalProof = $true
  requiresPostPublishProof = $true
  requiresReleaseCloseOwnerDecision = $true
  publicDownloadCannotSubstitutePostPublishProof = $true
  postPublishValidationReadyCannotSubstituteProofCandidateReady = $true
  bundleHashCannotSubstituteFinalCloseDecision = $true
  strictCloseOutputCannotCloseIssue = $true
  sourceArtifacts = @($sourceArtifacts)
  sourceArtifactEvidenceCount = $sourceArtifactEvidence.Count
  sourceArtifactEvidenceMissingCount = $sourceArtifactEvidenceMissingCount
  sourceArtifactEvidenceSha256Count = $sourceArtifactEvidenceSha256Count
  sourceArtifactEvidenceNonProofBoundaryCount = $sourceArtifactEvidenceNonProofBoundaryCount
  sourceArtifactEvidence = @($sourceArtifactEvidence)
  boundary = "Final owner execution package maps owner actions plus the eight-step release-close real input chain to commands, input contracts, hard gates, expected result artifacts, and validators. It is owner handoff only; public package download proof alone is not post-publish CleanConsumer proof, post-publish validation-ready without proofCandidateReady is not proof, release evidence bundle hash only is not final close decision, strict close validator output alone cannot close the issue, and this package does not publish, does not promote runtime proof, does not verify post-publish proof, cannot close the release issue, and is not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-package.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-package.md"
$record | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($step in $executionSteps) {
  "| ``$(ConvertTo-MarkdownCell $step.id)`` | ``$(ConvertTo-MarkdownCell $step.laneId)`` | ``$(ConvertTo-MarkdownCell $step.state)`` | ``$($step.requiredInputCount)`` | ``$($step.expectedResultArtifactCount)`` | $(ConvertTo-MarkdownCell $step.promotionBoundary) |"
}

$hardGateRows = foreach ($gate in $ownerReleaseCloseHardGates) {
  "| ``$(ConvertTo-MarkdownCell $gate.id)`` | ``$($gate.order)`` | $(ConvertTo-MarkdownCell $gate.currentState) | $(ConvertTo-MarkdownCell $gate.requiredReadyState) | ``$($gate.requiredFieldCount)`` | ``$($gate.rejectedSubstituteCount)`` | ``$($gate.sourceReadinessSignalCount)`` | ``$($gate.blockedRealInputCount)`` | ``$($gate.proofCandidateReady)`` | ``$($gate.sourceLinkageReady)`` | $(ConvertTo-MarkdownCell $gate.blockedReason) |"
}

$sourceArtifactEvidenceRows = foreach ($artifact in $sourceArtifactEvidence) {
  "| ``$(ConvertTo-MarkdownCell $artifact.id)`` | ``$(ConvertTo-MarkdownCell $artifact.artifactPath)`` | ``$($artifact.exists)`` | ``$(ConvertTo-MarkdownCell $artifact.sha256)`` | ``$(ConvertTo-MarkdownCell $artifact.state)`` | ``$($artifact.proofCandidateReady)`` | ``$($artifact.sourceProofLinkageReady)`` | ``$($artifact.canCloseReleaseIssue)`` |"
}

$sections = foreach ($step in $executionSteps) {
  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("### $($step.id)")
  $lines.Add("")
  $lines.Add("- actionRequiredId: ``$($step.actionRequiredId)``")
  $lines.Add("- sourceArtifact: ``$($step.inputContract.sourceArtifact)``")
  $lines.Add("- expected result artifacts: ``$($step.expectedResultArtifacts -join ', ')``")
  $lines.Add("- owner commands:")
  foreach ($command in $step.ownerCommands) {
    $lines.Add("  - ``$command``")
  }
  $lines.Add("- validator commands:")
  foreach ($command in $step.validatorCommands) {
    $lines.Add("  - ``$command``")
  }
  $lines.Add("- forbidden substitutes: ``$($step.forbiddenSubstitutes -join ', ')``")
  $lines.Add("- boundary: $(ConvertTo-MarkdownCell $step.promotionBoundary)")
  $lines -join "`r`n"
}

$markdown = @"
# Final Owner Execution Package

Generated at: ``$($record.generatedAtUtc)``

## Summary

- recordKind: ``$($record.recordKind)``
- packageState: ``$($record.packageState)``
- sourceWorklistState: ``$($record.sourceWorklistState)``
- sourceWorklistValidationState: ``$($record.sourceWorklistValidationState)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentState: ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentState)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState: ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus: ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount: ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount: ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount: ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount)``
- finalOwnerExecutionOneScreenPackValidationState: ``$($record.finalOwnerExecutionOneScreenPackValidationState)``
- finalOwnerStrictCloseExecutionOrderValidationState: ``$($record.finalOwnerStrictCloseExecutionOrderValidationState)``
- ownerPublicPublishResultCandidateValidationState: ``$($record.ownerPublicPublishResultCandidateValidationState)``
- publicPackageDownloadProofCandidateValidationState: ``$($record.publicPackageDownloadProofCandidateValidationState)``
- postPublishCleanConsumerProofResultValidationState: ``$($record.postPublishCleanConsumerProofResultValidationState)``
- releaseIssueCloseOwnerDecisionValidationState: ``$($record.releaseIssueCloseOwnerDecisionValidationState)``
- finalCloseStrictValidatorOutputState: ``$($record.finalCloseStrictValidatorOutputState)``
- releaseEvidenceBundleSha256: ``$($record.releaseEvidenceBundleSha256)``
- releaseCloseRealInputChainCount: ``$($record.releaseCloseRealInputChainCount)``
- releaseCloseRealInputChainRequiredFieldCount: ``$($record.releaseCloseRealInputChainRequiredFieldCount)``
- releaseCloseRealInputChainRejectedSubstituteCount: ``$($record.releaseCloseRealInputChainRejectedSubstituteCount)``
- releaseCloseRealInputChainSourceReadinessSignalCount: ``$($record.releaseCloseRealInputChainSourceReadinessSignalCount)``
- releaseCloseRealInputChainBlockedRealInputCount: ``$($record.releaseCloseRealInputChainBlockedRealInputCount)``
- ownerReleaseCloseHardGateCount: ``$($record.ownerReleaseCloseHardGateCount)``
- executionStepCount: ``$($record.executionStepCount)``
- blockedExecutionStepCount: ``$($record.blockedExecutionStepCount)``
- publicPackageDownloadProofCandidateReady: ``$($record.publicPackageDownloadProofCandidateReady)``
- postPublishProofCandidateReady: ``$($record.postPublishProofCandidateReady)``
- postPublishProofSourceLinkageReady: ``$($record.postPublishProofSourceLinkageReady)``
- publicDownloadCannotSubstitutePostPublishProof: ``True``
- postPublishValidationReadyCannotSubstituteProofCandidateReady: ``True``
- bundleHashCannotSubstituteFinalCloseDecision: ``True``
- strictCloseOutputCannotCloseIssue: ``True``
- sourceArtifactEvidenceCount: ``$($record.sourceArtifactEvidenceCount)``
- sourceArtifactEvidenceMissingCount: ``$($record.sourceArtifactEvidenceMissingCount)``
- sourceArtifactEvidenceSha256Count: ``$($record.sourceArtifactEvidenceSha256Count)``
- sourceArtifactEvidenceNonProofBoundaryCount: ``$($record.sourceArtifactEvidenceNonProofBoundaryCount)``
- performsPublish: ``False``
- canPromoteRuntimeProof: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Source Artifact Evidence

| ID | Artifact | Exists | SHA256 | State | Proof Candidate Ready | Source Linkage Ready | Can Close |
| --- | --- | --- | --- | --- | --- | --- | --- |
$($sourceArtifactEvidenceRows -join "`r`n")

## Release Close Hard Gates

| Gate | Order | Current State | Required Ready State | Required Fields | Rejected Substitutes | Source Signals | Blocked Inputs | Proof Candidate Ready | Source Linkage Ready | Blocked Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
$($hardGateRows -join "`r`n")

## Execution Steps

| Action | Lane | State | Required Inputs | Result Artifacts | Boundary |
| --- | --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Owner Step Details

$($sections -join "`r`n`r`n")

## Boundary

$($record.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Final owner execution package written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackageState=$($record.packageState) ExecutionSteps=$($record.executionStepCount) Blocked=$($record.blockedExecutionStepCount)"
