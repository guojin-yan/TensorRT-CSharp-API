[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict,
  [switch]$FailOnNotReady
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function ConvertTo-FlatStringLines {
  param([AllowNull()][object]$Value)

  foreach ($item in @($Value)) {
    if ($null -eq $item) {
      ""
    }
    elseif ($item -is [string]) {
      $item
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { [string]$child }
    }
    else {
      [string]$item
    }
  }
}
function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $lines = @(ConvertTo-FlatStringLines -Value $InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-GateCheck {
  param([string]$Id, [bool]$Passed, [string]$SourceArtifact, [string]$RequiredEvidence)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    sourceArtifact = $SourceArtifact
    requiredEvidence = $RequiredEvidence
    ownerActionRequired = -not $Passed
  }
}

$externalCandidate = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-execution-result-candidate.json"
$postPublishCandidate = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-candidate.json"
$rollbackReview = Read-JsonOrNull "artifacts\final-release\rollback-review.json"
$releaseRollbackReview = Read-JsonOrNull "artifacts\final-release\release-rollback-review.json"
$finalOwnerRollbackReviewImport = Read-JsonOrNull "artifacts\final-release\final-owner-rollback-review-import.json"
$finalOwnerRollbackReviewValidation = Read-JsonOrNull "artifacts\final-release\final-owner-rollback-review-validation.json"
$finalCloseDecision = Read-JsonOrNull "artifacts\final-release\final-close-decision.json"
$releaseIssueCloseRecord = Read-JsonOrNull "artifacts\final-release\release-issue-close-record.json"
$finalOwnerCloseDecisionImport = Read-JsonOrNull "artifacts\final-release\final-owner-close-decision-import.json"
$finalOwnerCloseDecisionValidation = Read-JsonOrNull "artifacts\final-release\final-owner-close-decision-validation.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"
$closeReadiness = Read-JsonOrNull "artifacts\final-release\final-owner-execution-close-readiness-from-real-input.json"
$closeReadinessValidation = Read-JsonOrNull "artifacts\final-release\final-owner-execution-close-readiness-from-real-input-validation.json"

$rollbackPresent = ($null -ne $rollbackReview -or $null -ne $releaseRollbackReview) -or
  ([bool](Get-PropertyOrDefault -Object $finalOwnerRollbackReviewImport -Name "rollbackReviewReady" -DefaultValue $false) -and
    [string](Get-PropertyOrDefault -Object $finalOwnerRollbackReviewValidation -Name "validationState" -DefaultValue "") -eq "final-owner-rollback-review-validation-ready-non-proof")
$finalDecisionPresent = ($null -ne $finalCloseDecision -or $null -ne $releaseIssueCloseRecord) -or
  ([bool](Get-PropertyOrDefault -Object $finalOwnerCloseDecisionImport -Name "finalCloseDecisionReady" -DefaultValue $false) -and
    [string](Get-PropertyOrDefault -Object $finalOwnerCloseDecisionValidation -Name "validationState" -DefaultValue "") -eq "final-owner-close-decision-validation-ready-non-proof")
$externalReady = [bool](Get-PropertyOrDefault -Object $externalCandidate -Name "proofCandidateReady" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $externalCandidate -Name "isRuntimeExecutionProof" -DefaultValue $false)
$postPublishReady = [bool](Get-PropertyOrDefault -Object $postPublishCandidate -Name "proofCandidateReady" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $postPublishCandidate -Name "isPostPublishProof" -DefaultValue $false)
$auditClean = $null -ne $classificationAudit -and [int](Get-PropertyOrDefault -Object $classificationAudit -Name "findingCount" -DefaultValue 999) -eq 0
$closeReady = $null -ne $closeReadiness -and [bool](Get-PropertyOrDefault -Object $closeReadiness -Name "canCloseReleaseIssue" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $closeReadiness -Name "blockedReadinessCheckCount" -DefaultValue 999) -eq 0
$closeValidationReady = $null -ne $closeReadinessValidation -and [bool](Get-PropertyOrDefault -Object $closeReadinessValidation -Name "canCloseReleaseIssue" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $closeReadinessValidation -Name "failedActionRequiredCount" -DefaultValue 999) -eq 0

$checks = @(
  New-GateCheck "external-clean-consumer-proof-candidate-ready" $externalReady "artifacts/final-release/external-clean-consumer-execution-result-candidate.json" "External CleanConsumer proof candidate must be ready and marked as real runtime execution proof."
  New-GateCheck "post-publish-clean-consumer-proof-candidate-ready" $postPublishReady "artifacts/final-release/post-publish-clean-consumer-proof-result-candidate.json" "PostPublish proof candidate must be ready and marked as post-publish proof."
  New-GateCheck "rollback-review-present" $rollbackPresent "artifacts/final-release/rollback-review.json" "Rollback review artifact must be present."
  New-GateCheck "final-close-decision-present" $finalDecisionPresent "artifacts/final-release/final-close-decision.json" "Final close decision artifact must be present."
  New-GateCheck "release-evidence-classification-audit-clean" $auditClean "artifacts/final-release/release-evidence-classification-audit.json" "Release evidence classification audit must have findingCount=0."
  New-GateCheck "final-owner-close-readiness-all-checks-pass" ($closeReady -and $closeValidationReady) "artifacts/final-release/final-owner-execution-close-readiness-from-real-input.json" "Close readiness and validation must both show canCloseReleaseIssue=true with zero blocked/action-required checks."
)

$blockedChecks = @($checks | Where-Object { -not [bool]$_.passed })
$ready = $blockedChecks.Count -eq 0
$gateState = if ($ready) { "ready-final-owner-real-proof-convergence" } else { "blocked-final-owner-real-proof-convergence-required" }

$record = [pscustomobject]@{
  recordKind = "final-owner-real-proof-convergence-gate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  gateState = $gateState
  gateCheckCount = $checks.Count
  blockedGateCheckCount = $blockedChecks.Count
  readyForFinalClose = $ready
  externalCleanConsumerProofReady = $externalReady
  postPublishProofReady = $postPublishReady
  rollbackReviewPresent = $rollbackPresent
  rollbackReviewImportState = [string](Get-PropertyOrDefault -Object $finalOwnerRollbackReviewImport -Name "importState" -DefaultValue "missing-final-owner-rollback-review-import")
  rollbackReviewValidationState = [string](Get-PropertyOrDefault -Object $finalOwnerRollbackReviewValidation -Name "validationState" -DefaultValue "missing-final-owner-rollback-review-validation")
  finalCloseDecisionPresent = $finalDecisionPresent
  finalCloseDecisionImportState = [string](Get-PropertyOrDefault -Object $finalOwnerCloseDecisionImport -Name "importState" -DefaultValue "missing-final-owner-close-decision-import")
  finalCloseDecisionValidationState = [string](Get-PropertyOrDefault -Object $finalOwnerCloseDecisionValidation -Name "validationState" -DefaultValue "missing-final-owner-close-decision-validation")
  classificationAuditFindingCount = [int](Get-PropertyOrDefault -Object $classificationAudit -Name "findingCount" -DefaultValue 999)
  closeReadinessBlockedCount = [int](Get-PropertyOrDefault -Object $closeReadiness -Name "blockedReadinessCheckCount" -DefaultValue 999)
  closeReadinessFailedActionRequiredCount = [int](Get-PropertyOrDefault -Object $closeReadinessValidation -Name "failedActionRequiredCount" -DefaultValue 999)
  failedBlockerCountIsNotProof = $true
  checks = @($checks)
  ownerActionRequired = -not $ready
  passed = $ready
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $ready
  canPublishPublicly = $false
  canCloseReleaseIssue = $ready
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $ready
  boundary = "Final Owner real proof convergence gate reads proof candidates and owner governance artifacts only. Template, candidate, runbook, command pack, contract, gap matrix, and failedBlockerCount=0 are not sufficient proof; this gate is not publish approval and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-real-proof-convergence-gate.json"
$markdownPath = Join-Path $OutputRoot "final-owner-real-proof-convergence-gate.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($check in $checks) {
  "| ``$($check.id)`` | ``$($check.passed)`` | $(ConvertTo-MarkdownCell $check.requiredEvidence) |"
}

Write-Utf8File -LiteralPath $markdownPath -InputObject @(
  "# Final Owner Real Proof Convergence Gate",
  "",
  "- gateState: ``$gateState``",
  "- blockedGateCheckCount: ``$($blockedChecks.Count)``",
  "- readyForFinalClose: ``$ready``",
  "- failedBlockerCountIsNotProof: ``True``",
  "",
  "| ID | Passed | Required Evidence |",
  "|---|---:|---|",
  @($rows),
  "",
  "## Boundary",
  "",
  $record.boundary
)

Write-Host "FinalOwnerRealProofConvergenceGateState=$gateState Blocked=$($blockedChecks.Count)"
if ($FailOnNotReady.IsPresent -and -not $ready) {
  throw "Final Owner real proof convergence gate is not ready."
}
if ($Strict.IsPresent -and $checks.Count -lt 6) {
  throw "Final Owner real proof convergence gate did not evaluate all required checks."
}
