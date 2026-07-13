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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Test-StateReady {
  param([AllowNull()][object]$State)
  $text = [string]$State
  return $text.IndexOf("ready", [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $text.IndexOf("accepted", [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $text.IndexOf("passed", [StringComparison]::OrdinalIgnoreCase) -ge 0
}

function New-GateCheck {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$SourceArtifact,
    [string]$RequiredEvidence,
    [string]$BlockingReason
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    sourceArtifact = $SourceArtifact
    requiredEvidence = $RequiredEvidence
    blockingReason = if ($Passed) { "" } else { $BlockingReason }
    ownerActionRequired = -not $Passed
  }
}

$stagingImport = Read-JsonOrNull "artifacts\final-release\owner-real-proof-staging-workspace-import.json"
$stagingValidation = Read-JsonOrNull "artifacts\final-release\owner-real-proof-staging-workspace-validation.json"
$publicPublishCandidateValidation = Read-JsonOrNull "artifacts\final-release\owner-public-publish-execution-result-candidate-validation.json"
$postPublishProofValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-real-proof-from-owner-result-validation.json"
$finalCloseApprovalValidation = Read-JsonOrNull "artifacts\final-release\final-release-close-approval-real-input-from-owner-result-validation.json"
$acceptanceGate = Read-JsonOrNull "artifacts\final-release\final-public-publish-acceptance-gate.json"
$finalOwnerConvergenceGate = Read-JsonOrNull "artifacts\final-release\final-owner-real-proof-convergence-gate.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"

$stagingReady = $null -ne $stagingImport -and $null -ne $stagingValidation -and
  [bool](Get-PropertyOrDefault -Object $stagingImport -Name "readyForStrictImport" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $stagingValidation -Name "readyForStrictImport" -DefaultValue $false) -and
  [int](Get-PropertyOrDefault -Object $stagingImport -Name "failedBlockerCount" -DefaultValue 999) -eq 0 -and
  [int](Get-PropertyOrDefault -Object $stagingImport -Name "failedActionRequiredCount" -DefaultValue 999) -eq 0 -and
  [int](Get-PropertyOrDefault -Object $stagingValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0
$publicPublishAccepted = $null -ne $publicPublishCandidateValidation -and
  (Test-StateReady (Get-PropertyOrDefault -Object $publicPublishCandidateValidation -Name "validationState" -DefaultValue "")) -and
  [int](Get-PropertyOrDefault -Object $publicPublishCandidateValidation -Name "candidateItemCount" -DefaultValue 0) -gt 0 -and
  [int](Get-PropertyOrDefault -Object $publicPublishCandidateValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0 -and
  [int](Get-PropertyOrDefault -Object $publicPublishCandidateValidation -Name "failedActionRequiredCount" -DefaultValue 999) -eq 0
$postPublishAccepted = $null -ne $postPublishProofValidation -and
  (Test-StateReady (Get-PropertyOrDefault -Object $postPublishProofValidation -Name "validationState" -DefaultValue "")) -and
  [int](Get-PropertyOrDefault -Object $postPublishProofValidation -Name "readyProofCount" -DefaultValue 0) -gt 0 -and
  [int](Get-PropertyOrDefault -Object $postPublishProofValidation -Name "blockedProofLaneCount" -DefaultValue 999) -eq 0 -and
  [int](Get-PropertyOrDefault -Object $postPublishProofValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0 -and
  [int](Get-PropertyOrDefault -Object $postPublishProofValidation -Name "failedActionRequiredCount" -DefaultValue 999) -eq 0
$finalCloseApprovalAccepted = $null -ne $finalCloseApprovalValidation -and
  (Test-StateReady (Get-PropertyOrDefault -Object $finalCloseApprovalValidation -Name "validationState" -DefaultValue "")) -and
  [int](Get-PropertyOrDefault -Object $finalCloseApprovalValidation -Name "readyCloseApprovalCount" -DefaultValue 0) -gt 0 -and
  [int](Get-PropertyOrDefault -Object $finalCloseApprovalValidation -Name "blockedApprovalFieldCount" -DefaultValue 999) -eq 0 -and
  [int](Get-PropertyOrDefault -Object $finalCloseApprovalValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0 -and
  [int](Get-PropertyOrDefault -Object $finalCloseApprovalValidation -Name "failedActionRequiredCount" -DefaultValue 999) -eq 0
$acceptanceReady = $null -ne $acceptanceGate -and
  [string](Get-PropertyOrDefault -Object $acceptanceGate -Name "gateState" -DefaultValue "") -eq "ready-final-public-publish-acceptance" -and
  [bool](Get-PropertyOrDefault -Object $acceptanceGate -Name "readyForPublicReleaseClose" -DefaultValue $false) -and
  [int](Get-PropertyOrDefault -Object $acceptanceGate -Name "blockedGateCheckCount" -DefaultValue 999) -eq 0
$finalOwnerConvergenceReady = $null -ne $finalOwnerConvergenceGate -and
  [string](Get-PropertyOrDefault -Object $finalOwnerConvergenceGate -Name "gateState" -DefaultValue "") -eq "ready-final-owner-real-proof-convergence" -and
  [bool](Get-PropertyOrDefault -Object $finalOwnerConvergenceGate -Name "readyForFinalClose" -DefaultValue $false) -and
  [int](Get-PropertyOrDefault -Object $finalOwnerConvergenceGate -Name "blockedGateCheckCount" -DefaultValue 999) -eq 0
$auditClean = $null -ne $classificationAudit -and
  [string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "") -eq "classification-audit-passed-non-proof-boundaries-intact" -and
  [int](Get-PropertyOrDefault -Object $classificationAudit -Name "findingCount" -DefaultValue 999) -eq 0

$checks = @(
  New-GateCheck "owner-real-proof-staging-workspace-ready" $stagingReady "artifacts/final-release/owner-real-proof-staging-workspace-validation.json" "Owner staging workspace must supply all required evidence files and hashes for strict import." "Owner staging workspace is missing or still action-required."
  New-GateCheck "owner-public-publish-execution-result-accepted" $publicPublishAccepted "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json" "Owner public publish execution result must be imported with real public URL/hash/transcripts." "Owner public publish result is missing or still candidate-only."
  New-GateCheck "post-publish-clean-consumer-real-proof-accepted" $postPublishAccepted "artifacts/final-release/post-publish-clean-consumer-real-proof-from-owner-result-validation.json" "Public-channel PostPublish CleanConsumer restore/build/smoke proof must be accepted." "PostPublish CleanConsumer proof is missing or blocked."
  New-GateCheck "final-release-close-approval-real-input-accepted" $finalCloseApprovalAccepted "artifacts/final-release/final-release-close-approval-real-input-from-owner-result-validation.json" "Owner final release close approval must be accepted after real public publish and PostPublish proof." "Final close approval real input is missing or blocked."
  New-GateCheck "final-public-publish-acceptance-ready" $acceptanceReady "artifacts/final-release/final-public-publish-acceptance-gate.json" "Final public publish acceptance gate must be ready." "Final public publish acceptance gate is still blocked."
  New-GateCheck "final-owner-real-proof-convergence-ready" $finalOwnerConvergenceReady "artifacts/final-release/final-owner-real-proof-convergence-gate.json" "Final Owner real proof convergence gate must be ready." "Final Owner convergence is still blocked."
  New-GateCheck "release-evidence-classification-audit-clean" $auditClean "artifacts/final-release/release-evidence-classification-audit.json" "Release evidence classification audit must remain clean." "Release evidence classification audit is missing or blocked."
)

$blockedChecks = @($checks | Where-Object { -not [bool]$_.passed })
$ready = $blockedChecks.Count -eq 0
$gateState = if ($ready) { "ready-owner-real-evidence-end-to-end-release" } else { "blocked-owner-real-evidence-end-to-end-release-required" }

$record = [ordered]@{
  recordKind = "owner-real-evidence-end-to-end-release-gate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  gateState = $gateState
  gateCheckCount = $checks.Count
  blockedGateCheckCount = $blockedChecks.Count
  readyForReleaseClose = $ready
  stagingWorkspaceReady = $stagingReady
  ownerPublicPublishExecutionResultAccepted = $publicPublishAccepted
  postPublishCleanConsumerProofAccepted = $postPublishAccepted
  finalReleaseCloseApprovalAccepted = $finalCloseApprovalAccepted
  finalPublicPublishAcceptanceReady = $acceptanceReady
  finalOwnerRealProofConvergenceReady = $finalOwnerConvergenceReady
  releaseEvidenceClassificationAuditClean = $auditClean
  failedBlockerCountIsNotProof = $true
  checks = @($checks)
  ownerActionRequired = -not $ready
  passed = $ready
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
  boundary = "Owner real evidence end-to-end release gate is read-only. It never executes dotnet nuget push, never stores tokens, never uploads packages, and never turns templates, candidates, dashboards, runbooks, command packs, local feed, ProjectReference, direct .nupkg, pre-publish smoke, or failedBlockerCount=0 into proof. It can become release-close-ready only after Owner staging evidence, real public publish result, public-channel PostPublish CleanConsumer proof, final close approval, final public publish acceptance, final Owner convergence, and classification audit all pass."
}

$jsonPath = Join-Path $OutputRoot "owner-real-evidence-end-to-end-release-gate.json"
$markdownPath = Join-Path $OutputRoot "owner-real-evidence-end-to-end-release-gate.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)

$rows = foreach ($check in $checks) {
  "| ``$($check.id)`` | ``$($check.passed)`` | $(ConvertTo-MarkdownCell $check.sourceArtifact) | $(ConvertTo-MarkdownCell $check.blockingReason) |"
}

Write-Utf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Real Evidence End-to-End Release Gate",
  "",
  "- gateState: ``$gateState``",
  "- blockedGateCheckCount: ``$($blockedChecks.Count)``",
  "- readyForReleaseClose: ``$ready``",
  "- performsPublish: ``False``",
  "- canPublishPublicly: ``False``",
  "- failedBlockerCountIsNotProof: ``True``",
  "",
  "| ID | Passed | Source Artifact | Blocking Reason |",
  "|---|---:|---|---|",
  @($rows),
  "",
  "## Boundary",
  "",
  $record.boundary
)

Write-Host "OwnerRealEvidenceEndToEndReleaseGateState=$gateState Blocked=$($blockedChecks.Count)"
if ($FailOnNotReady.IsPresent -and -not $ready) {
  throw "Owner real evidence end-to-end release gate is not ready."
}
if ($Strict.IsPresent -and $checks.Count -lt 7) {
  throw "Owner real evidence end-to-end release gate did not evaluate all required checks."
}
