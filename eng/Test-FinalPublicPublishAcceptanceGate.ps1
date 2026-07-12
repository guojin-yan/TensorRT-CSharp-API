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

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
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

$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"
$commandDryContractValidation = Read-JsonOrNull "artifacts\final-release\final-public-publish-command-dry-contract-validation.json"
$ownerPublicPublishCandidateValidation = Read-JsonOrNull "artifacts\final-release\owner-public-publish-execution-result-candidate-validation.json"
$postPublishProofValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-real-proof-from-owner-result-validation.json"
$finalCloseApprovalValidation = Read-JsonOrNull "artifacts\final-release\final-release-close-approval-real-input-from-owner-result-validation.json"
$finalOwnerConvergenceGate = Read-JsonOrNull "artifacts\final-release\final-owner-real-proof-convergence-gate.json"

$bundleBoundarySafe = $null -ne $releaseEvidenceBundle -and
  -not [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canPublishPublicly" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canCloseReleaseIssue" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "isRuntimeExecutionProof" -DefaultValue $true)
$auditClean = $null -ne $classificationAudit -and
  [string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "") -eq "classification-audit-passed-non-proof-boundaries-intact" -and
  [int](Get-PropertyOrDefault -Object $classificationAudit -Name "findingCount" -DefaultValue 999) -eq 0
$manualCommandContractSafe = $null -ne $commandDryContractValidation -and
  [string](Get-PropertyOrDefault -Object $commandDryContractValidation -Name "validationState" -DefaultValue "") -eq "blocked-public-publish-owner-manual-execution-required" -and
  [int](Get-PropertyOrDefault -Object $commandDryContractValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0 -and
  [bool](Get-PropertyOrDefault -Object $commandDryContractValidation -Name "isDryContract" -DefaultValue $false) -and
  -not [bool](Get-PropertyOrDefault -Object $commandDryContractValidation -Name "performsPublish" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $commandDryContractValidation -Name "canPublishPublicly" -DefaultValue $true)
$ownerPublicPublishResultAccepted = $null -ne $ownerPublicPublishCandidateValidation -and
  (Test-StateReady (Get-PropertyOrDefault -Object $ownerPublicPublishCandidateValidation -Name "validationState" -DefaultValue "")) -and
  [int](Get-PropertyOrDefault -Object $ownerPublicPublishCandidateValidation -Name "candidateItemCount" -DefaultValue 0) -gt 0 -and
  [int](Get-PropertyOrDefault -Object $ownerPublicPublishCandidateValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0 -and
  [int](Get-PropertyOrDefault -Object $ownerPublicPublishCandidateValidation -Name "failedActionRequiredCount" -DefaultValue 999) -eq 0
$postPublishProofAccepted = $null -ne $postPublishProofValidation -and
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
$finalOwnerConvergenceReady = $null -ne $finalOwnerConvergenceGate -and
  [string](Get-PropertyOrDefault -Object $finalOwnerConvergenceGate -Name "gateState" -DefaultValue "") -eq "ready-final-owner-real-proof-convergence" -and
  [bool](Get-PropertyOrDefault -Object $finalOwnerConvergenceGate -Name "readyForFinalClose" -DefaultValue $false) -and
  [int](Get-PropertyOrDefault -Object $finalOwnerConvergenceGate -Name "blockedGateCheckCount" -DefaultValue 999) -eq 0

$checks = @(
  New-GateCheck "release-evidence-bundle-boundary-safe" $bundleBoundarySafe "artifacts/final-release/release-evidence-bundle.json" "Release evidence bundle exists and still reports canPublishPublicly=false, canCloseReleaseIssue=false, and isRuntimeExecutionProof=false." "Refresh release evidence and keep aggregate bundle non-proof."
  New-GateCheck "release-evidence-classification-audit-clean" $auditClean "artifacts/final-release/release-evidence-classification-audit.json" "Classification audit must pass with findingCount=0." "Run Test-ReleaseEvidenceClassificationAudit.ps1 -Strict and resolve any classification boundary findings."
  New-GateCheck "manual-public-publish-command-contract-safe" $manualCommandContractSafe "artifacts/final-release/final-public-publish-command-dry-contract-validation.json" "Manual publish command dry contract must be structurally valid and non-executing." "Generate and validate the dry contract; automation must not execute dotnet nuget push."
  New-GateCheck "owner-public-publish-real-result-accepted" $ownerPublicPublishResultAccepted "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json" "Owner must import a real public publish result with package URL/hash/transcripts and zero blockers." "Owner public publish execution result is still missing or blocked."
  New-GateCheck "post-publish-clean-consumer-real-proof-accepted" $postPublishProofAccepted "artifacts/final-release/post-publish-clean-consumer-real-proof-from-owner-result-validation.json" "Owner must import accepted post-publish CleanConsumer proof from the public package source." "PostPublish clean consumer proof is still missing or blocked."
  New-GateCheck "final-release-close-approval-real-input-accepted" $finalCloseApprovalAccepted "artifacts/final-release/final-release-close-approval-real-input-from-owner-result-validation.json" "Owner must import real final close approval after public publish and PostPublish proof acceptance." "Final release close approval real input is still missing or blocked."
  New-GateCheck "final-owner-real-proof-convergence-ready" $finalOwnerConvergenceReady "artifacts/final-release/final-owner-real-proof-convergence-gate.json" "Final Owner real proof convergence gate must be ready with zero blocked checks." "Final Owner real proof convergence is still blocked."
)

$blockedChecks = @($checks | Where-Object { -not [bool]$_.passed })
$ready = $blockedChecks.Count -eq 0
$gateState = if ($ready) { "ready-final-public-publish-acceptance" } else { "blocked-final-public-publish-owner-evidence-required" }

$record = [ordered]@{
  recordKind = "final-public-publish-acceptance-gate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  gateState = $gateState
  gateCheckCount = $checks.Count
  blockedGateCheckCount = $blockedChecks.Count
  readyForPublicReleaseClose = $ready
  ownerPublicPublishResultAccepted = $ownerPublicPublishResultAccepted
  postPublishCleanConsumerProofAccepted = $postPublishProofAccepted
  finalCloseApprovalAccepted = $finalCloseApprovalAccepted
  finalOwnerRealProofConvergenceReady = $finalOwnerConvergenceReady
  failedBlockerCountIsNotProof = $true
  checks = @($checks)
  forbiddenNonProofSubstitutes = @(
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
  )
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
  boundary = "Final public publish acceptance gate is read-only. It never executes dotnet nuget push, never stores tokens, never uploads packages, and never turns templates, candidates, dashboards, command packs, local feed, ProjectReference, direct .nupkg, dry-run, manual approval, queued workflow, missing self-hosted runner, sidecar-only, TensorRtExec report, pre-publish smoke, or failedBlockerCount=0 into proof. It can become close-ready only after real Owner public publish result, public-channel PostPublish CleanConsumer proof, final close approval, final Owner convergence, and classification audit all pass."
}

$jsonPath = Join-Path $OutputRoot "final-public-publish-acceptance-gate.json"
$markdownPath = Join-Path $OutputRoot "final-public-publish-acceptance-gate.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)

$rows = foreach ($check in $checks) {
  "| ``$($check.id)`` | ``$($check.passed)`` | $(ConvertTo-MarkdownCell $check.sourceArtifact) | $(ConvertTo-MarkdownCell $check.requiredEvidence) | $(ConvertTo-MarkdownCell $check.blockingReason) | ``$($check.ownerActionRequired)`` |"
}

$markdownLines = @(
  "# Final Public Publish Acceptance Gate",
  "",
  "- gateState: ``$gateState``",
  "- blockedGateCheckCount: ``$($blockedChecks.Count)``",
  "- readyForPublicReleaseClose: ``$ready``",
  "- performsPublish: ``False``",
  "- canPublishPublicly: ``False``",
  "- failedBlockerCountIsNotProof: ``True``",
  "",
  "| ID | Passed | Source Artifact | Required Evidence | Blocking Reason | Owner Action Required |",
  "|---|---:|---|---|---|---:|"
)
$markdownLines += $rows
$markdownLines += @(
  "",
  "## Boundary",
  "",
  $record.boundary
)
Write-Utf8File -LiteralPath $markdownPath -InputObject $markdownLines

Write-Host "FinalPublicPublishAcceptanceGateState=$gateState Blocked=$($blockedChecks.Count)"
if ($FailOnNotReady.IsPresent -and -not $ready) {
  throw "Final public publish acceptance gate is not ready."
}
if ($Strict.IsPresent -and $checks.Count -lt 7) {
  throw "Final public publish acceptance gate did not evaluate all required checks."
}
