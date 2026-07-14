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
$scriptDir = Join-Path $RepositoryRoot "eng"

function Invoke-OwnerScript {
  param([string]$Name, [string[]]$Arguments = @())
  $commandArgs = @{
    RepositoryRoot = $RepositoryRoot
    OutputRoot = $OutputRoot
  }
  if ($Arguments -contains "-Strict") { $commandArgs.Strict = $true }
  & (Join-Path $scriptDir $Name) @commandArgs | Out-Null
}

function Read-FinalJsonOrNull {
  param([string]$Name)
  return Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath (Join-Path "artifacts\final-release" $Name)
}

function Get-FileSha256OrEmpty {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return "" }
  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function New-CloseGate {
  param([string]$Id, [string]$Title, [bool]$Ready, [string]$BlockedReason, [string[]]$SourceArtifacts = @())
  [pscustomobject]@{
    id = $Id
    title = $Title
    ready = $Ready
    acceptedRealInput = $Ready
    blockedReason = if ($Ready) { "" } else { $BlockedReason }
    ownerActionRequired = -not $Ready
    sourceArtifacts = @($SourceArtifacts)
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Release close bridge gate is read-only admission status; it does not publish, use tokens, close the release issue, or substitute real proof."
  }
}

Invoke-OwnerScript "Export-PostPublishProofValidatorBridge.ps1"
Invoke-OwnerScript "Test-PostPublishProofValidatorBridge.ps1" @("-Strict")
Invoke-OwnerScript "Import-OwnerRealProofStagingWorkspace.ps1"
Invoke-OwnerScript "Test-OwnerRealProofStagingWorkspace.ps1" @("-Strict")

$postPublishBridge = Read-FinalJsonOrNull "post-publish-proof-validator-bridge-validation.json"
$rollbackReview = Read-FinalJsonOrNull "final-owner-rollback-review-import.json"
$rollbackValidation = Read-FinalJsonOrNull "final-owner-rollback-review-validation.json"
$closeDecision = Read-FinalJsonOrNull "final-owner-close-decision-import.json"
$closeDecisionValidation = Read-FinalJsonOrNull "final-owner-close-decision-validation.json"
$ownerStagingImport = Read-FinalJsonOrNull "owner-real-proof-staging-workspace-import.json"
$ownerStagingValidation = Read-FinalJsonOrNull "owner-real-proof-staging-workspace-validation.json"
$classificationAudit = Read-FinalJsonOrNull "release-evidence-classification-audit.json"
$realOwnerProofConvergence = Read-FinalJsonOrNull "real-owner-proof-convergence-dashboard.json"
$realOwnerProofConvergenceValidation = Read-FinalJsonOrNull "real-owner-proof-convergence-dashboard-validation.json"

$postPublishReady = [bool](Get-PropertyOrDefault -Object $postPublishBridge -Name "allPostPublishInputsAccepted" -DefaultValue $false)
$rollbackReady = [bool](Get-PropertyOrDefault -Object $rollbackReview -Name "rollbackReviewReady" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $rollbackValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0
$closeDecisionReady = [bool](Get-PropertyOrDefault -Object $closeDecision -Name "finalCloseDecisionReady" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $closeDecisionValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0
$ownerStagingReady = [bool](Get-PropertyOrDefault -Object $ownerStagingImport -Name "readyForStrictImport" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $ownerStagingValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0
$releaseEvidenceBundleSha256 = Get-FileSha256OrEmpty "artifacts\final-release\release-evidence-bundle.json"
$classificationAuditSha256 = Get-FileSha256OrEmpty "artifacts\final-release\release-evidence-classification-audit.json"
$classificationAuditPassed = [string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "") -eq "classification-audit-passed-non-proof-boundaries-intact"
$realOwnerProofConvergenceReady = [bool](Get-PropertyOrDefault -Object $realOwnerProofConvergence -Name "allRealOwnerProofInputsAccepted" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $realOwnerProofConvergenceValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0

$gates = @(
  New-CloseGate -Id "post-publish-proof-validator-bridge" -Title "Post-publish proof validator bridge" -Ready $postPublishReady -BlockedReason "post-publish-proof-validator-bridge-not-accepted" -SourceArtifacts @("artifacts/final-release/post-publish-proof-validator-bridge-validation.json")
  New-CloseGate -Id "final-owner-rollback-review" -Title "Final Owner rollback review" -Ready $rollbackReady -BlockedReason "final-owner-rollback-review-missing" -SourceArtifacts @("artifacts/final-release/final-owner-rollback-review-validation.json")
  New-CloseGate -Id "final-owner-close-decision" -Title "Final Owner close decision" -Ready $closeDecisionReady -BlockedReason "final-owner-close-decision-missing" -SourceArtifacts @("artifacts/final-release/final-owner-close-decision-validation.json")
  New-CloseGate -Id "owner-real-proof-staging-workspace" -Title "Owner real proof staging workspace strict import" -Ready $ownerStagingReady -BlockedReason "owner-real-proof-staging-workspace-not-strict-ready" -SourceArtifacts @("artifacts/final-release/owner-real-proof-staging-workspace-validation.json")
  New-CloseGate -Id "release-evidence-bundle-sha256" -Title "Release evidence bundle SHA256" -Ready (Test-Sha256Text $releaseEvidenceBundleSha256) -BlockedReason "release-evidence-bundle-sha256-missing" -SourceArtifacts @("artifacts/final-release/release-evidence-bundle.json")
  New-CloseGate -Id "classification-audit-sha256" -Title "Classification audit SHA256" -Ready ((Test-Sha256Text $classificationAuditSha256) -and $classificationAuditPassed) -BlockedReason "classification-audit-sha256-or-pass-state-missing" -SourceArtifacts @("artifacts/final-release/release-evidence-classification-audit.json")
  New-CloseGate -Id "real-owner-proof-convergence" -Title "Real Owner proof nine-lane convergence" -Ready $realOwnerProofConvergenceReady -BlockedReason "real-owner-proof-nine-lane-convergence-not-ready" -SourceArtifacts @("artifacts/final-release/real-owner-proof-convergence-dashboard-validation.json")
)

$readyGateCount = @($gates | Where-Object { [bool]$_.ready }).Count
$blockedGateCount = $gates.Count - $readyGateCount
$acceptedRealInputIds = @($gates | Where-Object { [bool]$_.acceptedRealInput } | ForEach-Object { [string]$_.id })
$blockedReasons = @($gates | Where-Object { -not [bool]$_.ready } | ForEach-Object { [string]$_.blockedReason })
$rejectedNonProofStates = @(
  "template",
  "dashboard",
  "dry-run",
  "local-feed",
  "ProjectReference",
  "direct-nupkg",
  "queued-workflow",
  "staging-shape-valid-only",
  "public-package-hash-only",
  "validation-ready-without-owner-proof",
  "sample-build-only",
  "mock-output"
)
$ownerNextActions = @(
  "import-real-post-publish-clean-consumer-proof",
  "import-real-public-package-download-hashes",
  "import-article-publication-proof",
  "import-yolovision-real-model-proof",
  "import-final-owner-rollback-review",
  "import-final-owner-close-decision",
  "rerun-classification-audit-and-release-evidence-bundle"
)
$allCloseInputsReady = $gates.Count -gt 0 -and $readyGateCount -eq $gates.Count
$bridgeState = if ($allCloseInputsReady) { "release-close-final-bridge-ready-for-owner-manual-close-review-non-proof" } else { "blocked-release-close-final-bridge-real-owner-proof-required" }
$sourceArtifacts = @($gates | ForEach-Object { $_.sourceArtifacts } | Select-Object -Unique)

$record = [pscustomobject]@{
  recordKind = "release-close-final-bridge"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  bridgeState = $bridgeState
  gateCount = $gates.Count
  readyGateCount = $readyGateCount
  blockedGateCount = $blockedGateCount
  allCloseInputsReady = $allCloseInputsReady
  acceptedRealInputIds = @($acceptedRealInputIds)
  rejectedNonProofStates = @($rejectedNonProofStates)
  closeGates = @($gates)
  releaseEvidenceBundleSha256 = $releaseEvidenceBundleSha256
  classificationAuditSha256 = $classificationAuditSha256
  classificationAuditPassed = $classificationAuditPassed
  blockedReasonCount = $blockedReasons.Count
  blockedReasons = @($blockedReasons)
  ownerNextActions = @($ownerNextActions)
  sourceArtifacts = @($sourceArtifacts)
  publicPackageHashCannotSubstitutePostPublishProof = $true
  shapeValidCannotSubstituteReleaseCloseProof = $true
  ownerActionRequired = -not $allCloseInputsReady
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Release close final bridge cross-checks post-publish proof admission, final rollback review, final close decision, staging strict import, release evidence bundle hash, classification audit hash, and unified nine-lane Owner proof convergence for Owner review only. It does not publish, does not use tokens, does not run dotnet nuget push, does not close the release issue, and rejects template, dashboard, dry-run, local feed, ProjectReference, direct nupkg, queued workflow, staging shape-valid-only, public-package-hash-only, validation-ready, sample-build-only, and mock-output substitutes. It is not runtime proof, not post-publish proof, not release close approval, and not package push."
}

Write-Utf8File -LiteralPath (Join-Path $OutputRoot "release-close-final-bridge.json") -InputObject ($record | ConvertTo-Json -Depth 16)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "release-close-final-bridge.md") -InputObject @(
  "# Release Close Final Bridge",
  "",
  "- bridgeState: ``$bridgeState``",
  "- readyGateCount: ``$readyGateCount/$($gates.Count)``",
  "- blockedGateCount: ``$blockedGateCount``",
  "- canCloseReleaseIssue: ``False``",
  "- publicPackageHashCannotSubstitutePostPublishProof: ``True``",
  "",
  $record.boundary
)
Write-Host "ReleaseCloseFinalBridgeState=$bridgeState ReadyGates=$readyGateCount/$($gates.Count) BlockedGates=$blockedGateCount"
