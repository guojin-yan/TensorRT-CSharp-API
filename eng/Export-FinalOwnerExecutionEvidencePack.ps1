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

function New-PackGate {
  param(
    [string]$Id,
    [string]$Title,
    [bool]$Ready,
    [string]$BlockedReason,
    [string[]]$SourceArtifacts
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    ready = $Ready
    blockedReason = if ($Ready) { "" } else { $BlockedReason }
    sourceArtifacts = @($SourceArtifacts)
    ownerActionRequired = -not $Ready
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Final Owner execution evidence pack gate is read-only status. It does not publish, does not use tokens, does not close the release issue, and is not runtime proof, not post-publish proof, not release close approval, and not package push."
  }
}

Invoke-OwnerScript "Export-FinalOwnerExecutionInputSkeleton.ps1"
Invoke-OwnerScript "Test-FinalOwnerExecutionInputSkeleton.ps1" @("-Strict")
Invoke-OwnerScript "Export-ReleaseCloseFinalBridge.ps1"
Invoke-OwnerScript "Test-ReleaseCloseFinalBridge.ps1" @("-Strict")

$inputSkeletonValidation = Read-FinalJsonOrNull "final-owner-execution-input-skeleton-validation.json"
$githubCiValidation = Read-FinalJsonOrNull "github-ci-evidence-from-owner-input-validation.json"
$bundleHashValidation = Read-FinalJsonOrNull "release-evidence-bundle-hash-review-validation.json"
$classificationHashValidation = Read-FinalJsonOrNull "classification-audit-hash-review-validation.json"
$postPublishBridgeValidation = Read-FinalJsonOrNull "post-publish-proof-validator-bridge-validation.json"
$releaseCloseBridgeValidation = Read-FinalJsonOrNull "release-close-final-bridge-validation.json"
$realOwnerProofConvergenceValidation = Read-FinalJsonOrNull "real-owner-proof-convergence-dashboard-validation.json"

$inputSkeletonReady = [int](Get-PropertyOrDefault -Object $inputSkeletonValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0
$githubCiAccepted = [bool](Get-PropertyOrDefault -Object $githubCiValidation -Name "ciEvidenceAccepted" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $githubCiValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0
$bundleHashAccepted = [bool](Get-PropertyOrDefault -Object $bundleHashValidation -Name "reviewAccepted" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $bundleHashValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0
$classificationHashAccepted = [bool](Get-PropertyOrDefault -Object $classificationHashValidation -Name "reviewAccepted" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $classificationHashValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0
$postPublishAccepted = [bool](Get-PropertyOrDefault -Object $postPublishBridgeValidation -Name "allPostPublishInputsAccepted" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $postPublishBridgeValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0
$releaseCloseAccepted = [bool](Get-PropertyOrDefault -Object $releaseCloseBridgeValidation -Name "allCloseInputsReady" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $releaseCloseBridgeValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0
$realOwnerProofConvergenceAccepted = [bool](Get-PropertyOrDefault -Object $realOwnerProofConvergenceValidation -Name "allRealOwnerProofInputsAccepted" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $realOwnerProofConvergenceValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0

$gates = @(
  New-PackGate -Id "final-owner-execution-input-skeleton" -Title "Final Owner execution input skeleton" -Ready $inputSkeletonReady -BlockedReason "input-skeleton-invalid" -SourceArtifacts @("artifacts/final-release/final-owner-execution-input-skeleton-validation.json")
  New-PackGate -Id "github-ci-evidence-from-owner-input" -Title "GitHub CI evidence from Owner input" -Ready $githubCiAccepted -BlockedReason "github-ci-owner-evidence-required" -SourceArtifacts @("artifacts/final-release/github-ci-evidence-from-owner-input-validation.json")
  New-PackGate -Id "release-evidence-bundle-hash-review" -Title "Release evidence bundle hash review" -Ready $bundleHashAccepted -BlockedReason "owner-bundle-hash-review-required" -SourceArtifacts @("artifacts/final-release/release-evidence-bundle-hash-review-validation.json")
  New-PackGate -Id "classification-audit-hash-review" -Title "Classification audit hash review" -Ready $classificationHashAccepted -BlockedReason "owner-classification-audit-hash-review-required" -SourceArtifacts @("artifacts/final-release/classification-audit-hash-review-validation.json")
  New-PackGate -Id "post-publish-proof-validator-bridge" -Title "Post-publish proof validator bridge" -Ready $postPublishAccepted -BlockedReason "post-publish-proof-validator-bridge-not-accepted" -SourceArtifacts @("artifacts/final-release/post-publish-proof-validator-bridge-validation.json")
  New-PackGate -Id "release-close-final-bridge" -Title "Release close final bridge" -Ready $releaseCloseAccepted -BlockedReason "release-close-final-bridge-not-accepted" -SourceArtifacts @("artifacts/final-release/release-close-final-bridge-validation.json")
  New-PackGate -Id "real-owner-proof-convergence-dashboard" -Title "Real Owner proof nine-lane convergence dashboard" -Ready $realOwnerProofConvergenceAccepted -BlockedReason "real-owner-proof-nine-lane-convergence-not-ready" -SourceArtifacts @("artifacts/final-release/real-owner-proof-convergence-dashboard-validation.json")
)

$readyGateCount = @($gates | Where-Object { [bool]$_.ready }).Count
$blockedGateCount = $gates.Count - $readyGateCount
$blockedReasons = @($gates | Where-Object { -not [bool]$_.ready } | ForEach-Object { [string]$_.blockedReason })
$rejectedNonProofStates = @("template", "dashboard", "dry-run", "local-feed", "ProjectReference", "direct-nupkg", "queued-workflow", "local-build", "local-test", "hash-only", "validation-ready-without-owner-proof", "staging-shape-valid-only")
$ownerNextActions = @(
  "fill-final-owner-execution-input-skeleton",
  "import-real-github-ci-success-evidence",
  "review-current-release-evidence-bundle-sha256",
  "review-current-classification-audit-sha256",
  "import-real-public-package-url-hash-proof",
  "import-real-external-cleanconsumer-post-publish-proof",
  "import-article-publication-proof",
  "import-yolovision-real-model-proof",
  "import-final-owner-rollback-review",
  "import-final-owner-close-decision"
)
$allOwnerExecutionInputsReady = $gates.Count -gt 0 -and $readyGateCount -eq $gates.Count
$packState = if ($allOwnerExecutionInputsReady) { "final-owner-execution-evidence-pack-ready-for-owner-release-review-non-proof" } else { "blocked-final-owner-execution-evidence-pack-real-owner-proof-required" }
$sourceArtifacts = @($gates | ForEach-Object { $_.sourceArtifacts } | Select-Object -Unique)

$record = [pscustomobject]@{
  recordKind = "final-owner-execution-evidence-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  packState = $packState
  gateCount = $gates.Count
  readyGateCount = $readyGateCount
  blockedGateCount = $blockedGateCount
  allOwnerExecutionInputsReady = $allOwnerExecutionInputsReady
  gates = @($gates)
  blockedReasonCount = $blockedReasons.Count
  blockedReasons = @($blockedReasons)
  rejectedNonProofStates = @($rejectedNonProofStates)
  ownerNextActions = @($ownerNextActions)
  sourceArtifacts = @($sourceArtifacts)
  ownerActionRequired = -not $allOwnerExecutionInputsReady
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner execution evidence pack aggregates final Owner input skeleton, GitHub CI evidence, bundle hash review, classification audit hash review, post-publish bridge, release-close bridge, and unified nine-lane real Owner proof convergence for Owner review only. It rejects templates, dashboards, dry-runs, local feeds, ProjectReference, direct nupkg, queued workflows, local builds, local tests, hash-only records, validation-ready records, staging shape-valid-only, sample-build-only, and mock-output substitutes. It does not publish, does not use tokens, does not close the release issue, and is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

Write-Utf8File -LiteralPath (Join-Path $OutputRoot "final-owner-execution-evidence-pack.json") -InputObject ($record | ConvertTo-Json -Depth 14)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "final-owner-execution-evidence-pack.md") -InputObject @(
  "# Final Owner Execution Evidence Pack",
  "",
  "- packState: ``$packState``",
  "- readyGateCount: ``$readyGateCount/$($gates.Count)``",
  "- blockedGateCount: ``$blockedGateCount``",
  "- canPublishPublicly: ``False``",
  "- canCloseReleaseIssue: ``False``",
  "",
  $record.boundary
)
Write-Host "FinalOwnerExecutionEvidencePackState=$packState ReadyGates=$readyGateCount/$($gates.Count) BlockedGates=$blockedGateCount"
