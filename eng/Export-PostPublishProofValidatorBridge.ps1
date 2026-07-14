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

function New-BridgeLane {
  param(
    [string]$Id,
    [string]$Title,
    [AllowNull()][object]$Validator,
    [AllowNull()][object]$ShapeRecord,
    [string]$ValidatorAcceptedProperty = "ownerEvidenceAccepted",
    [string]$ShapeReadyProperty = "",
    [string]$BlockedReason = "real-owner-proof-required",
    [string[]]$SourceArtifacts = @()
  )

  $validatorPresent = $null -ne $Validator
  $validatorAccepted = [bool](Get-PropertyOrDefault -Object $Validator -Name $ValidatorAcceptedProperty -DefaultValue $false)
  $validatorFailedBlockers = [int](Get-PropertyOrDefault -Object $Validator -Name "failedBlockerCount" -DefaultValue 999)
  $validatorReady = $validatorPresent -and $validatorFailedBlockers -eq 0
  $shapeReady = $false
  if (-not [string]::IsNullOrWhiteSpace($ShapeReadyProperty)) {
    $shapeReady = [bool](Get-PropertyOrDefault -Object $ShapeRecord -Name $ShapeReadyProperty -DefaultValue $false)
  }

  $inputShapeReady = $validatorAccepted -or $shapeReady
  $proofReady = $validatorAccepted -and $validatorFailedBlockers -eq 0
  $reasons = New-Object System.Collections.Generic.List[string]
  if (-not $validatorPresent) { $reasons.Add("validator-missing") | Out-Null }
  if ($validatorFailedBlockers -gt 0) { $reasons.Add("validator-failed-blockers=$validatorFailedBlockers") | Out-Null }
  if (-not $inputShapeReady) { $reasons.Add($BlockedReason) | Out-Null }
  if ($shapeReady -and -not $proofReady) { $reasons.Add("shape-valid-is-not-proof") | Out-Null }

  [pscustomobject]@{
    id = $Id
    title = $Title
    inputShapeReady = $inputShapeReady
    stagingShapeReady = $shapeReady
    validatorReady = $validatorReady
    validatorOwnerEvidenceAccepted = $validatorAccepted
    proofReady = $proofReady
    blockedReason = if ($reasons.Count -gt 0) { [string]$reasons[0] } else { "" }
    blockedReasons = @($reasons.ToArray())
    ownerActionRequired = -not $proofReady
    sourceArtifacts = @($SourceArtifacts)
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Post-publish bridge lane inventories Owner evidence only; shape-valid, validation-ready, downloaded hash, staging admission, local build, and dashboard signals are not runtime proof, not post-publish proof, not release close approval, and not package push."
  }
}

Invoke-OwnerScript "Export-PublicPackageUrlHashProofValidator.ps1"
Invoke-OwnerScript "Test-PublicPackageUrlHashProofValidator.ps1" @("-Strict")
Invoke-OwnerScript "Export-ExternalCleanConsumerPostPublishProofValidator.ps1"
Invoke-OwnerScript "Test-ExternalCleanConsumerPostPublishProofValidator.ps1" @("-Strict")
Invoke-OwnerScript "Export-YoloVisionRealModelPostPublishProofValidator.ps1"
Invoke-OwnerScript "Test-YoloVisionRealModelPostPublishProofValidator.ps1" @("-Strict")
Invoke-OwnerScript "Export-ArticlePublicationProofValidator.ps1"
Invoke-OwnerScript "Test-ArticlePublicationProofValidator.ps1" @("-Strict")
Invoke-OwnerScript "Export-ReleaseCloseFinalBridgeProofValidator.ps1"
Invoke-OwnerScript "Test-ReleaseCloseFinalBridgeProofValidator.ps1" @("-Strict")
Invoke-OwnerScript "Export-OwnerPostPublishProofAcceptanceManifest.ps1"
Invoke-OwnerScript "Test-OwnerPostPublishProofAcceptanceManifest.ps1" @("-Strict")
Invoke-OwnerScript "Export-PublicPackageUrlHashDownloadVerification.ps1"
Invoke-OwnerScript "Test-PublicPackageUrlHashDownloadVerification.ps1" @("-Strict")
Invoke-OwnerScript "Import-ArticlePublicationProofFromStagingWorkspace.ps1"
Invoke-OwnerScript "Test-ArticlePublicationProofFromStagingWorkspace.ps1" @("-Strict")
Invoke-OwnerScript "Import-YoloVisionRealModelProofFromStagingWorkspace.ps1"
Invoke-OwnerScript "Test-YoloVisionRealModelProofFromStagingWorkspace.ps1" @("-Strict")

$publicPackageValidator = Read-FinalJsonOrNull "public-package-url-hash-proof-validator-validation.json"
$externalCleanConsumerValidator = Read-FinalJsonOrNull "external-clean-consumer-post-publish-proof-validator-validation.json"
$yoloValidator = Read-FinalJsonOrNull "yolovision-real-model-post-publish-proof-validator-validation.json"
$articleValidator = Read-FinalJsonOrNull "article-publication-proof-validator-validation.json"
$acceptanceManifest = Read-FinalJsonOrNull "owner-post-publish-proof-acceptance-manifest-validation.json"
$downloadVerification = Read-FinalJsonOrNull "public-package-url-hash-download-verification-validation.json"
$articleStaging = Read-FinalJsonOrNull "article-publication-proof-from-staging-workspace-validation.json"
$yoloStaging = Read-FinalJsonOrNull "yolovision-real-model-proof-from-staging-workspace-validation.json"

$lanes = @(
  New-BridgeLane -Id "public-package-url-hash" -Title "Public package URL/hash evidence" -Validator $publicPackageValidator -ShapeRecord $downloadVerification -ShapeReadyProperty "downloadVerificationReady" -BlockedReason "public-package-url-hash-owner-proof-required" -SourceArtifacts @("artifacts/final-release/public-package-url-hash-proof-validator-validation.json", "artifacts/final-release/public-package-url-hash-download-verification-validation.json")
  New-BridgeLane -Id "external-clean-consumer-post-publish" -Title "External CleanConsumer post-publish evidence" -Validator $externalCleanConsumerValidator -ShapeRecord $null -BlockedReason "external-clean-consumer-post-publish-proof-required" -SourceArtifacts @("artifacts/final-release/external-clean-consumer-post-publish-proof-validator-validation.json")
  New-BridgeLane -Id "article-publication" -Title "Article publication evidence" -Validator $articleValidator -ShapeRecord $articleStaging -ShapeReadyProperty "ownerEvidenceShapeValid" -BlockedReason "article-publication-owner-proof-required" -SourceArtifacts @("artifacts/final-release/article-publication-proof-validator-validation.json", "artifacts/final-release/article-publication-proof-from-staging-workspace-validation.json")
  New-BridgeLane -Id "yolovision-real-model" -Title "YoloVision real model evidence" -Validator $yoloValidator -ShapeRecord $yoloStaging -ShapeReadyProperty "ownerEvidenceShapeValid" -BlockedReason "yolovision-real-model-owner-proof-required" -SourceArtifacts @("artifacts/final-release/yolovision-real-model-post-publish-proof-validator-validation.json", "artifacts/final-release/yolovision-real-model-proof-from-staging-workspace-validation.json")
)

$laneCount = $lanes.Count
$inputShapeReadyCount = @($lanes | Where-Object { [bool]$_.inputShapeReady }).Count
$proofReadyCount = @($lanes | Where-Object { [bool]$_.proofReady }).Count
$stagingShapeReadyCount = @($lanes | Where-Object { [bool]$_.stagingShapeReady }).Count
$blockedLaneCount = $laneCount - $proofReadyCount
$acceptanceValidatorsAccepted = [bool](Get-PropertyOrDefault -Object $acceptanceManifest -Name "allValidatorsAccepted" -DefaultValue $false)
$publicPackageDownloadReady = [bool](Get-PropertyOrDefault -Object $downloadVerification -Name "downloadVerificationReady" -DefaultValue $false)
$publicPackageHashCannotSubstitutePostPublishProof = $true
$shapeValidCannotSubstitutePostPublishProof = $true
$allPostPublishInputsAccepted = $laneCount -gt 0 -and $proofReadyCount -eq $laneCount -and $acceptanceValidatorsAccepted
$blockedReasons = New-Object System.Collections.Generic.List[string]
if (-not $acceptanceValidatorsAccepted) { $blockedReasons.Add("owner-post-publish-proof-acceptance-manifest-not-accepted") | Out-Null }
if ($blockedLaneCount -gt 0) { $blockedReasons.Add("post-publish-proof-lanes-blocked=$blockedLaneCount/$laneCount") | Out-Null }
if (-not $publicPackageDownloadReady) { $blockedReasons.Add("public-package-download-hash-verification-not-ready") | Out-Null }

$bridgeState = if ($allPostPublishInputsAccepted) { "post-publish-proof-validator-bridge-ready-for-owner-close-review-non-proof" } else { "blocked-post-publish-proof-validator-bridge-real-owner-proof-required" }
$sourceArtifacts = @(
  "artifacts/final-release/public-package-url-hash-proof-validator-validation.json",
  "artifacts/final-release/external-clean-consumer-post-publish-proof-validator-validation.json",
  "artifacts/final-release/yolovision-real-model-post-publish-proof-validator-validation.json",
  "artifacts/final-release/article-publication-proof-validator-validation.json",
  "artifacts/final-release/owner-post-publish-proof-acceptance-manifest-validation.json",
  "artifacts/final-release/public-package-url-hash-download-verification-validation.json",
  "artifacts/final-release/article-publication-proof-from-staging-workspace-validation.json",
  "artifacts/final-release/yolovision-real-model-proof-from-staging-workspace-validation.json"
)

$record = [pscustomobject]@{
  recordKind = "post-publish-proof-validator-bridge"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  bridgeState = $bridgeState
  laneCount = $laneCount
  inputShapeReadyLaneCount = $inputShapeReadyCount
  stagingShapeReadyLaneCount = $stagingShapeReadyCount
  proofReadyLaneCount = $proofReadyCount
  blockedLaneCount = $blockedLaneCount
  allPostPublishInputsAccepted = $allPostPublishInputsAccepted
  acceptanceManifestValidatorsAccepted = $acceptanceValidatorsAccepted
  publicPackageDownloadReady = $publicPackageDownloadReady
  publicPackageHashCannotSubstitutePostPublishProof = $publicPackageHashCannotSubstitutePostPublishProof
  shapeValidCannotSubstitutePostPublishProof = $shapeValidCannotSubstitutePostPublishProof
  lanes = @($lanes)
  blockedReasonCount = $blockedReasons.Count
  blockedReasons = @($blockedReasons.ToArray())
  sourceArtifacts = @($sourceArtifacts)
  ownerActionRequired = -not $allPostPublishInputsAccepted
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Post-publish proof validator bridge aggregates public package download/hash, external CleanConsumer, article publication, YoloVision real-model, and staging admission signals for Owner review only. Downloaded hash, validation-ready, staging shape-valid, candidate, dashboard, dry-run, local feed, ProjectReference, and direct nupkg signals cannot substitute post-publish CleanConsumer runtime proof. This bridge does not publish, does not use tokens, does not run inference or CleanConsumer, does not close the release issue, and is not runtime proof, not post-publish proof, not release close approval, and not package push."
}

Write-Utf8File -LiteralPath (Join-Path $OutputRoot "post-publish-proof-validator-bridge.json") -InputObject ($record | ConvertTo-Json -Depth 16)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "post-publish-proof-validator-bridge.md") -InputObject @(
  "# Post-Publish Proof Validator Bridge",
  "",
  "- bridgeState: ``$bridgeState``",
  "- inputShapeReadyLaneCount: ``$inputShapeReadyCount/$laneCount``",
  "- proofReadyLaneCount: ``$proofReadyCount/$laneCount``",
  "- publicPackageHashCannotSubstitutePostPublishProof: ``True``",
  "- canCloseReleaseIssue: ``False``",
  "",
  $record.boundary
)
Write-Host "PostPublishProofValidatorBridgeState=$bridgeState InputShapeReady=$inputShapeReadyCount/$laneCount ProofReady=$proofReadyCount/$laneCount BlockedReasons=$($blockedReasons.Count)"
