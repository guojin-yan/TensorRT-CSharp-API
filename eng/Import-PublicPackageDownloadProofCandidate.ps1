[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-package-download-proof-input.template.json",
  [string]$InputValidationPath = "artifacts\final-release\public-package-download-proof-input-validation.json",
  [string]$GitHubActionsRunEvidenceValidationPath = "artifacts\final-release\github-actions-run-evidence-import-validation.json",
  [string]$OwnerPublicPublishResultValidationPath = "artifacts\final-release\owner-public-publish-execution-result-candidate-validation.json",
  [string]$PreReleaseReadinessMatrixPath = "artifacts\final-release\pre-release-package-proof-readiness-matrix.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Get-BoolPropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [bool]$DefaultValue
  )

  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue $DefaultValue
  if ($value -is [bool]) { return [bool]$value }

  $parsed = $false
  if ([bool]::TryParse(([string]$value).Trim(), [ref]$parsed)) { return $parsed }

  return $DefaultValue
}

function Read-JsonOrNull {
  param([string]$Path)

  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-PublicPackageDownloadProofInputTemplate.ps1") -RepositoryRoot $RepositoryRoot
}

$resolvedInputValidationPath = Resolve-RepositoryPath -Path $InputValidationPath
if (-not (Test-Path -LiteralPath $resolvedInputValidationPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Test-PublicPackageDownloadProofInput.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot -InputPath $InputPath
}

$inputRecord = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$inputValidation = Get-Content -LiteralPath $resolvedInputValidationPath -Raw -Encoding utf8 | ConvertFrom-Json
$resolvedGitHubActionsRunEvidenceValidationPath = Resolve-RepositoryPath -Path $GitHubActionsRunEvidenceValidationPath
$resolvedOwnerPublicPublishResultValidationPath = Resolve-RepositoryPath -Path $OwnerPublicPublishResultValidationPath
$githubActionsRunEvidenceValidation = if (Test-Path -LiteralPath $resolvedGitHubActionsRunEvidenceValidationPath -PathType Leaf) {
  Get-Content -LiteralPath $resolvedGitHubActionsRunEvidenceValidationPath -Raw -Encoding utf8 | ConvertFrom-Json
}
else {
  $null
}
$ownerPublicPublishResultValidation = if (Test-Path -LiteralPath $resolvedOwnerPublicPublishResultValidationPath -PathType Leaf) {
  Get-Content -LiteralPath $resolvedOwnerPublicPublishResultValidationPath -Raw -Encoding utf8 | ConvertFrom-Json
}
else {
  $null
}
$preReleaseReadinessMatrix = Read-JsonOrNull -Path $PreReleaseReadinessMatrixPath
$preReleaseReadinessMatrixState = [string](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "matrixState" -DefaultValue "missing-pre-release-package-proof-readiness-matrix")
$preReleaseReadinessBlockedLaneCount = [int](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "blockedLaneCount" -DefaultValue 999)
$preReleaseReadinessLanes = @((Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "lanes" -DefaultValue @()))
$preReleasePublicPackageDownloadLane = @($preReleaseReadinessLanes | Where-Object { ([string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")).Equals("public-package-download", [StringComparison]::OrdinalIgnoreCase) } | Select-Object -First 1)
$preReleasePublicPackageDownloadLaneReady = if ($preReleasePublicPackageDownloadLane.Count -gt 0) {
  Get-BoolPropertyOrDefault -Object $preReleasePublicPackageDownloadLane[0] -Name "ready" -DefaultValue $false
}
else {
  $false
}
$preReleaseCanPromotePublicProof = if ($preReleasePublicPackageDownloadLane.Count -gt 0) {
  Get-BoolPropertyOrDefault -Object $preReleasePublicPackageDownloadLane[0] -Name "canPromotePublicProof" -DefaultValue $false
}
else {
  $false
}

$inputValidationState = [string](Get-PropertyOrDefault -Object $inputValidation -Name "validationState" -DefaultValue "missing-public-package-download-proof-input-validation")
$inputValidationInputPath = [string](Get-PropertyOrDefault -Object $inputValidation -Name "inputPath" -DefaultValue "")
$inputValidationMatchesInput = $false
if (-not [string]::IsNullOrWhiteSpace($inputValidationInputPath)) {
  try {
    $validationInputFullPath = [IO.Path]::GetFullPath((Resolve-RepositoryPath -Path $inputValidationInputPath))
    $currentInputFullPath = [IO.Path]::GetFullPath($resolvedInputPath)
    $inputValidationMatchesInput = $validationInputFullPath.Equals($currentInputFullPath, [StringComparison]::OrdinalIgnoreCase)
  }
  catch {
    $inputValidationMatchesInput = $inputValidationInputPath.Equals($resolvedInputPath, [StringComparison]::OrdinalIgnoreCase)
  }
}
$sourceReady = $inputValidationState -eq "public-package-download-proof-input-ready" -and
  [bool](Get-PropertyOrDefault -Object $inputValidation -Name "publicPackageDownloadProofReady" -DefaultValue $false) -and
  $inputValidationMatchesInput
$candidateReady = $sourceReady -and $preReleaseCanPromotePublicProof

$sourceGitHubActionsRunEvidenceReady = if ($null -eq $githubActionsRunEvidenceValidation) {
  [bool](Get-PropertyOrDefault -Object $inputRecord -Name "sourceGitHubActionsRunEvidenceReady" -DefaultValue $false)
}
else {
  [bool](Get-PropertyOrDefault -Object $githubActionsRunEvidenceValidation -Name "githubActionsRunEvidenceReady" -DefaultValue $false)
}
$sourceWorkflowRunLogSha256 = if ($null -eq $githubActionsRunEvidenceValidation) {
  [string](Get-PropertyOrDefault -Object $inputRecord -Name "sourceWorkflowRunLogSha256" -DefaultValue "")
}
else {
  [string](Get-PropertyOrDefault -Object $githubActionsRunEvidenceValidation -Name "workflowRunLogSha256" -DefaultValue "")
}
$sourceArtifactManifestSha256 = if ($null -eq $githubActionsRunEvidenceValidation) {
  [string](Get-PropertyOrDefault -Object $inputRecord -Name "sourceArtifactManifestSha256" -DefaultValue "")
}
else {
  [string](Get-PropertyOrDefault -Object $githubActionsRunEvidenceValidation -Name "artifactManifestSha256" -DefaultValue "")
}
$sourceOwnerPublicPublishResultReady = if ($null -eq $ownerPublicPublishResultValidation) {
  [bool](Get-PropertyOrDefault -Object $inputRecord -Name "sourceOwnerPublicPublishResultReady" -DefaultValue $false)
}
else {
  [bool](Get-PropertyOrDefault -Object $ownerPublicPublishResultValidation -Name "proofCandidateReady" -DefaultValue $false)
}
$sourceOwnerPublicPackageUrl = if ($null -eq $ownerPublicPublishResultValidation) {
  [string](Get-PropertyOrDefault -Object $inputRecord -Name "sourceOwnerPublicPackageUrl" -DefaultValue "")
}
else {
  [string](Get-PropertyOrDefault -Object $ownerPublicPublishResultValidation -Name "publicPackageUrl" -DefaultValue "")
}
$sourceOwnerPublicPackageVersion = if ($null -eq $ownerPublicPublishResultValidation) {
  [string](Get-PropertyOrDefault -Object $inputRecord -Name "sourceOwnerPublicPackageVersion" -DefaultValue "")
}
else {
  [string](Get-PropertyOrDefault -Object $ownerPublicPublishResultValidation -Name "publicPackageVersion" -DefaultValue "")
}
$sourceOwnerPublicPackageSha256 = if ($null -eq $ownerPublicPublishResultValidation) {
  [string](Get-PropertyOrDefault -Object $inputRecord -Name "sourceOwnerPublicPackageSha256" -DefaultValue "")
}
else {
  [string](Get-PropertyOrDefault -Object $ownerPublicPublishResultValidation -Name "publicPackageSha256" -DefaultValue "")
}
$forbiddenSubstituteFindings = @((Get-PropertyOrDefault -Object $inputValidation -Name "forbiddenSubstituteFindings" -DefaultValue @()))

$projection = [ordered]@{
  managedPackageId = [string](Get-PropertyOrDefault -Object $inputRecord -Name "managedPackageId" -DefaultValue "")
  managedPackageVersion = [string](Get-PropertyOrDefault -Object $inputRecord -Name "managedPackageVersion" -DefaultValue "")
  managedPackagePageUrl = [string](Get-PropertyOrDefault -Object $inputRecord -Name "managedPackagePageUrl" -DefaultValue "")
  managedPackageDownloadUrl = [string](Get-PropertyOrDefault -Object $inputRecord -Name "managedPackageDownloadUrl" -DefaultValue "")
  runtimePackageId = [string](Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackageId" -DefaultValue "")
  runtimePackageVersion = [string](Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackageVersion" -DefaultValue "")
  runtimePackageKey = [string](Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackageKey" -DefaultValue "")
  runtimePackagePageUrl = [string](Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackagePageUrl" -DefaultValue "")
  runtimePackageDownloadUrl = [string](Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackageDownloadUrl" -DefaultValue "")
  publicPackageSourceKind = [string](Get-PropertyOrDefault -Object $inputRecord -Name "publicPackageSourceKind" -DefaultValue "")
  publicPackageSourceUrl = [string](Get-PropertyOrDefault -Object $inputRecord -Name "publicPackageSourceUrl" -DefaultValue "")
  downloadedManagedNupkgPath = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedManagedNupkgPath" -DefaultValue "")
  downloadedManagedNupkgSha256 = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedManagedNupkgSha256" -DefaultValue "")
  downloadedManagedNupkgSizeBytes = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedManagedNupkgSizeBytes" -DefaultValue "")
  downloadedRuntimeNupkgPath = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedRuntimeNupkgPath" -DefaultValue "")
  downloadedRuntimeNupkgSha256 = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedRuntimeNupkgSha256" -DefaultValue "")
  downloadedRuntimeNupkgSizeBytes = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedRuntimeNupkgSizeBytes" -DefaultValue "")
  githubReleaseUrl = [string](Get-PropertyOrDefault -Object $inputRecord -Name "githubReleaseUrl" -DefaultValue "")
  githubReleaseAssetUrl = [string](Get-PropertyOrDefault -Object $inputRecord -Name "githubReleaseAssetUrl" -DefaultValue "")
  githubReleaseAssetDownloadedPath = [string](Get-PropertyOrDefault -Object $inputRecord -Name "githubReleaseAssetDownloadedPath" -DefaultValue "")
  githubReleaseAssetSha256 = [string](Get-PropertyOrDefault -Object $inputRecord -Name "githubReleaseAssetSha256" -DefaultValue "")
  githubReleaseAssetSizeBytes = [string](Get-PropertyOrDefault -Object $inputRecord -Name "githubReleaseAssetSizeBytes" -DefaultValue "")
  downloadCommand = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadCommand" -DefaultValue "")
  downloadedAtUtc = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedAtUtc" -DefaultValue "")
  capturedAtUtc = [string](Get-PropertyOrDefault -Object $inputRecord -Name "capturedAtUtc" -DefaultValue "")
  ownerName = [string](Get-PropertyOrDefault -Object $inputRecord -Name "ownerName" -DefaultValue "")
  ownerReviewer = [string](Get-PropertyOrDefault -Object $inputRecord -Name "ownerReviewer" -DefaultValue "")
  ownerAuthorizationState = [string](Get-PropertyOrDefault -Object $inputRecord -Name "ownerAuthorizationState" -DefaultValue "")
  sourceGitHubActionsRunEvidenceValidationPath = $resolvedGitHubActionsRunEvidenceValidationPath
  sourceOwnerPublicPublishResultValidationPath = $resolvedOwnerPublicPublishResultValidationPath
  sourceGitHubActionsRunEvidenceReady = $sourceGitHubActionsRunEvidenceReady
  sourceWorkflowRunLogSha256 = $sourceWorkflowRunLogSha256
  sourceArtifactManifestSha256 = $sourceArtifactManifestSha256
  sourceOwnerPublicPublishResultReady = $sourceOwnerPublicPublishResultReady
  sourceOwnerPublicPackageUrl = $sourceOwnerPublicPackageUrl
  sourceOwnerPublicPackageVersion = $sourceOwnerPublicPackageVersion
  sourceOwnerPublicPackageSha256 = $sourceOwnerPublicPackageSha256
  preReleaseReadinessMatrixPath = $PreReleaseReadinessMatrixPath
  preReleaseReadinessMatrixState = $preReleaseReadinessMatrixState
  preReleaseReadinessBlockedLaneCount = $preReleaseReadinessBlockedLaneCount
  preReleasePublicPackageDownloadLaneReady = $preReleasePublicPackageDownloadLaneReady
  preReleaseCanPromotePublicProof = $preReleaseCanPromotePublicProof
  forbiddenSubstituteFindings = @($forbiddenSubstituteFindings)
}

$candidateItems = @()
if ($candidateReady) {
  $candidate = [ordered]@{
    candidateId = "public-package-download-proof-candidate-001"
    candidateState = "public-package-download-proof-candidate-imported"
  }
  foreach ($key in $projection.Keys) { $candidate[$key] = $projection[$key] }
  $candidate["boundary"] = "Candidate only for public package download proof. It is not runtime proof, not package-consumer runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
  $candidateItems = @([pscustomobject]$candidate)
}

$record = [ordered]@{
  recordKind = "public-package-download-proof-candidate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  candidateState = if ($candidateReady) { "public-package-download-proof-candidate-imported" } else { "blocked-public-package-download-proof-required" }
  sourceInputPath = $resolvedInputPath
  sourceInputValidationPath = $resolvedInputValidationPath
  sourceInputValidationState = $inputValidationState
  sourceInputValidationInputPath = $inputValidationInputPath
  sourceInputValidationMatchesInput = $inputValidationMatchesInput
  sourcePublicPackageDownloadProofReady = $sourceReady
  preReleaseReadinessMatrixPath = $PreReleaseReadinessMatrixPath
  preReleaseReadinessMatrixState = $preReleaseReadinessMatrixState
  preReleaseReadinessBlockedLaneCount = $preReleaseReadinessBlockedLaneCount
  preReleasePublicPackageDownloadLaneReady = $preReleasePublicPackageDownloadLaneReady
  preReleaseCanPromotePublicProof = $preReleaseCanPromotePublicProof
  candidateItemCount = $candidateItems.Count
  readyCandidateCount = if ($candidateReady) { 1 } else { 0 }
  blockedCandidateCount = if ($candidateReady) { 0 } else { 1 }
  publicPackageDownloadProofCandidateReady = $candidateReady
  proofCandidateReady = $candidateReady
  candidateItems = @($candidateItems)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canPublishGitHubPackages = $false
  canCloseReleaseIssue = $false
  canClaimRuntimeProof = $false
  canClaimPackageConsumerRuntimeProof = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isGitHubActionsProof = $false
  whyNotProof = @(
    "Candidate import does not execute package download, restore, build, smoke, publish, or issue close.",
    "Candidate import only projects already validated public download evidence into the remote proof lane.",
    "Candidate import cannot substitute GitHub Actions run proof, owner public publish result, post-publish clean consumer proof, or release close approval."
  )
  boundary = "Public package download proof candidate only. It is not runtime proof, not package-consumer runtime proof, not post-publish proof, not publish approval, not release close approval, not GitHub Actions proof, not package push, and cannot close the release."
}
foreach ($key in $projection.Keys) { $record[$key] = $projection[$key] }

$jsonPath = Join-Path $OutputRoot "public-package-download-proof-candidate.json"
$markdownPath = Join-Path $OutputRoot "public-package-download-proof-candidate.md"

$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Public Package Download Proof Candidate

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| candidateState | ``$($record.candidateState)`` |
| sourceInputValidationState | ``$($record.sourceInputValidationState)`` |
| sourceInputValidationMatchesInput | ``$($record.sourceInputValidationMatchesInput)`` |
| sourceGitHubActionsRunEvidenceReady | ``$($record.sourceGitHubActionsRunEvidenceReady)`` |
| sourceOwnerPublicPublishResultReady | ``$($record.sourceOwnerPublicPublishResultReady)`` |
| preReleaseReadinessMatrixState | ``$($record.preReleaseReadinessMatrixState)`` |
| preReleasePublicPackageDownloadLaneReady | ``$($record.preReleasePublicPackageDownloadLaneReady)`` |
| preReleaseCanPromotePublicProof | ``$($record.preReleaseCanPromotePublicProof)`` |
| managedPackagePageUrl | ``$($record.managedPackagePageUrl)`` |
| managedPackageDownloadUrl | ``$($record.managedPackageDownloadUrl)`` |
| runtimePackagePageUrl | ``$($record.runtimePackagePageUrl)`` |
| runtimePackageDownloadUrl | ``$($record.runtimePackageDownloadUrl)`` |
| githubReleaseUrl | ``$($record.githubReleaseUrl)`` |
| githubReleaseAssetUrl | ``$($record.githubReleaseAssetUrl)`` |
| candidateItemCount | ``$($record.candidateItemCount)`` |
| readyCandidateCount | ``$($record.readyCandidateCount)`` |
| blockedCandidateCount | ``$($record.blockedCandidateCount)`` |
| publicPackageDownloadProofCandidateReady | ``$($record.publicPackageDownloadProofCandidateReady)`` |
| proofCandidateReady | ``$($record.proofCandidateReady)`` |
| performsPublish | ``$($record.performsPublish)`` |
| usesPublishToken | ``$($record.usesPublishToken)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |
| isRuntimeExecutionProof | ``$($record.isRuntimeExecutionProof)`` |
| isPostPublishProof | ``$($record.isPostPublishProof)`` |
| isReleaseCloseProof | ``$($record.isReleaseCloseProof)`` |

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public package download proof candidate written to $jsonPath"
Write-Host "CandidateState=$($record.candidateState) ReadyCandidateCount=$($record.readyCandidateCount) BlockedCandidateCount=$($record.blockedCandidateCount)"
