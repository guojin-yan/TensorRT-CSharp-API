[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-package-download-proof-input.template.json",
  [string]$InputValidationPath = "artifacts\final-release\public-package-download-proof-input-validation.json",
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

$inputValidationState = [string](Get-PropertyOrDefault -Object $inputValidation -Name "validationState" -DefaultValue "missing-public-package-download-proof-input-validation")
$sourceReady = $inputValidationState -eq "public-package-download-proof-input-ready" -and
  [bool](Get-PropertyOrDefault -Object $inputValidation -Name "publicPackageDownloadProofReady" -DefaultValue $false)

$candidateItems = @()
if ($sourceReady) {
  $candidateItems = @([pscustomobject]@{
      candidateId = "public-package-download-proof-candidate-001"
      candidateState = "public-package-download-proof-candidate-imported"
      managedPackageId = [string](Get-PropertyOrDefault -Object $inputRecord -Name "managedPackageId" -DefaultValue "")
      managedPackageVersion = [string](Get-PropertyOrDefault -Object $inputRecord -Name "managedPackageVersion" -DefaultValue "")
      runtimePackageId = [string](Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackageId" -DefaultValue "")
      runtimePackageVersion = [string](Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackageVersion" -DefaultValue "")
      runtimePackageKey = [string](Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackageKey" -DefaultValue "")
      publicPackageSourceKind = [string](Get-PropertyOrDefault -Object $inputRecord -Name "publicPackageSourceKind" -DefaultValue "")
      publicPackageSourceUrl = [string](Get-PropertyOrDefault -Object $inputRecord -Name "publicPackageSourceUrl" -DefaultValue "")
      downloadedManagedNupkgPath = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedManagedNupkgPath" -DefaultValue "")
      downloadedManagedNupkgSha256 = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedManagedNupkgSha256" -DefaultValue "")
      downloadedRuntimeNupkgPath = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedRuntimeNupkgPath" -DefaultValue "")
      downloadedRuntimeNupkgSha256 = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedRuntimeNupkgSha256" -DefaultValue "")
      downloadCommand = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadCommand" -DefaultValue "")
      downloadedAtUtc = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedAtUtc" -DefaultValue "")
      ownerName = [string](Get-PropertyOrDefault -Object $inputRecord -Name "ownerName" -DefaultValue "")
      ownerAuthorizationState = [string](Get-PropertyOrDefault -Object $inputRecord -Name "ownerAuthorizationState" -DefaultValue "")
      boundary = "Candidate only for public package download proof. It is not runtime proof, not package-consumer runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
    })
}

$record = [pscustomobject]@{
  recordKind = "public-package-download-proof-candidate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  candidateState = if ($sourceReady) { "public-package-download-proof-candidate-imported" } else { "blocked-public-package-download-proof-required" }
  sourceInputPath = $resolvedInputPath
  sourceInputValidationPath = $resolvedInputValidationPath
  sourceInputValidationState = $inputValidationState
  sourcePublicPackageDownloadProofReady = $sourceReady
  candidateItemCount = $candidateItems.Count
  readyCandidateCount = if ($sourceReady) { 1 } else { 0 }
  blockedCandidateCount = if ($sourceReady) { 0 } else { 1 }
  publicPackageDownloadProofCandidateReady = $sourceReady
  proofCandidateReady = $sourceReady
  candidateItems = @($candidateItems)
  managedPackageId = [string](Get-PropertyOrDefault -Object $inputRecord -Name "managedPackageId" -DefaultValue "")
  managedPackageVersion = [string](Get-PropertyOrDefault -Object $inputRecord -Name "managedPackageVersion" -DefaultValue "")
  runtimePackageId = [string](Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackageId" -DefaultValue "")
  runtimePackageVersion = [string](Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackageVersion" -DefaultValue "")
  runtimePackageKey = [string](Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackageKey" -DefaultValue "")
  publicPackageSourceKind = [string](Get-PropertyOrDefault -Object $inputRecord -Name "publicPackageSourceKind" -DefaultValue "")
  publicPackageSourceUrl = [string](Get-PropertyOrDefault -Object $inputRecord -Name "publicPackageSourceUrl" -DefaultValue "")
  downloadedManagedNupkgPath = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedManagedNupkgPath" -DefaultValue "")
  downloadedManagedNupkgSha256 = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedManagedNupkgSha256" -DefaultValue "")
  downloadedRuntimeNupkgPath = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedRuntimeNupkgPath" -DefaultValue "")
  downloadedRuntimeNupkgSha256 = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedRuntimeNupkgSha256" -DefaultValue "")
  downloadCommand = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadCommand" -DefaultValue "")
  downloadedAtUtc = [string](Get-PropertyOrDefault -Object $inputRecord -Name "downloadedAtUtc" -DefaultValue "")
  ownerName = [string](Get-PropertyOrDefault -Object $inputRecord -Name "ownerName" -DefaultValue "")
  ownerAuthorizationState = [string](Get-PropertyOrDefault -Object $inputRecord -Name "ownerAuthorizationState" -DefaultValue "")
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
