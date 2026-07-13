[CmdletBinding()]
param(
  [string]$OutputPath = "artifacts\final-release\pre-release-package-proof-readiness-matrix.json",
  [string]$MarkdownOutputPath = "artifacts\final-release\pre-release-package-proof-readiness-matrix.md",
  [string]$CurrentHeadPackageDryRunPreflightPath = "artifacts\final-release\current-head-package-dry-run-preflight.json",
  [string]$DispatchPackValidationPath = "artifacts\final-release\current-head-package-dry-run-owner-dispatch-pack-validation.json",
  [string]$SourceQualityEvidenceValidationPath = "artifacts\final-release\github-actions-source-quality-run-evidence-validation\github-actions-run-evidence-import-validation.json",
  [string]$PublicPackageDownloadProofInputValidationPath = "artifacts\final-release\public-package-download-proof-input-validation.json",
  [string]$PackageConsumerRuntimeOwnerInputValidationPath = "artifacts\final-release\package-consumer-runtime-proof-owner-input-validation.json",
  [string]$PostPublishProofValidationPath = "artifacts\final-release\post-publish-clean-consumer-proof-result-validation.json",
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

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepoPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)

  $resolvedPath = Resolve-RepoPath -Path $Path
  if ([string]::IsNullOrWhiteSpace($resolvedPath) -or -not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

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
  if ($value -is [bool]) {
    return [bool]$value
  }

  $parsed = $false
  if ([bool]::TryParse(([string]$value).Trim(), [ref]$parsed)) {
    return $parsed
  }

  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-Lane {
  param(
    [string]$Id,
    [string]$Title,
    [string]$State,
    [bool]$Ready,
    [string]$SourceArtifact,
    [string]$RequiredEvidence,
    [string]$BlockedReason,
    [string]$ValidatorPath,
    [int]$FailedBlockerCount = 0,
    [int]$FailedActionRequiredCount = 0,
    [bool]$CanPromotePublicProof = $false,
    [bool]$CanPromoteRuntimeProof = $false,
    [bool]$CanPromotePostPublishProof = $false
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    state = $State
    ready = $Ready
    sourceArtifact = $SourceArtifact
    requiredEvidence = $RequiredEvidence
    requiredProof = $RequiredEvidence
    blockedReason = $BlockedReason
    validatorPath = $ValidatorPath
    failedBlockerCount = $FailedBlockerCount
    failedActionRequiredCount = $FailedActionRequiredCount
    performsPublish = $false
    canPromotePublicProof = $CanPromotePublicProof
    canPromoteRuntimeProof = $CanPromoteRuntimeProof
    canPromotePostPublishProof = $CanPromotePostPublishProof
    canPromoteProof = $CanPromotePublicProof -or $CanPromoteRuntimeProof -or $CanPromotePostPublishProof
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isPackageConsumerRuntimeProof = $false
    isPostPublishProof = $false
  }
}

function Get-ValidationBlockedReason {
  param(
    [AllowNull()][object]$Validation,
    [string]$MissingReason,
    [string]$ReadyReason = "none"
  )

  if ($null -eq $Validation) {
    return $MissingReason
  }

  $state = [string](Get-PropertyOrDefault -Object $Validation -Name "validationState" -DefaultValue "")
  $failedBlockers = [int](Get-PropertyOrDefault -Object $Validation -Name "failedBlockerCount" -DefaultValue 0)
  $failedActionRequired = [int](Get-PropertyOrDefault -Object $Validation -Name "failedActionRequiredCount" -DefaultValue 0)
  if ($failedBlockers -eq 0 -and $failedActionRequired -eq 0) {
    return $ReadyReason
  }

  return "validationState=$state; failedBlockerCount=$failedBlockers; failedActionRequiredCount=$failedActionRequired"
}

$sourceQualityValidation = Read-JsonOrNull -Path $SourceQualityEvidenceValidationPath
$packageDryRunPreflight = Read-JsonOrNull -Path $CurrentHeadPackageDryRunPreflightPath
$dispatchPackValidation = Read-JsonOrNull -Path $DispatchPackValidationPath
$publicDownloadValidation = Read-JsonOrNull -Path $PublicPackageDownloadProofInputValidationPath
$consumerRuntimeValidation = Read-JsonOrNull -Path $PackageConsumerRuntimeOwnerInputValidationPath
$postPublishValidation = Read-JsonOrNull -Path $PostPublishProofValidationPath

$sourceQualityReady = Get-BoolPropertyOrDefault -Object $sourceQualityValidation -Name "sourceQualityRunEvidenceReady" -DefaultValue $false
$packageDryRunReady = Get-BoolPropertyOrDefault -Object $packageDryRunPreflight -Name "canClaimGitHubActionsPackageDryRunPackForCurrentHead" -DefaultValue $false
$dispatchPackReadyForOwner = $null -ne $dispatchPackValidation -and
  ([string](Get-PropertyOrDefault -Object $dispatchPackValidation -Name "validationState" -DefaultValue "")).Contains("dispatch-pack", [StringComparison]::OrdinalIgnoreCase) -and
  ([int](Get-PropertyOrDefault -Object $dispatchPackValidation -Name "failedBlockerCount" -DefaultValue 1)) -eq 0
$publicDownloadReady = Get-BoolPropertyOrDefault -Object $publicDownloadValidation -Name "publicPackageDownloadProofReady" -DefaultValue $false
$consumerRuntimeReady = Get-BoolPropertyOrDefault -Object $consumerRuntimeValidation -Name "cleanOwnerInputReady" -DefaultValue $false
$postPublishReady = Get-BoolPropertyOrDefault -Object $postPublishValidation -Name "postPublishProofReady" -DefaultValue $false
$sourceQualityBlockedReason = if ($sourceQualityReady) { "none" } else { "Current source-quality evidence is missing or not ready." }
$dispatchPackBlockedReason = if ($dispatchPackReadyForOwner) { "none; still not proof" } else { "Owner dispatch pack has not been generated or validated." }
$publicDownloadBlockedReason = Get-ValidationBlockedReason -Validation $publicDownloadValidation -MissingReason "Public package download validator output is missing; real public package URL/download/hash owner evidence is required."
$consumerRuntimeBlockedReason = Get-ValidationBlockedReason -Validation $consumerRuntimeValidation -MissingReason "Package consumer runtime owner input validation is missing; clean external consumer runtime evidence is required."
$postPublishBlockedReason = Get-ValidationBlockedReason -Validation $postPublishValidation -MissingReason "Post-publish proof validation is missing; real published package clean consumer evidence is required."

$lanes = @(
  New-Lane -Id "source-quality-ci" -Title "Source-quality CI evidence" -State ([string](Get-PropertyOrDefault -Object $sourceQualityValidation -Name "validationState" -DefaultValue "missing-source-quality-validation")) -Ready $sourceQualityReady -SourceArtifact $SourceQualityEvidenceValidationPath -RequiredEvidence "Successful source-quality run for current HEAD; source-quality only and not package proof." -BlockedReason $sourceQualityBlockedReason -ValidatorPath "eng\Test-GitHubActionsRunEvidenceImport.ps1 -Strict" -FailedBlockerCount ([int](Get-PropertyOrDefault -Object $sourceQualityValidation -Name "failedBlockerCount" -DefaultValue 0)) -FailedActionRequiredCount ([int](Get-PropertyOrDefault -Object $sourceQualityValidation -Name "failedActionRequiredCount" -DefaultValue 0))
  New-Lane -Id "current-head-package-dry-run" -Title "Current HEAD package-managed dry-run" -State ([string](Get-PropertyOrDefault -Object $packageDryRunPreflight -Name "state" -DefaultValue "missing-current-head-package-dry-run-preflight")) -Ready $packageDryRunReady -SourceArtifact $CurrentHeadPackageDryRunPreflightPath -RequiredEvidence "workflow_dispatch package-managed dry-run for current HEAD with publish flags disabled, imported and validated from run artifacts." -BlockedReason ([string](Get-PropertyOrDefault -Object $packageDryRunPreflight -Name "blockedReason" -DefaultValue "Owner authorization and current-head dry-run run are required.")) -ValidatorPath "eng\Export-CurrentHeadPackageDryRunPreflight.ps1"
  New-Lane -Id "owner-dispatch-pack" -Title "Owner non-publish dry-run dispatch pack" -State ([string](Get-PropertyOrDefault -Object $dispatchPackValidation -Name "validationState" -DefaultValue "missing-current-head-package-dry-run-owner-dispatch-pack-validation")) -Ready $dispatchPackReadyForOwner -SourceArtifact $DispatchPackValidationPath -RequiredEvidence "Owner-reviewed command pack only; executing it still requires explicit Owner action and later artifact import." -BlockedReason $dispatchPackBlockedReason -ValidatorPath "eng\Test-CurrentHeadPackageDryRunOwnerDispatchPack.ps1 -Strict" -FailedBlockerCount ([int](Get-PropertyOrDefault -Object $dispatchPackValidation -Name "failedBlockerCount" -DefaultValue 1)) -FailedActionRequiredCount ([int](Get-PropertyOrDefault -Object $dispatchPackValidation -Name "failedActionRequiredCount" -DefaultValue 1))
  New-Lane -Id "public-package-download" -Title "Public package download proof" -State ([string](Get-PropertyOrDefault -Object $publicDownloadValidation -Name "validationState" -DefaultValue "missing-public-package-download-proof-input-validation")) -Ready $publicDownloadReady -SourceArtifact $PublicPackageDownloadProofInputValidationPath -RequiredEvidence "Public NuGet/GitHub Packages package URLs, downloaded nupkg hashes, and source proof linkage." -BlockedReason $publicDownloadBlockedReason -ValidatorPath "eng\Test-PublicPackageDownloadProofInput.ps1 -Strict" -FailedBlockerCount ([int](Get-PropertyOrDefault -Object $publicDownloadValidation -Name "failedBlockerCount" -DefaultValue 0)) -FailedActionRequiredCount ([int](Get-PropertyOrDefault -Object $publicDownloadValidation -Name "failedActionRequiredCount" -DefaultValue 1)) -CanPromotePublicProof $publicDownloadReady
  New-Lane -Id "clean-external-package-consumer-runtime" -Title "Clean external package consumer runtime proof" -State ([string](Get-PropertyOrDefault -Object $consumerRuntimeValidation -Name "validationState" -DefaultValue "missing-package-consumer-runtime-owner-input-validation")) -Ready $consumerRuntimeReady -SourceArtifact $PackageConsumerRuntimeOwnerInputValidationPath -RequiredEvidence "Repository-external clean consumer restore/build/runtime smoke logs with public package source, hash-matched packages, host metadata, and no ProjectReference/local feed/direct nupkg." -BlockedReason $consumerRuntimeBlockedReason -ValidatorPath "eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -FailedBlockerCount ([int](Get-PropertyOrDefault -Object $consumerRuntimeValidation -Name "failedBlockerCount" -DefaultValue 0)) -FailedActionRequiredCount ([int](Get-PropertyOrDefault -Object $consumerRuntimeValidation -Name "failedActionRequiredCount" -DefaultValue 1)) -CanPromoteRuntimeProof $consumerRuntimeReady
  New-Lane -Id "post-publish-clean-consumer-proof" -Title "Post-publish clean consumer proof" -State ([string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-proof-validation")) -Ready $postPublishReady -SourceArtifact $PostPublishProofValidationPath -RequiredEvidence "After actual public publication, separate clean consumer install/run proof from public channel." -BlockedReason $postPublishBlockedReason -ValidatorPath "eng\Test-PostPublishCleanConsumerProofResult.ps1 -Strict" -FailedBlockerCount ([int](Get-PropertyOrDefault -Object $postPublishValidation -Name "failedBlockerCount" -DefaultValue 0)) -FailedActionRequiredCount ([int](Get-PropertyOrDefault -Object $postPublishValidation -Name "failedActionRequiredCount" -DefaultValue 1)) -CanPromotePostPublishProof $postPublishReady
)

$readyLanes = @($lanes | Where-Object { $_.ready })
$blockedLanes = @($lanes | Where-Object { -not $_.ready })
$matrixState = if ($blockedLanes.Count -eq 0) {
  "pre-release-package-proof-ready"
}
else {
  "blocked-real-public-package-and-runtime-proof-required"
}

$matrix = [pscustomobject]@{
  recordKind = "pre-release-package-proof-readiness-matrix"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  matrixState = $matrixState
  currentHead = [string](Get-PropertyOrDefault -Object $packageDryRunPreflight -Name "currentHead" -DefaultValue "")
  sourceQualityRunId = [string](Get-PropertyOrDefault -Object $packageDryRunPreflight -Name "sourceQualityRunId" -DefaultValue "")
  readyLaneCount = $readyLanes.Count
  blockedLaneCount = $blockedLanes.Count
  currentHeadPackageDryRunReady = $packageDryRunReady
  ownerDispatchPackReadyForOwner = $dispatchPackReadyForOwner
  publicPackageDownloadProofReady = $publicDownloadReady
  packageConsumerRuntimeProofReady = $consumerRuntimeReady
  postPublishProofReady = $postPublishReady
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteProof = $false
  lanes = @($lanes)
  sourceArtifacts = @(
    $CurrentHeadPackageDryRunPreflightPath,
    $DispatchPackValidationPath,
    $SourceQualityEvidenceValidationPath,
    $PublicPackageDownloadProofInputValidationPath,
    $PackageConsumerRuntimeOwnerInputValidationPath,
    $PostPublishProofValidationPath
  )
  safetyBoundary = "This matrix is a readiness classifier only. It does not publish packages, run workflow_dispatch, execute runtime smoke, close release issues, or promote package-consumer/post-publish proof. Source-quality, dispatch packs, templates, preflights, local artifacts, dry-runs, queued runs, and owner-action placeholders remain non-proof."
}

$resolvedOutputPath = Resolve-RepoPath -Path $OutputPath
$resolvedMarkdownOutputPath = Resolve-RepoPath -Path $MarkdownOutputPath
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($resolvedOutputPath)) | Out-Null

$matrix | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $resolvedOutputPath -Encoding utf8

$laneRows = $matrix.lanes | ForEach-Object {
  "| ``$(ConvertTo-MarkdownCell $_.id)`` | $(ConvertTo-MarkdownCell $_.title) | ``$($_.state)`` | ``$($_.ready)`` | ``$($_.failedBlockerCount)`` | ``$($_.failedActionRequiredCount)`` | ``$(ConvertTo-MarkdownCell $_.validatorPath)`` | $(ConvertTo-MarkdownCell $_.blockedReason) |"
}

$markdown = @"
# Pre-Release Package Proof Readiness Matrix

生成时间：$($matrix.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| matrixState | ``$($matrix.matrixState)`` |
| currentHead | ``$($matrix.currentHead)`` |
| sourceQualityRunId | ``$($matrix.sourceQualityRunId)`` |
| readyLaneCount | ``$($matrix.readyLaneCount)`` |
| blockedLaneCount | ``$($matrix.blockedLaneCount)`` |
| currentHeadPackageDryRunReady | ``$($matrix.currentHeadPackageDryRunReady)`` |
| ownerDispatchPackReadyForOwner | ``$($matrix.ownerDispatchPackReadyForOwner)`` |
| publicPackageDownloadProofReady | ``$($matrix.publicPackageDownloadProofReady)`` |
| packageConsumerRuntimeProofReady | ``$($matrix.packageConsumerRuntimeProofReady)`` |
| postPublishProofReady | ``$($matrix.postPublishProofReady)`` |
| performsPublish | ``$($matrix.performsPublish)`` |
| canPublishPublicly | ``$($matrix.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($matrix.canCloseReleaseIssue)`` |

## Lanes

| ID | Title | State | Ready | Blockers | Action Required | Validator | Blocked Reason |
|---|---|---|---|---:|---:|---|---|
$($laneRows -join "`r`n")

## Safety Boundary

$($matrix.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $resolvedMarkdownOutputPath -Encoding utf8

Write-Host "Pre-release package proof readiness matrix written:"
Write-Host "  Json=$resolvedOutputPath"
Write-Host "  Markdown=$resolvedMarkdownOutputPath"
Write-Host "MatrixState=$($matrix.matrixState) ReadyLaneCount=$($matrix.readyLaneCount) BlockedLaneCount=$($matrix.blockedLaneCount)"
