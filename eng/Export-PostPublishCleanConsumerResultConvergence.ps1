[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

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
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ConvergenceLane {
  param(
    [string]$Id,
    [string]$State,
    [string]$RequiredState,
    [string[]]$MissingFields,
    [string]$OwnerNextAction,
    [string]$Validator,
    [string]$SourceArtifact
  )

  $ready = [string]::Equals($State, $RequiredState, [StringComparison]::OrdinalIgnoreCase) -and @($MissingFields).Count -eq 0
  [pscustomobject]@{
    laneId = $Id
    state = $State
    requiredState = $RequiredState
    ready = $ready
    convergenceState = if ($ready) { "ready" } else { "blocked-post-publish-clean-consumer-result-required" }
    missingFields = @($MissingFields)
    ownerNextAction = $OwnerNextAction
    validator = $Validator
    sourceArtifact = $SourceArtifact
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
  }
}

$publicPublishImportValidation = Read-JsonOrNull "artifacts\final-release\public-publish-result-import-validation.json"
$publicPublishImport = Read-JsonOrNull "artifacts\final-release\public-publish-result-import.json"
$postPublishOwnerValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-owner-input-validation.json"
$postPublishRecordValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$packageConsumerRuntimeProof = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-record-validation.json"
$packageConsumerSummary = Read-JsonOrNull "artifacts\package-consumer\package-consumer-validation-summary.json"
$finalPostPublishAuditPackValidation = Read-JsonOrNull "artifacts\final-release\final-post-publish-audit-pack-validation.json"
$cleanConsumerScan = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-project-scan.json"

$publicPublishImportState = [string](Get-PropertyOrDefault -Object $publicPublishImportValidation -Name "validationState" -DefaultValue "missing-public-publish-result-import-validation")
$postPublishOwnerState = [string](Get-PropertyOrDefault -Object $postPublishOwnerValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-owner-input-validation")
$postPublishRecordState = [string](Get-PropertyOrDefault -Object $postPublishRecordValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$packageConsumerRuntimeState = [string](Get-PropertyOrDefault -Object $packageConsumerRuntimeProof -Name "validationState" -DefaultValue "missing-package-consumer-runtime-proof-record-validation")
$packageConsumerSummaryState = [string](Get-PropertyOrDefault -Object $packageConsumerSummary -Name "ValidationState" -DefaultValue "missing-package-consumer-validation-summary")
$finalPostPublishAuditState = [string](Get-PropertyOrDefault -Object $finalPostPublishAuditPackValidation -Name "validationState" -DefaultValue "missing-final-post-publish-audit-pack-validation")
$cleanConsumerScanState = [string](Get-PropertyOrDefault -Object $cleanConsumerScan -Name "scanState" -DefaultValue "missing-post-publish-clean-consumer-project-scan")

$localFeedReferenceCount = [int](Get-PropertyOrDefault -Object $cleanConsumerScan -Name "localFeedReferenceCount" -DefaultValue -1)
$projectReferenceCount = [int](Get-PropertyOrDefault -Object $cleanConsumerScan -Name "projectReferenceCount" -DefaultValue -1)
$directNupkgReferenceCount = [int](Get-PropertyOrDefault -Object $cleanConsumerScan -Name "directNupkgReferenceCount" -DefaultValue -1)

$publicImportMissing = @()
if (-not [string]::Equals($publicPublishImportState, "public-publish-result-import-ready", [StringComparison]::OrdinalIgnoreCase)) {
  $publicImportMissing += "real public publish result import"
}

$postPublishMissing = @()
if (-not [string]::Equals($postPublishOwnerState, "owner-input-ready-for-record-projection", [StringComparison]::OrdinalIgnoreCase)) {
  $postPublishMissing += "post-publish owner input"
}
if (-not [string]::Equals($postPublishRecordState, "post-publish-verification-ready", [StringComparison]::OrdinalIgnoreCase)) {
  $postPublishMissing += "post-publish clean consumer proof"
}

$packageConsumerMissing = @()
if (-not [string]::Equals($packageConsumerRuntimeState, "package-consumer-runtime-proof-record-ready", [StringComparison]::OrdinalIgnoreCase)) {
  $packageConsumerMissing += "package-consumer runtime proof record"
}
if ($packageConsumerSummaryState -notin @("package-consumer-validation-ready", "ready")) {
  $packageConsumerMissing += "package-consumer validation summary"
}

$cleanScanMissing = @()
if ($localFeedReferenceCount -ne 0) { $cleanScanMissing += "no local feed references" }
if ($projectReferenceCount -ne 0) { $cleanScanMissing += "no ProjectReference entries" }
if ($directNupkgReferenceCount -ne 0) { $cleanScanMissing += "no direct nupkg references" }
if (-not [string]::Equals($cleanConsumerScanState, "post-publish-clean-consumer-project-scan-ready", [StringComparison]::OrdinalIgnoreCase)) {
  $cleanScanMissing += "clean consumer project scan"
}

$auditMissing = @()
if (-not [string]::Equals($finalPostPublishAuditState, "final-post-publish-audit-ready", [StringComparison]::OrdinalIgnoreCase)) {
  $auditMissing += "final post-publish audit pack ready"
}

$lanes = @(
  New-ConvergenceLane -Id "public-publish-result-import" -State $publicPublishImportState -RequiredState "public-publish-result-import-ready" -MissingFields $publicImportMissing -OwnerNextAction "Owner fills real public package URL, timestamp, SHA256, channel, transcript path/hash, and review confirmations." -Validator "Test-PublicPublishResultImport.ps1 -Strict" -SourceArtifact "artifacts/final-release/public-publish-result-import-validation.json"
  New-ConvergenceLane -Id "post-publish-verification" -State $postPublishRecordState -RequiredState "post-publish-verification-ready" -MissingFields $postPublishMissing -OwnerNextAction "Owner runs clean consumer restore/build/runtime smoke from public package source and records logs, hashes, host metadata, and command summaries." -Validator "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" -SourceArtifact "artifacts/final-release/post-publish-verification-validation.json"
  New-ConvergenceLane -Id "package-consumer-runtime-proof" -State $packageConsumerRuntimeState -RequiredState "package-consumer-runtime-proof-record-ready" -MissingFields $packageConsumerMissing -OwnerNextAction "Owner supplies package-consumer runtime proof from a compatible host using public package sources." -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict" -SourceArtifact "artifacts/final-release/package-consumer-runtime-proof-record-validation.json"
  New-ConvergenceLane -Id "clean-consumer-source-scan" -State $cleanConsumerScanState -RequiredState "post-publish-clean-consumer-project-scan-ready" -MissingFields $cleanScanMissing -OwnerNextAction "Owner ensures clean consumer has no local feed, ProjectReference, direct nupkg, or private package source leakage." -Validator "Export-PostPublishCleanConsumerProjectScan.ps1; Test-PostPublishCleanConsumerProjectScan.ps1 -Strict" -SourceArtifact "artifacts/final-release/post-publish-clean-consumer-project-scan.json"
  New-ConvergenceLane -Id "final-post-publish-audit-pack" -State $finalPostPublishAuditState -RequiredState "final-post-publish-audit-ready" -MissingFields $auditMissing -OwnerNextAction "Owner re-runs final post-publish audit pack after public publish and clean consumer proof are real." -Validator "Test-FinalPostPublishAuditPack.ps1 -Strict" -SourceArtifact "artifacts/final-release/final-post-publish-audit-pack-validation.json"
)

$blocked = @($lanes | Where-Object { -not [bool]$_.ready })
$ready = @($lanes | Where-Object { [bool]$_.ready })

$record = [pscustomobject]@{
  recordKind = "post-publish-clean-consumer-result-convergence"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  convergenceState = if ($blocked.Count -eq 0) { "post-publish-clean-consumer-result-convergence-ready" } else { "blocked-post-publish-clean-consumer-result-required" }
  laneCount = $lanes.Count
  blockedLaneCount = $blocked.Count
  readyLaneCount = $ready.Count
  localFeedReferenceCount = $localFeedReferenceCount
  projectReferenceCount = $projectReferenceCount
  directNupkgReferenceCount = $directNupkgReferenceCount
  publicPublishResultImportValidationState = $publicPublishImportState
  postPublishVerificationOwnerInputValidationState = $postPublishOwnerState
  postPublishVerificationValidationState = $postPublishRecordState
  packageConsumerRuntimeProofValidationState = $packageConsumerRuntimeState
  finalPostPublishAuditPackValidationState = $finalPostPublishAuditState
  lanes = $lanes
  sourceArtifacts = @(
    "artifacts/final-release/public-publish-result-import.json",
    "artifacts/final-release/public-publish-result-import-validation.json",
    "artifacts/final-release/post-publish-verification-owner-input-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/package-consumer-runtime-proof-record-validation.json",
    "artifacts/package-consumer/package-consumer-validation-summary.json",
    "artifacts/final-release/final-post-publish-audit-pack-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-project-scan.json"
  )
  notExecutedByAutomation = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Post-publish clean consumer result convergence is gap aggregation only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "post-publish-clean-consumer-result-convergence.json"
$markdownPath = Join-Path $artifactRoot "post-publish-clean-consumer-result-convergence.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $record.lanes | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.laneId) | $(ConvertTo-MarkdownCell $_.state) | $(ConvertTo-MarkdownCell $_.requiredState) | ``$($_.ready)`` | $(ConvertTo-MarkdownCell ($_.missingFields -join ', ')) | $(ConvertTo-MarkdownCell $_.validator) |"
}

$markdown = @"
# Post-Publish Clean Consumer Result Convergence

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| convergenceState | ``$($record.convergenceState)`` |
| laneCount | ``$($record.laneCount)`` |
| blockedLaneCount | ``$($record.blockedLaneCount)`` |
| readyLaneCount | ``$($record.readyLaneCount)`` |
| localFeedReferenceCount | ``$($record.localFeedReferenceCount)`` |
| projectReferenceCount | ``$($record.projectReferenceCount)`` |
| directNupkgReferenceCount | ``$($record.directNupkgReferenceCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Lanes

| Lane | Current State | Required State | Ready | Missing Fields | Validator |
|---|---|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish clean consumer result convergence written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ConvergenceState=$($record.convergenceState) Lanes=$($record.laneCount) Blocked=$($record.blockedLaneCount) Ready=$($record.readyLaneCount)"
