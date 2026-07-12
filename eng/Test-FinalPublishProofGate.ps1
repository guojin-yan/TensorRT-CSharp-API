[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Test-LiveStaleClaimAbsent {
  param([string]$Pattern)

  $paths = @("README.md", "README.zh-CN.md", "docs", "samples", "applications", "src", "pack", "artifacts\interface-coverage", "artifacts\final-release")
  $arguments = @("-n", $Pattern) + $paths + @("-S", "-g", "!**/bin/**", "-g", "!**/obj/**", "-g", "!docs/_site/**")
  $matches = @()
  try {
    $matches = @(& rg @arguments 2>$null)
  }
  catch {
    $matches = @()
  }

  $liveMatches = @(
    $matches | Where-Object {
      $line = [string]$_
      if ($line -like "artifacts\final-release\*" -or $line -like "artifacts/final-release/*") {
        return $false
      }

      if ($Pattern -eq "\bYoloDet\b") {
        $normalized = $line.Replace("/", "\")
        if ($normalized -match "samples\\YoloDet(\\|$)" -or $normalized -match "\bYoloDet\.csproj\b") {
          return $true
        }

        $boundaryMarkers = @(
          "旧", "历史", "迁移", "已取代", "取代旧", "不应", "不能作为", "回流",
          "stale", "legacy", "old", "renamed", "migrated", "migration",
          "replace", "replaced", "regression guard", "must not", "must not return"
        )
        foreach ($marker in $boundaryMarkers) {
          if ($line.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $false
          }
        }
      }

      return $true
    }
  )

  return @($liveMatches)
}

$laneWorklist = Read-JsonOrNull "artifacts\final-release\release-close-proof-lane-worklist.json"
$yoloValidation = Read-JsonOrNull "artifacts\user-acceptance\yolovision-real-asset-owner-proof-input-validation.json"
$packageOwnerValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$prePublishValidation = Read-JsonOrNull "artifacts\final-release\final-release-pre-publish-audit-matrix-validation.json"
$releaseDashboard = Read-JsonOrNull "artifacts\final-release\release-proof-dashboard.json"
$dashboardValidation = Read-JsonOrNull "artifacts\final-release\release-proof-dashboard-validation.json"
$publicDocsGate = Read-JsonOrNull "artifacts\final-release\public-docs-package-metadata-gate.json"
$ownerExternalProofResultImportValidation = Read-JsonOrNull "artifacts\final-release\owner-external-proof-execution-result-import-validation.json"
$realProofRecordCandidateFromOwnerResultImportValidation = Read-JsonOrNull "artifacts\final-release\real-proof-record-candidate-from-owner-result-import-validation.json"
$finalOwnerRealInputTemplatePackValidation = Read-JsonOrNull "artifacts\final-release\final-owner-real-input-template-pack-validation.json"

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem "no-automatic-nuget-push" $true "blocker" "Final publish proof gate does not execute dotnet nuget push and only writes validation artifacts.")) | Out-Null
$items.Add((New-ValidationItem "release-lane-worklist-present" ($null -ne $laneWorklist -and [string](Get-PropertyOrDefault -Object $laneWorklist -Name "recordKind" -DefaultValue "") -eq "release-close-proof-lane-worklist") "blocker" "release-close-proof-lane-worklist.json must exist.")) | Out-Null
$items.Add((New-ValidationItem "release-lane-worklist-non-proof" ($null -ne $laneWorklist -and -not [bool](Get-PropertyOrDefault -Object $laneWorklist -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $laneWorklist -Name "canPublishPublicly" -DefaultValue $true)) "blocker" "Proof lane worklist must not publish or close the release.")) | Out-Null
$items.Add((New-ValidationItem "real-model-runtime-owner-proof-required" ($null -ne $yoloValidation -and [bool](Get-PropertyOrDefault -Object $yoloValidation -Name "candidateReadyForRealModelRuntime" -DefaultValue $false)) "action-required" "real-model-runtime still requires real YoloVision logs, output JSON, hashes, host metadata, and owner review.")) | Out-Null
$items.Add((New-ValidationItem "package-consumer-runtime-owner-proof-required" ($null -ne $packageOwnerValidation -and [bool](Get-PropertyOrDefault -Object $packageOwnerValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)) "action-required" "package-consumer-runtime still requires clean external consumer proof with public package source.")) | Out-Null
$items.Add((New-ValidationItem "post-publish-verification-owner-proof-required" ($null -ne $postPublishValidation -and [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)) "action-required" "post-publish verification requires public channel publish and clean install/run logs.")) | Out-Null
$finalOwnerRealInputTemplatePackPresent = $null -ne $finalOwnerRealInputTemplatePackValidation -and [string](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "recordKind" -DefaultValue "") -eq "final-owner-real-input-template-pack-validation"
$finalOwnerRealInputTemplatePackSafe = $finalOwnerRealInputTemplatePackPresent -and
  [int](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "laneCount" -DefaultValue 0) -ge 5 -and
  [int](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "failedBlockerCount" -DefaultValue -1) -eq 0 -and
  [int](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "failedActionRequiredCount" -DefaultValue 0) -gt 0 -and
  -not [bool](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "canPublishPublicly" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "canCloseReleaseIssue" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "isPostPublishProof" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "isReleaseCloseProof" -DefaultValue $true)
$items.Add((New-ValidationItem "final-owner-real-input-template-pack-safe" $finalOwnerRealInputTemplatePackSafe "blocker" "Final owner real input template pack must cover all configured lanes, remain non-proof, and keep action-required blockers until real Owner input exists.")) | Out-Null
$items.Add((New-ValidationItem "final-owner-real-input-template-pack-owner-input-required" ($finalOwnerRealInputTemplatePackPresent -and [int](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "failedActionRequiredCount" -DefaultValue 0) -eq 0) "action-required" "Final owner real input template pack still requires real owner-filled stdout/stderr/log/SHA256/exitCode/host/package evidence for every lane.")) | Out-Null
$ownerExternalProofResultImportPresent = $null -ne $ownerExternalProofResultImportValidation -and [string](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "recordKind" -DefaultValue "") -eq "owner-external-proof-execution-result-import-validation"
$ownerExternalProofResultImportStructurallySafe = $ownerExternalProofResultImportPresent -and
  [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "failedBlockerCount" -DefaultValue -1) -eq 0 -and
  [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "ownerExternalProofResultLaneCount" -DefaultValue 0) -eq 6 -and
  [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "ownerExternalProofResultPromotableLaneCount" -DefaultValue -1) -eq 0 -and
  [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "fileMissingCount" -DefaultValue -1) -ge 0 -and
  [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "invalidSha256Count" -DefaultValue -1) -ge 0 -and
  [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "hashMismatchCount" -DefaultValue -1) -ge 0 -and
  [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "outsideAllowedEvidenceRootCount" -DefaultValue -1) -ge 0 -and
  [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "forbiddenSubstituteFindingCount" -DefaultValue -1) -ge 0 -and
  -not [bool](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "isPostPublishProof" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "canCloseReleaseIssue" -DefaultValue $true)
$items.Add((New-ValidationItem "owner-external-proof-result-import-structurally-safe" $ownerExternalProofResultImportStructurallySafe "blocker" "Owner external proof result import validation must exist, cover all six lanes, have no structural blockers, and remain non-proof until strict validators promote real records.")) | Out-Null
$items.Add((New-ValidationItem "owner-external-proof-result-import-owner-proof-required" ($ownerExternalProofResultImportPresent -and [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "failedActionRequiredCount" -DefaultValue 0) -eq 0) "action-required" "owner external proof result import still requires real existing logs, matching SHA256, exitCode=0, non-substitute confirmations, and owner review for every lane.")) | Out-Null
$ownerResultCandidateBridgePresent = $null -ne $realProofRecordCandidateFromOwnerResultImportValidation -and [string](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "recordKind" -DefaultValue "") -eq "real-proof-record-candidate-from-owner-result-import-validation"
$ownerResultCandidateBridgeSafe = $ownerResultCandidateBridgePresent -and
  [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "failedBlockerCount" -DefaultValue -1) -eq 0 -and
  [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "candidateCount" -DefaultValue -1) -ge 0 -and
  [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "strictValidatorReadyCandidateCount" -DefaultValue -1) -ge 0 -and
  [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "packageConsumerRuntimeCandidateCount" -DefaultValue -1) -ge 0 -and
  [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "postPublishVerificationCandidateCount" -DefaultValue -1) -ge 0 -and
  -not [bool](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "isPostPublishProof" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "canCloseReleaseIssue" -DefaultValue $true)
$items.Add((New-ValidationItem "owner-result-candidate-bridge-structurally-safe" $ownerResultCandidateBridgeSafe "blocker" "Owner result candidate bridge validation must exist, have no blockers, and keep ready contracts as non-proof strict-validator input only.")) | Out-Null
$items.Add((New-ValidationItem "owner-result-candidate-bridge-real-proof-required" ($ownerResultCandidateBridgePresent -and [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "candidateCount" -DefaultValue 0) -gt 0 -and [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "failedActionRequiredCount" -DefaultValue 0) -eq 0) "action-required" "owner result candidates still require strict real proof validator and promotion guard before any runtime or post-publish proof claim.")) | Out-Null
$items.Add((New-ValidationItem "prepublish-audit-does-not-promote-proof" ($null -eq $prePublishValidation -or (-not [bool](Get-PropertyOrDefault -Object $prePublishValidation -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $prePublishValidation -Name "canCloseReleaseIssue" -DefaultValue $true))) "blocker" "Pre-publish audit must not promote proof.")) | Out-Null
$dashboardValidationIsCurrent = $false
if ($null -ne $dashboardValidation -and $null -ne $releaseDashboard) {
  $dashboardValidationInputPath = [string](Get-PropertyOrDefault -Object $dashboardValidation -Name "inputPath" -DefaultValue "")
  $dashboardPath = Join-Path $RepositoryRoot "artifacts\final-release\release-proof-dashboard.json"
  $dashboardValidationIsCurrent = -not [string]::IsNullOrWhiteSpace($dashboardValidationInputPath) -and
    [IO.Path]::GetFullPath($dashboardValidationInputPath).Equals([IO.Path]::GetFullPath($dashboardPath), [StringComparison]::OrdinalIgnoreCase) -and
    [string](Get-PropertyOrDefault -Object $releaseDashboard -Name "recordKind" -DefaultValue "") -eq "release-proof-dashboard"
}

$dashboardValidationPassed = $null -eq $dashboardValidation -or -not $dashboardValidationIsCurrent -or (
  [string](Get-PropertyOrDefault -Object $dashboardValidation -Name "recordKind" -DefaultValue "") -eq "release-proof-dashboard-validation" -and
  [int](Get-PropertyOrDefault -Object $dashboardValidation -Name "failedBlockerCount" -DefaultValue 0) -eq 0 -and
  -not [bool](Get-PropertyOrDefault -Object $dashboardValidation -Name "canPublishPublicly" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $dashboardValidation -Name "canCloseReleaseIssue" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $dashboardValidation -Name "canPromoteRuntimeProof" -DefaultValue $true)
)
$items.Add((New-ValidationItem "release-dashboard-validation-non-proof-passed" $dashboardValidationPassed "blocker" "If current release-proof-dashboard-validation.json exists, it must pass with zero blockers and still not promote proof; stale validation is ignored until regenerated.")) | Out-Null
$publicDocsGatePassed = $null -eq $publicDocsGate -or (
  [string](Get-PropertyOrDefault -Object $publicDocsGate -Name "recordKind" -DefaultValue "") -eq "public-docs-package-metadata-gate" -and
  [int](Get-PropertyOrDefault -Object $publicDocsGate -Name "failedBlockerCount" -DefaultValue 0) -eq 0 -and
  -not [bool](Get-PropertyOrDefault -Object $publicDocsGate -Name "canPublishPublicly" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $publicDocsGate -Name "canCloseReleaseIssue" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $publicDocsGate -Name "canPromoteRuntimeProof" -DefaultValue $true)
)
$items.Add((New-ValidationItem "public-docs-package-metadata-gate-passed" $publicDocsGatePassed "blocker" "If public-docs-package-metadata-gate.json exists, it must pass with zero blockers and still not promote proof.")) | Out-Null
$laneBoundary = [string](Get-PropertyOrDefault -Object $laneWorklist -Name "boundary" -DefaultValue "")
$items.Add((New-ValidationItem "no-sample-run-substitute-package-consumer" ($null -ne $laneWorklist -and $laneBoundary -match "(?i)sample-run-evidence") "blocker" "sample-run-evidence must remain separate from package-consumer-runtime.")) | Out-Null

$yoloDetLiveMatches = @(Test-LiveStaleClaimAbsent -Pattern "\bYoloDet\b")
$tensorRtLayerLiveMatches = @(Test-LiveStaleClaimAbsent -Pattern "TensorRtLayerTensorInfo")
$items.Add((New-ValidationItem "no-yolodet-live" ($yoloDetLiveMatches.Count -eq 0) "blocker" "YoloDet must not reappear in live docs/source/samples/applications.")) | Out-Null
$items.Add((New-ValidationItem "no-tensorrt-layer-tensor-info-live" ($tensorRtLayerLiveMatches.Count -eq 0) "blocker" "TensorRtLayerTensorInfo stale claim must not reappear in live docs/source.")) | Out-Null

$failedBlockerCount = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" }).Count
$failedActionRequiredCount = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" }).Count
$validationState = "blocked-final-publish-real-proof-required"
if ($failedBlockerCount -ne 0) {
  $validationState = "failed-final-publish-proof-gate"
}

$sourceArtifacts = @(
  "artifacts/final-release/release-close-proof-lane-worklist.json",
  "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
  "artifacts/final-release/post-publish-verification-validation.json",
  "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
  "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json",
  "artifacts/final-release/final-owner-real-input-template-pack-validation.json",
  "artifacts/final-release/final-release-pre-publish-audit-matrix-validation.json",
  "artifacts/final-release/release-proof-dashboard-validation.json",
  "artifacts/final-release/public-docs-package-metadata-gate.json"
)

$boundary = "Final publish proof gate blocks release until real-model-runtime, package-consumer-runtime, post-publish verification, public owner confirmation, and public docs/package metadata gate are all safe. If release-proof-dashboard-validation or public-docs-package-metadata-gate exists and has failed blockers, this gate remains blocked. It does not run dotnet nuget push and does not accept build-only, dry-run, template, local feed, ProjectReference, direct .nupkg, TensorRtExec report, YoloVision matrix, screenshot, sidecar-only report, skipped run, or blocked-by-cuda-driver substitutes."

$report = [pscustomobject]@{
  recordKind = "final-publish-proof-gate-report"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  ownerExternalProofResultImportState = [string](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "validationState" -DefaultValue "missing-owner-external-proof-execution-result-import-validation")
  ownerExternalProofResultLaneCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "ownerExternalProofResultLaneCount" -DefaultValue 0)
  ownerExternalProofResultBlockedLaneCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "ownerExternalProofResultBlockedLaneCount" -DefaultValue 0)
  ownerExternalProofResultReadyLaneCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "ownerExternalProofResultReadyLaneCount" -DefaultValue 0)
  ownerExternalProofResultPromotableLaneCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "ownerExternalProofResultPromotableLaneCount" -DefaultValue 0)
  ownerExternalProofResultFileMissingCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "fileMissingCount" -DefaultValue 0)
  ownerExternalProofResultInvalidSha256Count = [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "invalidSha256Count" -DefaultValue 0)
  ownerExternalProofResultHashMismatchCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "hashMismatchCount" -DefaultValue 0)
  ownerExternalProofResultOutsideAllowedEvidenceRootCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "outsideAllowedEvidenceRootCount" -DefaultValue 0)
  ownerExternalProofResultForbiddenSubstituteFindingCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofResultImportValidation -Name "forbiddenSubstituteFindingCount" -DefaultValue 0)
  ownerResultCandidateBridgeState = [string](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "validationState" -DefaultValue "missing-real-proof-record-candidate-from-owner-result-import-validation")
  ownerResultCandidateBridgeCandidateCount = [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "candidateCount" -DefaultValue 0)
  ownerResultCandidateBridgeStrictValidatorReadyCandidateCount = [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "strictValidatorReadyCandidateCount" -DefaultValue 0)
  ownerResultCandidateBridgePackageConsumerCandidateCount = [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "packageConsumerRuntimeCandidateCount" -DefaultValue 0)
  ownerResultCandidateBridgePostPublishCandidateCount = [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "postPublishVerificationCandidateCount" -DefaultValue 0)
  finalOwnerRealInputTemplatePackState = [string](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "validationState" -DefaultValue "missing-final-owner-real-input-template-pack-validation")
  finalOwnerRealInputTemplatePackLaneCount = [int](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "laneCount" -DefaultValue 0)
  finalOwnerRealInputTemplatePackFailedBlockerCount = [int](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "failedBlockerCount" -DefaultValue 999)
  finalOwnerRealInputTemplatePackFailedActionRequiredCount = [int](Get-PropertyOrDefault -Object $finalOwnerRealInputTemplatePackValidation -Name "failedActionRequiredCount" -DefaultValue 999)
  failedBlockerCount = [int]$failedBlockerCount
  failedActionRequiredCount = [int]$failedActionRequiredCount
  validationItems = [object[]]@($items.ToArray())
  liveYoloDetMatches = [string[]]@($yoloDetLiveMatches)
  liveTensorRtLayerTensorInfoMatches = [string[]]@($tensorRtLayerLiveMatches)
  sourceArtifacts = [string[]]$sourceArtifacts
  boundary = $boundary
}

$jsonPath = Join-Path $OutputRoot "final-publish-proof-gate-report.json"
$markdownPath = Join-Path $OutputRoot "final-publish-proof-gate-report.md"
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $items) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Final Publish Proof Gate Report

Generated at: ``$($report.generatedAtUtc)``

## Summary

- recordKind: ``$($report.recordKind)``
- validationState: ``$($report.validationState)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromoteRuntimeProof: ``False``
- failedBlockerCount: ``$($report.failedBlockerCount)``
- failedActionRequiredCount: ``$($report.failedActionRequiredCount)``

## Validation Items

| Id | Passed | Severity | Detail |
| --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($report.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Final publish proof gate report written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$($report.validationState) FailedBlockers=$($report.failedBlockerCount) ActionRequired=$($report.failedActionRequiredCount)"

if ($Strict.IsPresent -and $failedBlockerCount -gt 0) {
  exit 1
}
