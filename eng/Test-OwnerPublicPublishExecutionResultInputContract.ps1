[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory,
  [string]$InputPath,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPublicPublishExecutionResultCommon.ps1")

$ctx = Initialize-OwnerPublicPublishContext -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-input-contract.json" }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Export-OwnerPublicPublishExecutionResultInputContract.ps1") -RepositoryRoot $ctx.RepositoryRoot -OutputDirectory $ctx.OutputDirectory
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = $record | ConvertTo-Json -Depth 32
$fields = @((Get-OwnerPropertyOrDefault -Object $record -Name "requiredFields" -DefaultValue @()))
$blockedFields = @($fields | Where-Object { -not [bool](Get-OwnerPropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$dualPackageRouteFields = @($fields | Where-Object {
    @("nugetSmallBridgeCoreRoute", "githubPackagesFullRuntimeRoute") -contains [string](Get-OwnerPropertyOrDefault -Object $_ -Name "group" -DefaultValue "")
  })
$requiredMarkers = @(
  "nugetSmallBridgeCoreOwnerAuthorizationUrl", "nugetSmallBridgeCorePublicPackageUrl",
  "nugetSmallBridgeCorePackageId", "nugetSmallBridgeCorePackageVersion",
  "nugetSmallBridgeCoreDownloadedNupkgSha256", "nugetSmallBridgeCoreCleanExternalConsumerLogPath",
  "nugetSmallBridgeCoreCleanExternalConsumerLogSha256", "nugetSmallBridgeCorePostPublishCleanConsumerProofLogSha256",
  "githubPackagesFullRuntimeOwnerAuthorizationUrl", "githubPackagesFullRuntimeRestoreSourceUrl",
  "githubPackagesFullRuntimePackageId", "githubPackagesFullRuntimePackageVersion",
  "githubPackagesFullRuntimePackageKey", "githubPackagesFullRuntimePackageSha256",
  "githubPackagesFullRuntimeDllResolutionReportPath", "githubPackagesFullRuntimeCleanRuntimeSmokeLogSha256",
  "publicPackageUrl", "publicPackageSha256", "githubReleaseAssetUrl", "githubReleaseAssetSha256",
  "nugetPushTranscriptPath", "nugetPushTranscriptSha256", "cleanConsumerRestoreStdoutPath",
  "cleanConsumerBuildStdoutPath", "cleanConsumerRuntimeSmokeStdoutPath", "strictValidatorOutputPath",
  "hostOs", "hostGpuName", "hostCudaVersion", "hostTensorRtVersion", "hostCudnnVersion",
  "packageManagedPackageId", "packageNativeBridgePackageId", "packageRuntimePackageId",
  "releaseNotesPath", "rollbackDecision", "releaseIssueCloseDecision", "finalPublicPackageUrlApproval",
  "ownerReviewer", "ownerSignature", "noLocalFeedConfirmation", "noProjectReferenceConfirmation",
  "noDirectNupkgConfirmation", "noBuildOnlyConfirmation", "noDependencyProbeOnlyConfirmation",
  "noDryRunOnlyConfirmation", "noCandidateDashboardRunbookSubstitutionConfirmation"
)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem -Id "record-kind" -Passed ([string](Get-OwnerPropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-public-publish-execution-result-input-contract") -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "state-blocked" -Passed ([string](Get-OwnerPropertyOrDefault -Object $record -Name "contractState" -DefaultValue "") -eq "blocked-owner-public-publish-execution-result-input-required") -Severity "blocker" -Detail "Contract must remain blocked until real Owner input exists.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "required-fields-at-least-100" -Passed ($fields.Count -ge 100) -Severity "blocker" -Detail "Contract must expose at least 100 required Owner fields.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "dual-package-route-fields" -Passed ([int](Get-OwnerPropertyOrDefault -Object $record -Name "dualPackageRouteCount" -DefaultValue 0) -eq 2 -and $dualPackageRouteFields.Count -ge 18) -Severity "blocker" -Detail "Contract must expose both NuGet managed and GitHub Packages bridge-only route fields; legacy field names do not permit vendor runtime packages.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "dual-package-route-compatibility-boundary" -Passed (
  [bool](Get-OwnerPropertyOrDefault -Object $record -Name "legacyDualPackageFieldNamesPreserved" -DefaultValue $false) -and
  [bool](Get-OwnerPropertyOrDefault -Object $record -Name "vendorRuntimePackagesForbidden" -DefaultValue $false) -and
  @((Get-OwnerPropertyOrDefault -Object $record -Name "currentDualPackageRouteIds" -DefaultValue @())) -contains "nuget-managed-plus-bridge-packages" -and
  @((Get-OwnerPropertyOrDefault -Object $record -Name "currentDualPackageRouteIds" -DefaultValue @())) -contains "github-release-managed-plus-bridge-assets") -Severity "blocker" -Detail "Legacy field names must map explicitly to current managed plus bridge-only routes and forbid vendor runtime packages.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "all-fields-blocked-by-default" -Passed ($blockedFields.Count -eq $fields.Count) -Severity "action-required" -Detail "Owner must provide real values for every required field.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "non-proof-flags" -Passed ((-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isReleaseReady" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "Contract must be non-proof, non-publish, and non-close.")) | Out-Null

foreach ($marker in $requiredMarkers) {
  $items.Add((New-OwnerValidationItem -Id "marker-$marker" -Passed ($raw.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Required marker missing: $marker")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validation = [ordered]@{
  recordKind = "owner-public-publish-execution-result-input-contract-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = if ($failedBlockers.Count -eq 0) { "blocked-owner-public-publish-execution-result-input-required" } else { "failed-owner-public-publish-execution-result-input-contract" }
  contractState = [string](Get-OwnerPropertyOrDefault -Object $record -Name "contractState" -DefaultValue "")
  requiredFieldCount = $fields.Count
  blockedRequiredFieldCount = $blockedFields.Count
  readyRequiredFieldCount = 0
  dualPackageRouteCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "dualPackageRouteCount" -DefaultValue 0)
  dualPackageRouteRequiredFieldCount = $dualPackageRouteFields.Count
  dualPackageRouteReadyFieldCount = 0
  dualPackageRouteBlockedFieldCount = $dualPackageRouteFields.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = [Math]::Max($failedActionRequired.Count, 1)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "Owner public publish execution result input contract validation only. It is not proof, not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-input-contract-validation.json"
$markdownPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-input-contract-validation.md"
Write-OwnerUtf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 32)
Write-OwnerUtf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Public Publish Execution Result Input Contract Validation",
  "",
  "- validationState: ``$($validation.validationState)``",
  "- requiredFieldCount: ``$($validation.requiredFieldCount)``",
  "- dualPackageRouteCount: ``$($validation.dualPackageRouteCount)``",
  "- dualPackageRouteRequiredFieldCount: ``$($validation.dualPackageRouteRequiredFieldCount)``",
  "- blockedRequiredFieldCount: ``$($validation.blockedRequiredFieldCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- failedActionRequiredCount: ``$($validation.failedActionRequiredCount)``",
  "",
  "> $($validation.boundary)"
)

Write-Host "ValidationState=$($validation.validationState)"
Write-Host "RequiredFieldCount=$($validation.requiredFieldCount)"
Write-Host "FailedBlockerCount=$($validation.failedBlockerCount)"
if ($Strict -and $failedBlockers.Count -gt 0) { throw "Owner public publish execution result input contract validation failed." }
