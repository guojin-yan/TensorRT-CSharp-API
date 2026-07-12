[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-verification-owner-input.template.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
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

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)

  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Test-RelativeFileHash {
  param([AllowNull()][object]$RelativePath, [AllowNull()][object]$Sha256)

  $pathText = [string]$RelativePath
  $hashText = [string]$Sha256
  if ((Test-IsPlaceholder -Value $pathText) -or -not (Test-Sha256Format -Value $hashText)) { return $false }
  $resolved = Resolve-InputPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $false }
  return [string]::Equals((Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash, $hashText, [System.StringComparison]::OrdinalIgnoreCase)
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

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Post-publish verification owner input not found: $resolvedInputPath"
}

$recordRaw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$record = $recordRaw | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$ownerInputState = [string](Get-PropertyOrDefault -Object $record -Name "ownerInputState" -DefaultValue "")
$proofLineId = [string](Get-PropertyOrDefault -Object $record -Name "proofLineId" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$declaredProof = [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishVerificationProof" -DefaultValue $true)
$packageIdentity = Get-PropertyOrDefault -Object $record -Name "packageIdentity" -DefaultValue $null
$hostInfo = Get-PropertyOrDefault -Object $record -Name "host" -DefaultValue $null

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "post-publish-verification-owner-input") -Severity "blocker" -Detail "recordKind must be post-publish-verification-owner-input.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-line-id" -Passed ($proofLineId -eq "post-publish-verification") -Severity "blocker" -Detail "proofLineId must be post-publish-verification.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue -and -not $declaredProof) -Severity "blocker" -Detail "Owner input must not publish, approve public release, prove post-publish verification, or close the release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-validator-command" -Passed (([string](Get-PropertyOrDefault -Object $record -Name "strictValidationCommand" -DefaultValue "")).Contains("Test-PostPublishVerificationRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and ([string](Get-PropertyOrDefault -Object $record -Name "strictValidationCommand" -DefaultValue "")).Contains("-FailOnNotProof", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "strictValidationCommand must use Test-PostPublishVerificationRecord.ps1 -FailOnNotProof.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes-listed" -Passed ($recordRaw.Contains("local feed", [StringComparison]::OrdinalIgnoreCase) -and $recordRaw.Contains("ProjectReference", [StringComparison]::OrdinalIgnoreCase) -and $recordRaw.Contains("direct .nupkg", [StringComparison]::OrdinalIgnoreCase) -and $recordRaw.Contains("manual approval", [StringComparison]::OrdinalIgnoreCase) -and $recordRaw.Contains("queued GitHub Actions run", [StringComparison]::OrdinalIgnoreCase) -and $recordRaw.Contains("missing self-hosted runner", [StringComparison]::OrdinalIgnoreCase) -and $recordRaw.Contains("sidecar-only", [StringComparison]::OrdinalIgnoreCase) -and $recordRaw.Contains("TensorRtExec report", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Owner input must list forbidden non-proof substitutes.")) | Out-Null

foreach ($field in @("selectedChannel", "channelSourceUri", "publishedPackageUrl", "packagePageUrl", "downloadedManagedPackagePath", "downloadedRuntimePackagePath", "runtimeNativeAssetResolutionReportPath", "ownerVerificationDecision", "rollbackReviewPath", "forbiddenSubstituteScanPath", "cleanConsumerRoot", "consumerProjectName", "consumerProjectPath", "restoreCommand", "buildCommand", "smokeCommand", "stdoutSummary", "stderrSummary", "restoreLogPath", "nativeAssetListingPath", "dependencyProbeLogPath", "smokeLogPath", "managedPackageSource", "runtimePackageSource", "ownerName", "reviewerName", "publishedVersion")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be filled with a real post-publish value.")) | Out-Null
}

foreach ($field in @("managedPackageId", "managedPackageVersion", "runtimePackageId", "runtimePackageVersion", "managedPackageUrl", "runtimePackageUrl", "managedPackageSha256Source", "runtimePackageSha256Source", "managedPackageDownloadTimestampUtc", "runtimePackageDownloadTimestampUtc")) {
  $items.Add((New-ValidationItem -Id "package-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $packageIdentity -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "packageIdentity.$field must be filled.")) | Out-Null
}

foreach ($field in @("managedNupkgSha256", "runtimeNupkgSha256")) {
  $items.Add((New-ValidationItem -Id "package-$field" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $packageIdentity -Name $field -DefaultValue "")) -Severity "action-required" -Detail "packageIdentity.$field must be a 64-character SHA256.")) | Out-Null
}

foreach ($field in @("ownerName", "machineName", "osDescription", "gpuName", "driverVersion", "cudaDriverSupportedRuntime", "cudaRuntimeVersion", "tensorRtRuntimeVersion", "tensorRtLine", "cudnnVersion")) {
  $items.Add((New-ValidationItem -Id "host-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hostInfo -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "host.$field must be filled.")) | Out-Null
}

foreach ($field in @("restoreLogSha256", "nativeAssetListingSha256", "dependencyProbeLogSha256", "smokeLogSha256", "downloadedManagedPackageSha256", "downloadedRuntimePackageSha256", "runtimeNativeAssetResolutionReportSha256", "rollbackReviewSha256", "forbiddenSubstituteScanSha256")) {
  $items.Add((New-ValidationItem -Id "sha-$field" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must be a 64-character SHA256.")) | Out-Null
}

$smokeCommand = [string](Get-PropertyOrDefault -Object $record -Name "smokeCommand" -DefaultValue "")
$runtimePackageKey = [string](Get-PropertyOrDefault -Object $record -Name "runtimePackageKey" -DefaultValue "")
$items.Add((New-ValidationItem -Id "smoke-command-runtime-key" -Passed ($smokeCommand.Contains("--runtime-package-key", [StringComparison]::Ordinal) -and $smokeCommand.Contains($runtimePackageKey, [StringComparison]::Ordinal)) -Severity "action-required" -Detail "smokeCommand must include --runtime-package-key and the target runtime package key.")) | Out-Null

$cleanConsumerScanPath = [string](Get-PropertyOrDefault -Object $record -Name "cleanConsumerProjectScanPath" -DefaultValue "")
$cleanConsumerScanSha256 = [string](Get-PropertyOrDefault -Object $record -Name "cleanConsumerProjectScanSha256" -DefaultValue "")
$items.Add((New-ValidationItem -Id "clean-consumer-scan-hash" -Passed (Test-RelativeFileHash -RelativePath $cleanConsumerScanPath -Sha256 $cleanConsumerScanSha256) -Severity "action-required" -Detail "clean consumer scan path and SHA256 must exist and match.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-owner-input"
}
elseif ($failedActionRequired.Count -gt 0 -or $ownerInputState -like "*template*") {
  "blocked-owner-input-required"
}
else {
  "owner-input-ready-for-record-projection"
}

$validation = [pscustomobject]@{
  recordKind = "post-publish-verification-owner-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  isValidOwnerInputShape = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPublishPublicly = $false
  isPostPublishVerificationProof = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Owner input validation only. It cannot publish packages, prove post-publish verification, or close the release issue. Local feed, ProjectReference, direct .nupkg, dashboard, dry-run, manual approval, queued workflow, missing runner, sidecar-only, and TensorRtExec report are non-proof substitutes."
}

$jsonPath = Join-Path $OutputRoot "post-publish-verification-owner-input-validation.json"
$markdownPath = Join-Path $OutputRoot "post-publish-verification-owner-input-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Post-Publish Verification Owner Input Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| isValidOwnerInputShape | ``$($validation.isValidOwnerInputShape)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| isPostPublishVerificationProof | ``$($validation.isPostPublishVerificationProof)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| performsPublish | ``$($validation.performsPublish)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Safety Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish verification owner input validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) PerformsPublish=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Post-publish verification owner input has blocker validation failures."
}
