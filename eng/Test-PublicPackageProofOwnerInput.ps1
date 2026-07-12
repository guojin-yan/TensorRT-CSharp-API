[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-package-proof-owner-input.template.json",
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

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Public package proof owner input not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$githubRelease = Get-PropertyOrDefault -Object $record -Name "githubRelease" -DefaultValue $null
$managed = Get-PropertyOrDefault -Object $record -Name "managedPackage" -DefaultValue $null
$runtime = Get-PropertyOrDefault -Object $record -Name "runtimePackage" -DefaultValue $null
$cleanExternalConsumer = Get-PropertyOrDefault -Object $record -Name "cleanExternalConsumer" -DefaultValue $null
$hostMetadata = Get-PropertyOrDefault -Object $record -Name "hostMetadata" -DefaultValue $null
$ownerReview = Get-PropertyOrDefault -Object $record -Name "ownerReview" -DefaultValue $null
$confirmation = Get-PropertyOrDefault -Object $record -Name "ownerConfirmation" -DefaultValue $null
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-package-proof-owner-input") -Severity "blocker" -Detail "recordKind must be public-package-proof-owner-input.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Owner input must not publish, promote proof, prove post-publish state, or close the release issue.")) | Out-Null

$items.Add((New-ValidationItem -Id "field-nugetPackageSource" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "nugetPackageSource" -DefaultValue ""))) -Severity "action-required" -Detail "nugetPackageSource must identify the real public package source used by the clean external consumer.")) | Out-Null

foreach ($field in @("releaseUrl", "tagName", "managedAssetPath", "runtimeAssetPath")) {
  $items.Add((New-ValidationItem -Id "github-release-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $githubRelease -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "githubRelease.$field must be filled from the real GitHub Release asset surface.")) | Out-Null
}

foreach ($field in @("managedAssetSha256", "runtimeAssetSha256")) {
  $items.Add((New-ValidationItem -Id "github-release-$field" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $githubRelease -Name $field -DefaultValue "")) -Severity "action-required" -Detail "githubRelease.$field must be a real 64-character SHA256.")) | Out-Null
}

foreach ($prefix in @("managed", "runtime")) {
  $package = if ($prefix -eq "managed") { $managed } else { $runtime }
  foreach ($field in @("packageId", "version", "publicSourceUrl", "registryUrl", "packageUrl", "nupkgPath", "sha256Source", "publicDownloadUrl")) {
    $items.Add((New-ValidationItem -Id "$prefix-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $package -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$prefix package $field must be filled from the real public package channel.")) | Out-Null
  }
  $items.Add((New-ValidationItem -Id "$prefix-nupkgSha256" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $package -Name "nupkgSha256" -DefaultValue "")) -Severity "action-required" -Detail "$prefix package nupkgSha256 must be a real 64-character SHA256.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$prefix-publicDownloadSha256" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $package -Name "publicDownloadSha256" -DefaultValue "")) -Severity "action-required" -Detail "$prefix package publicDownloadSha256 must be a real 64-character SHA256.")) | Out-Null
}

foreach ($field in @("root", "projectPath", "restoreCommand", "buildCommand", "smokeCommand", "restoreLogPath", "buildLogPath", "smokeLogPath", "stdoutLogPath", "stderrLogPath", "exitCode")) {
  $items.Add((New-ValidationItem -Id "clean-consumer-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $cleanExternalConsumer -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "cleanExternalConsumer.$field must be filled from a clean external consumer run, not local feed, ProjectReference, direct .nupkg, build-only, or dry-run evidence.")) | Out-Null
}

foreach ($field in @("restoreLogSha256", "buildLogSha256", "smokeLogSha256", "stdoutLogSha256")) {
  $items.Add((New-ValidationItem -Id "clean-consumer-$field" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $cleanExternalConsumer -Name $field -DefaultValue "")) -Severity "action-required" -Detail "cleanExternalConsumer.$field must be a real 64-character SHA256.")) | Out-Null
}

$stderrSha = [string](Get-PropertyOrDefault -Object $cleanExternalConsumer -Name "stderrLogSha256" -DefaultValue "")
$stderrShaReady = (Test-Sha256Format -Value $stderrSha) -or ($stderrSha -eq "no-stderr-emitted")
$items.Add((New-ValidationItem -Id "clean-consumer-stderrLogSha256" -Passed $stderrShaReady -Severity "action-required" -Detail "cleanExternalConsumer.stderrLogSha256 must be a real 64-character SHA256 or the explicit token no-stderr-emitted.")) | Out-Null

$cleanConsumerRoot = [string](Get-PropertyOrDefault -Object $cleanExternalConsumer -Name "root" -DefaultValue "")
$items.Add((New-ValidationItem -Id "clean-consumer-root-outside-repository" -Passed ((-not (Test-IsPlaceholder -Value $cleanConsumerRoot)) -and (-not $cleanConsumerRoot.Contains($RepositoryRoot, [StringComparison]::OrdinalIgnoreCase))) -Severity "action-required" -Detail "cleanExternalConsumer.root must point outside the repository so public package proof cannot be confused with ProjectReference or local repo evidence.")) | Out-Null

$commands = @(
  [string](Get-PropertyOrDefault -Object $cleanExternalConsumer -Name "restoreCommand" -DefaultValue ""),
  [string](Get-PropertyOrDefault -Object $cleanExternalConsumer -Name "buildCommand" -DefaultValue ""),
  [string](Get-PropertyOrDefault -Object $cleanExternalConsumer -Name "smokeCommand" -DefaultValue "")
)
$commandText = ($commands -join " ")
$forbiddenTokens = @("ProjectReference", "local feed", "--source .\", "--source ./", "artifacts\", "artifacts/", ".nupkg")
foreach ($token in $forbiddenTokens) {
  $items.Add((New-ValidationItem -Id "forbidden-command-token-$($token.Replace('\','slash').Replace('/','slash').Replace(' ','-').Replace('.','dot'))" -Passed (-not $commandText.Contains($token, [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "cleanExternalConsumer commands must not contain forbidden substitute token '$token'.")) | Out-Null
}

foreach ($field in @("ownerName", "machineName", "osDescription", "architecture", "gpuName", "cudaDriverVersion", "cudaRuntimeVersion", "cudnnVersion", "tensorRtVersion", "tensorRtLine")) {
  $items.Add((New-ValidationItem -Id "host-metadata-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hostMetadata -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "hostMetadata.$field must be filled from the real owner host.")) | Out-Null
}

foreach ($field in @("reviewer", "reviewedAtUtc", "approvalState")) {
  $items.Add((New-ValidationItem -Id "owner-review-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $ownerReview -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "ownerReview.$field must be filled by Owner.")) | Out-Null
}

foreach ($field in @("publishTimestampUtc", "ownerReviewer", "ownerReviewTimestampUtc")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be filled by Owner.")) | Out-Null
}

foreach ($field in @("confirmsPublicPackageSource", "confirmsNoLocalFeed", "confirmsNoProjectReference", "confirmsNoDirectNupkgReference", "confirmsPackageHashesReviewed", "confirmsPublicDownload", "confirmsGithubReleaseAssetsReviewed", "confirmsCleanExternalConsumerRestoreBuildSmoke", "confirmsStdoutStderrSha256Reviewed", "confirmsHostMetadataReviewed")) {
  $items.Add((New-ValidationItem -Id "confirmation-$field" -Passed ([bool](Get-PropertyOrDefault -Object $confirmation -Name $field -DefaultValue $false)) -Severity "action-required" -Detail "ownerConfirmation.$field must be true after real public package review.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-public-package-proof-owner-input"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-public-package-proof-owner-input-required"
}
else {
  "public-package-proof-owner-input-ready"
}

$validation = [pscustomobject]@{
  recordKind = "public-package-proof-owner-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  canPromoteRuntimeProof = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Public package owner input validation only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$jsonPath = Join-Path $OutputRoot "public-package-proof-owner-input-validation.json"
$markdownPath = Join-Path $OutputRoot "public-package-proof-owner-input-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Public Package Proof Owner Input Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public package proof owner input validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Public package proof owner input has blocker validation failures."
}
