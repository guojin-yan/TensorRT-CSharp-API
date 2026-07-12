[CmdletBinding()]
param(
  [string]$OwnerInputPath = "artifacts\final-release\post-publish-verification-owner-input.template.json",
  [string]$TemplatePath = "artifacts\final-release\post-publish-verification-record-template.json",
  [string]$OutputPath = "artifacts\final-release\post-publish-verification-record.json",
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

$templateFullPath = Resolve-RepositoryPath -Path $TemplatePath
if (-not (Test-Path -LiteralPath $templateFullPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-PostPublishVerificationRecordTemplate.ps1")
}

if (-not (Test-Path -LiteralPath $templateFullPath -PathType Leaf)) {
  throw "Post-publish verification record template was not found: $templateFullPath"
}

$ownerInputFullPath = Resolve-RepositoryPath -Path $OwnerInputPath
if (-not (Test-Path -LiteralPath $ownerInputFullPath -PathType Leaf)) {
  throw "Post-publish verification owner input was not found: $ownerInputFullPath"
}

$record = Get-Content -LiteralPath $templateFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$ownerInput = Get-Content -LiteralPath $ownerInputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$ownerPackageIdentity = Get-PropertyOrDefault -Object $ownerInput -Name "packageIdentity" -DefaultValue $null
$ownerHost = Get-PropertyOrDefault -Object $ownerInput -Name "host" -DefaultValue $null

$record.recordKind = "post-publish-verification-record"
$record.templateOnly = $false
$record.verificationState = "owner-action-required"
$record.postPublishProofClassification = "owner-action-required"
$record.runtimePackageKey = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimePackageKey" -DefaultValue $record.runtimePackageKey)
$record.selectedChannel = [string](Get-PropertyOrDefault -Object $ownerInput -Name "selectedChannel" -DefaultValue $record.selectedChannel)
$record.channelSourceUri = [string](Get-PropertyOrDefault -Object $ownerInput -Name "channelSourceUri" -DefaultValue $record.channelSourceUri)
$record.publishedPackageUrl = [string](Get-PropertyOrDefault -Object $ownerInput -Name "publishedPackageUrl" -DefaultValue $record.publishedPackageUrl)
$record.packagePageUrl = [string](Get-PropertyOrDefault -Object $ownerInput -Name "packagePageUrl" -DefaultValue $record.packagePageUrl)
$record.downloadedManagedPackagePath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "downloadedManagedPackagePath" -DefaultValue $record.downloadedManagedPackagePath)
$record.downloadedManagedPackageSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "downloadedManagedPackageSha256" -DefaultValue $record.downloadedManagedPackageSha256)
$record.downloadedRuntimePackagePath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "downloadedRuntimePackagePath" -DefaultValue $record.downloadedRuntimePackagePath)
$record.downloadedRuntimePackageSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "downloadedRuntimePackageSha256" -DefaultValue $record.downloadedRuntimePackageSha256)
$record.runtimeNativeAssetResolutionReportPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimeNativeAssetResolutionReportPath" -DefaultValue $record.runtimeNativeAssetResolutionReportPath)
$record.runtimeNativeAssetResolutionReportSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimeNativeAssetResolutionReportSha256" -DefaultValue $record.runtimeNativeAssetResolutionReportSha256)
$record.ownerVerificationDecision = [string](Get-PropertyOrDefault -Object $ownerInput -Name "ownerVerificationDecision" -DefaultValue $record.ownerVerificationDecision)
$record.rollbackReviewPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "rollbackReviewPath" -DefaultValue $record.rollbackReviewPath)
$record.rollbackReviewSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "rollbackReviewSha256" -DefaultValue $record.rollbackReviewSha256)
$record.forbiddenSubstituteScanPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "forbiddenSubstituteScanPath" -DefaultValue $record.forbiddenSubstituteScanPath)
$record.forbiddenSubstituteScanSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "forbiddenSubstituteScanSha256" -DefaultValue $record.forbiddenSubstituteScanSha256)
$record.cleanConsumerRoot = [string](Get-PropertyOrDefault -Object $ownerInput -Name "cleanConsumerRoot" -DefaultValue $record.cleanConsumerRoot)
$record.consumerProjectName = [string](Get-PropertyOrDefault -Object $ownerInput -Name "consumerProjectName" -DefaultValue $record.consumerProjectName)
$record.consumerProjectPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "consumerProjectPath" -DefaultValue $record.consumerProjectPath)
$record.restoreCommand = [string](Get-PropertyOrDefault -Object $ownerInput -Name "restoreCommand" -DefaultValue $record.restoreCommand)
$record.buildCommand = [string](Get-PropertyOrDefault -Object $ownerInput -Name "buildCommand" -DefaultValue $record.buildCommand)
$record.smokeCommand = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeCommand" -DefaultValue $record.smokeCommand)
$record.stdoutSummary = [string](Get-PropertyOrDefault -Object $ownerInput -Name "stdoutSummary" -DefaultValue $record.stdoutSummary)
$record.stderrSummary = [string](Get-PropertyOrDefault -Object $ownerInput -Name "stderrSummary" -DefaultValue $record.stderrSummary)
$record.restoreLogPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "restoreLogPath" -DefaultValue $record.restoreLogPath)
$record.restoreLogSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "restoreLogSha256" -DefaultValue $record.restoreLogSha256)
$record.nativeAssetListingPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "nativeAssetListingPath" -DefaultValue $record.nativeAssetListingPath)
$record.nativeAssetListingSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "nativeAssetListingSha256" -DefaultValue $record.nativeAssetListingSha256)
$record.dependencyProbeLogPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "dependencyProbeLogPath" -DefaultValue $record.dependencyProbeLogPath)
$record.dependencyProbeLogSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "dependencyProbeLogSha256" -DefaultValue $record.dependencyProbeLogSha256)
$record.runtimeSmokeLogPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeLogPath" -DefaultValue $record.runtimeSmokeLogPath)
$record.runtimeSmokeLogSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeLogSha256" -DefaultValue $record.runtimeSmokeLogSha256)
$record.smokeLogPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeLogPath" -DefaultValue $record.smokeLogPath)
$record.smokeLogSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeLogSha256" -DefaultValue $record.smokeLogSha256)
$record.managedPackageSource = [string](Get-PropertyOrDefault -Object $ownerInput -Name "managedPackageSource" -DefaultValue $record.managedPackageSource)
$record.runtimePackageSource = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimePackageSource" -DefaultValue $record.runtimePackageSource)
$record.expectedRuntimePackageKey = [string](Get-PropertyOrDefault -Object $ownerInput -Name "expectedRuntimePackageKey" -DefaultValue $record.expectedRuntimePackageKey)
$record.noProjectReference = [bool](Get-PropertyOrDefault -Object $ownerInput -Name "noProjectReference" -DefaultValue $false)
$record.noLocalPackageSource = [bool](Get-PropertyOrDefault -Object $ownerInput -Name "noLocalPackageSource" -DefaultValue $false)
$record.noLocalNupkgPackageReference = [bool](Get-PropertyOrDefault -Object $ownerInput -Name "noLocalNupkgPackageReference" -DefaultValue $false)
$record.nativeAssetsCopied = [bool](Get-PropertyOrDefault -Object $ownerInput -Name "nativeAssetsCopied" -DefaultValue $false)
$record.dependencyProbePassed = [bool](Get-PropertyOrDefault -Object $ownerInput -Name "dependencyProbePassed" -DefaultValue $false)
$record.runtimeSmokePassed = [bool](Get-PropertyOrDefault -Object $ownerInput -Name "runtimeSmokePassed" -DefaultValue $false)
$record.runtimeSmokeExitCode = Get-PropertyOrDefault -Object $ownerInput -Name "runtimeSmokeExitCode" -DefaultValue $null
$record.smokeStatus = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeStatus" -DefaultValue "owner-action-required")
$record.performsPublish = $false
$record.isPostPublishVerificationProof = $false
$record.canCloseReleaseIssue = $false
$record.ownerName = [string](Get-PropertyOrDefault -Object $ownerInput -Name "ownerName" -DefaultValue $record.ownerName)
$record.reviewerName = [string](Get-PropertyOrDefault -Object $ownerInput -Name "reviewerName" -DefaultValue $record.reviewerName)
$record.publishedVersion = [string](Get-PropertyOrDefault -Object $ownerInput -Name "publishedVersion" -DefaultValue $record.publishedVersion)

foreach ($field in @("managedPackageId", "managedPackageVersion", "runtimePackageId", "runtimePackageVersion", "managedPackageUrl", "runtimePackageUrl", "managedNupkgSha256", "runtimeNupkgSha256", "managedPackageSha256Source", "runtimePackageSha256Source", "managedPackageDownloadTimestampUtc", "runtimePackageDownloadTimestampUtc")) {
  $record.packageIdentity.$field = [string](Get-PropertyOrDefault -Object $ownerPackageIdentity -Name $field -DefaultValue $record.packageIdentity.$field)
}

foreach ($field in @("ownerName", "machineName", "osDescription", "gpuName", "driverVersion", "cudaDriverSupportedRuntime", "cudaRuntimeVersion", "tensorRtRuntimeVersion", "tensorRtLine", "cudnnVersion")) {
  $record.host.$field = [string](Get-PropertyOrDefault -Object $ownerHost -Name $field -DefaultValue $record.host.$field)
  $record.hostMetadata.$field = $record.host.$field
}

$record | Add-Member -NotePropertyName ownerInputPath -NotePropertyValue $OwnerInputPath -Force
$record | Add-Member -NotePropertyName ownerInputOverlayApplied -NotePropertyValue $true -Force
$record | Add-Member -NotePropertyName cleanConsumerProjectScanPath -NotePropertyValue ([string](Get-PropertyOrDefault -Object $ownerInput -Name "cleanConsumerProjectScanPath" -DefaultValue "")) -Force
$record | Add-Member -NotePropertyName strictValidationCommand -NotePropertyValue ([string](Get-PropertyOrDefault -Object $ownerInput -Name "strictValidationCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof")) -Force

$outputFullPath = Resolve-RepositoryPath -Path $OutputPath
$outputRoot = [System.IO.Path]::GetDirectoryName($outputFullPath)
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $outputFullPath -Encoding utf8

$markdownPath = [System.IO.Path]::ChangeExtension($outputFullPath, ".md")
$markdown = @"
# Post-Publish Verification Record From Owner Input

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| recordKind | ``$($record.recordKind)`` |
| verificationState | ``$($record.verificationState)`` |
| postPublishProofClassification | ``$($record.postPublishProofClassification)`` |
| ownerInputOverlayApplied | ``True`` |
| performsPublish | ``$($record.performsPublish)`` |
| isPostPublishVerificationProof | ``$($record.isPostPublishVerificationProof)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Boundary

该 record 只是把 Owner input 投影为 strict validator 可检查的 post-publish record。除非真实公开包源、clean consumer、日志 SHA256、host metadata 和 smoke 结果全部通过 `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`，否则仍不是 proof。
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish verification record written from owner input:"
Write-Host "  Json=$outputFullPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ProofClassification=$($record.postPublishProofClassification) IsProof=$($record.isPostPublishVerificationProof) CanCloseReleaseIssue=$($record.canCloseReleaseIssue)"
