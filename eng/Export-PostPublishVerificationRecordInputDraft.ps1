[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-verification-record-template.json",
  [string]$RepositoryRoot,
  [string]$CleanConsumerProjectScanPath = "artifacts\final-release\post-publish-clean-consumer-project-scan.json",
  [string]$OutputPath = "artifacts\final-release\post-publish-verification-record.input-draft.json",
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$SelectedChannel = "",
  [string]$ChannelSourceUri = "",
  [string]$CleanConsumerRoot = "",
  [string]$ConsumerProjectName = "",
  [string]$ConsumerProjectPath = "",
  [string]$ManagedPackageId = "",
  [string]$ManagedPackageVersion = "",
  [string]$RuntimePackageId = "",
  [string]$RuntimePackageVersion = "",
  [string]$ManagedPackageUrl = "",
  [string]$RuntimePackageUrl = "",
  [string]$ManagedPackageSha256Source = "",
  [string]$RuntimePackageSha256Source = "",
  [string]$ManagedPackageDownloadTimestampUtc = "",
  [string]$RuntimePackageDownloadTimestampUtc = "",
  [string]$RestoreLogPath = "",
  [string]$NativeAssetListingPath = "",
  [string]$DependencyProbeLogPath = "",
  [string]$SmokeLogPath = "",
  [string]$RestoreCommand = "",
  [string]$BuildCommand = "",
  [string]$SmokeCommand = "",
  [string]$ManagedPackageSource = "",
  [string]$RuntimePackageSource = ""
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

  if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
  if ([IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-Sha256OrEmpty {
  param([string]$Path)

  $resolvedPath = Resolve-RepositoryPath -Path $Path
  if ([string]::IsNullOrWhiteSpace($resolvedPath) -or -not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return ""
  }

  $stream = [System.IO.File]::OpenRead($resolvedPath)
  try {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
      return -join ($sha.ComputeHash($stream) | ForEach-Object { $_.ToString("x2") })
    }
    finally {
      $sha.Dispose()
    }
  }
  finally {
    $stream.Dispose()
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Post-publish verification template '$InputPath' was not found."
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$scanPath = Resolve-RepositoryPath -Path $CleanConsumerProjectScanPath
$scan = $null
$scanExists = Test-Path -LiteralPath $scanPath -PathType Leaf
if ($scanExists) {
  $scan = Get-Content -LiteralPath $scanPath -Raw -Encoding utf8 | ConvertFrom-Json
}

$restoreLogSha256 = Get-Sha256OrEmpty -Path $RestoreLogPath
$nativeAssetListingSha256 = Get-Sha256OrEmpty -Path $NativeAssetListingPath
$dependencyProbeLogSha256 = Get-Sha256OrEmpty -Path $DependencyProbeLogPath
$smokeLogSha256 = Get-Sha256OrEmpty -Path $SmokeLogPath

$scanPassed = $scanExists -and [bool]$scan.scanPassed
$scanConsumerProjectPath = if ($scanExists) { [string]$scan.resolvedProjectPath } else { "" }
$scanConsumerProjectName = if ($scanExists -and -not [string]::IsNullOrWhiteSpace($scan.resolvedProjectPath)) { [System.IO.Path]::GetFileNameWithoutExtension([string]$scan.resolvedProjectPath) } else { "" }
$scanHasLocalPackageSource = if ($scanExists -and $scan.PSObject.Properties.Name -contains "hasLocalPackageSource") { [bool]$scan.hasLocalPackageSource } else { $null }
$scanHasLocalNupkgPackageReference = if ($scanExists -and $scan.PSObject.Properties.Name -contains "hasLocalNupkgPackageReference") { [bool]$scan.hasLocalNupkgPackageReference } else { $null }
$scanLocalPackageSourceCount = if ($scanExists -and $scan.PSObject.Properties.Name -contains "localPackageSourceCount") { [int]$scan.localPackageSourceCount } else { $null }
$scanLocalNupkgPackageReferenceCount = if ($scanExists -and $scan.PSObject.Properties.Name -contains "localNupkgPackageReferenceCount") { [int]$scan.localNupkgPackageReferenceCount } else { $null }

if ([string]::IsNullOrWhiteSpace($ConsumerProjectPath) -and -not [string]::IsNullOrWhiteSpace($scanConsumerProjectPath)) { $ConsumerProjectPath = $scanConsumerProjectPath }
if ([string]::IsNullOrWhiteSpace($ConsumerProjectName) -and -not [string]::IsNullOrWhiteSpace($scanConsumerProjectName)) { $ConsumerProjectName = $scanConsumerProjectName }
if ([string]::IsNullOrWhiteSpace($CleanConsumerRoot) -and -not [string]::IsNullOrWhiteSpace($ConsumerProjectPath)) { $CleanConsumerRoot = [System.IO.Path]::GetDirectoryName($ConsumerProjectPath) }

$record.recordKind = "post-publish-verification-record-input-draft"
$record.templateOnly = $false
$record.verificationState = "owner-action-required"
$record.postPublishProofClassification = "owner-action-required"
$record.selectedChannel = $SelectedChannel
$record.channelSourceUri = $ChannelSourceUri
$record.cleanConsumerRoot = $CleanConsumerRoot
$record.consumerProjectName = $ConsumerProjectName
$record.consumerProjectPath = $ConsumerProjectPath
$record.restoreCommand = $RestoreCommand
$record.buildCommand = $BuildCommand
$record.smokeCommand = $SmokeCommand
$record.restoreLogPath = $RestoreLogPath
$record.restoreLogSha256 = $restoreLogSha256
$record.nativeAssetListingPath = $NativeAssetListingPath
$record.nativeAssetListingSha256 = $nativeAssetListingSha256
$record.dependencyProbeLogPath = $DependencyProbeLogPath
$record.dependencyProbeLogSha256 = $dependencyProbeLogSha256
$record.smokeLogPath = $SmokeLogPath
$record.smokeLogSha256 = $smokeLogSha256
$record.managedPackageSource = $ManagedPackageSource
$record.runtimePackageSource = $RuntimePackageSource
$record.expectedRuntimePackageKey = $RuntimePackageKey
$record.noProjectReference = if ($scanExists) { -not [bool]$scan.projectReferenceCount } else { $null }
$record.nativeAssetsCopied = $null
$record.dependencyProbePassed = $null
$record.runtimeSmokePassed = $false
$record.runtimeSmokeExitCode = $null
$record.smokeStatus = "owner-action-required"
$record.performsPublish = $false
$record.isPostPublishVerificationProof = $false
$record.canCloseReleaseIssue = $false
$record.packageIdentity.managedPackageId = $ManagedPackageId
$record.packageIdentity.managedPackageVersion = $ManagedPackageVersion
$record.packageIdentity.runtimePackageId = $RuntimePackageId
$record.packageIdentity.runtimePackageVersion = $RuntimePackageVersion
$record.packageIdentity.managedPackageUrl = $ManagedPackageUrl
$record.packageIdentity.runtimePackageUrl = $RuntimePackageUrl
$record.packageIdentity.managedNupkgSha256 = ""
$record.packageIdentity.runtimeNupkgSha256 = ""
$record.packageIdentity.managedPackageSha256Source = $ManagedPackageSha256Source
$record.packageIdentity.runtimePackageSha256Source = $RuntimePackageSha256Source
$record.packageIdentity.managedPackageDownloadTimestampUtc = $ManagedPackageDownloadTimestampUtc
$record.packageIdentity.runtimePackageDownloadTimestampUtc = $RuntimePackageDownloadTimestampUtc
$record | Add-Member -NotePropertyName noLocalPackageSource -NotePropertyValue $(if ($scanExists -and $null -ne $scanHasLocalPackageSource) { -not $scanHasLocalPackageSource } else { $null }) -Force
$record | Add-Member -NotePropertyName noLocalNupkgPackageReference -NotePropertyValue $(if ($scanExists -and $null -ne $scanHasLocalNupkgPackageReference) { -not $scanHasLocalNupkgPackageReference } else { $null }) -Force
$record | Add-Member -NotePropertyName localPackageSourceCount -NotePropertyValue $scanLocalPackageSourceCount -Force
$record | Add-Member -NotePropertyName localNupkgPackageReferenceCount -NotePropertyValue $scanLocalNupkgPackageReferenceCount -Force
$record | Add-Member -NotePropertyName cleanConsumerProjectScanPath -NotePropertyValue $CleanConsumerProjectScanPath -Force
$record | Add-Member -NotePropertyName cleanConsumerProjectScanPassed -NotePropertyValue $scanPassed -Force
$record | Add-Member -NotePropertyName cleanConsumerProjectScanHasLocalPackageSource -NotePropertyValue $scanHasLocalPackageSource -Force
$record | Add-Member -NotePropertyName cleanConsumerProjectScanHasLocalNupkgPackageReference -NotePropertyValue $scanHasLocalNupkgPackageReference -Force
$record | Add-Member -NotePropertyName inputDraftOnly -NotePropertyValue $true -Force
$record | Add-Member -NotePropertyName ownerActionRequired -NotePropertyValue $true -Force

$outputFullPath = Resolve-RepositoryPath -Path $OutputPath
$outputRoot = [System.IO.Path]::GetDirectoryName($outputFullPath)
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $outputFullPath -Encoding utf8

$markdownPath = [System.IO.Path]::ChangeExtension($outputFullPath, ".md")
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Post-Publish Verification Record Input Draft")
$lines.Add("")
$lines.Add("- record kind: ``post-publish-verification-record-input-draft``")
$lines.Add("- input draft only: ``True``")
$lines.Add("- post-publish verification proof: ``False``")
$lines.Add("- can close release issue: ``False``")
$lines.Add("- clean consumer scan path: ``$CleanConsumerProjectScanPath``")
$lines.Add("- clean consumer scan passed: ``$scanPassed``")
$lines.Add("- clean consumer scan has local package source: ``$scanHasLocalPackageSource``")
$lines.Add("- clean consumer scan has local .nupkg reference: ``$scanHasLocalNupkgPackageReference``")
$lines.Add("- restore log SHA256: ``$restoreLogSha256``")
$lines.Add("- native asset listing SHA256: ``$nativeAssetListingSha256``")
$lines.Add("- dependency probe log SHA256: ``$dependencyProbeLogSha256``")
$lines.Add("- smoke log SHA256: ``$smokeLogSha256``")
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add("- This draft may compute log SHA256 values, but it is not proof.")
$lines.Add("- Local package sources, local feeds, repository artifact folders, and direct .nupkg references keep the draft non-proof.")
$lines.Add("- Owner must fill real package URLs, package SHA256 values, host metadata, stdout/stderr summaries, smoke result, owner/reviewer, and published version.")
$lines.Add("- Run Test-PostPublishVerificationRecord.ps1 with -RequireExistingLog -FailOnNotProof before release issue close readiness can change.")
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish verification record input draft written to $outputFullPath"
Write-Host "Post-publish verification record input draft written to $markdownPath"
Write-Host "InputDraftOnly=True IsPostPublishVerificationProof=False CanCloseReleaseIssue=False CleanConsumerProjectScanPassed=$scanPassed"
