[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\public-release-consumer\win-x64-trt10.11-cuda12.9-cudnn9.22\public-release-bridge-package-consumer.json",
  [string]$OutputDirectory,
  [string]$ExpectedSourceRuntimeKey,
  [string]$RepositoryRoot,
  [switch]$RequireReferencedFiles,
  [switch]$FailOnNotEvidence,
  [switch]$Strict
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
$RepositoryRoot = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Resolve-RecordedPath {
  param([AllowNull()][object]$Path)

  $text = [string]$Path
  if ([string]::IsNullOrWhiteSpace($text)) { return "" }
  if ([System.IO.Path]::IsPathRooted($text)) { return [System.IO.Path]::GetFullPath($text) }
  return [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $text))
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [Parameter(Mandatory = $true)][string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object -or $Object.PSObject.Properties.Name -notcontains $Name) {
    return $DefaultValue
  }
  return $Object.PSObject.Properties[$Name].Value
}

function Add-ValidationItem {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][bool]$Passed,
    [Parameter(Mandatory = $true)][ValidateSet("blocker", "evidence-required")][string]$Severity,
    [Parameter(Mandatory = $true)][string]$Detail
  )

  $script:Items.Add([pscustomobject]@{
      id = $Id
      passed = $Passed
      severity = $Severity
      detail = $Detail
    }) | Out-Null
}

function Test-Sha256 {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match '^[0-9a-fA-F]{64}$'
}

function Test-GitCommit {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match '^[0-9a-fA-F]{40}$'
}

function Test-PathOutsideRepository {
  param([AllowNull()][object]$Path)

  try {
    $resolved = Resolve-RecordedPath -Path $Path
    if ([string]::IsNullOrWhiteSpace($resolved)) { return $false }
    return -not ($resolved.Equals($RepositoryRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
      $resolved.StartsWith($RepositoryRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase))
  }
  catch {
    return $false
  }
}

function Test-StringArrayEqual {
  param(
    [AllowNull()][object[]]$Actual,
    [AllowNull()][object[]]$Expected
  )

  $actualValues = @($Actual | ForEach-Object { [string]$_ } | Sort-Object)
  $expectedValues = @($Expected | ForEach-Object { [string]$_ } | Sort-Object)
  return ($actualValues -join "`n") -ceq ($expectedValues -join "`n")
}

function Get-NuGetMetadata {
  param([Parameter(Mandatory = $true)][string]$Path)

  $archive = [System.IO.Compression.ZipFile]::OpenRead($Path)
  try {
    $nuspecEntries = @($archive.Entries | Where-Object {
        $_.FullName.EndsWith('.nuspec', [System.StringComparison]::OrdinalIgnoreCase)
      })
    if ($nuspecEntries.Count -ne 1) {
      throw "Expected exactly one nuspec in '$Path'; found $($nuspecEntries.Count)."
    }

    $reader = [System.IO.StreamReader]::new($nuspecEntries[0].Open())
    try { [xml]$nuspec = $reader.ReadToEnd() } finally { $reader.Dispose() }
    $metadataNode = $nuspec.SelectSingleNode("//*[local-name()='metadata']")
    $idNode = $nuspec.SelectSingleNode("//*[local-name()='metadata']/*[local-name()='id']")
    $versionNode = $nuspec.SelectSingleNode("//*[local-name()='metadata']/*[local-name()='version']")
    $repositoryNode = $nuspec.SelectSingleNode("//*[local-name()='metadata']/*[local-name()='repository']")
    if ($null -eq $metadataNode -or $null -eq $idNode -or $null -eq $versionNode -or $null -eq $repositoryNode) {
      throw "NuGet metadata is incomplete in '$Path'."
    }

    return [pscustomobject]@{
      id = [string]$idNode.InnerText
      version = [string]$versionNode.InnerText
      repositoryUrl = [string]$repositoryNode.GetAttribute('url')
      repositoryCommit = [string]$repositoryNode.GetAttribute('commit')
      nativeEntries = @($archive.Entries | Where-Object {
          $_.FullName -match '^runtimes/[^/]+/native/[^/]+$'
        } | ForEach-Object { [string]$_.FullName })
    }
  }
  finally {
    $archive.Dispose()
  }
}

$resolvedInputPath = Resolve-RecordedPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Public Release bridge consumer report not found: $resolvedInputPath"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Split-Path -Parent $resolvedInputPath
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot $OutputDirectory
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$script:Items = New-Object System.Collections.Generic.List[object]
$repository = [string](Get-PropertyOrDefault -Object $record -Name 'repository' -DefaultValue '')
$sourceRuntimeKey = [string](Get-PropertyOrDefault -Object $record -Name 'sourceRuntimeKey' -DefaultValue '')
$managed = Get-PropertyOrDefault -Object (Get-PropertyOrDefault -Object $record -Name 'releaseAssets' -DefaultValue $null) -Name 'managed' -DefaultValue $null
$bridge = Get-PropertyOrDefault -Object (Get-PropertyOrDefault -Object $record -Name 'releaseAssets' -DefaultValue $null) -Name 'bridge' -DefaultValue $null
$consumer = Get-PropertyOrDefault -Object $record -Name 'consumer' -DefaultValue $null

$manifestPath = Join-Path $RepositoryRoot 'pack\runtime-split\split-runtime-packages.manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$bridgeDefinitions = @($manifest.packages | Where-Object {
    [string]$_.sourceRuntimeKey -eq $sourceRuntimeKey -and [string]$_.role -eq 'bridge'
  })
$bridgeDefinition = if ($bridgeDefinitions.Count -eq 1) { $bridgeDefinitions[0] } else { $null }
$expectedRepositoryUrl = "https://github.com/$repository"
$expectedBridgeId = if ($null -eq $bridgeDefinition) { '' } else { [string]$bridgeDefinition.packageId }
$expectedBridgeNativeEntries = if ($null -eq $bridgeDefinition) {
  @()
}
else {
  @($bridgeDefinition.assets | ForEach-Object { "runtimes/$([string]$bridgeDefinition.rid)/native/$([string]$_)" })
}

Add-ValidationItem 'record-kind' ([string](Get-PropertyOrDefault -Object $record -Name 'recordKind' -DefaultValue '') -eq 'public-release-bridge-package-consumer') 'blocker' 'recordKind must identify the public Release bridge consumer report.'
Add-ValidationItem 'schema-version' ([int](Get-PropertyOrDefault -Object $record -Name 'schemaVersion' -DefaultValue 0) -eq 1) 'blocker' 'schemaVersion must be 1.'
Add-ValidationItem 'repository-format' ($repository -match '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') 'blocker' 'repository must use owner/name form.'
Add-ValidationItem 'source-runtime-key' (-not [string]::IsNullOrWhiteSpace($sourceRuntimeKey)) 'blocker' 'sourceRuntimeKey is required.'
Add-ValidationItem 'expected-source-runtime-key' ([string]::IsNullOrWhiteSpace($ExpectedSourceRuntimeKey) -or $sourceRuntimeKey.Equals($ExpectedSourceRuntimeKey, [System.StringComparison]::OrdinalIgnoreCase)) 'blocker' 'sourceRuntimeKey must match the requested runtime line.'
Add-ValidationItem 'bridge-definition' ($bridgeDefinitions.Count -eq 1) 'blocker' 'sourceRuntimeKey must resolve to exactly one bridge package definition.'
Add-ValidationItem 'bridge-only-channel' (
  [string](Get-PropertyOrDefault -Object $record -Name 'publicationPolicy' -DefaultValue '') -eq 'bridge-only' -and
  [string](Get-PropertyOrDefault -Object $record -Name 'channel' -DefaultValue '') -eq 'github-release-assets') 'blocker' 'The report must describe the GitHub Release managed plus bridge-only channel.'
Add-ValidationItem 'no-publish-or-close-side-effects' (
  -not [bool](Get-PropertyOrDefault -Object $record -Name 'performsPublish' -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name 'canPublishPublicly' -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name 'canCloseReleaseIssue' -DefaultValue $true)) 'blocker' 'Consumer evidence must not publish, approve publication, or close the release.'
Add-ValidationItem 'proof-boundaries' (
  -not [bool](Get-PropertyOrDefault -Object $record -Name 'isPackageConsumerRuntimeProof' -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name 'isPostPublishProof' -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name 'currentHeadBindingVerified' -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name 'canPromoteCurrentHeadPackageConsumerProof' -DefaultValue $true)) 'blocker' 'Public Release asset evidence cannot claim current-HEAD, package-consumer-runtime, or post-publish proof.'
Add-ValidationItem 'vendor-runtime-boundary' (
  -not [bool](Get-PropertyOrDefault -Object $record -Name 'vendorRuntimeBundled' -DefaultValue $true) -and
  [bool](Get-PropertyOrDefault -Object $record -Name 'systemInstalledVendorDependenciesRequired' -DefaultValue $false)) 'blocker' 'NVIDIA runtime libraries must remain machine-installed prerequisites.'
Add-ValidationItem 'package-reference-boundary' (
  [bool](Get-PropertyOrDefault -Object $record -Name 'restoreUsesDownloadedAssetStaging' -DefaultValue $false) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name 'stagingIsLocallyBuiltPackageFeed' -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name 'directNupkgReferenceUsed' -DefaultValue $true) -and
  [bool](Get-PropertyOrDefault -Object $record -Name 'packageReferenceOnly' -DefaultValue $false)) 'blocker' 'Restore must use verified downloaded staging through PackageReference only.'
Add-ValidationItem 'consumer-root-outside-repository' (
  [bool](Get-PropertyOrDefault -Object $record -Name 'consumerRootOutsideRepository' -DefaultValue $false) -and
  (Test-PathOutsideRepository -Path (Get-PropertyOrDefault -Object $consumer -Name 'outputRoot' -DefaultValue ''))) 'blocker' 'The clean consumer root must be outside the source repository.'

$assetFileEvidencePassed = $true
$packageMetadataEvidencePassed = $true
foreach ($assetCase in @(
    [pscustomobject]@{ Role = 'managed'; Asset = $managed; ExpectedId = 'JYPPX.TensorRT.CSharp.API'; ExpectedNativeEntries = @() },
    [pscustomobject]@{ Role = 'bridge'; Asset = $bridge; ExpectedId = $expectedBridgeId; ExpectedNativeEntries = @($expectedBridgeNativeEntries) }
  )) {
  $role = [string]$assetCase.Role
  $asset = $assetCase.Asset
  $expectedId = [string]$assetCase.ExpectedId
  $releaseTag = [string](Get-PropertyOrDefault -Object $asset -Name 'releaseTag' -DefaultValue '')
  $assetName = [string](Get-PropertyOrDefault -Object $asset -Name 'assetName' -DefaultValue '')
  $assetUrl = [string](Get-PropertyOrDefault -Object $asset -Name 'assetUrl' -DefaultValue '')
  $releaseUrl = [string](Get-PropertyOrDefault -Object $asset -Name 'releaseUrl' -DefaultValue '')
  $githubDigest = [string](Get-PropertyOrDefault -Object $asset -Name 'githubDigest' -DefaultValue '')
  $downloadedSha256 = [string](Get-PropertyOrDefault -Object $asset -Name 'downloadedSha256' -DefaultValue '')
  $downloadedPath = Resolve-RecordedPath -Path (Get-PropertyOrDefault -Object $asset -Name 'downloadedPath' -DefaultValue '')
  $packageId = [string](Get-PropertyOrDefault -Object $asset -Name 'packageId' -DefaultValue '')
  $packageVersion = [string](Get-PropertyOrDefault -Object $asset -Name 'packageVersion' -DefaultValue '')
  $repositoryUrl = [string](Get-PropertyOrDefault -Object $asset -Name 'repositoryUrl' -DefaultValue '')
  $repositoryCommit = [string](Get-PropertyOrDefault -Object $asset -Name 'repositoryCommit' -DefaultValue '')
  $lengthBytes = [long](Get-PropertyOrDefault -Object $asset -Name 'lengthBytes' -DefaultValue 0)
  $expectedAssetName = "$packageId.$packageVersion.nupkg"

  Add-ValidationItem "$role-package-identity" (
    -not [string]::IsNullOrWhiteSpace($expectedId) -and
    $packageId.Equals($expectedId, [System.StringComparison]::OrdinalIgnoreCase) -and
    -not [string]::IsNullOrWhiteSpace($packageVersion) -and
    $assetName.Equals($expectedAssetName, [System.StringComparison]::OrdinalIgnoreCase)) 'blocker' "$role package id, version, and asset name must match the manifest identity."
  Add-ValidationItem "$role-public-urls" (
    -not [string]::IsNullOrWhiteSpace($releaseTag) -and
    $releaseUrl.Equals("https://github.com/$repository/releases/tag/$releaseTag", [System.StringComparison]::OrdinalIgnoreCase) -and
    $assetUrl.Equals("https://github.com/$repository/releases/download/$releaseTag/$assetName", [System.StringComparison]::OrdinalIgnoreCase)) 'blocker' "$role package URLs must be immutable public GitHub Release URLs."
  Add-ValidationItem "$role-digest-record" (
    (Test-Sha256 -Value $downloadedSha256) -and
    $githubDigest -match '^sha256:[0-9a-fA-F]{64}$' -and
    $githubDigest.Substring('sha256:'.Length).Equals($downloadedSha256, [System.StringComparison]::OrdinalIgnoreCase) -and
    $lengthBytes -gt 0) 'blocker' "$role GitHub digest, downloaded SHA256, and length must be internally consistent."
  Add-ValidationItem "$role-repository-provenance" (
    $repositoryUrl.Equals($expectedRepositoryUrl, [System.StringComparison]::OrdinalIgnoreCase) -and
    (Test-GitCommit -Value $repositoryCommit)) 'blocker' "$role nuspec repository URL and commit must identify the formal repository."
  Add-ValidationItem "$role-recorded-native-assets" (
    Test-StringArrayEqual -Actual @(Get-PropertyOrDefault -Object $asset -Name 'nativeEntries' -DefaultValue @()) -Expected @($assetCase.ExpectedNativeEntries)) 'blocker' "$role recorded native entries must contain only the project-owned bridge asset set."
  Add-ValidationItem "$role-download-outside-repository" (Test-PathOutsideRepository -Path $downloadedPath) 'blocker' "$role downloaded nupkg staging path must be outside the source repository."

  $fileExists = -not [string]::IsNullOrWhiteSpace($downloadedPath) -and (Test-Path -LiteralPath $downloadedPath -PathType Leaf)
  $missingSeverity = if ($RequireReferencedFiles.IsPresent) { 'blocker' } else { 'evidence-required' }
  Add-ValidationItem "$role-downloaded-file-exists" $fileExists $missingSeverity "$role downloaded nupkg must be available for independent hash and nuspec verification."
  if (-not $fileExists) {
    $assetFileEvidencePassed = $false
    $packageMetadataEvidencePassed = $false
    continue
  }

  $file = Get-Item -LiteralPath $downloadedPath
  $actualSha256 = (Get-FileHash -LiteralPath $downloadedPath -Algorithm SHA256).Hash
  $fileMatch = [long]$file.Length -eq $lengthBytes -and $actualSha256.Equals($downloadedSha256, [System.StringComparison]::OrdinalIgnoreCase)
  Add-ValidationItem "$role-downloaded-file-integrity" $fileMatch 'blocker' "$role nupkg length and SHA256 must match the report and GitHub digest."
  $assetFileEvidencePassed = $assetFileEvidencePassed -and $fileMatch

  try {
    $metadata = Get-NuGetMetadata -Path $downloadedPath
    $metadataMatch = $metadata.id.Equals($packageId, [System.StringComparison]::OrdinalIgnoreCase) -and
      $metadata.version.Equals($packageVersion, [System.StringComparison]::OrdinalIgnoreCase) -and
      $metadata.repositoryUrl.Equals($repositoryUrl, [System.StringComparison]::OrdinalIgnoreCase) -and
      $metadata.repositoryCommit.Equals($repositoryCommit, [System.StringComparison]::OrdinalIgnoreCase) -and
      (Test-StringArrayEqual -Actual @($metadata.nativeEntries) -Expected @($assetCase.ExpectedNativeEntries))
    Add-ValidationItem "$role-nupkg-metadata" $metadataMatch 'blocker' "$role nupkg metadata must independently match id, version, repository provenance, and native entries."
    $packageMetadataEvidencePassed = $packageMetadataEvidencePassed -and $metadataMatch
  }
  catch {
    Add-ValidationItem "$role-nupkg-metadata" $false 'blocker' "$role nupkg metadata could not be verified: $($_.Exception.Message)"
    $packageMetadataEvidencePassed = $false
  }
}

$managedCommit = [string](Get-PropertyOrDefault -Object $managed -Name 'repositoryCommit' -DefaultValue '')
$bridgeCommit = [string](Get-PropertyOrDefault -Object $bridge -Name 'repositoryCommit' -DefaultValue '')
$sourceCommitsAligned = (Test-GitCommit -Value $managedCommit) -and $managedCommit.Equals($bridgeCommit, [System.StringComparison]::OrdinalIgnoreCase)
$reportedCommitAlignment = [bool](Get-PropertyOrDefault -Object $record -Name 'packageSourceCommitAligned' -DefaultValue $false)
$crossCommitOverride = [bool](Get-PropertyOrDefault -Object $record -Name 'crossCommitDiagnosticOverride' -DefaultValue $false)
Add-ValidationItem 'reported-commit-alignment' ($reportedCommitAlignment -eq $sourceCommitsAligned) 'blocker' 'Reported packageSourceCommitAligned must equal the independently computed nuspec commit comparison.'
Add-ValidationItem 'cross-commit-diagnostic-gate' (
  $sourceCommitsAligned -or $crossCommitOverride) 'blocker' 'Different managed and bridge commits require the explicit diagnostic-only override.'

$runtimeReportPath = Resolve-RecordedPath -Path (Get-PropertyOrDefault -Object $consumer -Name 'runtimeReportPath' -DefaultValue '')
$runtimeReportSha256 = [string](Get-PropertyOrDefault -Object $consumer -Name 'runtimeReportSha256' -DefaultValue '')
$runtimeReportExists = -not [string]::IsNullOrWhiteSpace($runtimeReportPath) -and (Test-Path -LiteralPath $runtimeReportPath -PathType Leaf)
$runtimeMissingSeverity = if ($RequireReferencedFiles.IsPresent) { 'blocker' } else { 'evidence-required' }
Add-ValidationItem 'runtime-report-sha256-format' (Test-Sha256 -Value $runtimeReportSha256) 'blocker' 'The outer record must pin the runtime consumer JSON SHA256.'
Add-ValidationItem 'runtime-report-exists' $runtimeReportExists $runtimeMissingSeverity 'The runtime consumer JSON must be available for independent verification.'

$runtimeReportIntegrityPassed = $false
$runtimeReportConsistent = $false
$runtimeSmokePassed = $false
$runtimeDiagnosticMode = $false
if ($runtimeReportExists -and (Test-Sha256 -Value $runtimeReportSha256)) {
  $actualRuntimeSha256 = (Get-FileHash -LiteralPath $runtimeReportPath -Algorithm SHA256).Hash
  $runtimeReportIntegrityPassed = $actualRuntimeSha256.Equals($runtimeReportSha256, [System.StringComparison]::OrdinalIgnoreCase)
  Add-ValidationItem 'runtime-report-hash-match' $runtimeReportIntegrityPassed 'blocker' 'Runtime consumer JSON SHA256 must match the outer report.'
  try {
    $runtime = Get-Content -LiteralPath $runtimeReportPath -Raw -Encoding utf8 | ConvertFrom-Json
    $runtimeSmokePassed = [bool](Get-PropertyOrDefault -Object $runtime -Name 'isRuntimeExecutionProof' -DefaultValue $false) -and
      [int](Get-PropertyOrDefault -Object $runtime -Name 'exitCode' -DefaultValue -1) -eq 0
    $runtimeDiagnosticMode = [bool](Get-PropertyOrDefault -Object $runtime -Name 'installedVendorAssetHashingSkipped' -DefaultValue $false) -or
      [bool](Get-PropertyOrDefault -Object $runtime -Name 'installedVendorAssetInventorySkipped' -DefaultValue $false) -or
      -not [bool](Get-PropertyOrDefault -Object $runtime -Name 'nativeAssetHashesComplete' -DefaultValue $false)
    $runtimePackages = Get-PropertyOrDefault -Object $runtime -Name 'packages' -DefaultValue $null
    $runtimeManaged = Get-PropertyOrDefault -Object $runtimePackages -Name 'managed' -DefaultValue $null
    $runtimeBridge = Get-PropertyOrDefault -Object $runtimePackages -Name 'bridge' -DefaultValue $null
    $runtimeReportConsistent = [string](Get-PropertyOrDefault -Object $runtime -Name 'sourceRuntimeKey' -DefaultValue '') -eq $sourceRuntimeKey -and
      [string](Get-PropertyOrDefault -Object $runtimeManaged -Name 'id' -DefaultValue '') -eq [string](Get-PropertyOrDefault -Object $managed -Name 'packageId' -DefaultValue '') -and
      [string](Get-PropertyOrDefault -Object $runtimeManaged -Name 'version' -DefaultValue '') -eq [string](Get-PropertyOrDefault -Object $managed -Name 'packageVersion' -DefaultValue '') -and
      [string](Get-PropertyOrDefault -Object $runtimeBridge -Name 'id' -DefaultValue '') -eq [string](Get-PropertyOrDefault -Object $bridge -Name 'packageId' -DefaultValue '') -and
      [string](Get-PropertyOrDefault -Object $runtimeBridge -Name 'version' -DefaultValue '') -eq [string](Get-PropertyOrDefault -Object $bridge -Name 'packageVersion' -DefaultValue '') -and
      [string](Get-PropertyOrDefault -Object $runtimeManaged -Name 'sha256' -DefaultValue '').Equals([string](Get-PropertyOrDefault -Object $managed -Name 'downloadedSha256' -DefaultValue ''), [System.StringComparison]::OrdinalIgnoreCase) -and
      [string](Get-PropertyOrDefault -Object $runtimeBridge -Name 'sha256' -DefaultValue '').Equals([string](Get-PropertyOrDefault -Object $bridge -Name 'downloadedSha256' -DefaultValue ''), [System.StringComparison]::OrdinalIgnoreCase) -and
      -not [bool](Get-PropertyOrDefault -Object $runtime -Name 'isPackageConsumerRuntimeProof' -DefaultValue $true) -and
      [bool](Get-PropertyOrDefault -Object $record -Name 'runtimeSmokePassed' -DefaultValue $false) -eq $runtimeSmokePassed -and
      [bool](Get-PropertyOrDefault -Object $record -Name 'isRuntimeExecutionProof' -DefaultValue $false) -eq $runtimeSmokePassed
    Add-ValidationItem 'runtime-report-consistency' $runtimeReportConsistent 'blocker' 'Runtime key, package identities, hashes, smoke result, and proof boundary must match the outer report.'
  }
  catch {
    Add-ValidationItem 'runtime-report-consistency' $false 'blocker' "Runtime consumer JSON could not be validated: $($_.Exception.Message)"
  }
}
else {
  Add-ValidationItem 'runtime-report-hash-match' $false $runtimeMissingSeverity 'Runtime consumer JSON SHA256 could not be verified.'
  Add-ValidationItem 'runtime-report-consistency' $false $runtimeMissingSeverity 'Runtime consumer JSON could not be compared with the outer report.'
}

$logEvidencePassed = $true
foreach ($logCase in @(
    [pscustomobject]@{ Id = 'stdout'; PathName = 'invocationStdoutPath'; HashName = 'invocationStdoutSha256' },
    [pscustomobject]@{ Id = 'stderr'; PathName = 'invocationStderrPath'; HashName = 'invocationStderrSha256' }
  )) {
  $logPath = Resolve-RecordedPath -Path (Get-PropertyOrDefault -Object $consumer -Name $logCase.PathName -DefaultValue '')
  $logSha256 = [string](Get-PropertyOrDefault -Object $consumer -Name $logCase.HashName -DefaultValue '')
  $logExists = -not [string]::IsNullOrWhiteSpace($logPath) -and (Test-Path -LiteralPath $logPath -PathType Leaf)
  $logPassed = $logExists -and (Test-Sha256 -Value $logSha256) -and
    (Get-FileHash -LiteralPath $logPath -Algorithm SHA256).Hash.Equals($logSha256, [System.StringComparison]::OrdinalIgnoreCase)
  Add-ValidationItem "$($logCase.Id)-log-hash-match" $logPassed $runtimeMissingSeverity "$($logCase.Id) invocation log must exist and match its recorded SHA256."
  $logEvidencePassed = $logEvidencePassed -and $logPassed
}

$policyGatePassed = $false
$managedPath = Resolve-RecordedPath -Path (Get-PropertyOrDefault -Object $managed -Name 'downloadedPath' -DefaultValue '')
$bridgePath = Resolve-RecordedPath -Path (Get-PropertyOrDefault -Object $bridge -Name 'downloadedPath' -DefaultValue '')
if ((Test-Path -LiteralPath $managedPath -PathType Leaf) -and (Test-Path -LiteralPath $bridgePath -PathType Leaf)) {
  try {
    $policyOutput = @(& (Join-Path $RepositoryRoot 'eng\Test-ExternalVendorRuntimePackagePolicy.ps1') -PackagePath @($managedPath, $bridgePath) -RepositoryRoot $RepositoryRoot 2>&1)
    $policyGatePassed = $true
  }
  catch {
    $policyOutput = @($_.Exception.Message)
  }
}
else {
  $policyOutput = @('Downloaded package files were unavailable for the independent bridge-only policy gate.')
}
Add-ValidationItem 'independent-bridge-only-policy' $policyGatePassed $runtimeMissingSeverity 'The downloaded package pair must independently pass the external vendor runtime policy gate.'

$reportedClassification = [string](Get-PropertyOrDefault -Object $record -Name 'proofClassification' -DefaultValue '')
$reportedPublicEvidence = [bool](Get-PropertyOrDefault -Object $record -Name 'isPublicReleaseAssetConsumerEvidence' -DefaultValue $false)
$reportedProvenance = [bool](Get-PropertyOrDefault -Object $record -Name 'publicReleaseAssetProvenanceVerified' -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $record -Name 'remoteDigestVerified' -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $record -Name 'packageIdentityVerified' -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $record -Name 'externalVendorRuntimePackagePolicyPassed' -DefaultValue $false)
$expectedClassification = if ($sourceCommitsAligned -and $runtimeSmokePassed -and -not $runtimeDiagnosticMode) {
  'public-release-assets-compatible-host-runtime'
}
elseif (-not $sourceCommitsAligned) {
  'cross-commit-public-assets-diagnostic-only'
}
else {
  'public-release-assets-runtime-failed'
}
Add-ValidationItem 'classification-consistency' ($reportedClassification -eq $expectedClassification) 'blocker' 'proofClassification must follow same-commit, diagnostic, and runtime-smoke boundaries.'
Add-ValidationItem 'diagnostic-mode-boundary' (
  $sourceCommitsAligned -or (
    $crossCommitOverride -and
    $runtimeDiagnosticMode -and
    -not $reportedPublicEvidence -and
    -not [bool](Get-PropertyOrDefault -Object $record -Name 'canPromoteCurrentHeadPackageConsumerProof' -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $record -Name 'isPackageConsumerRuntimeProof' -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $record -Name 'isPostPublishProof' -DefaultValue $true))) 'blocker' 'Cross-commit execution must remain explicitly diagnostic and non-promotable.'

$failedBlockersBeforeClaim = @($script:Items | Where-Object { -not $_.passed -and $_.severity -eq 'blocker' })
$referencedFilesFullyVerified = $assetFileEvidencePassed -and $packageMetadataEvidencePassed -and
  $runtimeReportIntegrityPassed -and $runtimeReportConsistent -and $logEvidencePassed -and $policyGatePassed
$computedPublicEvidence = $failedBlockersBeforeClaim.Count -eq 0 -and
  $referencedFilesFullyVerified -and
  $reportedProvenance -and
  $sourceCommitsAligned -and
  -not $crossCommitOverride -and
  $runtimeSmokePassed -and
  -not $runtimeDiagnosticMode
Add-ValidationItem 'public-evidence-claim-consistency' ($reportedPublicEvidence -eq $computedPublicEvidence) 'blocker' 'isPublicReleaseAssetConsumerEvidence must equal the independently validated result.'

$boundary = [string](Get-PropertyOrDefault -Object $record -Name 'boundary' -DefaultValue '')
Add-ValidationItem 'written-boundary' (
  $boundary.IndexOf('post-publish', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
  (($sourceCommitsAligned -and $boundary.IndexOf('current HEAD', [System.StringComparison]::OrdinalIgnoreCase) -ge 0) -or
    (-not $sourceCommitsAligned -and $boundary.IndexOf('diagnostic-only', [System.StringComparison]::OrdinalIgnoreCase) -ge 0))) 'blocker' 'The written boundary must name post-publish exclusion and the applicable current-HEAD or diagnostic-only limit.'

$failedBlockers = @($script:Items | Where-Object { -not $_.passed -and $_.severity -eq 'blocker' })
$failedEvidenceItems = @($script:Items | Where-Object { -not $_.passed -and $_.severity -eq 'evidence-required' })
$canPromotePublicReleaseAssetConsumerEvidence = $failedBlockers.Count -eq 0 -and
  $failedEvidenceItems.Count -eq 0 -and $computedPublicEvidence
$diagnosticOnly = -not $sourceCommitsAligned -and $failedBlockers.Count -eq 0
$validationState = if ($canPromotePublicReleaseAssetConsumerEvidence) {
  'verified-public-release-assets-compatible-host-runtime'
}
elseif ($diagnosticOnly) {
  'verified-cross-commit-diagnostic-only'
}
elseif ($failedBlockers.Count -gt 0) {
  'invalid-public-release-bridge-consumer-report'
}
else {
  'verified-provenance-runtime-not-evidence'
}

$validation = [pscustomobject]@{
  schemaVersion = 1
  recordKind = 'public-release-bridge-package-consumer-validation'
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
  inputPath = $resolvedInputPath
  inputSha256 = (Get-FileHash -LiteralPath $resolvedInputPath -Algorithm SHA256).Hash.ToLowerInvariant()
  validationState = $validationState
  repository = $repository
  sourceRuntimeKey = $sourceRuntimeKey
  managedRepositoryCommit = $managedCommit
  bridgeRepositoryCommit = $bridgeCommit
  sourceCommitsAligned = $sourceCommitsAligned
  diagnosticOnly = $diagnosticOnly
  referencedFilesRequired = $RequireReferencedFiles.IsPresent
  referencedFilesFullyVerified = $referencedFilesFullyVerified
  runtimeReportSha256Verified = $runtimeReportIntegrityPassed
  invocationLogHashesVerified = $logEvidencePassed
  bridgeOnlyPolicyPassed = $policyGatePassed
  canPromotePublicReleaseAssetConsumerEvidence = $canPromotePublicReleaseAssetConsumerEvidence
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  currentHeadBindingVerified = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  failedBlockerCount = $failedBlockers.Count
  failedEvidenceItemCount = $failedEvidenceItems.Count
  failOnNotEvidenceRequested = $FailOnNotEvidence.IsPresent
  validationItems = @($script:Items.ToArray())
  policyValidationOutput = @($policyOutput | ForEach-Object { [string]$_ })
  boundary = 'This validator independently checks public Release asset hashes, package provenance, same-commit pairing, runtime-report and log hashes, bridge-only policy, and diagnostic non-promotion. It does not create package-consumer-runtime proof, bind packages to current HEAD, publish, or establish post-publish proof.'
}

$jsonPath = Join-Path $OutputDirectory 'public-release-bridge-package-consumer-validation.json'
$markdownPath = Join-Path $OutputDirectory 'public-release-bridge-package-consumer-validation.md'
$validation | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$rows = @($validation.validationItems | ForEach-Object {
    "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $(([string]$_.detail).Replace('|', '\|')) |"
  })
$markdown = @"
# Public Release Bridge Package Consumer Validation

| Field | Value |
| --- | --- |
| State | ``$validationState`` |
| Runtime key | ``$sourceRuntimeKey`` |
| Same source commit | ``$sourceCommitsAligned`` |
| Diagnostic only | ``$diagnosticOnly`` |
| Referenced files verified | ``$referencedFilesFullyVerified`` |
| Public Release asset consumer evidence | ``$canPromotePublicReleaseAssetConsumerEvidence`` |
| Package-consumer-runtime proof | ``False`` |
| Post-publish proof | ``False`` |

## Checks

| ID | Passed | Severity | Detail |
| --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@
Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Host "Public Release bridge consumer validation written: $jsonPath"
Write-Host "Public Release bridge consumer validation written: $markdownPath"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Public Release bridge consumer validation failed with $($failedBlockers.Count) blocker(s)."
}
if ($FailOnNotEvidence.IsPresent -and -not $canPromotePublicReleaseAssetConsumerEvidence) {
  throw "Public Release bridge consumer report is not promotable public asset evidence. State=$validationState Blockers=$($failedBlockers.Count) EvidenceItems=$($failedEvidenceItems.Count)."
}
