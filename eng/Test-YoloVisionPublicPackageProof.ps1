[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$InputPath,
  [string]$RepositoryRoot,
  [string]$ExpectedRuntimePackageKey,
  [string]$ExpectedPackageVersion = "4.0.0",
  [string]$ExpectedHandoffPath,
  [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if ([string]::IsNullOrWhiteSpace($ExpectedHandoffPath)) {
  $ExpectedHandoffPath = Join-Path $RepositoryRoot "artifacts\interface-coverage\yolovision-public-package-owner-handoff.json"
}
$ExpectedHandoffPath = [IO.Path]::GetFullPath($ExpectedHandoffPath)

$InputPath = [IO.Path]::GetFullPath($InputPath)
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  throw "Public package proof input does not exist: $InputPath"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path (Split-Path -Parent $InputPath) "yolovision-public-package-proof-validation.json"
}
$OutputPath = [IO.Path]::GetFullPath($OutputPath)

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$failures = [Collections.Generic.List[object]]::new()
function Require-Condition {
  param([bool]$Passed, [string]$Id, [string]$Detail)
  if (-not $Passed) {
    $failures.Add([pscustomobject]@{ id = $Id; detail = $Detail })
  }
}
function Get-PropertyValue {
  param([object]$Object, [string]$Name, [object]$DefaultValue = $null)
  if ($null -eq $Object -or $null -eq $Object.PSObject.Properties[$Name]) { return $DefaultValue }
  return $Object.$Name
}
function Get-ExpectedPackageUrl {
  param([string]$PackageId, [string]$Version)
  $flatContainerUrl = "https://api.nuget.org/v3-flatcontainer"
  return "$flatContainerUrl/$($PackageId.ToLowerInvariant())/$Version/$($PackageId.ToLowerInvariant()).$Version.nupkg"
}

$feedUrl = [string](Get-PropertyValue $record "publicFeedUrl" "")
Require-Condition ([string](Get-PropertyValue $record "evidenceClassification" "") -eq "public-package-consumer-runtime") "classification" "evidenceClassification must be public-package-consumer-runtime."
Require-Condition ($feedUrl -eq "https://api.nuget.org/v3/index.json") "public-feed" "publicFeedUrl must be the NuGet v3 service index."
Require-Condition ([string](Get-PropertyValue $record "packageSourceMode" "") -eq "public-feed-only") "source-mode" "packageSourceMode must be public-feed-only."

$runtimePackageKey = [string](Get-PropertyValue $record "runtimePackageKey" "")
if (-not [string]::IsNullOrWhiteSpace($ExpectedRuntimePackageKey)) {
  Require-Condition ($runtimePackageKey -eq $ExpectedRuntimePackageKey) "runtime-key" "runtimePackageKey does not match the requested key."
}
$version = [string](Get-PropertyValue $record "packageVersion" "")
Require-Condition ($version -eq $ExpectedPackageVersion) "package-version" "packageVersion must match the expected version."

$packages = @((Get-PropertyValue $record "packages" @()))
Require-Condition ($packages.Count -eq 3) "package-count" "Exactly managed API, YoloVision, and selected bridge packages are required."
$handoff = if (Test-Path -LiteralPath $ExpectedHandoffPath -PathType Leaf) {
  Get-Content -LiteralPath $ExpectedHandoffPath -Raw -Encoding utf8 | ConvertFrom-Json
} else {
  $null
}
Require-Condition ($null -ne $handoff) "expected-handoff" "Expected owner handoff is required for package hash comparison."
$splitManifest = Get-Content -LiteralPath (Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json") -Raw -Encoding utf8 | ConvertFrom-Json
$bridgeEntries = @($splitManifest.packages | Where-Object {
  [string]::Equals([string]$_.sourceRuntimeKey, $runtimePackageKey, [StringComparison]::OrdinalIgnoreCase) -and
  [string]::Equals([string]$_.role, "bridge", [StringComparison]::OrdinalIgnoreCase)
})
Require-Condition ($bridgeEntries.Count -eq 1) "bridge-manifest" "runtimePackageKey must resolve to exactly one bridge package manifest entry."
$expectedBridgeId = if ($bridgeEntries.Count -eq 1) { [string]$bridgeEntries[0].packageId } else { "" }
$expectedIds = @("JYPPX.TensorRT.CSharp.API", "JYPPX.TensorRT.CSharp.API.YoloVision", $expectedBridgeId)
foreach ($expectedId in $expectedIds) {
  if ([string]::IsNullOrWhiteSpace($expectedId)) { continue }
  $matches = @($packages | Where-Object { [string](Get-PropertyValue $_ "id" "") -eq $expectedId -and [string](Get-PropertyValue $_ "version" "") -eq $version })
  Require-Condition ($matches.Count -eq 1) "package-$($expectedId.ToLowerInvariant())" "Expected package identity is missing or duplicated: $expectedId."
}
foreach ($package in $packages) {
  $packageId = [string](Get-PropertyValue $package "id" "missing-package-id")
  $packageVersion = [string](Get-PropertyValue $package "version" "")
  $packageSha256 = [string](Get-PropertyValue $package "sha256" "")
  $expectedUrl = Get-ExpectedPackageUrl -PackageId $packageId -Version $packageVersion
  Require-Condition ([string](Get-PropertyValue $package "sourceUrl" "") -eq $expectedUrl) "package-url-$packageId" "Package source URL does not match the exact public flat-container URL."
  Require-Condition ([bool](Get-PropertyValue $package "downloadedFromPublicFeed" $false)) "package-download-$packageId" "Package must be marked downloadedFromPublicFeed=true."
  Require-Condition ($packageSha256 -match '^[0-9a-fA-F]{64}$') "package-sha-$packageId" "Package SHA256 must be a 64-hex digest."
  if ($null -ne $handoff) {
    $expectedPackages = @($handoff.packages | Where-Object { $_.id -eq $packageId -and $_.version -eq $packageVersion })
    Require-Condition ($expectedPackages.Count -eq 1 -and [string]$expectedPackages[0].localSha256 -ieq $packageSha256) "package-handoff-hash-$packageId" "Downloaded package hash must match the frozen owner handoff hash."
  }
  $downloadedPath = [string](Get-PropertyValue $package "downloadedNupkgPath" "")
  $downloadedPathExists = -not [string]::IsNullOrWhiteSpace($downloadedPath) -and (Test-Path -LiteralPath $downloadedPath -PathType Leaf)
  Require-Condition $downloadedPathExists "package-file-$packageId" "Downloaded nupkg path must exist for hash verification."
  if ($downloadedPathExists) {
    $actualHash = (Get-FileHash -LiteralPath $downloadedPath -Algorithm SHA256).Hash
    Require-Condition ($actualHash -ieq $packageSha256) "package-hash-$packageId" "Downloaded nupkg SHA256 does not match the supplied digest."
  }
  $metadataPath = [string](Get-PropertyValue $package "nugetMetadataPath" "")
  $metadataPathExists = -not [string]::IsNullOrWhiteSpace($metadataPath) -and (Test-Path -LiteralPath $metadataPath -PathType Leaf)
  Require-Condition $metadataPathExists "package-metadata-$packageId" "Copied NuGet .nupkg.metadata evidence must exist."
  if ($metadataPathExists) {
    $metadata = Get-Content -LiteralPath $metadataPath -Raw -Encoding utf8 | ConvertFrom-Json
    Require-Condition ([string](Get-PropertyValue $metadata "source" "") -eq $feedUrl) "package-metadata-source-$packageId" "NuGet metadata source must match the public feed."
    $metadataHash = (Get-FileHash -LiteralPath $metadataPath -Algorithm SHA256).Hash
    Require-Condition ($metadataHash -ieq [string](Get-PropertyValue $package "nugetMetadataSha256" "")) "package-metadata-hash-$packageId" "NuGet metadata evidence hash does not match the record."
  }
}

$consumer = Get-PropertyValue $record "consumer" $null
Require-Condition ([int](Get-PropertyValue $consumer "projectReferenceCount" -1) -eq 0) "project-reference" "Consumer must have zero ProjectReference entries."
Require-Condition ([int](Get-PropertyValue $consumer "restoredProjectLibraryCount" -1) -eq 0) "restored-project-library" "Restore graph must contain zero project libraries."
Require-Condition ([string](Get-PropertyValue $consumer "workspaceDrive" "C:") -ne "C:") "workspace-drive" "External consumer workspace must not use C:."
Require-Condition ([bool](Get-PropertyValue $consumer "workspaceRemovedAfterValidation" $false)) "workspace-cleanup" "Consumer workspace must be removed after validation."
Require-Condition ([string](Get-PropertyValue $consumer "packageSources" "") -eq "public-feed-only") "consumer-source-mode" "Consumer record must state public-feed-only sources."

$runtime = Get-PropertyValue $record "runtime" $null
Require-Condition ([bool](Get-PropertyValue $runtime "passed" $false)) "runtime-passed" "Runtime row must explicitly pass."
Require-Condition ([string](Get-PropertyValue $runtime "passedMarker" "") -eq "YoloVision Passed=True") "runtime-marker" "YoloVision runtime marker is missing."
Require-Condition ([string](Get-PropertyValue $runtime "packageConsumerMarker" "") -eq "YoloVisionPackageConsumer ProjectReference=False") "consumer-marker" "Package consumer marker is missing."
Require-Condition ([int](Get-PropertyValue $runtime "predictionCount" 0) -gt 0) "predictions" "Runtime must produce at least one prediction."

$boundary = Get-PropertyValue $record "boundary" $null
Require-Condition ([bool](Get-PropertyValue $boundary "isPackageConsumerRuntimeProof" $false) -and
  [bool](Get-PropertyValue $boundary "packagesDownloadedFromPublicFeed" $false)) "proof-boundary" "Public package consumer proof flags must both be true."
Require-Condition (-not [bool](Get-PropertyValue $boundary "performsPublish" $true)) "publish-side-effect" "Proof validator rejects records that perform publishing."
Require-Condition (-not [bool](Get-PropertyValue $boundary "canCloseReleaseIssue" $true)) "release-close-side-effect" "Proof validator rejects release-close claims."

$validation = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-public-package-proof-validation"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  inputPath = $InputPath
  validationState = if ($failures.Count -eq 0) { "passed-public-package-consumer-proof" } else { "blocked-public-package-consumer-proof" }
  passed = $failures.Count -eq 0
  failureCount = $failures.Count
  failures = @($failures)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
}
$validation | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding utf8
$markdownPath = [IO.Path]::ChangeExtension($OutputPath, ".md")
@(
  "# YoloVision Public Package Proof Validation",
  "",
  "- state: ``$($validation.validationState)``",
  "- failures: ``$($validation.failureCount)``",
  "- performs publish: ``False``",
  "- can close release issue: ``False``",
  "",
  "This validator admits only a real public-feed consumer record with exact package URLs, downloaded nupkg SHA256 checks, zero ProjectReference/project libraries, a real YoloVision marker, and explicit non-publish boundaries."
) | Set-Content -LiteralPath $markdownPath -Encoding utf8
Write-Host "ValidationState=$($validation.validationState) FailureCount=$($validation.failureCount) PerformsPublish=False"
Write-Host "Report=$OutputPath"
if ($failures.Count -ne 0) { exit 1 }
