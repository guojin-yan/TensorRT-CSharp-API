[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$MatrixPath,
  [string]$SurfaceAuditPath,
  [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if ([string]::IsNullOrWhiteSpace($MatrixPath)) { $MatrixPath = Join-Path $RepositoryRoot "artifacts\yolovision\yolox-local-package-consumer-matrix\yolox-local-package-consumer-runtime-matrix.json" }
if ([string]::IsNullOrWhiteSpace($SurfaceAuditPath)) { $SurfaceAuditPath = Join-Path $RepositoryRoot "artifacts\yolovision\package-surface-audit\yolovision-package-surface-audit.json" }
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { $OutputDirectory = Join-Path $RepositoryRoot "artifacts\interface-coverage" }
$MatrixPath = [IO.Path]::GetFullPath($MatrixPath)
$SurfaceAuditPath = [IO.Path]::GetFullPath($SurfaceAuditPath)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
foreach ($path in @($MatrixPath, $SurfaceAuditPath)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Handoff input does not exist: $path" }
}
$matrix = Get-Content -LiteralPath $MatrixPath -Raw -Encoding utf8 | ConvertFrom-Json
$surface = Get-Content -LiteralPath $SurfaceAuditPath -Raw -Encoding utf8 | ConvertFrom-Json
if ([string]$matrix.evidenceClassification -ne "local-package-consumer-runtime-matrix") { throw "Unexpected matrix classification." }
if (-not [bool]$surface.valid) { throw "YoloVision surface audit must pass before handoff export." }

$feedUrl = "https://api.nuget.org/v3/index.json"
$flatContainerUrl = "https://api.nuget.org/v3-flatcontainer"
$githubFeedUrl = "https://nuget.pkg.github.com/guojin-yan/index.json"
function Get-PublicUrl {
  param([string]$PackageId, [string]$Version)
  return "$flatContainerUrl/$($PackageId.ToLowerInvariant())/$Version/$($PackageId.ToLowerInvariant()).$Version.nupkg"
}

$packageRows = [Collections.Generic.List[object]]::new()
foreach ($package in @($matrix.sharedPackages) + @($matrix.bridgePackages | Select-Object -Unique -Property id,version,length,sha256)) {
  $runtimeRow = @($matrix.rows | Where-Object { $_.bridgePackageId -eq $package.id } | Select-Object -First 1)
  $isRuntimeSpecific = $runtimeRow.Count -eq 1
  $runtimeEligible = -not $isRuntimeSpecific -or [bool]$runtimeRow[0].runtimePassed
  $packageRows.Add([pscustomobject][ordered]@{
    id = [string]$package.id
    version = [string]$package.version
    length = [int64]$package.length
    localSha256 = [string]$package.sha256
    expectedNuGetFlatContainerUrl = Get-PublicUrl -PackageId ([string]$package.id) -Version ([string]$package.version)
    expectedGitHubPackagesSource = $githubFeedUrl
    ownerMustConfirmPublicDownload = $true
    localYoloVisionRuntimeState = if ($isRuntimeSpecific) { [string]$runtimeRow[0].state } else { "shared-package" }
    eligibleForYoloVisionPublicRuntimeHandoff = $runtimeEligible
    ownerMustRebuildAndRefreezeBeforeYoloVisionPublish = $isRuntimeSpecific -and -not $runtimeEligible
  })
}

$runtimeKeys = @($matrix.rows | Select-Object -ExpandProperty runtimePackageKey -Unique)
$publicConsumerCandidateRuntimeKeys = @($matrix.rows | Where-Object { [bool]$_.runtimePassed } | Select-Object -ExpandProperty runtimePackageKey -Unique)
$blockedRuntimeKeys = @($matrix.rows | Where-Object { -not [bool]$_.runtimePassed } | Select-Object -ExpandProperty runtimePackageKey -Unique)
$commands = foreach ($key in $publicConsumerCandidateRuntimeKeys) {
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionPublicPackageConsumer.ps1 -RuntimePackageKey $key -PackageVersion $($matrix.packageVersion) -PublicFeedUrl $feedUrl -OutputRoot E:\\yolovision-public-consumer\\$key -ProofInputPath E:\\yolovision-public-proof\\$key\\public-proof-record.json"
}
$validatorCommands = foreach ($key in $publicConsumerCandidateRuntimeKeys) {
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionPublicPackageProof.ps1 -InputPath E:\\yolovision-public-proof\\$key\\public-proof-record.json -ExpectedRuntimePackageKey $key -ExpectedPackageVersion $($matrix.packageVersion)"
}

$handoff = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-public-package-owner-handoff"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  state = "owner-action-required-public-feed-not-executed"
  packageVersion = [string]$matrix.packageVersion
  packages = @($packageRows)
  runtimePackageKeys = $runtimeKeys
  publicConsumerCandidateRuntimeKeys = $publicConsumerCandidateRuntimeKeys
  blockedRuntimeKeys = $blockedRuntimeKeys
  publicRoutes = @(
    [pscustomobject][ordered]@{ name = "nuget-org"; source = $feedUrl; pushCommand = "OWNER ONLY: dotnet nuget push <nupkg> --source https://api.nuget.org/v3/index.json"; executed = $false },
    [pscustomobject][ordered]@{ name = "github-packages"; source = $githubFeedUrl; pushCommand = "OWNER ONLY: dotnet nuget push <nupkg> --source $githubFeedUrl"; executed = $false }
  )
  cleanExternalCommands = @($commands)
  strictValidatorCommands = @($validatorCommands)
  requiredProofFields = @(
    "publicFeedUrl", "packageSourceMode", "runtimePackageKey", "packageVersion",
    "packages[].sourceUrl", "packages[].downloadedFromPublicFeed", "packages[].downloadedNupkgPath", "packages[].sha256", "packages[].nugetMetadataPath", "packages[].nugetMetadataSha256",
    "consumer.projectReferenceCount", "consumer.restoredProjectLibraryCount", "consumer.workspaceRemovedAfterValidation",
    "runtime.passed", "runtime.passedMarker", "runtime.predictionCount",
    "boundary.isPackageConsumerRuntimeProof", "boundary.packagesDownloadedFromPublicFeed"
  )
  rejectedSubstitutes = @(
    "local-file-feed", "direct .nupkg reference", "ProjectReference", "bridge-only compile proof", "dependency-probe-only", "owner template without downloaded files", "publish command output without clean consumer runtime"
  )
  sourceMatrix = [IO.Path]::GetRelativePath($RepositoryRoot, $MatrixPath).Replace('\', '/')
  surfaceAudit = [IO.Path]::GetRelativePath($RepositoryRoot, $SurfaceAuditPath).Replace('\', '/')
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$jsonPath = Join-Path $OutputDirectory "yolovision-public-package-owner-handoff.json"
$handoff | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$mdPath = [IO.Path]::ChangeExtension($jsonPath, ".md")
@(
  "# YoloVision Public Package Owner Handoff",
  "",
  "- state: ``$($handoff.state)``",
  "- package version: ``$($handoff.packageVersion)``",
  "- requested runtime keys: ``$($runtimeKeys -join ', ')``",
  "- public consumer candidates: ``$($publicConsumerCandidateRuntimeKeys -join ', ')``",
  "- blocked pending rebuild: ``$($blockedRuntimeKeys -join ', ')``",
  "- NuGet expected source: ``$feedUrl``",
  "- GitHub Packages expected source: ``$githubFeedUrl``",
  "- publish executed: ``False``",
  "",
  "## Package identities",
  "",
  "| ID | Version | Local SHA256 | Expected public URL |",
  "| --- | --- | --- | --- |",
  $(foreach ($package in $packageRows) { "| ``$($package.id)`` | ``$($package.version)`` | ``$($package.localSha256)`` | $($package.expectedNuGetFlatContainerUrl) |" }),
  "",
  "## Boundary",
  "",
  "This is a read-only owner handoff. It does not authorize, execute, or simulate NuGet/GitHub Packages push. Only a real public-feed clean consumer with exact downloaded nupkg hashes may be passed to the strict validator."
) | Set-Content -LiteralPath $mdPath -Encoding utf8
Write-Host "HandoffState=$($handoff.state) PackageCount=$($packageRows.Count) RuntimeKeyCount=$($runtimeKeys.Count) PerformsPublish=False"
Write-Host "Handoff=$jsonPath"
