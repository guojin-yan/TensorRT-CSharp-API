[CmdletBinding()]
param(
  [string]$DownloadRoot,
  [string]$PackageConsumerProofPath = "artifacts\package-consumer\bridge-runtime\win-x64-trt11.0-cuda12.9-cudnn9.22\bridge-package-runtime-consumer-proof.json",
  [string]$SmokeLogRoot = "artifacts\real-case\trt11-cuda12-compatible-host-proof",
  [string]$OutputRoot = "artifacts\real-case\trt11-cuda12-compatible-host-proof",
  [string]$RepositoryRoot,
  [switch]$AllowIncomplete
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($DownloadRoot)) {
  $DownloadRoot = Join-Path (Split-Path -Parent $RepositoryRoot) "downloads\trt11-cuda12.9-v4.0.6156"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-InputPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return [IO.Path]::GetFullPath($Path)
  }

  return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
}

function Get-RequiredProperty {
  param(
    [object]$Object,
    [string]$Name
  )

  if ($null -eq $Object -or $Object.PSObject.Properties.Name -notcontains $Name) {
    throw "Required property '$Name' is missing."
  }

  return $Object.PSObject.Properties[$Name].Value
}

function Get-FileEvidence {
  param(
    [string]$Name,
    [string]$Path,
    [string]$ExpectedSha256 = ""
  )

  $exists = Test-Path -LiteralPath $Path -PathType Leaf
  $length = [int64]0
  $sha256 = ""
  if ($exists) {
    $item = Get-Item -LiteralPath $Path
    $length = [int64]$item.Length
    $sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
  }

  $matchesExpectedSha256 = [string]::IsNullOrWhiteSpace($ExpectedSha256) -or
    ($exists -and $sha256.Equals($ExpectedSha256, [StringComparison]::OrdinalIgnoreCase))

  return [pscustomobject]@{
    name = $Name
    path = $Path
    exists = $exists
    length = $length
    sha256 = $sha256
    expectedSha256 = $ExpectedSha256
    matchesExpectedSha256 = $matchesExpectedSha256
  }
}

function Get-SmokeLogEvidence {
  param(
    [string]$Name,
    [string]$Path,
    [string[]]$RequiredMarkers,
    [string[]]$ForbiddenMarkers = @()
  )

  $file = Get-FileEvidence -Name $Name -Path $Path
  $content = if ($file.exists) { Get-Content -LiteralPath $Path -Raw -Encoding utf8 } else { "" }
  $missingMarkers = @($RequiredMarkers | Where-Object { -not $content.Contains($_, [StringComparison]::Ordinal) })
  $presentForbiddenMarkers = @($ForbiddenMarkers | Where-Object { $content.Contains($_, [StringComparison]::Ordinal) })

  return [pscustomobject]@{
    name = $file.name
    path = $file.path
    exists = $file.exists
    length = $file.length
    sha256 = $file.sha256
    passed = $file.exists -and $missingMarkers.Count -eq 0 -and $presentForbiddenMarkers.Count -eq 0
    requiredMarkers = @($RequiredMarkers)
    missingMarkers = @($missingMarkers)
    forbiddenMarkers = @($ForbiddenMarkers)
    presentForbiddenMarkers = @($presentForbiddenMarkers)
  }
}

$resolvedProofPath = Resolve-InputPath $PackageConsumerProofPath
$resolvedSmokeRoot = Resolve-InputPath $SmokeLogRoot
$resolvedOutputRoot = Resolve-InputPath $OutputRoot
New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $resolvedProofPath -PathType Leaf)) {
  throw "Bridge package runtime consumer proof not found: $resolvedProofPath"
}

$packageProof = Get-Content -LiteralPath $resolvedProofPath -Raw -Encoding utf8 | ConvertFrom-Json

$releasePackages = @(
  [pscustomobject]@{
    role = "base"
    fileName = "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.4.0.6156.nupkg"
    expectedSha256 = "F1E1E896B5066472DD900CBD830950781E2215D967BA47BC97B3D103377FD0F3"
  },
  [pscustomobject]@{
    role = "tensorRtRuntime"
    fileName = "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.TensorRtRuntime.4.0.6156.nupkg"
    expectedSha256 = "93E8CA4FD0B95CFB49C3E2CDC6BB94AFA8126853BE66D0B5013D60875C325A2C"
  },
  [pscustomobject]@{
    role = "tensorRtBuilderSm75Sm86"
    fileName = "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.TensorRtBuilder.Sm75Sm86.4.0.6156.nupkg"
    expectedSha256 = "FE26D320160AF0B8CCD76F044429CE106DF8D89FE89B62859693FBC80FCD79CB"
  }
)

$releasePackageEvidence = @($releasePackages | ForEach-Object {
  Get-FileEvidence -Name $_.role -Path (Join-Path $DownloadRoot $_.fileName) -ExpectedSha256 $_.expectedSha256
})

$smokeLogs = @(
  Get-SmokeLogEvidence -Name "plugin-registry-inventory" -Path (Join-Path $resolvedSmokeRoot "plugin-registry-inventory.log") `
    -RequiredMarkers @("CreatorFieldCollection Included=False", "PluginRegistryInventorySmokeRunner Passed=True") `
    -ForbiddenMarkers @("Unexpected Internal Error", "Skipped=True Reason=EnvironmentProbe")
  Get-SmokeLogEvidence -Name "network-builder" -Path (Join-Path $resolvedSmokeRoot "network-builder.log") `
    -RequiredMarkers @("TensorRtLine=11", "Enqueue=True OutputMatch=True")
  Get-SmokeLogEvidence -Name "onnx-to-engine" -Path (Join-Path $resolvedSmokeRoot "onnx-to-engine.log") `
    -RequiredMarkers @("BuilderCaps FastFp16=Unavailable:NotSupported", "Enqueue=True OutputMatch=True")
  Get-SmokeLogEvidence -Name "inference-bindings" -Path (Join-Path $resolvedSmokeRoot "inference-bindings.log") `
    -RequiredMarkers @("ExecuteV2", "EnqueueV3", "OutputMatch=True")
)

$consumer = Get-RequiredProperty -Object $packageProof -Name "consumer"
$packages = Get-RequiredProperty -Object $packageProof -Name "packages"
$managedPackage = Get-RequiredProperty -Object $packages -Name "managed"
$bridgePackage = Get-RequiredProperty -Object $packages -Name "bridge"
$hostRecord = Get-RequiredProperty -Object $packageProof -Name "host"
$runtimeDiagnostic = Get-RequiredProperty -Object $packageProof -Name "runtimeCreateDiagnostic"

$managedPackageEvidence = Get-FileEvidence -Name "managed-local-package" -Path ([string](Get-RequiredProperty -Object $managedPackage -Name "path")) -ExpectedSha256 ([string](Get-RequiredProperty -Object $managedPackage -Name "sha256"))
$bridgePackageEvidence = Get-FileEvidence -Name "bridge-only-local-package" -Path ([string](Get-RequiredProperty -Object $bridgePackage -Name "path")) -ExpectedSha256 ([string](Get-RequiredProperty -Object $bridgePackage -Name "sha256"))

$releasePackagesReady = @($releasePackageEvidence | Where-Object { -not $_.exists -or -not $_.matchesExpectedSha256 }).Count -eq 0
$smokesReady = @($smokeLogs | Where-Object { -not $_.passed }).Count -eq 0
$packageConsumerReady =
  ([string](Get-RequiredProperty -Object $packageProof -Name "proofClassification")) -eq "compatible-host-bridge-package-runtime" -and
  ([string](Get-RequiredProperty -Object $packageProof -Name "smokeStatus")) -eq "passed" -and
  [bool](Get-RequiredProperty -Object $packageProof -Name "enqueueCompleted") -and
  [bool](Get-RequiredProperty -Object $packageProof -Name "identityOutputMatch") -and
  [bool](Get-RequiredProperty -Object $consumer -Name "consumerRootOutsideRepository") -and
  [bool](Get-RequiredProperty -Object $consumer -Name "usesPackageReferenceOnly") -and
  -not [bool](Get-RequiredProperty -Object $consumer -Name "usesProjectReference") -and
  [bool](Get-RequiredProperty -Object $runtimeDiagnostic -Name "returnedNonNull") -and
  $managedPackageEvidence.matchesExpectedSha256 -and
  $bridgePackageEvidence.matchesExpectedSha256

$proofPassed = $releasePackagesReady -and $smokesReady -and $packageConsumerReady
$state = if ($proofPassed) { "passed-compatible-host-engineering-proof" } else { "incomplete-compatible-host-engineering-proof" }

$record = [ordered]@{
  recordKind = "trt11-cuda12.9-compatible-host-proof"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  state = $state
  releaseTag = "v4.0.6156"
  releaseUrl = "https://github.com/guojin-yan/TensorRT-CSharp-API/releases/tag/v4.0.6156"
  sourceRuntimeKey = "win-x64-trt11.0-cuda12.9-cudnn9.22"
  proofClassification = "compatible-host-bridge-package-runtime"
  releaseAssetsDownloadedReadOnly = $true
  releasePackagesReady = $releasePackagesReady
  smokesReady = $smokesReady
  packageConsumerReady = $packageConsumerReady
  engineSerializedBytes = [int](Get-RequiredProperty -Object $packageProof -Name "engineSerializedBytes")
  enqueueCompleted = [bool](Get-RequiredProperty -Object $packageProof -Name "enqueueCompleted")
  identityOutputMatch = [bool](Get-RequiredProperty -Object $packageProof -Name "identityOutputMatch")
  runtimeCreateReturnedNonNull = [bool](Get-RequiredProperty -Object $runtimeDiagnostic -Name "returnedNonNull")
  consumerRootOutsideRepository = [bool](Get-RequiredProperty -Object $consumer -Name "consumerRootOutsideRepository")
  usesPackageReferenceOnly = [bool](Get-RequiredProperty -Object $consumer -Name "usesPackageReferenceOnly")
  usesProjectReference = [bool](Get-RequiredProperty -Object $consumer -Name "usesProjectReference")
  host = $hostRecord
  releasePackages = @($releasePackageEvidence)
  localPackages = @($managedPackageEvidence, $bridgePackageEvidence)
  smokeLogs = @($smokeLogs)
  packageConsumerProofPath = $resolvedProofPath
  isRuntimeExecutionProof = $proofPassed
  isPackageConsumerRuntimeProof = $false
  canPromoteCompatibleHostRuntimeProof = $proofPassed
  canPromoteRuntimeProof = $false
  isPublicCleanPackageConsumerProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  performsPublish = $false
  usesPublishToken = $false
  pushesNuGet = $false
  uploadsGitHubRelease = $false
  closesIssue = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  proofBoundary = "This record proves local execution from managed and bridge-only packages against read-only TensorRT 11/CUDA 12.9 vendor assets on one compatible host. The packages are local feeds and existing release downloads, so this is not public clean package-consumer proof, post-publish proof, owner approval, or permission to publish or close an issue."
}

$jsonPath = Join-Path $resolvedOutputRoot "trt11-cuda12.9-compatible-host-proof.json"
$markdownPath = Join-Path $resolvedOutputRoot "trt11-cuda12.9-compatible-host-proof.md"
[IO.File]::WriteAllText($jsonPath, (($record | ConvertTo-Json -Depth 12) + [Environment]::NewLine), $utf8)

$releaseRows = $releasePackageEvidence | ForEach-Object {
  "| ``$($_.name)`` | ``$($_.exists)`` | ``$($_.matchesExpectedSha256)`` | ``$($_.sha256)`` |"
}
$smokeRows = $smokeLogs | ForEach-Object {
  "| ``$($_.name)`` | ``$($_.passed)`` | ``$($_.sha256)`` |"
}

$markdown = @"
# TRT11 / CUDA 12.9 Compatible-Host Proof

| Field | Value |
|---|---|
| state | ``$state`` |
| proofClassification | ``compatible-host-bridge-package-runtime`` |
| runtimeCreateReturnedNonNull | ``$($record.runtimeCreateReturnedNonNull)`` |
| engineSerializedBytes | ``$($record.engineSerializedBytes)`` |
| enqueueCompleted | ``$($record.enqueueCompleted)`` |
| identityOutputMatch | ``$($record.identityOutputMatch)`` |
| isRuntimeExecutionProof | ``$($record.isRuntimeExecutionProof)`` |
| isPackageConsumerRuntimeProof | ``false`` |
| canPublishPublicly | ``false`` |
| canCloseReleaseIssue | ``false`` |

## Existing Release Packages

| Role | Exists | SHA256 Match | SHA256 |
|---|---|---|---|
$($releaseRows -join [Environment]::NewLine)

## Runtime Smokes

| Smoke | Passed | Log SHA256 |
|---|---|---|
$($smokeRows -join [Environment]::NewLine)

## Host

- GPU: ``$($hostRecord.gpuName)``
- Driver: ``$($hostRecord.driverVersion)``
- Runtime: ``$($hostRecord.runtimeEnvironmentLine)``

## Boundary

$($record.proofBoundary)
"@

[IO.File]::WriteAllText($markdownPath, $markdown + [Environment]::NewLine, $utf8)

Write-Host "TRT11/CUDA12.9 compatible-host proof written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "State=$state ReleasePackagesReady=$releasePackagesReady SmokesReady=$smokesReady PackageConsumerReady=$packageConsumerReady CanPublishPublicly=False"

if (-not $proofPassed -and -not $AllowIncomplete) {
  throw "TRT11/CUDA12.9 compatible-host proof is incomplete. Use -AllowIncomplete only to inspect the failed checks."
}
