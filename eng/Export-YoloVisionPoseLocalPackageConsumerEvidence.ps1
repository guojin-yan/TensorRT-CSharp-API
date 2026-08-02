[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$RuntimePackageKey = "win-x64-trt10.11-cuda12.9-cudnn9.22",
  [string]$ReportPath,
  [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if ([string]::IsNullOrWhiteSpace($ReportPath)) {
  $ReportPath = Join-Path $RepositoryRoot "artifacts\yolovision\yolov8n-pose-local-package-consumer\$RuntimePackageKey\yolov8n-pose-local-package-consumer-runtime.json"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot "samples\assets\yolovision-yolov8n-pose-local-package-consumer-runtime-evidence.json"
}
$ReportPath = [IO.Path]::GetFullPath($ReportPath)
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
$utf8 = [Text.UTF8Encoding]::new($false)

function Assert-Equal {
  param($Actual, $Expected, [string]$Description)
  if ($Actual -ne $Expected) {
    throw "$Description mismatch. Expected='$Expected' Actual='$Actual'."
  }
}

if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
  throw "Pose local-package report does not exist: $ReportPath"
}
$report = Get-Content -LiteralPath $ReportPath -Raw -Encoding utf8 | ConvertFrom-Json
Assert-Equal $report.schemaVersion 6 "Report schemaVersion"
Assert-Equal $report.recordKind "yolovision-yolov8n-pose-local-package-consumer-runtime" "Report recordKind"
Assert-Equal $report.validationState "passed-local-package-consumer-runtime" "Validation state"
Assert-Equal $report.evidenceClassification "local-package-consumer-runtime" "Evidence classification"
Assert-Equal $report.scenario "yolov8-pose" "Scenario"
Assert-Equal $report.consumer.projectReferenceCount 0 "ProjectReference count"
Assert-Equal $report.consumer.directAssemblyReferenceCount 0 "Direct assembly reference count"
Assert-Equal $report.consumer.restoredProjectLibraryCount 0 "Restored project library count"
Assert-Equal $report.consumer.packageSourceKind "local-file-feed-only" "Package source kind"
Assert-Equal $report.consumer.packageSourceIsolation "one-selected-nupkg-per-feed" "Package source isolation"
Assert-Equal $report.consumer.packageSourceCount 3 "Package source count"
Assert-Equal $report.consumer.restoredPackageHashesMatchSelected $true "Restored package hash match"
Assert-Equal @($report.consumer.restoredPackageHashChecks).Count 3 "Restored package hash check count"
Assert-Equal $report.consumer.workspaceDrive "E:" "Consumer workspace drive"
Assert-Equal $report.consumer.workspaceRemovedAfterValidation $true "Consumer workspace removal"
Assert-Equal $report.consumer.nativeBridgePathEnvironmentVariableSet $false "JYPPX_NATIVE_BRIDGE_PATH state"
Assert-Equal @($report.packages).Count 3 "Package count"
Assert-Equal $report.nativeDependency.tensorRtCudaAndCudnnAreExternalDependencies $true "External NVIDIA runtime dependency boundary"
Assert-Equal $report.nativeDependency.vendorRuntimePackageEntryCount 0 "Vendor runtime package entry count"
Assert-Equal $report.nativeDependency.bridgeNativePackageEntryCount 1 "Bridge native package entry count"
Assert-Equal $report.assets.assetsRemainOnEDrive $true "Heavy asset drive boundary"
Assert-Equal $report.assets.modelSha256 "ed1e8d2d2aeb8a2c66e642a16295a72a2990393e3a3843325537da7e11c8a899" "ONNX SHA256"
Assert-Equal $report.assets.modelWeightsSha256 "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212" "Weights SHA256"
Assert-Equal $report.assets.labelsSha256 "4d4aaea7bee6be2f675d9b53a9195ca36dfe6429f7479f29155da522a6c85930" "Labels SHA256"
Assert-Equal $report.assets.imageSha256 "6cdb4b6728a36516826f9adb9387774a6b5db0a49837d515c9045324e04e8688" "PPM image SHA256"
Assert-Equal $report.assets.sourceImageSha256 "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63" "Source image SHA256"
Assert-Equal $report.assets.referenceOutput0Sha256 "73752d18797b5c9336359b99eef4cadfb8d263e5925fd489648ce3072ef440f1" "Raw reference SHA256"
Assert-Equal $report.assets.referenceInputTensorSha256 "46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d" "Authoritative input tensor SHA256"
Assert-Equal $report.assets.tensorSha256 "46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d" "Generated input tensor SHA256"
Assert-Equal $report.pose.preprocessingValidation.matchesAuthoritativeTensor $true "Preprocessing tensor identity"
Assert-Equal $report.pose.rawTensorReferenceValidation.comparedElementCount 470400 "Raw compared value count"
Assert-Equal $report.pose.rawTensorReferenceValidation.mismatchCount 0 "Raw mismatch count"
Assert-Equal $report.pose.rawTensorReferenceValidation.passed $true "Raw reference validation"
Assert-Equal $report.pose.independentPostprocessValidation.predictionCount 4 "Pose prediction count"
Assert-Equal $report.pose.independentPostprocessValidation.passed $true "Independent pose comparison"
Assert-Equal @($report.pose.independentPostprocessValidation.comparisons).Count 4 "Independent pose comparison count"
Assert-Equal @($report.pose.independentPostprocessValidation.comparisons | Where-Object { -not $_.passed }).Count 0 "Independent pose comparison failures"
Assert-Equal $report.pose.controlledRawReferenceValidation.exitCode 1 "Controlled negative exit code"
Assert-Equal $report.pose.controlledRawReferenceValidation.mismatchCount 1 "Controlled negative mismatch count"
Assert-Equal $report.pose.controlledRawReferenceValidation.firstMismatchIndex 0 "Controlled negative first mismatch"
Assert-Equal $report.pose.controlledRawReferenceValidation.failClosed $true "Controlled negative fail-closed state"
Assert-Equal $report.boundary.isLocalPackageConsumerRuntimeEvidence $true "Local package-consumer evidence boundary"
Assert-Equal $report.boundary.isPackageConsumerRuntimeProof $false "Package-consumer proof promotion boundary"
Assert-Equal $report.boundary.isPublicPackageProof $false "Public package proof boundary"
Assert-Equal $report.boundary.isPostPublishProof $false "Post-publish proof boundary"
Assert-Equal $report.boundary.ownerReleaseAcceptance $false "Owner acceptance boundary"
Assert-Equal $report.boundary.releaseProof $false "Release proof boundary"
Assert-Equal $report.boundary.performsPublish $false "Publish boundary"
Assert-Equal $report.boundary.uploadsAssets $false "Asset upload boundary"

$evidence = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-local-package-consumer-runtime-evidence"
  recordName = "yolovision-yolov8n-pose-local-package-consumer-trt10.11"
  generatedAtUtc = [DateTimeOffset]::Parse([string]$report.generatedAtUtc).ToUniversalTime().ToString("O")
  templateOnly = $false
  sampleName = "YoloVision.PackageConsumer"
  family = "yolov8"
  task = "pose"
  proofClassification = "local-package-consumer-runtime"
  repositoryHeadAtExecution = [string]$report.sourceCommit
  fullLocalReportSha256 = (Get-FileHash -LiteralPath $ReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
  packageConsumer = [pscustomobject][ordered]@{
    targetFramework = [string]$report.consumer.targetFramework
    packageSourceKind = "local-file-feed-only"
    packageSourceIsolation = "one-selected-nupkg-per-feed"
    remotePackageSourceCount = 0
    packageCount = 3
    packages = @($report.packages | ForEach-Object {
      [pscustomobject][ordered]@{
        role = [string]$_.role
        id = [string]$_.id
        version = [string]$_.version
        length = [long]$_.length
        sha256 = [string]$_.sha256
      }
    })
    projectReferenceCount = 0
    directAssemblyReferenceCount = 0
    restoredProjectLibraryCount = 0
    restoredPackageHashesMatchSelected = $true
    nativeBridgeCopiedByNuGet = [bool]$report.nativeDependency.bridgeCopiedByNuGet
    nativeBridgePathEnvironmentVariableSet = $false
    tensorRtCudaAndCudnnAreExternalDependencies = $true
    vendorRuntimePackageEntryCount = 0
    bridgeNativePackageEntryCount = 1
    workspaceDrive = "E:"
    workspaceRemovedAfterValidation = $true
    restoreExitCode = [int]$report.consumer.restoreExitCode
    buildExitCode = [int]$report.consumer.buildExitCode
    runtimeExitCode = [int]$report.consumer.runtimeExitCode
  }
  assets = [pscustomobject][ordered]@{
    weightsSha256 = [string]$report.assets.modelWeightsSha256
    onnxSha256 = [string]$report.assets.modelSha256
    labelsSha256 = [string]$report.assets.labelsSha256
    imageSha256 = [string]$report.assets.imageSha256
    sourceImageSha256 = [string]$report.assets.sourceImageSha256
    authoritativeInputTensorSha256 = [string]$report.assets.referenceInputTensorSha256
    generatedInputTensorSha256 = [string]$report.assets.tensorSha256
    inputTensorElementCount = 1228800
    rawReferenceSha256 = [string]$report.assets.referenceOutput0Sha256
    onnxStoredUnderWorkspaceModelsDirectory = $true
    heavyAssetsRemainOutsideGit = $true
    publicRedistributionOwnerApproval = $false
  }
  runtimeEnvironment = [pscustomobject][ordered]@{
    os = [string]$report.host.os
    gpu = [string]$report.host.gpu
    driverVersion = [string]$report.host.driverVersion
    dotnetVersion = [string]$report.host.dotnetVersion
    tensorRtVersion = [string]$report.nativeDependency.bridgeBuildTensorRtVersion
    cudaToolkitVersion = [string]$report.nativeDependency.bridgeBuildCudaToolkitVersion
    nativeBridgeSha256 = [string]$report.nativeDependency.bridgeSha256
  }
  modelContract = $report.pose.modelContract
  preprocessingValidation = $report.pose.preprocessingValidation
  rawTensorReferenceValidation = $report.pose.rawTensorReferenceValidation
  independentPostprocessValidation = $report.pose.independentPostprocessValidation
  controlledNegativeValidation = $report.pose.controlledRawReferenceValidation
  proofBoundary = [pscustomobject][ordered]@{
    localPackageConsumerRuntimeEvidence = $true
    sourceTreeRuntimeProof = $false
    publicPackageProof = $false
    packagesDownloadedFromPublicFeed = $false
    postPublishProof = $false
    publicRedistributionOwnerApproval = $false
    ownerReleaseAcceptance = $false
    releaseProof = $false
    performsPublish = $false
    uploadsAssets = $false
  }
  notes = @(
    "This record proves a clean local-file-feed PackageReference restore/build/run, not a public-feed package consumer.",
    "The managed API, YoloVision, and bridge-only nupkg hashes are fixed; CUDA, cuDNN, and TensorRT are external host dependencies.",
    "The AGPL-3.0-only weights, ONNX, images, tensor, raw reference, SVG, logs, and full report remain on E drive outside Git and are not approved for repository redistribution.",
    "The converted ONNX remains under the outer workspace models directory for a future separately governed Model Zoo.",
    "Local package-consumer evidence does not substitute for public-package, post-publish, Owner acceptance, or release proof."
  )
}

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
[IO.File]::WriteAllText($OutputPath, ($evidence | ConvertTo-Json -Depth 16) + "`n", $utf8)
Write-Host "EvidenceClassification=$($evidence.proofClassification) Packages=$($evidence.packageConsumer.packageCount) RawValues=$($evidence.rawTensorReferenceValidation.comparedElementCount) Poses=$($evidence.independentPostprocessValidation.predictionCount)"
Write-Host "PublicPackageProof=False PostPublishProof=False OwnerReleaseAcceptance=False PerformsPublish=False"
Write-Host "Evidence=$OutputPath"
