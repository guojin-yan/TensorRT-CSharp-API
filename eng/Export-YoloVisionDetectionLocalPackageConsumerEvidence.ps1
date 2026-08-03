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
  $ReportPath = Join-Path $RepositoryRoot "artifacts\yolovision\yolov8n-det-local-package-consumer\$RuntimePackageKey\yolov8n-det-local-package-consumer-runtime.json"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot "samples\assets\yolovision-yolov8n-det-local-package-consumer-runtime-evidence.json"
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
  throw "Detection local-package report does not exist: $ReportPath"
}
$report = Get-Content -LiteralPath $ReportPath -Raw -Encoding utf8 | ConvertFrom-Json
Assert-Equal $report.schemaVersion 8 "Report schemaVersion"
Assert-Equal $report.recordKind "yolovision-yolov8n-det-local-package-consumer-runtime" "Report recordKind"
Assert-Equal $report.validationState "passed-local-package-consumer-runtime" "Validation state"
Assert-Equal $report.evidenceClassification "local-package-consumer-runtime" "Evidence classification"
Assert-Equal $report.scenario "yolov8-detection" "Scenario"
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
Assert-Equal $report.assets.modelSha256 "db28a49ffbb0425f39ae56252e7e0b43d06b357416c7da58872e285560b4221e" "ONNX SHA256"
Assert-Equal $report.assets.modelWeightsSha256 "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36" "Weights SHA256"
Assert-Equal $report.assets.labelsSha256 "bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a" "Labels SHA256"
Assert-Equal $report.assets.imageSha256 "6cdb4b6728a36516826f9adb9387774a6b5db0a49837d515c9045324e04e8688" "PPM image SHA256"
Assert-Equal $report.assets.sourceImageSha256 "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63" "Source image SHA256"
Assert-Equal $report.assets.referenceOutput0Sha256 "6b7d8acb790fde1d41aa503e8d6a09ead9e4ea8e447f503b8fddab998c7387d6" "Raw reference SHA256"
Assert-Equal $report.assets.referenceInputTensorSha256 "46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d" "Authoritative input tensor SHA256"
Assert-Equal $report.assets.tensorSha256 "46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d" "Generated input tensor SHA256"
Assert-Equal $report.officialDetection.preprocessingValidation.matchesAuthoritativeTensor $true "Preprocessing tensor identity"
Assert-Equal $report.officialDetection.rawTensorReferenceValidation.comparedElementCount 705600 "Raw compared value count"
Assert-Equal $report.officialDetection.rawTensorReferenceValidation.mismatchCount 0 "Raw mismatch count"
Assert-Equal $report.officialDetection.rawTensorReferenceValidation.passed $true "Raw reference validation"
Assert-Equal $report.officialDetection.independentPostprocessValidation.predictionCount 5 "Detection prediction count"
Assert-Equal $report.officialDetection.independentPostprocessValidation.passed $true "Independent detection comparison"
Assert-Equal @($report.officialDetection.independentPostprocessValidation.comparisons).Count 5 "Independent detection comparison count"
Assert-Equal @($report.officialDetection.independentPostprocessValidation.comparisons | Where-Object { -not $_.passed }).Count 0 "Independent detection comparison failures"
if ([double]$report.officialDetection.independentPostprocessValidation.minimumObservedBoxIoU -lt 0.995 -or
    [double]$report.officialDetection.independentPostprocessValidation.maximumObservedScoreError -gt 0.01) {
  throw "Independent detection comparison exceeds the approved thresholds."
}
Assert-Equal @($report.officialDetection.independentPostprocessValidation.comparisons | Where-Object { [int]$_.classId -notin @(0, 5) -or [string]$_.className -notin @("person", "bus") }).Count 0 "Independent detection class contract"
Assert-Equal @($report.officialDetection.independentPostprocessValidation.comparisons | Where-Object { [int]$_.classId -eq 0 }).Count 4 "Person prediction count"
Assert-Equal @($report.officialDetection.independentPostprocessValidation.comparisons | Where-Object { [int]$_.classId -eq 5 }).Count 1 "Bus prediction count"
Assert-Equal $report.officialDetection.controlledRawReferenceValidation.exitCode 1 "Controlled negative exit code"
Assert-Equal $report.officialDetection.controlledRawReferenceValidation.mismatchCount 1 "Controlled negative mismatch count"
Assert-Equal $report.officialDetection.controlledRawReferenceValidation.firstMismatchIndex 0 "Controlled negative first mismatch"
Assert-Equal $report.officialDetection.controlledRawReferenceValidation.failClosed $true "Controlled negative fail-closed state"
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
  recordName = "yolovision-yolov8n-det-local-package-consumer-trt10.11"
  generatedAtUtc = [DateTimeOffset]::Parse([string]$report.generatedAtUtc).ToUniversalTime().ToString("O")
  templateOnly = $false
  sampleName = "YoloVision.PackageConsumer"
  family = "yolov8"
  task = "det"
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
    pinnedIndependentReferenceSha256 = "eae23c95dafee3904fb8adc519e90045929491ccbea613a32ed69f2290e405c0"
    annotatedReferenceImageSha256 = "909011789107f1573a75efa0ecd84dbc92ffd032cc2936f22f0319bede5c1dd3"
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
  modelContract = $report.officialDetection.modelContract
  preprocessingValidation = $report.officialDetection.preprocessingValidation
  rawTensorReferenceValidation = $report.officialDetection.rawTensorReferenceValidation
  independentPostprocessValidation = $report.officialDetection.independentPostprocessValidation
  controlledNegativeValidation = $report.officialDetection.controlledRawReferenceValidation
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
Write-Host "EvidenceClassification=$($evidence.proofClassification) Packages=$($evidence.packageConsumer.packageCount) RawValues=$($evidence.rawTensorReferenceValidation.comparedElementCount) DetectionPredictions=$($evidence.independentPostprocessValidation.predictionCount)"
Write-Host "PublicPackageProof=False PostPublishProof=False OwnerReleaseAcceptance=False PerformsPublish=False"
Write-Host "Evidence=$OutputPath"
