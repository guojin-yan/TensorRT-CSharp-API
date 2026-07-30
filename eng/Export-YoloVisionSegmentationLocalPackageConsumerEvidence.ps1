[CmdletBinding()]
param(
  [string]$RepositoryRoot,
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
  $ReportPath = Join-Path $RepositoryRoot "artifacts\yolovision\yolov8n-seg-local-package-consumer\win-x64-trt10.11-cuda12.9-cudnn9.22\yolov8n-seg-local-package-consumer-runtime.json"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot "samples\assets\yolovision-yolov8n-seg-local-package-consumer-runtime-evidence.json"
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

function Assert-False {
  param($Actual, [string]$Description)
  if ($Actual -ne $false) {
    throw "$Description must be false."
  }
}

if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
  throw "YOLOv8 segmentation local package-consumer report does not exist: $ReportPath"
}
$report = Get-Content -LiteralPath $ReportPath -Raw -Encoding utf8 | ConvertFrom-Json
Assert-Equal $report.schemaVersion 3 "Report schemaVersion"
Assert-Equal $report.recordKind "yolovision-yolov8n-seg-local-package-consumer-runtime" "Report recordKind"
Assert-Equal $report.validationState "passed-local-package-consumer-runtime" "Report validationState"
Assert-Equal $report.evidenceClassification "local-package-consumer-runtime" "Report evidenceClassification"
Assert-Equal $report.scenario "yolov8-segmentation" "Report scenario"
Assert-Equal @($report.packages).Count 3 "Package count"
Assert-Equal $report.consumer.projectReferenceCount 0 "ProjectReference count"
Assert-Equal $report.consumer.directAssemblyReferenceCount 0 "Direct assembly reference count"
Assert-Equal $report.consumer.restoredProjectLibraryCount 0 "Restored project library count"
Assert-Equal $report.consumer.packageSourceKind "local-file-feed-only" "Package source kind"
Assert-Equal $report.consumer.packageSourceCount 3 "Package source count"
Assert-Equal $report.consumer.workspaceDrive "E:" "Consumer workspace drive"
Assert-Equal $report.consumer.workspaceRemovedAfterValidation $true "Consumer workspace removal"
Assert-Equal $report.consumer.nativeBridgePathEnvironmentVariableSet $false "JYPPX_NATIVE_BRIDGE_PATH state"
Assert-Equal $report.consumer.restoreExitCode 0 "Restore exit code"
Assert-Equal $report.consumer.buildExitCode 0 "Build exit code"
Assert-Equal $report.consumer.runtimeExitCode 0 "Runtime exit code"
Assert-Equal $report.nativeDependency.packageContainsTensorRtCudaOrCudnn $false "Vendor runtime package content state"
Assert-Equal $report.nativeDependency.vendorRuntimePackageEntryCount 0 "Vendor runtime package entry count"
Assert-Equal $report.nativeDependency.bridgeNativePackageEntryCount 1 "Bridge native package entry count"
Assert-Equal $report.runtime.predictionCount 4 "Segmentation prediction count"
Assert-Equal $report.segmentation.rawTensorReferenceValidation.tensorCount 2 "Raw reference tensor count"
Assert-Equal $report.segmentation.rawTensorReferenceValidation.comparedElementCount 1793600 "Raw compared value count"
Assert-Equal $report.segmentation.rawTensorReferenceValidation.mismatchCount 0 "Raw mismatch count"
Assert-Equal $report.segmentation.rawTensorReferenceValidation.passed $true "Raw reference result"
Assert-Equal $report.segmentation.maskArtifacts.predictionCount 4 "Mask artifact prediction count"
Assert-Equal $report.segmentation.maskArtifacts.sourceThresholdedMaskCount 4 "Source thresholded mask count"
Assert-Equal $report.segmentation.independentPostprocessValidation.evidenceClassification "local-package-consumer-runtime" "Independent comparison classification"
Assert-Equal $report.segmentation.independentPostprocessValidation.predictionCount 4 "Independent comparison prediction count"
Assert-Equal $report.segmentation.independentPostprocessValidation.passed $true "Independent comparison result"
Assert-Equal $report.segmentation.controlledRawReferenceValidation.exitCode 1 "Controlled raw-reference exit code"
Assert-Equal $report.segmentation.controlledRawReferenceValidation.mismatchCount 1 "Controlled raw-reference mismatch count"
Assert-Equal $report.segmentation.controlledRawReferenceValidation.firstMismatchIndex 0 "Controlled raw-reference first mismatch"
Assert-Equal $report.segmentation.controlledRawReferenceValidation.failClosed $true "Controlled raw-reference fail-closed state"
Assert-Equal $report.segmentation.controlledMaskIntegrityValidation.exitCode 1 "Controlled mask exit code"
Assert-Equal $report.segmentation.controlledMaskIntegrityValidation.failClosed $true "Controlled mask fail-closed state"
foreach ($name in @(
  "isSourceTreeRuntimeProof",
  "isPackageConsumerRuntimeProof",
  "isPublicPackageProof",
  "packagesDownloadedFromPublicFeed",
  "publicRedistributionOwnerApproval",
  "canPromotePackageConsumerRuntime",
  "canPublishPublicly",
  "isPostPublishProof",
  "ownerReleaseAcceptance",
  "releaseProof",
  "canCloseReleaseIssue",
  "performsPublish",
  "uploadsAssets"
)) {
  Assert-False $report.boundary.$name "Boundary '$name'"
}

$expectedPackageRoles = @("managed-api", "yolovision", "bridge-only")
$actualPackageRoles = @($report.packages | ForEach-Object { [string]$_.role })
Assert-Equal @(Compare-Object -ReferenceObject $expectedPackageRoles -DifferenceObject $actualPackageRoles).Count 0 "Package roles"
foreach ($package in @($report.packages)) {
  if ([string]$package.sha256 -notmatch '^[0-9a-f]{64}$' -or [long]$package.length -le 0) {
    throw "Package '$($package.id)' must have a positive length and lowercase SHA256."
  }
}

$comparisonSummaries = @(
  foreach ($comparison in @($report.segmentation.independentPostprocessValidation.comparisons)) {
    if (-not $comparison.passed -or [double]$comparison.boxIoU -lt 0.995 -or [double]$comparison.maskIoU -lt 0.99) {
      throw "Independent comparison for '$($comparison.className)' did not pass the committed thresholds."
    }
    [pscustomobject][ordered]@{
      classId = [int]$comparison.classId
      className = [string]$comparison.className
      maximumBoxCoordinateAbsoluteError = [double]$comparison.maximumBoxCoordinateAbsoluteError
      scoreAbsoluteError = [double]$comparison.scoreAbsoluteError
      boxIoU = [double]$comparison.boxIoU
      maskIoU = [double]$comparison.maskIoU
      passed = $true
    }
  }
)

$evidence = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-local-package-consumer-runtime-evidence"
  recordName = "yolovision-yolov8n-seg-local-package-consumer-trt10.11"
  generatedAtUtc = [string]$report.generatedAtUtc
  templateOnly = $false
  sampleName = "YoloVision.PackageConsumer"
  proofClassification = "local-package-consumer-runtime"
  repositoryHeadAtExecution = [string]$report.sourceCommit
  fullLocalReportSha256 = (Get-FileHash -LiteralPath $ReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
  packageConsumer = [pscustomobject][ordered]@{
    targetFramework = [string]$report.consumer.targetFramework
    packageSourceKind = [string]$report.consumer.packageSourceKind
    remotePackageSourceCount = 0
    packageCount = @($report.packages).Count
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
    nativeBridgeCopiedByNuGet = [bool]$report.nativeDependency.bridgeCopiedByNuGet
    nativeBridgePathEnvironmentVariableSet = $false
    tensorRtCudaAndCudnnAreExternalDependencies = $true
    vendorRuntimePackageEntryCount = 0
    bridgeNativePackageEntryCount = 1
    workspaceDrive = "E:"
    workspaceRemovedAfterValidation = $true
    restoreExitCode = 0
    buildExitCode = 0
    runtimeExitCode = 0
  }
  assets = [pscustomobject][ordered]@{
    modelSha256 = [string]$report.assets.modelSha256
    modelWeightsSha256 = [string]$report.assets.modelWeightsSha256
    labelsSha256 = [string]$report.assets.labelsSha256
    imageSha256 = [string]$report.assets.imageSha256
    inputTensorElementCount = [long]($report.assets.tensorLength / 4)
    inputTensorSha256 = [string]$report.assets.tensorSha256
    referenceOutput0Sha256 = [string]$report.assets.referenceOutput0Sha256
    referenceOutput1Sha256 = [string]$report.assets.referenceOutput1Sha256
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
  modelContract = $report.segmentation.modelContract
  rawTensorReferenceValidation = $report.segmentation.rawTensorReferenceValidation
  maskArtifacts = [pscustomobject][ordered]@{
    runtimeManifestSha256 = [string]$report.segmentation.maskArtifacts.runtimeManifestSha256
    archivedManifestSha256 = [string]$report.segmentation.maskArtifacts.archivedManifestSha256
    artifactFileCount = [int]$report.segmentation.maskArtifacts.archivedFileCount
    predictionCount = [int]$report.segmentation.maskArtifacts.predictionCount
    sourceThresholdedMaskCount = [int]$report.segmentation.maskArtifacts.sourceThresholdedMaskCount
    spatialTransformApplied = [bool]$report.segmentation.maskArtifacts.spatialTransformApplied
  }
  independentPostprocessValidation = [pscustomobject][ordered]@{
    evidenceClassification = "local-package-consumer-runtime"
    referenceFramework = [string]$report.segmentation.independentPostprocessValidation.referenceFramework
    pythonVersion = [string]$report.segmentation.independentPostprocessValidation.pythonVersion
    ultralyticsVersion = [string]$report.segmentation.independentPostprocessValidation.ultralyticsVersion
    torchVersion = [string]$report.segmentation.independentPostprocessValidation.torchVersion
    referenceSha256 = [string]$report.segmentation.independentPostprocessValidation.referenceSha256
    comparisonSha256 = [string]$report.segmentation.independentPostprocessValidation.comparisonSha256
    thresholds = $report.segmentation.independentPostprocessValidation.thresholds
    predictionCount = $comparisonSummaries.Count
    comparisons = $comparisonSummaries
    passed = $true
  }
  controlledNegativeValidation = $report.segmentation.controlledRawReferenceValidation
  controlledArtifactIntegrityValidation = $report.segmentation.controlledMaskIntegrityValidation
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
    "The AGPL-3.0-only weights, ONNX, references, masks, logs, and runtime outputs remain outside Git and are not approved for public redistribution.",
    "Local package-consumer evidence does not substitute for public-package, post-publish, Owner acceptance, or release proof."
  )
}

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
[IO.File]::WriteAllText($OutputPath, ($evidence | ConvertTo-Json -Depth 16) + "`n", $utf8)
Write-Host "EvidenceClassification=$($evidence.proofClassification) Packages=$($evidence.packageConsumer.packageCount) RawValues=$($evidence.rawTensorReferenceValidation.comparedElementCount) Predictions=$($evidence.independentPostprocessValidation.predictionCount)"
Write-Host "PublicPackageProof=False PostPublishProof=False OwnerReleaseAcceptance=False PerformsPublish=False"
Write-Host "Evidence=$OutputPath"
