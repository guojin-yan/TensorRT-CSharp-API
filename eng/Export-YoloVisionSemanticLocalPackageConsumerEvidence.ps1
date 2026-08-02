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
  $ReportPath = Join-Path $RepositoryRoot "artifacts\yolovision\lraspp-semantic-local-package-consumer\win-x64-trt10.11-cuda12.9-cudnn9.22\lraspp-semantic-local-package-consumer-runtime.json"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot "samples\assets\yolovision-lraspp-semantic-local-package-consumer-runtime-evidence.json"
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
  throw "LRASPP semantic local package-consumer report does not exist: $ReportPath"
}
$report = Get-Content -LiteralPath $ReportPath -Raw -Encoding utf8 | ConvertFrom-Json
Assert-Equal $report.schemaVersion 4 "Report schemaVersion"
Assert-Equal $report.recordKind "yolovision-lraspp-semantic-local-package-consumer-runtime" "Report recordKind"
Assert-Equal $report.validationState "passed-local-package-consumer-runtime" "Report validationState"
Assert-Equal $report.evidenceClassification "local-package-consumer-runtime" "Report evidenceClassification"
Assert-Equal $report.scenario "torchvision-lraspp-semantic" "Report scenario"
Assert-Equal @($report.packages).Count 3 "Package count"
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
Assert-Equal $report.consumer.restoreExitCode 0 "Restore exit code"
Assert-Equal $report.consumer.buildExitCode 0 "Build exit code"
Assert-Equal $report.consumer.runtimeExitCode 0 "Runtime exit code"
Assert-Equal $report.nativeDependency.packageContainsTensorRtCudaOrCudnn $false "Vendor runtime package content state"
Assert-Equal $report.nativeDependency.vendorRuntimePackageEntryCount 0 "Vendor runtime package entry count"
Assert-Equal $report.nativeDependency.bridgeNativePackageEntryCount 1 "Bridge native package entry count"
Assert-Equal $report.runtime.predictionCount 1 "Semantic result count"
Assert-Equal $report.semantic.rawTensorReferenceValidation.tensorCount 1 "Raw reference tensor count"
Assert-Equal $report.semantic.rawTensorReferenceValidation.comparedElementCount 2150400 "Raw compared value count"
Assert-Equal $report.semantic.rawTensorReferenceValidation.mismatchCount 0 "Raw mismatch count"
Assert-Equal $report.semantic.rawTensorReferenceValidation.passed $true "Raw reference result"
Assert-Equal $report.semantic.classIndexArtifactValidation.pixelCount 102400 "Class-index pixel count"
Assert-Equal $report.semantic.classIndexArtifactValidation.mismatchCount 0 "Class-index mismatch count"
Assert-Equal $report.semantic.classIndexArtifactValidation.firstMismatchIndex -1 "Class-index first mismatch"
Assert-Equal $report.semantic.classIndexArtifactValidation.classIndexMatches $true "Class-index comparison"
Assert-Equal $report.semantic.classIndexArtifactValidation.histogramMatches $true "Histogram comparison"
Assert-Equal $report.semantic.classIndexArtifactValidation.classIndexSha256 "fdd15b95222eadf137fc6880e56990aa507ee7d2429f471deaae9eab31268414" "Class-index SHA256"
Assert-Equal $report.semantic.controlledRawReferenceValidation.exitCode 1 "Controlled raw-reference exit code"
Assert-Equal $report.semantic.controlledRawReferenceValidation.mismatchCount 1 "Controlled raw-reference mismatch count"
Assert-Equal $report.semantic.controlledRawReferenceValidation.firstMismatchIndex 0 "Controlled raw-reference first mismatch"
Assert-Equal $report.semantic.controlledRawReferenceValidation.failClosed $true "Controlled raw-reference fail-closed state"
Assert-Equal $report.semantic.controlledClassIndexIntegrityValidation.exitCode 1 "Controlled class-index exit code"
Assert-Equal $report.semantic.controlledClassIndexIntegrityValidation.failClosed $true "Controlled class-index fail-closed state"
if (@($report.semantic.controlledClassIndexIntegrityValidation.findings | Where-Object { $_ -eq "artifact-sha256" }).Count -ne 1) {
  throw "Controlled class-index negative must report artifact-sha256 exactly once."
}
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
$generatedAtUtc = if ($report.generatedAtUtc -is [DateTime]) {
  $report.generatedAtUtc.ToUniversalTime().ToString("O")
}
else {
  [DateTimeOffset]::Parse([string]$report.generatedAtUtc, [Globalization.CultureInfo]::InvariantCulture).ToUniversalTime().ToString("O")
}

$evidence = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-local-package-consumer-runtime-evidence"
  recordName = "yolovision-lraspp-semantic-local-package-consumer-trt10.11"
  generatedAtUtc = $generatedAtUtc
  templateOnly = $false
  sampleName = "YoloVision.PackageConsumer"
  family = "torchvision-lraspp-mobilenet-v3-large"
  task = "sem"
  proofClassification = "local-package-consumer-runtime"
  repositoryHeadAtExecution = [string]$report.sourceCommit
  fullLocalReportSha256 = (Get-FileHash -LiteralPath $ReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
  packageConsumer = [pscustomobject][ordered]@{
    targetFramework = [string]$report.consumer.targetFramework
    packageSourceKind = [string]$report.consumer.packageSourceKind
    packageSourceIsolation = [string]$report.consumer.packageSourceIsolation
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
    restoredPackageHashesMatchSelected = $true
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
    weightsSha256 = [string]$report.assets.modelWeightsSha256
    onnxSha256 = [string]$report.assets.modelSha256
    labelsSha256 = [string]$report.assets.labelsSha256
    imageSha256 = [string]$report.assets.imageSha256
    inputTensorElementCount = [long]($report.assets.tensorLength / 4)
    inputTensorSha256 = [string]$report.assets.tensorSha256
    rawReferenceSha256 = [string]$report.assets.referenceOutput0Sha256
    classIndexReferenceSha256 = [string]$report.assets.referenceClassIndexSha256
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
  modelContract = $report.semantic.modelContract
  rawTensorReferenceValidation = $report.semantic.rawTensorReferenceValidation
  classIndexArtifactValidation = $report.semantic.classIndexArtifactValidation
  controlledNegativeValidation = $report.semantic.controlledRawReferenceValidation
  controlledArtifactIntegrityValidation = $report.semantic.controlledClassIndexIntegrityValidation
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
    "The official weights, ONNX, image, tensor, raw reference, class-index binary, SVG, logs, and full report remain on E drive outside Git and are not approved for repository redistribution.",
    "The converted ONNX remains under the outer workspace models directory for a future separately governed Model Zoo.",
    "Local package-consumer evidence does not substitute for public-package, post-publish, Owner acceptance, or release proof."
  )
}

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
[IO.File]::WriteAllText($OutputPath, ($evidence | ConvertTo-Json -Depth 16) + "`n", $utf8)
Write-Host "EvidenceClassification=$($evidence.proofClassification) Packages=$($evidence.packageConsumer.packageCount) RawValues=$($evidence.rawTensorReferenceValidation.comparedElementCount) Pixels=$($evidence.classIndexArtifactValidation.pixelCount)"
Write-Host "PublicPackageProof=False PostPublishProof=False OwnerReleaseAcceptance=False PerformsPublish=False"
Write-Host "Evidence=$OutputPath"
