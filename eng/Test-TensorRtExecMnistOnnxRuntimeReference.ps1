[CmdletBinding()]
param(
  [string]$OnnxRuntimeVersion = "1.23.2",
  [string]$GlobalPackageRoot,
  [string]$ModelPath,
  [string]$InputPath,
  [string]$TensorRtRoot = $env:JYPPX_TENSORRT_ROOT,
  [string]$TensorRtReferencePath,
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [switch]$KeepWorkspace,
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
else {
  $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
}

$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Get-Sha256 {
  param([Parameter(Mandatory = $true)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-RelativePath {
  param([Parameter(Mandatory = $true)][string]$Root, [Parameter(Mandatory = $true)][string]$Path)
  $rootPath = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
  $rootUri = [Uri]$rootPath
  $pathUri = [Uri][IO.Path]::GetFullPath($Path)
  return [Uri]::UnescapeDataString($rootUri.MakeRelativeUri($pathUri).ToString()).Replace('\', '/')
}

function Test-PathWithin {
  param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Parent)
  $resolvedPath = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
  $resolvedParent = [IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
  return $resolvedPath.StartsWith(
    $resolvedParent + [IO.Path]::DirectorySeparatorChar,
    [StringComparison]::OrdinalIgnoreCase)
}

function Remove-SafeWorkspace {
  param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$AllowedRoot)
  if (-not (Test-PathWithin -Path $Path -Parent $AllowedRoot)) {
    throw "Refusing to remove an ONNX Runtime workspace outside the allowed root: $Path"
  }
  if (Test-Path -LiteralPath $Path) {
    Remove-Item -LiteralPath $Path -Recurse -Force
  }
}

if ([string]::IsNullOrWhiteSpace($GlobalPackageRoot)) {
  $GlobalPackageRoot = Join-Path $env:USERPROFILE ".nuget\packages"
}
else { $GlobalPackageRoot = [IO.Path]::GetFullPath($GlobalPackageRoot) }

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
  $ModelPath = Join-Path (Split-Path -Parent $RepositoryRoot) "models\OnnxToEngine\MNIST\nvidia-tensorrt-10.11\mnist.onnx"
}
else { $ModelPath = [IO.Path]::GetFullPath($ModelPath) }

if ([string]::IsNullOrWhiteSpace($TensorRtRoot)) {
  $TensorRtRoot = if (-not [string]::IsNullOrWhiteSpace($env:TENSORRT_ROOT)) {
    $env:TENSORRT_ROOT
  }
  else {
    $env:TENSORRT_PATH
  }
}
if ([string]::IsNullOrWhiteSpace($TensorRtRoot) -or
    -not (Test-Path -LiteralPath (Join-Path $TensorRtRoot "data\mnist\README.md") -PathType Leaf)) {
  throw "TensorRT sample data was not found. Set -TensorRtRoot or JYPPX_TENSORRT_ROOT."
}
$TensorRtRoot = (Resolve-Path -LiteralPath $TensorRtRoot).Path

if ([string]::IsNullOrWhiteSpace($InputPath)) {
  $InputPath = Join-Path $RepositoryRoot "artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\digit-7\mnist-trt10-7-input-f32.bin"
}
else { $InputPath = [IO.Path]::GetFullPath($InputPath) }

if ([string]::IsNullOrWhiteSpace($TensorRtReferencePath)) {
  $TensorRtReferencePath = Join-Path $RepositoryRoot "artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\digit-7\mnist-trt10-7.reference.json"
}
else { $TensorRtReferencePath = [IO.Path]::GetFullPath($TensorRtReferencePath) }

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path (Split-Path -Parent $RepositoryRoot) ".onnxruntime-reference-work"
}
else { $OutputRoot = [IO.Path]::GetFullPath($OutputRoot) }

if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\digit-7\onnxruntime-cpu-reference"
}
else { $ReportDirectory = [IO.Path]::GetFullPath($ReportDirectory) }

if ([IO.Path]::GetPathRoot($OutputRoot).TrimEnd('\') -eq $env:SystemDrive.TrimEnd('\')) {
  throw "The ONNX Runtime workspace must be outside the system drive: $OutputRoot"
}

foreach ($requiredFile in @($ModelPath, $InputPath, $TensorRtReferencePath)) {
  if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
    throw "Required MNIST artifact is missing: $requiredFile"
  }
}

$packageSpecs = @(
  @{ id = "microsoft.ml.onnxruntime"; version = $OnnxRuntimeVersion; file = "microsoft.ml.onnxruntime.$OnnxRuntimeVersion.nupkg" },
  @{ id = "microsoft.ml.onnxruntime.managed"; version = $OnnxRuntimeVersion; file = "microsoft.ml.onnxruntime.managed.$OnnxRuntimeVersion.nupkg" },
  @{ id = "system.numerics.tensors"; version = "9.0.0"; file = "system.numerics.tensors.9.0.0.nupkg" },
  @{ id = "system.memory"; version = "4.5.5"; file = "system.memory.4.5.5.nupkg" }
)

$sourcePackages = [Collections.Generic.List[object]]::new()
foreach ($spec in $packageSpecs) {
  $packagePath = Join-Path $GlobalPackageRoot "$($spec.id)\$($spec.version)\$($spec.file)"
  if (-not (Test-Path -LiteralPath $packagePath -PathType Leaf)) {
    throw "Required cached NuGet package is missing; this runner will not download it: $packagePath"
  }
  $sourcePackages.Add([pscustomobject]@{
      id = $spec.id
      version = $spec.version
      path = $packagePath
      length = (Get-Item -LiteralPath $packagePath).Length
      sha256 = Get-Sha256 -Path $packagePath
    }) | Out-Null
}

$workspaceRoot = Join-Path $OutputRoot "mnist-onnxruntime-cpu-$OnnxRuntimeVersion"
Remove-SafeWorkspace -Path $workspaceRoot -AllowedRoot $OutputRoot
$feedDirectory = Join-Path $workspaceRoot "feed"
$restoreDirectory = Join-Path $workspaceRoot ".nuget\packages"
$projectDirectory = Join-Path $workspaceRoot "consumer"
$dotnetHome = Join-Path $workspaceRoot ".dotnet-home"
New-Item -ItemType Directory -Path $feedDirectory,$restoreDirectory,$projectDirectory,$dotnetHome,$ReportDirectory -Force | Out-Null

$workspaceRemoved = $false
try {
  foreach ($package in $sourcePackages) {
    Copy-Item -LiteralPath $package.path -Destination (Join-Path $feedDirectory ([IO.Path]::GetFileName($package.path)))
  }

  $templateRoot = Join-Path $RepositoryRoot "tests\fixtures\mnist-onnx-runtime-reference"
  $projectPath = Join-Path $projectDirectory "Mnist.OnnxRuntimeReference.csproj"
  $programPath = Join-Path $projectDirectory "Program.cs"
  $project = Get-Content -LiteralPath (Join-Path $templateRoot "Mnist.OnnxRuntimeReference.csproj.template") -Raw -Encoding utf8
  $project = $project.Replace("__ONNXRUNTIME_VERSION__", $OnnxRuntimeVersion)
  [IO.File]::WriteAllText($projectPath, $project, $utf8)
  Copy-Item -LiteralPath (Join-Path $templateRoot "Program.cs") -Destination $programPath

  $nugetConfigPath = Join-Path $workspaceRoot "NuGet.Config"
  $escapedFeed = [Security.SecurityElement]::Escape($feedDirectory)
  $nugetConfig = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="local-cached-packages" value="$escapedFeed" />
  </packageSources>
</configuration>
"@
  [IO.File]::WriteAllText($nugetConfigPath, $nugetConfig, $utf8)

  $referenceOutputPath = Join-Path (Split-Path -Parent $TensorRtReferencePath) "mnist-trt10-7.onnxruntime-cpu.reference.json"
  $rawOutputPath = Join-Path $ReportDirectory "mnist-onnxruntime-cpu-output.raw"
  $runReportPath = Join-Path $ReportDirectory "mnist-onnxruntime-cpu-run-report.json"
  $stdoutPath = Join-Path $ReportDirectory "mnist-onnxruntime-cpu.stdout.log"
  $stderrPath = Join-Path $ReportDirectory "mnist-onnxruntime-cpu.stderr.log"
  foreach ($path in @($referenceOutputPath, $rawOutputPath, $runReportPath, $stdoutPath, $stderrPath)) {
    if (Test-Path -LiteralPath $path -PathType Leaf) { Remove-Item -LiteralPath $path -Force }
  }
  Get-ChildItem -LiteralPath $ReportDirectory -File -Filter "onnxruntime_profile__*.json" -ErrorAction SilentlyContinue |
    Remove-Item -Force

  $previousNugetPackages = $env:NUGET_PACKAGES
  $previousDotnetHome = $env:DOTNET_CLI_HOME
  $previousSkipFirstTime = $env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE
  $previousDotnetNoLogo = $env:DOTNET_NOLOGO
  $previousTelemetryOptOut = $env:DOTNET_CLI_TELEMETRY_OPTOUT
  try {
    $env:NUGET_PACKAGES = $restoreDirectory
    $env:DOTNET_CLI_HOME = $dotnetHome
    $env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = "1"
    $env:DOTNET_NOLOGO = "1"
    $env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"

    & dotnet restore $projectPath --configfile $nugetConfigPath --packages $restoreDirectory --force --no-cache
    $restoreExitCode = $LASTEXITCODE
    if ($restoreExitCode -ne 0) { throw "Offline ONNX Runtime consumer restore failed with exit code $restoreExitCode." }

    & dotnet build $projectPath -c Release --no-restore --nologo
    $buildExitCode = $LASTEXITCODE
    if ($buildExitCode -ne 0) { throw "Offline ONNX Runtime consumer build failed with exit code $buildExitCode." }

    & dotnet run --project $projectPath -c Release --no-build -- `
      $ModelPath $InputPath $TensorRtReferencePath $referenceOutputPath $rawOutputPath $runReportPath `
      1> $stdoutPath 2> $stderrPath
    $runtimeExitCode = $LASTEXITCODE
  }
  finally {
    $env:NUGET_PACKAGES = $previousNugetPackages
    $env:DOTNET_CLI_HOME = $previousDotnetHome
    $env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = $previousSkipFirstTime
    $env:DOTNET_NOLOGO = $previousDotnetNoLogo
    $env:DOTNET_CLI_TELEMETRY_OPTOUT = $previousTelemetryOptOut
  }

  $stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw -Encoding utf8 } else { "" }
  $stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw -Encoding utf8 } else { "" }
  if ($runtimeExitCode -ne 0) {
    throw "ONNX Runtime reference execution failed with exit code $runtimeExitCode.`nSTDOUT:`n$stdout`nSTDERR:`n$stderr"
  }
  foreach ($marker in @(
      "OnnxRuntimeReference=Passed",
      "ProviderValidated=True",
      "ProfileProviders=CPUExecutionProvider",
      "DeterministicOutput=True",
      "TensorRtReferenceComparisonPassed=True",
      "PredictedIndex=7")) {
    if ($stdout.IndexOf($marker, [StringComparison]::Ordinal) -lt 0) {
      throw "ONNX Runtime reference output is missing marker '$marker'."
    }
  }

  $runReport = Get-Content -LiteralPath $runReportPath -Raw -Encoding utf8 | ConvertFrom-Json
  if (-not [bool]$runReport.success -or -not [bool]$runReport.providerValidated -or
      -not [bool]$runReport.deterministicOutput -or -not [bool]$runReport.tensorRtComparison.passed) {
    throw "ONNX Runtime run report did not pass all runtime gates."
  }

  $sidecarPath = $referenceOutputPath.Replace(".reference.json", ".reference.sidecar.json")
  $sidecar = [ordered]@{
    schemaVersion = "tensorrtexec-mnist-onnxruntime-reference-sidecar.v1"
    state = "independent-onnxruntime-cpu-reference-candidate-owner-review-required"
    sourceClassification = "onnxruntime-cpu-$OnnxRuntimeVersion-derived-unreviewed"
    onnxRuntime = [ordered]@{
      version = [string]$runReport.runtime.version
      requestedProvider = [string]$runReport.runtime.requestedProvider
      profileProviders = @($runReport.runtime.profileProviders)
      repositoryCommit = "a83fc4d58cb48eb68890dd689f94f28288cf2278"
      packageCacheSourceMetadata = "https://api.nuget.org/v3/index.json"
      packages = @($sourcePackages | ForEach-Object { [ordered]@{ id = $_.id; version = $_.version; length = $_.length; sha256 = $_.sha256 } })
      managedAssemblySha256 = [string]$runReport.runtime.managedAssemblySha256
      nativeLibrarySha256 = [string]$runReport.runtime.nativeLibrarySha256
    }
    model = [ordered]@{ sha256 = [string]$runReport.model.sha256; ownerRedistributionApproved = $false }
    input = [ordered]@{ sha256 = [string]$runReport.input.sha256; elementCount = [int]$runReport.input.elementCount }
    output = [ordered]@{
      referenceSha256 = Get-Sha256 -Path $referenceOutputPath
      rawSha256 = [string]$runReport.output.rawSha256
      deterministic = [bool]$runReport.deterministicOutput
      predictedIndex = [int]$runReport.output.predictedIndex
    }
    tensorRtComparison = [ordered]@{
      referenceSha256 = [string]$runReport.tensorRtComparison.referenceSha256
      comparedElementCount = [int]$runReport.tensorRtComparison.comparedElementCount
      mismatchCount = [int]$runReport.tensorRtComparison.mismatchCount
      firstMismatchIndex = [int]$runReport.tensorRtComparison.firstMismatchIndex
      maximumAbsoluteError = [single]$runReport.tensorRtComparison.maximumAbsoluteError
      maximumRelativeError = [single]$runReport.tensorRtComparison.maximumRelativeError
      passed = [bool]$runReport.tensorRtComparison.passed
    }
    profiling = [ordered]@{ sha256 = [string]$runReport.profile.sha256; providers = @($runReport.profile.providers) }
    execution = [ordered]@{
      remoteSourcesCleared = $true
      sourcePackageCacheReadOnly = $true
      isolatedRestoreCache = $true
      workspaceOnSystemDrive = $false
      restoreExitCode = $restoreExitCode
      buildExitCode = $buildExitCode
      runtimeExitCode = $runtimeExitCode
      projectSha256 = Get-Sha256 -Path $projectPath
      programSha256 = Get-Sha256 -Path $programPath
      stdoutSha256 = Get-Sha256 -Path $stdoutPath
      stderrSha256 = Get-Sha256 -Path $stderrPath
    }
    ownerReview = [ordered]@{
      status = "not-provided"
      acceptedAsGoldenReference = $false
      acceptedForRepositoryRedistribution = $false
      canPromoteRealModelRuntime = $false
    }
    proofBoundary = "CPUExecutionProvider profiling proves execution independent from TensorRT, but the model, input, reference, license, and redistribution remain unreviewed Owner inputs; this is not public-package, post-publish, or release proof."
  }
  $sidecar | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $sidecarPath -Encoding utf8

}
finally {
  if (-not $KeepWorkspace -and (Test-Path -LiteralPath $workspaceRoot)) {
    Remove-SafeWorkspace -Path $workspaceRoot -AllowedRoot $OutputRoot
    $workspaceRemoved = -not (Test-Path -LiteralPath $workspaceRoot)
    if ($workspaceRemoved -and (Test-Path -LiteralPath $OutputRoot -PathType Container) -and
        @(Get-ChildItem -LiteralPath $OutputRoot -Force).Count -eq 0) {
      Remove-Item -LiteralPath $OutputRoot -Force
    }
  }
  if ($Strict -and -not $KeepWorkspace -and -not $workspaceRemoved) {
    throw "ONNX Runtime consumer workspace cleanup was required but did not complete."
  }
}

$sidecarDocument = Get-Content -LiteralPath $sidecarPath -Raw -Encoding utf8 | ConvertFrom-Json
$evidencePath = Join-Path $RepositoryRoot "artifacts\interface-coverage\tensorrtexec-mnist-onnxruntime-reference-evidence.json"
$evidence = [ordered]@{
  schemaVersion = "tensorrtexec-mnist-onnxruntime-reference-evidence.v1"
  capturedDate = (Get-Date).ToString("yyyy-MM-dd")
  state = "independent-onnxruntime-cpu-reference-runtime-passed-owner-review-required"
  evidenceClassification = "independent-framework-reference-candidate-runtime"
  runtime = [ordered]@{
    name = "ONNX Runtime"
    version = [string]$runReport.runtime.version
    requestedProvider = [string]$runReport.runtime.requestedProvider
    availableProviders = @($runReport.runtime.availableProviders)
    profileProviders = @($runReport.runtime.profileProviders)
    providerValidated = [bool]$runReport.providerValidated
    repositoryCommit = [string]$sidecarDocument.onnxRuntime.repositoryCommit
    packages = @($sidecarDocument.onnxRuntime.packages)
    managedAssemblySha256 = [string]$runReport.runtime.managedAssemblySha256
    nativeLibrarySha256 = [string]$runReport.runtime.nativeLibrarySha256
  }
  model = [ordered]@{
    path = Get-RelativePath -Root $RepositoryRoot -Path $ModelPath
    sha256 = [string]$runReport.model.sha256
    sourceReadmePath = "<user-tensorrt-root>/data/mnist/README.md"
    sourceReadmeSha256 = Get-Sha256 -Path (Join-Path $TensorRtRoot "data\mnist\README.md")
    licenseReadmePath = "<user-tensorrt-root>/samples/sampleOnnxMNIST/README.md"
    licenseReadmeSha256 = Get-Sha256 -Path (Join-Path $TensorRtRoot "samples\sampleOnnxMNIST\README.md")
    ownerRedistributionApproved = $false
  }
  input = [ordered]@{
    path = Get-RelativePath -Root $RepositoryRoot -Path $InputPath
    sha256 = [string]$runReport.input.sha256
    tensorName = [string]$runReport.model.inputName
    shape = @($runReport.model.inputShape)
    elementCount = [int]$runReport.input.elementCount
    preprocessing = "float32 1-pixel/255"
  }
  reference = [ordered]@{
    path = Get-RelativePath -Root $RepositoryRoot -Path $referenceOutputPath
    sha256 = Get-Sha256 -Path $referenceOutputPath
    sidecarPath = Get-RelativePath -Root $RepositoryRoot -Path $sidecarPath
    sidecarSha256 = Get-Sha256 -Path $sidecarPath
    tensorName = [string]$runReport.output.tensorName
    shape = @($runReport.output.shape)
    elementCount = [int]$runReport.output.elementCount
    sourceClassification = [string]$runReport.reference.sourceClassification
    rawOutputPath = Get-RelativePath -Root $RepositoryRoot -Path $rawOutputPath
    rawOutputSha256 = [string]$runReport.output.rawSha256
    deterministicOutput = [bool]$runReport.deterministicOutput
    predictedIndex = [int]$runReport.output.predictedIndex
  }
  tensorRtComparison = [ordered]@{
    referencePath = Get-RelativePath -Root $RepositoryRoot -Path $TensorRtReferencePath
    referenceSha256 = [string]$runReport.tensorRtComparison.referenceSha256
    comparedElementCount = [int]$runReport.tensorRtComparison.comparedElementCount
    mismatchCount = [int]$runReport.tensorRtComparison.mismatchCount
    firstMismatchIndex = [int]$runReport.tensorRtComparison.firstMismatchIndex
    maximumAbsoluteError = [single]$runReport.tensorRtComparison.maximumAbsoluteError
    maximumRelativeError = [single]$runReport.tensorRtComparison.maximumRelativeError
    absoluteTolerance = [single]$runReport.tensorRtComparison.absoluteTolerance
    relativeTolerance = [single]$runReport.tensorRtComparison.relativeTolerance
    passed = [bool]$runReport.tensorRtComparison.passed
  }
  execution = [ordered]@{
    remoteSourcesCleared = [bool]$sidecarDocument.execution.remoteSourcesCleared
    sourcePackagesCopiedToTemporaryEdriveFeed = $true
    sourcePackageCacheReadOnly = [bool]$sidecarDocument.execution.sourcePackageCacheReadOnly
    isolatedRestoreCache = [bool]$sidecarDocument.execution.isolatedRestoreCache
    workspaceOnSystemDrive = [bool]$sidecarDocument.execution.workspaceOnSystemDrive
    workspaceRemovedAfterValidation = $workspaceRemoved
    restoreExitCode = [int]$sidecarDocument.execution.restoreExitCode
    buildExitCode = [int]$sidecarDocument.execution.buildExitCode
    runtimeExitCode = [int]$sidecarDocument.execution.runtimeExitCode
    projectSha256 = [string]$sidecarDocument.execution.projectSha256
    programSha256 = [string]$sidecarDocument.execution.programSha256
    stdoutPath = Get-RelativePath -Root $RepositoryRoot -Path $stdoutPath
    stdoutSha256 = [string]$sidecarDocument.execution.stdoutSha256
    stderrPath = Get-RelativePath -Root $RepositoryRoot -Path $stderrPath
    stderrSha256 = [string]$sidecarDocument.execution.stderrSha256
    runReportPath = Get-RelativePath -Root $RepositoryRoot -Path $runReportPath
    runReportSha256 = Get-Sha256 -Path $runReportPath
    profilePath = Get-RelativePath -Root $RepositoryRoot -Path ([string]$runReport.profile.path)
    profileSha256 = [string]$runReport.profile.sha256
  }
  ownerReview = [ordered]@{
    status = "not-provided"
    acceptedAsGoldenReference = $false
    acceptedForRepositoryRedistribution = $false
    canPromoteRealModelRuntime = $false
  }
  proofBoundary = [ordered]@{
    independentFromTensorRtExecution = $true
    independentFrameworkReferenceCandidate = $true
    ownerReviewedGolden = $false
    repositoryRedistributionApproved = $false
    publicPackageProof = $false
    postPublishProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    statement = "Two deterministic ONNX Runtime CPU runs and CPUExecutionProvider profiling provide an execution path independent from TensorRT. Owner review for model/license/redistribution/reference acceptance remains absent, so this is an independent-framework reference candidate, not accepted real-model, public-package, post-publish, or release proof."
  }
}
$evidence | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $evidencePath -Encoding utf8

Write-Output "TensorRtExecMnistOnnxRuntimeReference=Passed"
Write-Output "Reference=$referenceOutputPath"
Write-Output "ReferenceSha256=$(Get-Sha256 -Path $referenceOutputPath)"
Write-Output "Sidecar=$sidecarPath"
Write-Output "SidecarSha256=$(Get-Sha256 -Path $sidecarPath)"
Write-Output "Evidence=$evidencePath"
Write-Output "EvidenceSha256=$(Get-Sha256 -Path $evidencePath)"
Write-Output "RunReport=$runReportPath"
Write-Output "MaximumAbsoluteError=$($runReport.tensorRtComparison.maximumAbsoluteError)"
Write-Output "MaximumRelativeError=$($runReport.tensorRtComparison.maximumRelativeError)"
