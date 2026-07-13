[CmdletBinding()]
param(
  [string]$OutputDirectory = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $repositoryRoot "artifacts\real-case\multi-version-onnx-runtime"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return [IO.Path]::GetFullPath($Path)
  }

  return [IO.Path]::GetFullPath((Join-Path $repositoryRoot ($Path -replace "/", "\")))
}

function Get-RepositoryRelativePath {
  param([Parameter(Mandatory = $true)][string]$Path)

  $fullPath = Resolve-RepositoryPath $Path
  return [IO.Path]::GetRelativePath($repositoryRoot, $fullPath).Replace("\", "/")
}

function New-EvidenceArtifact {
  param(
    [Parameter(Mandatory = $true)][string]$Kind,
    [Parameter(Mandatory = $true)][string]$Path,
    [bool]$Required = $false,
    [string]$Description = ""
  )

  $fullPath = Resolve-RepositoryPath $Path
  $exists = Test-Path -LiteralPath $fullPath -PathType Leaf
  $item = if ($exists) { Get-Item -LiteralPath $fullPath } else { $null }

  return [pscustomobject][ordered]@{
    kind = $Kind
    path = Get-RepositoryRelativePath $fullPath
    required = $Required
    exists = $exists
    lengthBytes = if ($exists) { [long]$item.Length } else { 0L }
    sha256 = if ($exists) { (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { "" }
    description = $Description
  }
}

function Read-JsonFile {
  param([Parameter(Mandatory = $true)][string]$Path)

  $fullPath = Resolve-RepositoryPath $Path
  return Get-Content -LiteralPath $fullPath -Raw | ConvertFrom-Json
}

function New-CaseRecord {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][int]$TensorRtLine,
    [Parameter(Mandatory = $true)][string]$TensorRtVersion,
    [Parameter(Mandatory = $true)][string]$CudaToolkitVersion,
    [Parameter(Mandatory = $true)][string]$Workload,
    [Parameter(Mandatory = $true)][string]$State,
    [Parameter(Mandatory = $true)][string]$ProofClassification,
    [Parameter(Mandatory = $true)][bool]$Success,
    [Parameter(Mandatory = $true)][bool]$InferenceRan,
    [Parameter(Mandatory = $true)][bool]$OutputMatch,
    [Parameter(Mandatory = $true)][string]$StatusDetail,
    [Parameter(Mandatory = $true)][object[]]$Evidence,
    [object]$TensorMetadata = $null,
    [object]$Result = $null,
    [string[]]$MissingRuntimeAssets = @()
  )

  return [pscustomobject][ordered]@{
    id = $Id
    tensorRtLine = $TensorRtLine
    tensorRtVersion = $TensorRtVersion
    cudaToolkitVersion = $CudaToolkitVersion
    workload = $Workload
    state = $State
    proofClassification = $ProofClassification
    success = $Success
    inferenceRan = $InferenceRan
    outputMatch = $OutputMatch
    statusDetail = $StatusDetail
    tensorMetadata = $TensorMetadata
    result = $Result
    missingRuntimeAssets = @($MissingRuntimeAssets)
    evidence = @($Evidence)
    existingEvidenceCount = @($Evidence | Where-Object exists).Count
    hashedEvidenceCount = @($Evidence | Where-Object { $_.exists -and $_.sha256.Length -eq 64 }).Count
    isPackageConsumerRuntimeProof = $false
    canPromotePackageConsumerRuntime = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    proofBoundary = "Source-tree build/runtime evidence is version-specific validation only. synthetic-input-runtime and real-model-runtime do not replace package-consumer-runtime, post-publish verification, or owner release authorization."
  }
}

function New-Trt8IdentityCase {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][string]$CudaToolkitVersion,
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$BridgePath
  )

  $logPath = Join-Path $Directory "inference-bindings-runtime.log"
  $logText = Get-Content -LiteralPath (Resolve-RepositoryPath $logPath) -Raw
  $versionMatch = [regex]::Match($logText, "TRT=(?<trt>[0-9.]+)\s+CUDA=(?<cuda>[0-9.]+)")
  $executionMatch = [regex]::Match($logText, "ElapsedMs=(?<elapsed>[0-9.]+)\s+OutputMatch=(?<match>True|False)")
  $passed = $logText.Contains("InferenceBindings Passed=True", [StringComparison]::Ordinal) -and
    $executionMatch.Success -and
    [bool]::Parse($executionMatch.Groups["match"].Value)

  $evidence = @(
    New-EvidenceArtifact -Kind "native-bridge" -Path $BridgePath -Required $true -Description "Version-specific native bridge."
    New-EvidenceArtifact -Kind "runtime-log" -Path $logPath -Required $true -Description "InferenceBindings identity runtime transcript."
  )

  $tensorMetadata = [pscustomobject][ordered]@{
    inputName = "Input"
    inputDataType = "Float"
    inputShape = @(2, 4)
    outputName = "Output"
    outputDataType = "Float"
    outputShape = @(2, 4)
  }

  $result = [pscustomobject][ordered]@{
    elapsedMilliseconds = if ($executionMatch.Success) { [double]::Parse($executionMatch.Groups["elapsed"].Value, [Globalization.CultureInfo]::InvariantCulture) } else { $null }
    expectedOutput = "identity"
    actualOutput = "identity"
  }

  return New-CaseRecord `
    -Id $Id `
    -TensorRtLine 8 `
    -TensorRtVersion $(if ($versionMatch.Success) { $versionMatch.Groups["trt"].Value } else { "8.6.1" }) `
    -CudaToolkitVersion $CudaToolkitVersion `
    -Workload "embedded-inference-bindings-identity" `
    -State $(if ($passed) { "passed" } else { "failed" }) `
    -ProofClassification "synthetic-input-runtime" `
    -Success $passed `
    -InferenceRan $passed `
    -OutputMatch $passed `
    -StatusDetail $(if ($passed) { "InferenceBindings runtime completed with OutputMatch=True." } else { "InferenceBindings runtime log did not contain a passing output match." }) `
    -Evidence $evidence `
    -TensorMetadata $tensorMetadata `
    -Result $result
}

function New-Trt10IdentityCase {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$BridgePath
  )

  $reportPath = Join-Path $Directory "identity-runtime-report.json"
  $outputPath = Join-Path $Directory "identity-output.json"
  $report = Read-JsonFile $reportPath
  $output = Read-JsonFile $outputPath
  $passed = [bool]$report.Success -and [bool]$report.InferenceRan -and [bool]$report.OutputMatch -and
    [string]$report.ProofClassification -eq "synthetic-input-runtime"

  $evidence = @(
    New-EvidenceArtifact -Kind "native-bridge" -Path $BridgePath -Required $true -Description "Version-specific native bridge."
    New-EvidenceArtifact -Kind "runtime-report" -Path $reportPath -Required $true -Description "Structured identity runtime report."
    New-EvidenceArtifact -Kind "runtime-log" -Path (Join-Path $Directory "identity-runtime.log") -Required $true -Description "Identity runtime transcript."
    New-EvidenceArtifact -Kind "serialized-engine" -Path (Join-Path $Directory "identity.plan") -Required $true -Description "Serialized identity engine."
    New-EvidenceArtifact -Kind "output-json" -Path $outputPath -Required $true -Description "Identity tensor output summary."
    New-EvidenceArtifact -Kind "timing-json" -Path (Join-Path $Directory "identity-times.json") -Required $true -Description "Runtime timing summary."
    New-EvidenceArtifact -Kind "engine-readback-json" -Path (Join-Path $Directory "identity-times.engine-readback.json") -Required $true -Description "Readonly engine metadata readback."
  )

  $tensorMetadata = [pscustomobject][ordered]@{
    outputName = [string]$output.TensorName
    outputShape = @($output.Shape)
    inputElementCount = [int]$output.InputElementCount
    outputElementCount = [int]$output.OutputElementCount
  }

  $result = [pscustomobject][ordered]@{
    elapsedMilliseconds = [double]$report.ElapsedMilliseconds
    normalizedCommandSha256 = [string]$report.NormalizedCommandSha256
    modelSource = [string]$report.ModelSource
  }

  return New-CaseRecord `
    -Id $Id `
    -TensorRtLine 10 `
    -TensorRtVersion ([string]$report.CapabilityProbe.TensorRtVersion) `
    -CudaToolkitVersion ([string]$report.CapabilityProbe.CudaToolkitVersion) `
    -Workload "embedded-dynamic-identity" `
    -State $(if ($passed) { "passed" } else { [string]$report.State }) `
    -ProofClassification ([string]$report.ProofClassification) `
    -Success $passed `
    -InferenceRan ([bool]$report.InferenceRan) `
    -OutputMatch ([bool]$report.OutputMatch) `
    -StatusDetail $(if ($passed) { "Engine roundtrip, enqueue, synchronization, and identity output match completed." } else { [string]$report.SkipReason }) `
    -Evidence $evidence `
    -TensorMetadata $tensorMetadata `
    -Result $result
}

function New-MnistCase {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$ReportFileName,
    [Parameter(Mandatory = $true)][string]$OutputFileName,
    [Parameter(Mandatory = $true)][string]$EngineFileName,
    [Parameter(Mandatory = $true)][string]$InputTensorFileName,
    [Parameter(Mandatory = $true)][string]$LogFileName,
    [Parameter(Mandatory = $true)][string]$BridgePath,
    [Parameter(Mandatory = $true)][string]$CudaToolkitVersion
  )

  $reportPath = Join-Path $Directory $ReportFileName
  $outputPath = Join-Path $Directory $OutputFileName
  $report = Read-JsonFile $reportPath
  $output = Read-JsonFile $outputPath
  $passed = [bool]$report.Success -and [bool]$report.InferenceRan -and [bool]$report.OutputMatch -and
    [int]$report.ExpectedDigit -eq [int]$report.PredictedDigit -and
    [double]$report.Confidence -ge [double]$report.MinimumConfidence -and
    [string]$report.ProofClassification -eq "real-model-runtime"

  $evidence = @(
    New-EvidenceArtifact -Kind "native-bridge" -Path $BridgePath -Required $true -Description "Version-specific native bridge."
    New-EvidenceArtifact -Kind "onnx-model" -Path ([string]$report.ModelPath) -Required $true -Description "External TensorRT MNIST ONNX model."
    New-EvidenceArtifact -Kind "input-pgm" -Path ([string]$report.InputPath) -Required $true -Description "P5 PGM input image."
    New-EvidenceArtifact -Kind "preprocessed-input-tensor" -Path (Join-Path $Directory $InputTensorFileName) -Required $true -Description "Float32 tensor after 1 - pixel / 255 preprocessing."
    New-EvidenceArtifact -Kind "runtime-report" -Path $reportPath -Required $true -Description "Structured MNIST runtime report."
    New-EvidenceArtifact -Kind "runtime-log" -Path (Join-Path $Directory $LogFileName) -Required $true -Description "MNIST runtime transcript."
    New-EvidenceArtifact -Kind "serialized-engine" -Path (Join-Path $Directory $EngineFileName) -Required $true -Description "Serialized MNIST engine."
    New-EvidenceArtifact -Kind "output-json" -Path $outputPath -Required $true -Description "Logits, probabilities, prediction, and confidence."
  )

  $tensorMetadata = [pscustomobject][ordered]@{
    inputName = [string]$report.InputTensorName
    inputDataType = [string]$report.InputDataType
    inputShape = @($report.InputShape)
    outputName = [string]$report.OutputTensorName
    outputDataType = [string]$report.OutputDataType
    outputShape = @($report.OutputShape)
  }

  $result = [pscustomobject][ordered]@{
    expectedDigit = [int]$output.ExpectedDigit
    predictedDigit = [int]$output.PredictedDigit
    confidence = [double]$output.Confidence
    minimumConfidence = [double]$output.MinimumConfidence
    elapsedMilliseconds = [double]$output.ElapsedMilliseconds
    modelSha256 = [string]$report.ModelSha256
    inputSha256 = [string]$report.InputSha256
    preprocessedInputSha256 = [string]$report.PreprocessedInputSha256
    engineSha256 = [string]$report.EngineSha256
  }

  return New-CaseRecord `
    -Id $Id `
    -TensorRtLine 10 `
    -TensorRtVersion "10.11.0" `
    -CudaToolkitVersion $CudaToolkitVersion `
    -Workload "external-mnist-digit-$($output.ExpectedDigit)" `
    -State $(if ($passed) { "passed" } else { [string]$report.State }) `
    -ProofClassification ([string]$report.ProofClassification) `
    -Success $passed `
    -InferenceRan ([bool]$report.InferenceRan) `
    -OutputMatch ([bool]$report.OutputMatch) `
    -StatusDetail $(if ($passed) { "External MNIST inference matched digit $($output.ExpectedDigit) with confidence $([double]$output.Confidence)." } else { [string]$report.SkipReason }) `
    -Evidence $evidence `
    -TensorMetadata $tensorMetadata `
    -Result $result
}

function New-Trt8MnistBlockedCase {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][string]$TensorRtRoot,
    [Parameter(Mandatory = $true)][string]$CudaToolkitVersion,
    [Parameter(Mandatory = $true)][string]$BridgePath,
    [Parameter(Mandatory = $true)][string]$CMakeCachePath,
    [Parameter(Mandatory = $true)][string]$CudnnRuntimePath
  )

  $evidence = @(
    New-EvidenceArtifact -Kind "native-bridge" -Path $BridgePath -Required $true -Description "TRT8 native bridge; identity runtime remains usable."
    New-EvidenceArtifact -Kind "cmake-cache" -Path $CMakeCachePath -Required $true -Description "Build configuration records JYPPX_CUDNN8_RUNTIME_DLL-NOTFOUND."
    New-EvidenceArtifact -Kind "onnx-parser-runtime" -Path (Join-Path $TensorRtRoot "lib\nvonnxparser.dll") -Required $true -Description "TensorRT 8 ONNX parser runtime."
    New-EvidenceArtifact -Kind "cudnn8-runtime" -Path $CudnnRuntimePath -Required $true -Description "Required cuDNN8 runtime for parser enablement."
    New-EvidenceArtifact -Kind "onnx-model" -Path (Join-Path $TensorRtRoot "data\mnist\mnist.onnx") -Required $true -Description "External TensorRT MNIST ONNX model."
    New-EvidenceArtifact -Kind "input-pgm" -Path (Join-Path $TensorRtRoot "data\mnist\7.pgm") -Required $true -Description "Representative MNIST input."
  )

  $missing = @($evidence | Where-Object { $_.required -and -not $_.exists } | ForEach-Object path)

  return New-CaseRecord `
    -Id $Id `
    -TensorRtLine 8 `
    -TensorRtVersion "8.6.1" `
    -CudaToolkitVersion $CudaToolkitVersion `
    -Workload "external-mnist-digit-7" `
    -State "blocked-by-cudnn8-runtime-missing" `
    -ProofClassification "dependency-probe-only" `
    -Success $false `
    -InferenceRan $false `
    -OutputMatch $false `
    -StatusDetail "nvonnxparser is present, but CMake safely disabled TensorRT 8 ONNX parser integration because cudnn64_8.dll is unavailable." `
    -Evidence $evidence `
    -MissingRuntimeAssets $missing
}

function New-Trt11BlockedCase {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][string]$Workload,
    [Parameter(Mandatory = $true)][string]$BridgePath,
    [Parameter(Mandatory = $true)][string]$TensorRtRoot,
    [string]$LogPath = ""
  )

  $evidence = @(
    New-EvidenceArtifact -Kind "native-bridge" -Path $BridgePath -Required $true -Description "TRT11/CUDA12.9 bridge built from available headers and import libraries."
    New-EvidenceArtifact -Kind "nvinfer-runtime" -Path (Join-Path $TensorRtRoot "lib\nvinfer_11.dll") -Required $true -Description "Required TensorRT 11 runtime DLL."
    New-EvidenceArtifact -Kind "nvinfer-plugin-runtime" -Path (Join-Path $TensorRtRoot "lib\nvinfer_plugin_11.dll") -Required $true -Description "Required TensorRT 11 plugin runtime DLL."
    New-EvidenceArtifact -Kind "onnx-parser-runtime" -Path (Join-Path $TensorRtRoot "lib\nvonnxparser_11.dll") -Required $true -Description "Required TensorRT 11 ONNX parser runtime DLL."
  )

  if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
    $evidence += New-EvidenceArtifact -Kind "runtime-log" -Path $LogPath -Required $true -Description "Runtime probe transcript."
  }

  if ($Workload.StartsWith("external-mnist", [StringComparison]::Ordinal)) {
    $evidence += New-EvidenceArtifact -Kind "onnx-model" -Path (Join-Path $TensorRtRoot "data\mnist\mnist.onnx") -Required $false -Description "MNIST model candidate; runtime cannot be reached without TensorRT DLLs."
    $evidence += New-EvidenceArtifact -Kind "input-pgm" -Path (Join-Path $TensorRtRoot "data\mnist\7.pgm") -Required $false -Description "MNIST input candidate."
  }

  $missing = @($evidence | Where-Object { $_.required -and -not $_.exists } | ForEach-Object path)

  return New-CaseRecord `
    -Id $Id `
    -TensorRtLine 11 `
    -TensorRtVersion "11.0.0" `
    -CudaToolkitVersion "12.9" `
    -Workload $Workload `
    -State "blocked-by-runtime-assets-missing" `
    -ProofClassification "dependency-probe-only" `
    -Success $false `
    -InferenceRan $false `
    -OutputMatch $false `
    -StatusDetail "The CUDA12.9 TensorRT 11 vendor root contains headers/import libraries but not the required nvinfer_11, nvinfer_plugin_11, and nvonnxparser_11 runtime DLLs." `
    -Evidence $evidence `
    -MissingRuntimeAssets $missing
}

$cases = [Collections.Generic.List[object]]::new()

$cases.Add((New-Trt8IdentityCase `
  -Id "trt8-cuda11-identity" `
  -CudaToolkitVersion "11.8" `
  -Directory "artifacts\real-case\multi-version-onnx-runtime\trt8-cuda11-identity" `
  -BridgePath "build-out\win-x64-trt8-cuda11-release\bin\Release\jyppxtrtbridge.dll"))

$cases.Add((New-Trt8IdentityCase `
  -Id "trt8-cuda12-identity" `
  -CudaToolkitVersion "12.1" `
  -Directory "artifacts\real-case\multi-version-onnx-runtime\trt8-cuda12-identity" `
  -BridgePath "build-out\win-x64-trt8-cuda12-release\bin\Release\jyppxtrtbridge.dll"))

$cases.Add((New-Trt8MnistBlockedCase `
  -Id "trt8-cuda11-mnist-digit-7" `
  -TensorRtRoot "third_party\nvidia\TensorRT-8.6.1.6-cuda 11.8" `
  -CudaToolkitVersion "11.8" `
  -BridgePath "build-out\win-x64-trt8-cuda11-release\bin\Release\jyppxtrtbridge.dll" `
  -CMakeCachePath "build-out\win-x64-trt8-cuda11-release\CMakeCache.txt" `
  -CudnnRuntimePath "third_party\nvidia\cudnn-windows-x86_64-8.9.7.29_cuda11-archive\bin\cudnn64_8.dll"))

$cases.Add((New-Trt8MnistBlockedCase `
  -Id "trt8-cuda12-mnist-digit-7" `
  -TensorRtRoot "third_party\nvidia\TensorRT-8.6.1.6-cuda 12.1" `
  -CudaToolkitVersion "12.1" `
  -BridgePath "build-out\win-x64-trt8-cuda12-release\bin\Release\jyppxtrtbridge.dll" `
  -CMakeCachePath "build-out\win-x64-trt8-cuda12-release\CMakeCache.txt" `
  -CudnnRuntimePath "third_party\nvidia\cudnn-windows-x86_64-8.9.7.29_cuda12-archive\bin\cudnn64_8.dll"))

$cases.Add((New-Trt10IdentityCase `
  -Id "trt10-cuda11-identity" `
  -Directory "artifacts\real-case\multi-version-onnx-runtime\trt10-cuda11-identity" `
  -BridgePath "build-out\win-x64-trt10-cuda11-release\bin\Release\jyppxtrtbridge.dll"))

$cases.Add((New-MnistCase `
  -Id "trt10-cuda11-mnist-digit-7" `
  -Directory "artifacts\real-case\multi-version-onnx-runtime\trt10-cuda11-mnist\digit-7" `
  -ReportFileName "mnist-runtime-report.json" `
  -OutputFileName "mnist-output.json" `
  -EngineFileName "mnist-trt10-cuda11.plan" `
  -InputTensorFileName "mnist-input-f32.bin" `
  -LogFileName "mnist-runtime.log" `
  -BridgePath "build-out\win-x64-trt10-cuda11-release\bin\Release\jyppxtrtbridge.dll" `
  -CudaToolkitVersion "11.8"))

$cases.Add((New-Trt10IdentityCase `
  -Id "trt10-cuda12-identity" `
  -Directory "artifacts\real-case\multi-version-onnx-runtime\trt10-cuda12-identity" `
  -BridgePath "build-out\win-x64-trt10-cuda12-release\bin\Release\jyppxtrtbridge.dll"))

foreach ($digit in 0..9) {
  $directory = "artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\digit-$digit"
  $cases.Add((New-MnistCase `
    -Id "trt10-cuda12-mnist-digit-$digit" `
    -Directory $directory `
    -ReportFileName "mnist-trt10-$digit-runtime-report.json" `
    -OutputFileName "mnist-trt10-$digit-output.json" `
    -EngineFileName "mnist-trt10-$digit.plan" `
    -InputTensorFileName "mnist-trt10-$digit-input-f32.bin" `
    -LogFileName "mnist-trt10-$digit-runtime.log" `
    -BridgePath "build-out\win-x64-trt10-cuda12-release\bin\Release\jyppxtrtbridge.dll" `
    -CudaToolkitVersion "12.9"))
}

$trt11Root = "third_party\nvidia\TensorRT-11.0.0.114-cuda 12.9"
$trt11Bridge = "build-out\win-x64-trt11-cuda12-release\bin\Release\jyppxtrtbridge.dll"
$cases.Add((New-Trt11BlockedCase `
  -Id "trt11-cuda12-identity" `
  -Workload "embedded-dynamic-identity" `
  -BridgePath $trt11Bridge `
  -TensorRtRoot $trt11Root `
  -LogPath "artifacts\real-case\multi-version-onnx-runtime\trt11-cuda12-identity\identity-runtime.log"))

$cases.Add((New-Trt11BlockedCase `
  -Id "trt11-cuda12-mnist-digit-7" `
  -Workload "external-mnist-digit-7" `
  -BridgePath $trt11Bridge `
  -TensorRtRoot $trt11Root))

$passedCases = @($cases | Where-Object state -eq "passed")
$blockedCases = @($cases | Where-Object state -like "blocked-*")
$syntheticCases = @($cases | Where-Object proofClassification -eq "synthetic-input-runtime")
$realModelCases = @($cases | Where-Object proofClassification -eq "real-model-runtime")
$packageConsumerCases = @($cases | Where-Object isPackageConsumerRuntimeProof)

$gpuName = ""
$driverVersion = ""
try {
  $gpuLine = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
  if (-not [string]::IsNullOrWhiteSpace($gpuLine)) {
    $parts = $gpuLine -split ",", 2
    $gpuName = $parts[0].Trim()
    if ($parts.Count -gt 1) {
      $driverVersion = $parts[1].Trim()
    }
  }
}
catch {
  $gpuName = ""
  $driverVersion = ""
}

$matrix = [pscustomobject][ordered]@{
  recordKind = "multi-version-onnx-runtime-evidence-matrix"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  matrixState = if ($blockedCases.Count -eq 0) { "verified" } else { "partially-verified" }
  caseCount = $cases.Count
  passedCaseCount = $passedCases.Count
  blockedCaseCount = $blockedCases.Count
  syntheticRuntimeCaseCount = $syntheticCases.Count
  realModelRuntimeCaseCount = $realModelCases.Count
  packageConsumerRuntimeCaseCount = $packageConsumerCases.Count
  tensorRtLines = @(8, 10, 11)
  host = [pscustomobject][ordered]@{
    os = [Environment]::OSVersion.VersionString
    architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    gpuName = $gpuName
    nvidiaDriver = $driverVersion
  }
  cases = @($cases)
  blockedCaseIds = @($blockedCases | ForEach-Object id)
  proofClassifications = @(
    "dependency-probe-only",
    "synthetic-input-runtime",
    "real-model-runtime",
    "package-consumer-runtime"
  )
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  boundary = "This matrix proves only the recorded source-tree version/runtime cases. Blocked dependency probes are not runtime proof. synthetic-input-runtime is not real-model-runtime. Neither source-tree classification is package-consumer-runtime or post-publish verification."
}

$jsonPath = Join-Path $OutputDirectory "multi-version-runtime-evidence-matrix.json"
$markdownPath = Join-Path $OutputDirectory "multi-version-runtime-evidence-matrix.md"

$matrix | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = [Collections.Generic.List[string]]::new()
$lines.Add("# TensorRT/CUDA 多版本运行证据矩阵")
$lines.Add("")
$lines.Add("- 生成时间（UTC）：``$($matrix.generatedAtUtc)``")
$lines.Add("- 状态：``$($matrix.matrixState)``")
$lines.Add("- 案例总数：``$($matrix.caseCount)``")
$lines.Add("- 通过：``$($matrix.passedCaseCount)``")
$lines.Add("- 环境阻塞：``$($matrix.blockedCaseCount)``")
$lines.Add("- synthetic runtime：``$($matrix.syntheticRuntimeCaseCount)``")
$lines.Add("- real-model runtime：``$($matrix.realModelRuntimeCaseCount)``")
$lines.Add("- package-consumer runtime：``$($matrix.packageConsumerRuntimeCaseCount)``")
$lines.Add("- canPublishPublicly：``$($matrix.canPublishPublicly)``")
$lines.Add("- canCloseReleaseIssue：``$($matrix.canCloseReleaseIssue)``")
$lines.Add("")
$lines.Add("| 案例 | TensorRT | CUDA | 工作负载 | 状态 | 证据分类 | 输出匹配 |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- |")
foreach ($case in $cases) {
  $lines.Add("| ``$($case.id)`` | ``$($case.tensorRtVersion)`` | ``$($case.cudaToolkitVersion)`` | $($case.workload) | ``$($case.state)`` | ``$($case.proofClassification)`` | ``$($case.outputMatch)`` |")
}

$lines.Add("")
$lines.Add("## 证据哈希")
$lines.Add("")
foreach ($case in $cases) {
  $lines.Add("### $($case.id)")
  $lines.Add("")
  $lines.Add("- 状态：``$($case.state)``")
  $lines.Add("- 说明：$($case.statusDetail)")
  foreach ($artifact in $case.evidence) {
    $hashText = if ($artifact.exists) { $artifact.sha256 } else { "missing" }
    $lines.Add("- ``$($artifact.kind)``：``$($artifact.path)``；SHA256：``$hashText``")
  }
  if ($case.missingRuntimeAssets.Count -gt 0) {
    $lines.Add("- 缺失 runtime：``$($case.missingRuntimeAssets -join '``、``')``")
  }
  $lines.Add("")
}

$lines.Add("## Proof 边界")
$lines.Add("")
$lines.Add($matrix.boundary)
$lines.Add("")
$lines.Add("本矩阵不执行发布，不提供 owner authorization，也不将 source-tree runtime 证据晋级为 package-consumer-runtime。")

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Multi-version runtime evidence matrix written."
Write-Host "JSON=$jsonPath"
Write-Host "Markdown=$markdownPath"
Write-Host "Cases=$($matrix.caseCount) Passed=$($matrix.passedCaseCount) Blocked=$($matrix.blockedCaseCount)"
Write-Host "Synthetic=$($matrix.syntheticRuntimeCaseCount) RealModel=$($matrix.realModelRuntimeCaseCount) PackageConsumer=$($matrix.packageConsumerRuntimeCaseCount)"
Write-Host "CanPublishPublicly=$($matrix.canPublishPublicly) CanCloseReleaseIssue=$($matrix.canCloseReleaseIssue)"
