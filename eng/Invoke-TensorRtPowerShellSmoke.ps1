[CmdletBinding()]
param(
  [ValidateSet("8", "10", "11")]
  [string]$TensorRtLine = "11",
  [string]$BridgePath,
  [string]$TensorRtRoot,
  [string]$CudaRoot,
  [string]$CudnnRoot,
  [string]$Configuration = "Debug",
  [string]$TargetFramework = "net48",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [string]::IsNullOrWhiteSpace($BridgePath)) {
  $env:JYPPX_NATIVE_BRIDGE_PATH = $BridgePath
}

if (-not [string]::IsNullOrWhiteSpace($TensorRtRoot)) {
  $env:JYPPX_TENSORRT_ROOT = $TensorRtRoot
}

if (-not [string]::IsNullOrWhiteSpace($CudaRoot)) {
  $env:JYPPX_CUDA_ROOT = $CudaRoot
}

if (-not [string]::IsNullOrWhiteSpace($CudnnRoot)) {
  $env:JYPPX_CUDNN_ROOT = $CudnnRoot
}

$sharedAssembly = Join-Path $RepositoryRoot "src\JYPPX.Shared\bin\$Configuration\$TargetFramework\JYPPX.Shared.dll"
$cudaAssembly = Join-Path $RepositoryRoot "src\JYPPX.CudaSharp\bin\$Configuration\$TargetFramework\JYPPX.CudaSharp.dll"
$tensorRtAssembly = Join-Path $RepositoryRoot "src\JYPPX.TensorRtSharp\bin\$Configuration\$TargetFramework\JYPPX.TensorRtSharp.dll"

foreach ($assembly in @($sharedAssembly, $cudaAssembly, $tensorRtAssembly)) {
  if (-not (Test-Path -LiteralPath $assembly)) {
    throw "Assembly was not found: $assembly. Run dotnet build first."
  }

  Add-Type -Path $assembly
}

$line = switch ($TensorRtLine) {
  "8" { [JYPPX.TensorRtSharp.Shared.Interop.TensorRtApiLine]::TensorRt8 }
  "10" { [JYPPX.TensorRtSharp.Shared.Interop.TensorRtApiLine]::TensorRt10 }
  "11" { [JYPPX.TensorRtSharp.Shared.Interop.TensorRtApiLine]::TensorRt11 }
}

$snapshot = [JYPPX.TensorRtSharp.TensorRtEnvironmentProbe]::GetCurrent()
Write-Host "Bridge=$($snapshot.BuildInfo.BridgeName) TensorRT=$($snapshot.BuildInfo.TensorRtVersion) CUDA=$($snapshot.BuildInfo.CudaToolkitVersion) Line=$TensorRtLine"

$logger = [JYPPX.TensorRtSharp.TensorRtLogger]::new($line)
$runtime = [JYPPX.TensorRtSharp.TensorRtRuntime]::new($logger)
$builder = [JYPPX.TensorRtSharp.TensorRtBuilder]::new($logger)
$config = $builder.CreateBuilderConfig()
$trt11DeploymentProbe = "Skipped"
if ($TensorRtLine -eq "11") {
  $trt11ProbeItems = [System.Collections.Generic.List[string]]::new()
  $profile = $builder.CreateOptimizationProfile()
  try {
    $profile.SetShape(
      "input_0",
      [JYPPX.TensorRtSharp.TensorRtDims]::new([int[]]@(1, 1, 1, 1)),
      [JYPPX.TensorRtSharp.TensorRtDims]::new([int[]]@(1, 1, 1, 1)),
      [JYPPX.TensorRtSharp.TensorRtDims]::new([int[]]@(1, 1, 1, 1)))
    $shapeValueCountV2 = $profile.GetShapeValueCountV2("input_0")
    $shapeValuesV2 = $profile.GetShapeValuesV2("input_0", [JYPPX.TensorRtSharp.TensorRtOptimizationProfileSelector]::Min)
    $trt11ProbeItems.Add("ProfileShapeValuesV2=$shapeValueCountV2/$($shapeValuesV2.Count)")
  }
  finally {
    $profile.Dispose()
  }

  $directConfig = $builder.CreateBuilderConfig()
  $directNetwork = $builder.CreateNetwork()
  try {
    $directConfig.SetOptimizationLevel(0)
    $directConfig.ClearFlag([JYPPX.TensorRtSharp.TensorRtBuilderFlag]::Fp16)
    $pluginsSet = $directConfig.SetPluginsToSerialize([string[]]@())
    $directEngine = $builder.BuildEngineWithConfig($directNetwork, $directConfig)
    try {
      $trt11ProbeItems.Add("DirectEngineIOTensors=$($directEngine.IOTensorCount)")
    }
    finally {
      $directEngine.Dispose()
    }
    $trt11ProbeItems.Add("PluginsToSerialize=$pluginsSet")
  }
  finally {
    $directNetwork.Dispose()
    $directConfig.Dispose()
  }

  $kernelConfig = $builder.CreateBuilderConfig()
  $kernelNetwork = $builder.CreateNetwork()
  try {
    $kernelConfig.SetOptimizationLevel(0)
    try {
      $serializedWithKernelText = $builder.BuildSerializedNetworkWithKernelText($kernelNetwork, $kernelConfig)
      try {
        $kernelTextBytes = if ($serializedWithKernelText.KernelText -eq $null) { 0 } else { $serializedWithKernelText.KernelText.SizeInBytes }
        $engineFromKernelPlan = $runtime.Deserialize($serializedWithKernelText.Plan)
        try {
          $trt11ProbeItems.Add("KernelTextPlan=$($serializedWithKernelText.Plan.SizeInBytes)/$($serializedWithKernelText.Plan.DataType)")
          $trt11ProbeItems.Add("KernelTextBytes=$kernelTextBytes")
          $trt11ProbeItems.Add("KernelTextDeserializeIOTensors=$($engineFromKernelPlan.IOTensorCount)")
        }
        finally {
          $engineFromKernelPlan.Dispose()
        }
      }
      finally {
        $serializedWithKernelText.Dispose()
      }
    }
    catch {
      $trt11ProbeItems.Add("KernelText=Blocked:$($_.Exception.GetType().Name):$($_.Exception.Message)")
    }
  }
  finally {
    $kernelNetwork.Dispose()
    $kernelConfig.Dispose()
  }

  $trt11DeploymentProbe = [string]::Join(" ", $trt11ProbeItems)
}
$network = $builder.CreateNetwork()
$hostMemory = $builder.BuildSerializedNetwork($network, $config)

try {
  $serialized = $hostMemory.ToArray()
  $engine = $runtime.Deserialize($hostMemory)
  $context = $engine.CreateExecutionContext()
  $engineFromBytes = $runtime.Deserialize($serialized)

  try {
    $tensors = $engine.GetIOTensors()
    $trt11ContextProbe = "Skipped"
    if ($TensorRtLine -eq "11") {
      $trt11ContextItems = [System.Collections.Generic.List[string]]::new()
      $firstTensor = @($tensors | Select-Object -First 1)
      $firstInput = @($tensors | Where-Object { $_.IOMode -eq [JYPPX.TensorRtSharp.TensorRtIOMode]::Input } | Select-Object -First 1)
      $firstOutput = @($tensors | Where-Object { $_.IOMode -eq [JYPPX.TensorRtSharp.TensorRtIOMode]::Output } | Select-Object -First 1)
      if ($firstTensor.Count -gt 0) {
        $trt11ContextItems.Add("ClearTensor=$($context.ClearTensorAddress($firstTensor[0].Name))")
      }
      if ($firstInput.Count -gt 0) {
        $trt11ContextItems.Add("ClearInput=$($context.ClearInputTensorAddress($firstInput[0].Name))")
      }
      if ($firstOutput.Count -gt 0) {
        $trt11ContextItems.Add("ClearOutput=$($context.ClearOutputTensorAddress($firstOutput[0].Name))")
      }
      $context.ClearDeviceMemory()
      $trt11ContextItems.Add("ClearDeviceMemory=True")
      $trt11ContextItems.Add("ClearInputConsumedEvent=$($context.ClearInputConsumedEvent())")
      $context.SetAuxStreams([JYPPX.CudaSharp.CudaStream[]]@())
      $trt11ContextItems.Add("SetAuxStreams=True")
      $trt11ContextProbe = [string]::Join(" ", $trt11ContextItems)
    }
    $tensorPreview = [string]::Join("; ", @($tensors | ForEach-Object { "$($_.Index):$($_.Name):$($_.IOMode):$($_.DataType):$($_.Shape)" }))
    Write-Host "TensorRtPowerShellSmoke HostMemory=$($hostMemory.SizeInBytes)/$($hostMemory.DataType) SerializedBytes=$($serialized.Length) IOTensors=$($tensors.Count) ByteDeserializeIOTensors=$($engineFromBytes.IOTensorCount) Context=True Trt11Deployment=[$trt11DeploymentProbe] Trt11Context=[$trt11ContextProbe] [$tensorPreview]"
  }
  finally {
    $engineFromBytes.Dispose()
    $context.Dispose()
    $engine.Dispose()
  }
}
finally {
  $hostMemory.Dispose()
  $network.Dispose()
  $config.Dispose()
  $builder.Dispose()
  $runtime.Dispose()
  $logger.Dispose()
}
