[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$TensorRtPackageRoot,
  [string]$CudaToolkitRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
  [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($TensorRtPackageRoot)) {
  $candidate = Join-Path $RepositoryRoot "third_party\nvidia"
  if (Test-Path -LiteralPath $candidate -PathType Container) {
    $TensorRtPackageRoot = $candidate
  }
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\interface-coverage"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function ConvertTo-ApiToken {
  param([string]$Text)

  if ([string]::IsNullOrWhiteSpace($Text)) {
    return ""
  }

  $value = $Text -replace '^I(?=[A-Z])', ''
  $value = $value -creplace '([A-Z]+)([A-Z][a-z])', '$1-$2'
  $value = $value -creplace '([a-z0-9])([A-Z])', '$1-$2'
  $value = $value -replace '_', '-'
  $value = $value -replace '[^A-Za-z0-9]+', '-'
  $value = $value.Trim('-').ToLowerInvariant()
  return $value
}

function Remove-CxxComments {
  param([string]$Text)

  $withoutBlock = [regex]::Replace($Text, '/\*.*?\*/', '', [System.Text.RegularExpressions.RegexOptions]::Singleline)
  $withoutLine = [regex]::Replace($withoutBlock, '(^|[^:])//.*$', '$1', [System.Text.RegularExpressions.RegexOptions]::Multiline)
  return $withoutLine
}

function Get-ManifestApis {
  param([string]$Root)

  $manifestRoot = Join-Path $Root "native\manifests"
  $items = New-Object System.Collections.Generic.List[object]
  if (-not (Test-Path -LiteralPath $manifestRoot -PathType Container)) {
    return $items
  }

  Get-ChildItem -LiteralPath $manifestRoot -Recurse -Filter *.json |
    Where-Object { $_.Name -ne "bridge-api.schema.json" } |
    ForEach-Object {
      $manifestPath = $_.FullName
      $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
      foreach ($api in @($manifest.apis)) {
        $searchText = ConvertTo-ApiToken ("$($api.id) $($api.entryPoint)")
        $items.Add([pscustomobject]@{
          Module = [string]$manifest.module
          VersionLine = [string]$manifest.versionLine
          Id = [string]$api.id
          EntryPoint = [string]$api.entryPoint
          SearchText = $searchText
          ManifestPath = $manifestPath.Substring($Root.Length).TrimStart('\', '/')
        }) | Out-Null
      }
    }

  return $items
}

function Get-NativeExportNames {
  param([string]$Root)

  $nativeRoot = Join-Path $Root "native"
  $names = New-Object 'System.Collections.Generic.HashSet[string]'
  if (-not (Test-Path -LiteralPath $nativeRoot -PathType Container)) {
    return $names
  }

  $pattern = 'jyppx_[A-Za-z0-9_]+'
  Get-ChildItem -LiteralPath $nativeRoot -Recurse -Include *.h,*.hpp,*.cpp,*.inc -File |
    ForEach-Object {
      $text = Get-Content -LiteralPath $_.FullName -Raw -Encoding utf8
      foreach ($match in [regex]::Matches($text, $pattern)) {
        [void]$names.Add($match.Value)
      }
    }

  return $names
}

function Get-ManagedSourceText {
  param([string]$Root, [string]$ProjectFolder)

  $folder = Join-Path $Root $ProjectFolder
  if (-not (Test-Path -LiteralPath $folder -PathType Container)) {
    return ""
  }

  $builder = [System.Text.StringBuilder]::new()
  Get-ChildItem -LiteralPath $folder -Recurse -Filter *.cs |
    Where-Object { $_.FullName -notmatch '[\\/](bin|obj)[\\/]' -and $_.Name -notlike "*.Generated.g.cs" -and $_.FullName -notmatch '[\\/]Generated[\\/]' } |
    ForEach-Object {
      [void]$builder.AppendLine((Get-Content -LiteralPath $_.FullName -Raw -Encoding utf8))
    }

  return $builder.ToString()
}

function Get-TensorRtPackageInfo {
  param([string]$Root)

  $packages = New-Object System.Collections.Generic.List[object]
  if ([string]::IsNullOrWhiteSpace($Root) -or -not (Test-Path -LiteralPath $Root -PathType Container)) {
    return $packages
  }

  Get-ChildItem -LiteralPath $Root -Directory -Filter "TensorRT-*" | ForEach-Object {
    $includeRoot = Join-Path $_.FullName "include"
    if (-not (Test-Path -LiteralPath $includeRoot -PathType Container)) {
      return
    }

    $name = $_.Name
    $line = ""
    $version = ""
    $cuda = ""
    if ($name -match 'TensorRT-(\d+)\.(\d+)\.(\d+)\.(\d+)-cuda\s*([0-9.]+)') {
      $line = $matches[1]
      $version = "$($matches[1]).$($matches[2]).$($matches[3]).$($matches[4])"
      $cuda = $matches[5]
    }
    elseif ($name -match 'TensorRT-(\d+)\.(\d+)') {
      $line = $matches[1]
      $version = "$($matches[1]).$($matches[2])"
    }

    $packages.Add([pscustomobject]@{
      Package = $name
      VersionLine = $line
      Version = $version
      CudaVariant = $cuda
      IncludeRoot = $includeRoot
    }) | Out-Null
  }

  return $packages
}

function Get-TensorRtCategory {
  param([string]$ClassName, [string]$MethodName, [string]$Header)

  if ($Header -like "*NvOnnx*") { return "onnx-parser" }
  switch -Regex ($ClassName) {
    'BuilderConfig|Builder' { return "builder" }
    'Runtime|SerializationConfig|RuntimeConfig' { return "runtime-serialization" }
    'CudaEngine|ExecutionContext|EngineInspector' { return "engine-context" }
    'NetworkDefinition|Tensor|Layer|Loop|IfConditional' { return "network-layer" }
    'OptimizationProfile' { return "optimization-profile" }
    'Refitter' { return "refitter" }
    'Plugin' { return "plugin" }
    'ErrorRecorder|Profiler|Logger' { return "diagnostics" }
    default { return "other" }
  }
}

function Get-TensorRtClassAliases {
  param([string]$ClassName)

  $base = ConvertTo-ApiToken $ClassName
  $aliases = New-Object System.Collections.Generic.List[string]
  $aliases.Add($base) | Out-Null

  switch ($ClassName) {
    "IRuntime" { $aliases.Add("runtime") | Out-Null }
    "IRuntimeConfig" { $aliases.Add("runtime-config") | Out-Null }
    "ISerializationConfig" { $aliases.Add("serialization-config") | Out-Null }
    "IBuilder" { $aliases.Add("builder") | Out-Null }
    "IBuilderConfig" { $aliases.Add("builder-config") | Out-Null; $aliases.Add("config") | Out-Null }
    "INetworkDefinition" { $aliases.Add("network") | Out-Null; $aliases.Add("network-definition") | Out-Null }
    "IOptimizationProfile" { $aliases.Add("optimization-profile") | Out-Null; $aliases.Add("profile") | Out-Null }
    "IHostMemory" { $aliases.Add("host-memory") | Out-Null }
    "ICudaEngine" { $aliases.Add("engine") | Out-Null; $aliases.Add("cuda-engine") | Out-Null }
    "IExecutionContext" { $aliases.Add("execution-context") | Out-Null; $aliases.Add("context") | Out-Null }
    "IEngineInspector" { $aliases.Add("engine-inspector") | Out-Null; $aliases.Add("inspector") | Out-Null }
    "IRefitter" { $aliases.Add("refitter") | Out-Null }
    "IParser" { $aliases.Add("parser") | Out-Null; $aliases.Add("onnx-parser") | Out-Null }
    "IParserError" { $aliases.Add("parser-error") | Out-Null; $aliases.Add("parser") | Out-Null }
    "ITensor" { $aliases.Add("tensor") | Out-Null }
    "ILayer" { $aliases.Add("layer") | Out-Null }
    "ISoftMaxLayer" { $aliases.Add("softmax-layer") | Out-Null; $aliases.Add("softmax") | Out-Null }
    "ITopKLayer" { $aliases.Add("topk-layer") | Out-Null; $aliases.Add("topk") | Out-Null }
    "IElementWiseLayer" { $aliases.Add("elementwise-layer") | Out-Null; $aliases.Add("elementwise") | Out-Null }
    "IMatrixMultiplyLayer" { $aliases.Add("matrix-multiply-layer") | Out-Null; $aliases.Add("matrix-multiply") | Out-Null }
    "IMoELayer" { $aliases.Add("moe-layer") | Out-Null; $aliases.Add("moe") | Out-Null }
    default {
      if ($ClassName -match '^I(.+)Layer$') {
        $layerName = ConvertTo-ApiToken $matches[1]
        $aliases.Add("$layerName-layer") | Out-Null
        $aliases.Add($layerName) | Out-Null
      }
    }
  }

  return @($aliases | Select-Object -Unique)
}

function Get-MethodCandidates {
  param([string]$MethodName, [string]$ClassName = "")

  $candidates = New-Object System.Collections.Generic.List[string]
  $token = ConvertTo-ApiToken $MethodName
  if (-not [string]::IsNullOrWhiteSpace($token)) {
    $candidates.Add($token) | Out-Null
  }

  if ($MethodName -match '^(get|set|is|has|can|create|destroy|add|remove|mark|unmark|reset|clear|build|deserialize|serialize|parse|enqueue|execute|infer|report|refit)(.+)$') {
    $verb = ConvertTo-ApiToken $matches[1]
    $noun = ConvertTo-ApiToken $matches[2]
    if (-not [string]::IsNullOrWhiteSpace($noun)) {
      $candidates.Add($noun) | Out-Null
      $candidates.Add("$verb-$noun") | Out-Null
    }
  }

  if ($MethodName -match '^getNb(.+)$') {
    $noun = ConvertTo-ApiToken $matches[1]
    if (-not [string]::IsNullOrWhiteSpace($noun)) {
      $candidates.Add("$noun-count") | Out-Null
      $candidates.Add("get-$noun-count") | Out-Null
      $candidates.Add("count-$noun") | Out-Null
      if ($noun.EndsWith("s") -and $noun.Length -gt 1) {
        $singular = $noun.Substring(0, $noun.Length - 1)
        $candidates.Add("$singular-count") | Out-Null
        $candidates.Add("get-$singular-count") | Out-Null
        $candidates.Add("count-$singular") | Out-Null
      }
    }
  }

  if ($MethodName -match '^get(.+)V2$') {
    $noun = ConvertTo-ApiToken $matches[1]
    $candidates.Add("$noun-v2") | Out-Null
    $candidates.Add("get-$noun-v2") | Out-Null
  }

  $interfaceKey = if ([string]::IsNullOrWhiteSpace($ClassName)) { $MethodName } else { "$ClassName::$MethodName" }
  $aliasMap = @{
    "IBuilder::createBuilderConfig" = @("create-config", "builder-config-create")
    "IBuilder::createNetworkV2" = @("create-network", "network-create")
    "IBuilder::createOptimizationProfile" = @("create-optimization-profile")
    "IBuilder::buildSerializedNetwork" = @("build-serialized-network", "serialized-build")
    "IRuntime::deserializeCudaEngine" = @("runtime-deserialize-engine")
    "IParser::parse" = @("parse-from-memory", "parse-memory")
    "IParser::parseFromFile" = @("parse-from-file")
    "IParser::getNbErrors" = @("error-count", "get-error-count")
    "IParser::getError" = @("get-error", "error")
    "IParser::clearErrors" = @("clear-errors")
    "IParser::supportsOperator" = @("supports-operator")
    "IParser::getSubgraphNodes" = @("get-subgraph-node-count", "get-subgraph-node", "subgraph-node", "subgraph-nodes")
    "IParser::getUsedVCPluginLibraries" = @("get-used-vc-plugin-library-count", "get-used-vc-plugin-library", "used-vc-plugin-library", "used-vc-plugin-libraries")
    "ICudaEngine::getNbIOTensors" = @("io-tensor-count", "get-io-tensor-count")
    "ICudaEngine::getTensorName" = @("io-tensor-name", "tensor-name")
    "ICudaEngine::getTensorIOMode" = @("tensor-io-mode", "io-mode")
    "ICudaEngine::getTensorDataType" = @("tensor-data-type")
    "ICudaEngine::getTensorShape" = @("tensor-shape")
    "ICudaEngine::getTensorLocation" = @("tensor-location")
    "ICudaEngine::getTensorBytesPerComponent" = @("tensor-bytes-per-component")
    "ICudaEngine::getTensorBytesPerComponentV2" = @("tensor-bytes-per-component-v2", "tensor-bytes-per-component")
    "ICudaEngine::getTensorComponentsPerElement" = @("tensor-components-per-element")
    "ICudaEngine::getTensorComponentsPerElementV2" = @("tensor-components-per-element-v2", "tensor-components-per-element")
    "ICudaEngine::getTensorFormat" = @("tensor-format")
    "ICudaEngine::getTensorFormatV2" = @("tensor-format-v2", "tensor-format")
    "ICudaEngine::getTensorFormatDesc" = @("tensor-format-desc", "tensor-format-description")
    "ICudaEngine::getTensorFormatDescV2" = @("tensor-format-desc-v2", "tensor-format-desc")
    "ICudaEngine::getTensorVectorizedDim" = @("tensor-vectorized-dim")
    "ICudaEngine::getTensorVectorizedDimV2" = @("tensor-vectorized-dim-v2", "tensor-vectorized-dim")
    "ICudaEngine::getNbOptimizationProfiles" = @("optimization-profile-count", "get-optimization-profile-count")
    "ICudaEngine::getNbLayers" = @("layer-count", "get-layer-count")
    "ICudaEngine::getDeviceMemorySize" = @("device-memory-size")
    "ICudaEngine::getDeviceMemorySizeV2" = @("device-memory-size-v2")
    "ICudaEngine::getDeviceMemorySizeForProfile" = @("device-memory-size-for-profile")
    "ICudaEngine::getDeviceMemorySizeForProfileV2" = @("device-memory-size-for-profile-v2")
    "ICudaEngine::getProfileShape" = @("profile-shape")
    "ICudaEngine::getProfileDimensions" = @("profile-shape", "profile-dimensions")
    "ICudaEngine::getEngineCapability" = @("engine-capability")
    "ICudaEngine::getTacticSources" = @("tactic-sources")
    "ICudaEngine::getProfilingVerbosity" = @("profiling-verbosity")
    "ICudaEngine::createExecutionContext" = @("create-execution-context")
    "ICudaEngine::createExecutionContextWithoutDeviceMemory" = @("create-execution-context-without-device-memory")
    "ICudaEngine::serialize" = @("engine-serialize", "serialize-engine")
    "IHostMemory::data" = @("host-memory-copy-to-buffer", "copy-to-buffer")
    "ITimingCache::combine" = @("timing-cache-combine", "combine")
    "ITimingCache::reset" = @("timing-cache-reset", "reset")
    "ITimingCache::queryKeys" = @("timing-cache-query-key-count", "timing-cache-copy-keys", "query-keys")
    "ITimingCache::query" = @("timing-cache-query", "query")
    "ITimingCache::update" = @("timing-cache-update", "update")
    "Global::getNvOnnxParserVersion" = @("global-get-onnx-parser-version")
    "IExecutionContext::setInputShape" = @("set-input-shape", "input-shape")
    "IExecutionContext::getTensorShape" = @("tensor-shape")
    "IExecutionContext::getTensorStrides" = @("tensor-strides")
    "IExecutionContext::setTensorAddress" = @("set-tensor-address", "tensor-address")
    "IExecutionContext::setInputTensorAddress" = @("set-input-tensor-address")
    "IExecutionContext::setOutputTensorAddress" = @("set-output-tensor-address")
    "IExecutionContext::enqueueV3" = @("execution-context-enqueue-async")
    "IExecutionContext::setOptimizationProfileAsync" = @("set-optimization-profile-async", "optimization-profile-async")
    "IExecutionContext::getOptimizationProfile" = @("optimization-profile")
    "IExecutionContext::inferShapes" = @("infer-shapes")
    "IExecutionContext::allInputDimensionsSpecified" = @("all-input-dimensions-specified")
    "IExecutionContext::allInputShapesSpecified" = @("all-input-shapes-specified")
    "IExecutionContext::setEnqueueEmitsProfile" = @("set-enqueue-emits-profile")
    "IExecutionContext::getEnqueueEmitsProfile" = @("get-enqueue-emits-profile", "enqueue-emits-profile")
    "IExecutionContext::reportToProfiler" = @("report-to-profiler")
    "IExecutionContext::setDeviceMemory" = @("set-device-memory")
    "IExecutionContext::getDeviceMemorySize" = @("device-memory-size")
    "IExecutionContext::updateDeviceMemorySizeForShapes" = @("update-device-memory-size-for-shapes")
    "IExecutionContext::setPersistentCacheLimit" = @("set-persistent-cache-limit")
    "IExecutionContext::getPersistentCacheLimit" = @("get-persistent-cache-limit", "persistent-cache-limit")
    "IExecutionContext::setInputConsumedEvent" = @("set-input-consumed-event")
    "INetworkDefinition::addElementWise" = @("add-elementwise", "elementwise")
    "INetworkDefinition::addSoftMax" = @("add-softmax", "softmax")
    "INetworkDefinition::addTopK" = @("add-topk", "topk")
    "INetworkDefinition::addTopKV2" = @("add-topk-v2", "topk-v2")
    "INetworkDefinition::addRaggedSoftMax" = @("add-ragged-softmax", "ragged-softmax")
    "INetworkDefinition::addNMSV2" = @("add-nms", "nms")
    "INetworkDefinition::addNonZeroV2" = @("add-non-zero", "non-zero")
    "INetworkDefinition::addParametricReLU" = @("add-parametric-relu", "parametric-relu")
    "INetworkDefinition::addMoE" = @("add-moe", "moe")
    "INetworkDefinition::setWeightsName" = @("id:*network-set-weights-name")
    "IRefitter::getMissing" = @("missing-entries", "get-missing-entries")
    "IRefitter::getAll" = @("all-entries", "get-all-entries")
    "IRefitter::getMissingWeights" = @("missing-entries", "missing-count")
    "IRefitter::getAllWeights" = @("all-entries", "all-count")
    "IRefitter::getTensorsWithDynamicRange" = @("dynamic-range-tensor-count", "dynamic-range-tensor-entries")
    "IRefitter::setWeights" = @("set-weights")
    "IRefitter::refitCudaEngine" = @("refit-cuda-engine")
    "IReduceLayer::getReduceAxes" = @("axes", "get-axes")
    "IReduceLayer::setReduceAxes" = @("set-axes", "axes")
    "ITopKLayer::getReduceAxes" = @("axes", "get-axes")
    "ITopKLayer::setReduceAxes" = @("set-axes", "axes")
    "IGatherLayer::getGatherAxis" = @("axis", "get-axis")
    "IGatherLayer::setGatherAxis" = @("set-axis", "axis")
    "IGatherLayer::getNbElementWiseDims" = @("get-nb-elementwise-dims", "nb-elementwise-dims", "get-elementwise-dims", "elementwise-dims")
    "IGatherLayer::setNbElementWiseDims" = @("set-nb-elementwise-dims", "nb-elementwise-dims", "set-elementwise-dims", "elementwise-dims")
    "INMSLayer::getTopKBoxLimit" = @("get-topk-box-limit", "topk-box-limit")
    "INMSLayer::setTopKBoxLimit" = @("set-topk-box-limit", "topk-box-limit")
    "IOptimizationProfile::getDimensions" = @("get-shape", "profile-shape")
    "IOptimizationProfile::setDimensions" = @("set-shape", "profile-shape")
    "IActivationLayer::getActivationType" = @("type", "get-type", "activation-type")
    "IActivationLayer::setActivationType" = @("set-type", "type", "activation-type")
    "IPoolingLayer::getPoolingType" = @("type", "get-type", "pooling-type")
    "IPoolingLayer::setPoolingType" = @("set-type", "type", "pooling-type")
  }

  if ($aliasMap.ContainsKey($interfaceKey)) {
    foreach ($alias in $aliasMap[$interfaceKey]) {
      $candidates.Add($alias) | Out-Null
    }
  }

  return @($candidates | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
}

function Find-ExplicitTensorRtInterfaceAliasApis {
  param(
    [object[]]$ManifestApis,
    [string]$Module,
    [string]$VersionLine,
    [string]$InterfaceKey
  )

  $aliasMap = @{
    "Global::createInferBuilder_INTERNAL" = @("id:*builder-create")
    "Global::createInferRuntime_INTERNAL" = @("id:*runtime-create")
    "Global::createInferRefitter_INTERNAL" = @("id:*engine-create-refitter")
    "Global::createNvOnnxParser_INTERNAL" = @("id:*onnx-parser-create")
    "Global::createNvOnnxParserRefitter_INTERNAL" = @("id:*parser-refitter-create", "id:*parser-refitter-create-deferred")
    "Global::createONNXConfig" = @("id:*onnx-config-create")
    "Global::initLibNvInferPlugins" = @("id:*global-init-lib-nvinfer-plugins", "id:*global-init-lib-nvinfer-plugins-deferred")
    "Global::setInternalLibraryPath" = @("id:*global-set-internal-library-path-deferred")
    "Global::getBuilderPluginRegistry" = @("id:*builder-capability-plugin-registry-exists")
    "Global::getPluginRegistry" = @("id:*global-plugin-registry-exists")
    "IBuilder::getPluginRegistry" = @("id:*builder-plugin-registry-exists", "id:*builder-plugin-registry-get-creator-count", "id:*builder-plugin-registry-get-recursive-creator-count", "id:*builder-plugin-registry-has-error-recorder", "id:*builder-plugin-registry-is-parent-search-enabled", "id:*builder-plugin-creator-get-name", "id:*builder-plugin-creator-get-version", "id:*builder-plugin-creator-get-namespace", "id:*builder-plugin-creator-get-interface-info", "id:*builder-plugin-creator-get-field-count", "id:*builder-plugin-creator-get-field-name", "id:*builder-plugin-creator-get-field-metadata", "id:*builder-plugin-creator-lookup", "id:*builder-get-plugin-registry-deferred")
    "IRuntime::getPluginRegistry" = @("id:*runtime-plugin-registry-exists", "id:*runtime-plugin-registry-get-creator-count", "id:*runtime-plugin-registry-has-error-recorder", "id:*runtime-plugin-registry-is-parent-search-enabled", "id:*runtime-plugin-creator-get-name", "id:*runtime-plugin-creator-get-version", "id:*runtime-plugin-creator-get-namespace", "id:*runtime-plugin-creator-get-interface-info", "id:*runtime-plugin-creator-get-field-count", "id:*runtime-plugin-creator-get-field-name", "id:*runtime-plugin-creator-get-field-metadata", "id:*runtime-plugin-creator-lookup", "id:*runtime-get-plugin-registry-deferred")
    "IRuntime::deserializeCudaEngine" = @("id:*runtime-deserialize-engine")
    "IRuntime::destroy" = @("id:*trt-object-destroy")
    "IRefitter::destroy" = @("id:*trt-object-destroy")
    "IParser::destroy" = @("id:*trt-object-destroy")
    "IParser::setBuilderConfig" = @("id:*onnx-parser-set-builder-config-safe")
    "IOnnxConfig::destroy" = @("id:*trt-object-destroy")
    "INetworkDefinition::destroy" = @("id:*trt-object-destroy")
    "IHostMemory::destroy" = @("id:*trt-object-destroy")
    "IBuilder::destroy" = @("id:*trt-object-destroy", "id:*builder-destroy-deferred")
    "IBuilderConfig::destroy" = @("id:*trt-object-destroy", "id:*builder-config-destroy-deferred")
    "ICudaEngine::destroy" = @("id:*trt-object-destroy", "id:*cuda-engine-destroy-deferred")
    "IExecutionContext::destroy" = @("id:*trt-object-destroy", "id:*execution-context-destroy-deferred")
    "IExecutionContext::setAuxStreams" = @("id:*execution-context-set-aux-streams")
    "IBuilder::buildEngineWithConfig" = @("id:*builder-build-engine-with-config-owner-safe", "id:*builder-build-engine-with-config", "id:*builder-build-engine-with-config-deferred")
    "IGpuAllocator::free" = @("id:*gpu-allocator-free*")
    "IAlgorithm::getTimingMSec" = @("id:*algorithm-get-timing-msec*")
    "IBuilder::getErrorRecorder" = @("id:*builder-has-error-recorder", "id:*builder-clear-error-recorder")
    "IBuilder::getLogger" = @("id:*builder-has-logger", "id:*builder-get-logger-deferred")
    "IBuilder::setErrorRecorder" = @("id:*builder-has-error-recorder", "id:*builder-clear-error-recorder")
    "IBuilder::setGpuAllocator" = @("id:*builder-clear-gpu-allocator")
    "ICudaEngine::getErrorRecorder" = @("id:*engine-has-error-recorder", "id:*engine-clear-error-recorder")
    "ICudaEngine::setErrorRecorder" = @("id:*engine-has-error-recorder", "id:*engine-clear-error-recorder")
    "IEngineInspector::getErrorRecorder" = @("id:*engine-inspector-has-error-recorder", "id:*engine-inspector-clear-error-recorder")
    "IEngineInspector::setErrorRecorder" = @("id:*engine-inspector-has-error-recorder", "id:*engine-inspector-clear-error-recorder")
    "IExecutionContext::getErrorRecorder" = @("id:*execution-context-has-error-recorder", "id:*execution-context-clear-error-recorder")
    "IExecutionContext::setErrorRecorder" = @("id:*execution-context-has-error-recorder", "id:*execution-context-clear-error-recorder")
    "INetworkDefinition::getErrorRecorder" = @("id:*network-has-error-recorder", "id:*network-clear-error-recorder")
    "INetworkDefinition::setErrorRecorder" = @("id:*network-has-error-recorder", "id:*network-clear-error-recorder")
    "IPluginRegistry::getErrorRecorder" = @("id:*plugin-registry-has-error-recorder", "id:*plugin-registry-get-error-recorder-deferred")
    "IErrorRecorder::getNbErrors" = @("id:*runtime-get-error-recorder-snapshot-info", "id:*refitter-get-error-recorder-snapshot-info", "id:*builder-get-error-recorder-snapshot-info", "id:*network-get-error-recorder-snapshot-info", "id:*engine-inspector-get-error-recorder-snapshot-info", "id:*error-recorder-get-nb-errors-deferred")
    "IErrorRecorder::getErrorCode" = @("id:*runtime-get-error-recorder-error", "id:*refitter-get-error-recorder-error", "id:*builder-get-error-recorder-error", "id:*network-get-error-recorder-error", "id:*engine-inspector-get-error-recorder-error", "id:*error-recorder-get-error-code-deferred")
    "IErrorRecorder::getErrorDesc" = @("id:*runtime-get-error-recorder-error", "id:*refitter-get-error-recorder-error", "id:*builder-get-error-recorder-error", "id:*network-get-error-recorder-error", "id:*engine-inspector-get-error-recorder-error", "id:*error-recorder-get-error-desc-deferred")
    "IErrorRecorder::getInterfaceInfo" = @("id:*runtime-get-error-recorder-snapshot-info", "id:*refitter-get-error-recorder-snapshot-info", "id:*engine-get-error-recorder-snapshot-info", "id:*execution-context-get-error-recorder-snapshot-info", "id:*builder-get-error-recorder-snapshot-info", "id:*network-get-error-recorder-snapshot-info", "id:*engine-inspector-get-error-recorder-snapshot-info", "id:*error-recorder-get-interface-info-deferred")
    "IErrorRecorder::hasOverflowed" = @("id:*runtime-get-error-recorder-snapshot-info", "id:*refitter-get-error-recorder-snapshot-info", "id:*builder-get-error-recorder-snapshot-info", "id:*network-get-error-recorder-snapshot-info", "id:*engine-inspector-get-error-recorder-snapshot-info", "id:*error-recorder-has-overflowed-deferred")
    "IErrorRecorder::EnumMax" = @("id:*error-code-enum-max-metadata", "id:*error-recorder-enum-max-deferred")
    "IBuilderConfig::getAvgTimingIterations" = @("id:*builder-config-get-average-timing-iterations")
    "IBuilderConfig::setAvgTimingIterations" = @("id:*builder-config-set-average-timing-iterations")
    "IBuilderConfig::getBuilderOptimizationLevel" = @("id:*builder-config-get-optimization-level")
    "IBuilderConfig::setBuilderOptimizationLevel" = @("id:*builder-config-set-optimization-level")
    "IBuilderConfig::getFlags" = @("id:*builder-config-get-flags", "id:*builder-config-get-flags-deferred")
    "IBuilderConfig::getAlgorithmSelector" = @("id:*builder-config-has-algorithm-selector", "id:*builder-config-get-algorithm-selector-deferred")
    "IBuilderConfig::getInt8Calibrator" = @("id:*builder-config-has-int8-calibrator", "id:*builder-config-get-int8-calibrator-deferred")
    "IBuilderConfig::getPluginToSerialize" = @("id:*builder-config-get-plugin-to-serialize-caller-buffer", "id:*builder-config-get-plugin-to-serialize", "id:*builder-config-get-plugin-to-serialize-v2", "id:*builder-config-get-plugin-to-serialize-deferred")
    "IBuilderConfig::setPluginsToSerialize" = @("id:*builder-config-set-plugins-to-serialize-copied-paths", "id:*builder-config-set-plugins-to-serialize", "id:*builder-config-set-plugins-to-serialize-v2", "id:*builder-config-set-plugins-to-serialize-deferred")
    "IRuntime::getLogger" = @("id:*runtime-has-logger", "id:*runtime-get-logger-deferred")
    "IParser::getNbErrors" = @("id:*onnx-parser-get-error-count", "id:*parser-get-nb-errors")
    "IParser::getNbSubgraphs" = @("id:*onnx-parser-get-subgraph-count")
    "IParser::getLayerOutputTensor" = @("id:*onnx-parser-layer-output-tensor-exists", "id:*parser-get-layer-output-tensor-deferred")
    "IParser::getSubgraphNodes" = @("id:*onnx-parser-get-subgraph-node-count", "id:*onnx-parser-get-subgraph-node")
    "IParser::getUsedVCPluginLibraries" = @("id:*onnx-parser-get-used-vc-plugin-library-count", "id:*onnx-parser-get-used-vc-plugin-library")
    "IParser::parseWithWeightDescriptors" = @("id:*onnx-parser-parse-with-weight-descriptors", "id:*parser-parse-with-weight-descriptors-deferred")
    "IParserRefitter::getNbErrors" = @("id:*parser-refitter-get-error-count")
    "IExecutionContext::getOutputAllocator" = @("id:*execution-context-has-output-allocator", "id:*execution-context-clear-output-allocator")
    "IExecutionContext::setOutputAllocator" = @("id:*execution-context-has-output-allocator", "id:*execution-context-clear-output-allocator")
    "IExecutionContext::getTemporaryStorageAllocator" = @("id:*execution-context-has-temporary-storage-allocator", "id:*execution-context-clear-temporary-storage-allocator")
    "IExecutionContext::setTemporaryStorageAllocator" = @("id:*execution-context-has-temporary-storage-allocator", "id:*execution-context-clear-temporary-storage-allocator")
    "IExecutionContext::getDebugListener" = @("id:*execution-context-has-debug-listener", "id:*execution-context-clear-debug-listener")
    "IExecutionContext::setDebugListener" = @("id:*execution-context-has-debug-listener", "id:*execution-context-clear-debug-listener")
    "IExecutionContext::setInputShapeBinding" = @("id:*execution-context-set-input-shape-binding-copied-values", "id:*execution-context-set-input-shape-binding-deferred")
    "IExecutionContext::execute" = @("id:*execution-context-execute-legacy-safe")
    "IExecutionContext::executeV2" = @("id:*execution-context-execute-v2-safe")
    "IExecutionContext::enqueueV2" = @("id:*execution-context-enqueue-v2-safe")
    "IExecutionContext::enqueueV3" = @("id:*execution-context-enqueue-async")
    "IBuilderConfig::getProgressMonitor" = @("id:*builder-config-has-progress-monitor", "id:*builder-config-clear-progress-monitor", "id:*builder-config-set-progress-monitor")
    "IBuilderConfig::setProgressMonitor" = @("id:*builder-config-has-progress-monitor", "id:*builder-config-clear-progress-monitor", "id:*builder-config-set-progress-monitor")
    "IDebugListener::getInterfaceInfo" = @("id:*execution-context-debug-listener-get-interface-info", "id:*debug-listener-get-interface-info-deferred")
    "IGpuAllocator::getInterfaceInfo" = @("id:*execution-context-temporary-storage-allocator-get-interface-info", "id:*gpu-allocator-get-interface-info-deferred")
    "IProfiler::reportLayerTime" = @("id:*profiler-create-with-callback", "id:*profiler-emit-diagnostic", "id:*execution-context-set-profiler", "id:*profiler-report-layer-time-deferred")
    "IOutputAllocator::getInterfaceInfo" = @("id:*execution-context-output-allocator-get-interface-info", "id:*output-allocator-get-interface-info-deferred")
    "IProgressMonitor::getInterfaceInfo" = @("id:*progress-monitor-get-interface-info")
    "IProgressMonitor::phaseStart" = @("id:*progress-monitor-create-with-callback", "id:*progress-monitor-emit-diagnostic", "id:*builder-config-set-progress-monitor", "id:*progress-monitor-phase-start-deferred")
    "IProgressMonitor::stepComplete" = @("id:*progress-monitor-create-with-callback", "id:*progress-monitor-emit-diagnostic", "id:*builder-config-set-progress-monitor", "id:*progress-monitor-step-complete-deferred")
    "IProgressMonitor::phaseFinish" = @("id:*progress-monitor-create-with-callback", "id:*progress-monitor-emit-diagnostic", "id:*builder-config-set-progress-monitor", "id:*progress-monitor-phase-finish-deferred")
    "INetworkDefinition::addRNNv2" = @("id:*network-add-rnnv2*")
    "IRNNv2Layer::getBiasForGate" = @("id:*rnnv2-layer-get-bias-for-gate*", "id:*rnn-v2-layer-get-bias-for-gate*")
    "IRNNv2Layer::getCellState" = @("id:*rnnv2-layer-get-cell-state*", "id:*rnn-v2-layer-get-cell-state*")
    "IRNNv2Layer::getDataLength" = @("id:*rnnv2-layer-get-data-length*", "id:*rnn-v2-layer-get-data-length*")
    "IRNNv2Layer::getDirection" = @("id:*rnnv2-layer-get-direction*", "id:*rnn-v2-layer-get-direction*")
    "IRNNv2Layer::getHiddenSize" = @("id:*rnnv2-layer-get-hidden-size*", "id:*rnn-v2-layer-get-hidden-size*")
    "IRNNv2Layer::getHiddenState" = @("id:*rnnv2-layer-get-hidden-state*", "id:*rnn-v2-layer-get-hidden-state*")
    "IRNNv2Layer::getInputMode" = @("id:*rnnv2-layer-get-input-mode*", "id:*rnn-v2-layer-get-input-mode*")
    "IRNNv2Layer::getLayerCount" = @("id:*rnnv2-layer-get-layer-count*", "id:*rnn-v2-layer-get-layer-count*")
    "IRNNv2Layer::getMaxSeqLength" = @("id:*rnnv2-layer-get-max-seq-length*", "id:*rnn-v2-layer-get-max-seq-length*")
    "IRNNv2Layer::getOperation" = @("id:*rnnv2-layer-get-operation*", "id:*rnn-v2-layer-get-operation*")
    "IRNNv2Layer::getSequenceLengths" = @("id:*rnnv2-layer-get-sequence-lengths*", "id:*rnn-v2-layer-get-sequence-lengths*")
    "IRNNv2Layer::getWeightsForGate" = @("id:*rnnv2-layer-get-weights-for-gate*", "id:*rnn-v2-layer-get-weights-for-gate*")
    "IRNNv2Layer::setBiasForGate" = @("id:*rnnv2-layer-set-bias-for-gate*", "id:*rnn-v2-layer-set-bias-for-gate*")
    "IRNNv2Layer::setCellState" = @("id:*rnnv2-layer-set-cell-state*", "id:*rnn-v2-layer-set-cell-state*")
    "IRNNv2Layer::setDirection" = @("id:*rnnv2-layer-set-direction*", "id:*rnn-v2-layer-set-direction*")
    "IRNNv2Layer::setHiddenState" = @("id:*rnnv2-layer-set-hidden-state*", "id:*rnn-v2-layer-set-hidden-state*")
    "IRNNv2Layer::setInputMode" = @("id:*rnnv2-layer-set-input-mode*", "id:*rnn-v2-layer-set-input-mode*")
    "IRNNv2Layer::setOperation" = @("id:*rnnv2-layer-set-operation*", "id:*rnn-v2-layer-set-operation*")
    "IRNNv2Layer::setSequenceLengths" = @("id:*rnnv2-layer-set-sequence-lengths*", "id:*rnn-v2-layer-set-sequence-lengths*")
    "IRNNv2Layer::setWeightsForGate" = @("id:*rnnv2-layer-set-weights-for-gate*", "id:*rnn-v2-layer-set-weights-for-gate*")
    "IMoELayer::setInput" = @("id:*layer-set-input")
    "IRefitter::getTensorsWithDynamicRange" = @("id:*refitter-get-dynamic-range-tensor-count", "id:*refitter-get-dynamic-range-tensor-entries")
    "IRefitter::getLogger" = @("id:*refitter-has-logger*", "id:*refitter-get-logger-deferred")
    "IPluginRegistry::getBuilderSafePluginRegistry" = @("id:*builder-safe-plugin-registry-exists", "id:*builder-capability-plugin-registry-exists")
    "IPluginRegistry::setParentSearchEnabled" = @("id:*global-plugin-registry-set-parent-search-enabled")
    "IPluginV2Ext::getTensorRTVersion" = @("id:*plugin-v2-layer-get-tensor-rt-version")
    "IPluginV2IOExt::getTensorRTVersion" = @("id:*plugin-v2-layer-get-tensor-rt-version")
    "IPluginV2DynamicExt::supportsFormatCombination" = @("id:*plugin-v2-dynamic-ext-supports-format-combination-owner-scoped")
    "IPluginV2DynamicExt::canBroadcastInputAcrossBatch" = @("id:*plugin-v2-ext-can-broadcast-input-across-batch-owner-scoped", "id:*plugin-v2-dynamic-ext-can-broadcast-input-across-batch-deferred")
    "IPluginV2DynamicExt::isOutputBroadcastAcrossBatch" = @("id:*plugin-v2-ext-is-output-broadcast-across-batch-owner-scoped", "id:*plugin-v2-dynamic-ext-is-output-broadcast-across-batch-deferred")
    "IPluginV2IOExt::supportsFormatCombination" = @("id:*plugin-v2-io-ext-supports-format-combination-owner-scoped")
    "IPluginV3OneBuild::getAliasedInput" = @("id:*plugin-v3-one-build-get-aliased-input-owner-scoped")
    "IPluginV3OneBuild::getOutputDataTypes" = @("id:*plugin-v3-one-build-get-output-data-types-owner-scoped")
    "IPluginV3OneBuild::supportsFormatCombination" = @("id:*plugin-v3-one-build-supports-format-combination-owner-scoped")
    "IPluginV3OneRuntime::getFieldsToSerialize" = @("id:*plugin-v3-one-runtime-get-fields-to-serialize-*-owner-scoped")
    "IVersionedInterface::getAPILanguage" = @("id:*versioned-metadata", "id:*get-api-language")
    "IVersionedInterface::getInterfaceInfo" = @("id:*versioned-metadata", "id:*get-interface-info")
    "IPluginCreator::getPluginName" = @("id:*builder-capability-plugin-creator-get-name", "id:*builder-plugin-creator-get-name", "id:*global-plugin-creator-get-name", "id:*runtime-plugin-creator-get-name")
    "IPluginCreator::getPluginVersion" = @("id:*builder-capability-plugin-creator-get-version", "id:*builder-plugin-creator-get-version", "id:*global-plugin-creator-get-version", "id:*runtime-plugin-creator-get-version")
    "IPluginCreator::getPluginNamespace" = @("id:*builder-capability-plugin-creator-get-namespace", "id:*builder-plugin-creator-get-namespace", "id:*global-plugin-creator-get-namespace", "id:*runtime-plugin-creator-get-namespace")
    "IPluginCreator::getFieldNames" = @("id:*builder-capability-plugin-creator-get-field-count", "id:*builder-capability-plugin-creator-get-field-name", "id:*builder-capability-plugin-creator-get-field-metadata", "id:*builder-plugin-creator-get-field-count", "id:*builder-plugin-creator-get-field-name", "id:*builder-plugin-creator-get-field-metadata", "id:*global-plugin-creator-get-field-count", "id:*global-plugin-creator-get-field-name", "id:*global-plugin-creator-get-field-metadata", "id:*runtime-plugin-creator-get-field-count", "id:*runtime-plugin-creator-get-field-name", "id:*runtime-plugin-creator-get-field-metadata")
    "IPluginCreator::getInterfaceInfo" = @("id:*builder-capability-plugin-creator-get-interface-info", "id:*builder-plugin-creator-get-interface-info", "id:*global-plugin-creator-get-interface-info", "id:*runtime-plugin-creator-get-interface-info")
    "IPluginCreatorV3One::getPluginName" = @("id:*builder-capability-plugin-creator-get-name", "id:*builder-plugin-creator-get-name", "id:*global-plugin-creator-get-name", "id:*runtime-plugin-creator-get-name")
    "IPluginCreatorV3One::getPluginVersion" = @("id:*builder-capability-plugin-creator-get-version", "id:*builder-plugin-creator-get-version", "id:*global-plugin-creator-get-version", "id:*runtime-plugin-creator-get-version")
    "IPluginCreatorV3One::getPluginNamespace" = @("id:*builder-capability-plugin-creator-get-namespace", "id:*builder-plugin-creator-get-namespace", "id:*global-plugin-creator-get-namespace", "id:*runtime-plugin-creator-get-namespace")
    "IPluginCreatorV3One::getFieldNames" = @("id:*builder-capability-plugin-creator-get-field-count", "id:*builder-capability-plugin-creator-get-field-name", "id:*builder-capability-plugin-creator-get-field-metadata", "id:*builder-plugin-creator-get-field-count", "id:*builder-plugin-creator-get-field-name", "id:*builder-plugin-creator-get-field-metadata", "id:*global-plugin-creator-get-field-count", "id:*global-plugin-creator-get-field-name", "id:*global-plugin-creator-get-field-metadata", "id:*runtime-plugin-creator-get-field-count", "id:*runtime-plugin-creator-get-field-name", "id:*runtime-plugin-creator-get-field-metadata")
    "IPluginCreatorV3One::getInterfaceInfo" = @("id:*builder-capability-plugin-creator-get-interface-info", "id:*builder-plugin-creator-get-interface-info", "id:*global-plugin-creator-get-interface-info", "id:*runtime-plugin-creator-get-interface-info")
    "IPluginRegistry::getPluginCreator" = @("id:*plugin-creator-lookup", "id:*plugin-creator-lookup-get-interface-info", "id:*plugin-creator-lookup-get-field-count", "id:*plugin-creator-lookup-get-field-name", "id:*plugin-creator-lookup-get-field-metadata")
    "IPluginRegistry::getPluginCreatorList" = @("id:*plugin-registry-get-creator-count", "id:*plugin-creator-get-name", "id:*plugin-creator-get-version", "id:*plugin-creator-get-namespace", "id:*plugin-creator-get-interface-info", "id:*plugin-creator-get-field-count", "id:*plugin-creator-get-field-name", "id:*plugin-creator-get-field-metadata", "id:*plugin-registry-get-plugin-creator-list-deferred")
    "IPluginRegistry::getAllCreators" = @("id:*plugin-registry-get-creator-count", "id:*plugin-creator-get-name", "id:*plugin-creator-get-version", "id:*plugin-creator-get-namespace", "id:*plugin-creator-get-interface-info", "id:*plugin-creator-get-field-count")
    "IPluginRegistry::getAllCreatorsRecursive" = @("id:*plugin-registry-get-recursive-creator-count")
  }

  $deferredHistoryAliasMap = @{
    "Global::createONNXConfig" = @("id:*onnx-config-create-deferred", "id:*global-create-onnx-config-deferred")
    "Global::getBuilderPluginRegistry" = @("id:*global-get-builder-plugin-registry-deferred")
    "Global::getPluginRegistry" = @("id:*global-get-plugin-registry-deferred")
    "IExecutionContext::execute" = @("id:*execution-context-execute-deferred")
    "IExecutionContext::executeV2" = @("id:*execution-context-execute-v2-deferred")
    "IExecutionContext::enqueueV2" = @("id:*execution-context-enqueue-v2-deferred")
    "IExecutionContext::setAuxStreams" = @("id:*execution-context-set-aux-streams-deferred")
    "IPluginRegistry::getBuilderSafePluginRegistry" = @("id:*plugin-registry-get-builder-safe-plugin-registry-deferred")
    "IPluginRegistry::setParentSearchEnabled" = @("id:*plugin-registry-set-parent-search-enabled-deferred")
    "IPluginV2Ext::getTensorRTVersion" = @("id:*plugin-v2-ext-get-tensor-rt-version-deferred")
    "IPluginV2IOExt::getTensorRTVersion" = @("id:*plugin-v2-io-ext-get-tensor-rt-version-deferred")
    "IPluginV2DynamicExt::supportsFormatCombination" = @("id:*plugin-v2-dynamic-ext-supports-format-combination-deferred")
    "IPluginV2IOExt::supportsFormatCombination" = @("id:*plugin-v2-io-ext-supports-format-combination-deferred")
    "IPluginV3OneBuild::getAliasedInput" = @("id:*plugin-v3-one-build-get-aliased-input-deferred")
    "IPluginV3OneBuild::getOutputDataTypes" = @("id:*plugin-v3-one-build-get-output-data-types-deferred")
    "IPluginV3OneBuild::supportsFormatCombination" = @("id:*plugin-v3-one-build-supports-format-combination-deferred")
    "IPluginV3OneRuntime::getFieldsToSerialize" = @("id:*plugin-v3-one-runtime-get-fields-to-serialize-deferred")
    "IOnnxConfig::destroy" = @("id:*onnx-config-destroy-deferred")
    "IParser::setBuilderConfig" = @("id:*parser-set-builder-config-deferred")
    "IBuilderConfig::getAvgTimingIterations" = @("id:*builder-config-get-avg-timing-iterations-deferred")
    "IBuilderConfig::setAvgTimingIterations" = @("id:*builder-config-set-avg-timing-iterations-deferred")
    "IBuilderConfig::getBuilderOptimizationLevel" = @("id:*builder-config-get-builder-optimization-level-deferred")
    "IBuilderConfig::setBuilderOptimizationLevel" = @("id:*builder-config-set-builder-optimization-level-deferred")
    "IVersionedInterface::getAPILanguage" = @("id:*versioned-interface-get-api-language-deferred")
    "IVersionedInterface::getInterfaceInfo" = @("id:*versioned-interface-get-interface-info-deferred")
  }

  if (-not $aliasMap.ContainsKey($InterfaceKey)) {
    return @()
  }

  $aliases = @($aliasMap[$InterfaceKey])
  if ($deferredHistoryAliasMap.ContainsKey($InterfaceKey)) {
    $aliases += @($deferredHistoryAliasMap[$InterfaceKey])
  }

  $versionCandidates = @($VersionLine, "common", "")
  $matches = New-Object System.Collections.Generic.List[object]
  foreach ($api in $ManifestApis) {
    if ($api.Module -ne $Module) { continue }
    if ($versionCandidates -notcontains $api.VersionLine) { continue }

    foreach ($alias in $aliases) {
      if ($alias.StartsWith("id:")) {
        $idPattern = $alias.Substring(3)
        if ($api.Id -like $idPattern) {
          $matches.Add($api) | Out-Null
          break
        }
      }
      elseif ($api.SearchText.Contains($alias)) {
        $matches.Add($api) | Out-Null
        break
      }
    }
  }

  return @($matches | Sort-Object EntryPoint -Unique)
}

function Get-CSharpMethodCandidates {
  param([string]$MethodName)

  $candidates = New-Object System.Collections.Generic.List[string]
  $candidates.Add($MethodName) | Out-Null
  if ($MethodName -match '^(get|set|is|has|can|create|destroy|add|remove|mark|unmark|reset|clear|build|deserialize|serialize|parse|enqueue|execute|infer|report|refit)(.+)$') {
    $verb = $matches[1]
    $noun = $matches[2]
    $pascalVerb = $verb.Substring(0,1).ToUpperInvariant() + $verb.Substring(1)
    $candidates.Add("$pascalVerb$noun") | Out-Null
    $candidates.Add($noun) | Out-Null
  }
  return @($candidates | Select-Object -Unique)
}

function Find-MatchedManifestApis {
  param(
    [object[]]$ManifestApis,
    [string]$Module,
    [string]$VersionLine,
    [string]$ClassName,
    [string]$MethodName
  )

  $classAliases = Get-TensorRtClassAliases $ClassName
  $methodCandidates = Get-MethodCandidates -MethodName $MethodName -ClassName $ClassName
  $versionCandidates = @($VersionLine, "common", "")
  $interfaceKey = "$ClassName::$MethodName"
  $explicitOnlyOnMiss = @(
    "IBuilderConfig::getFlags"
  )

  if ($interfaceKey -in @(
    "Global::createONNXConfig",
    "Global::initLibNvInferPlugins",
    "Global::getBuilderPluginRegistry",
    "Global::getPluginRegistry",
    "IBuilder::buildEngineWithConfig",
    "IBuilder::destroy",
    "IBuilder::getPluginRegistry",
    "IBuilder::getErrorRecorder",
    "IBuilder::getLogger",
    "IBuilder::setErrorRecorder",
    "IBuilder::setGpuAllocator",
    "ICudaEngine::getErrorRecorder",
    "ICudaEngine::destroy",
    "ICudaEngine::setErrorRecorder",
    "IEngineInspector::getErrorRecorder",
    "IEngineInspector::setErrorRecorder",
    "IExecutionContext::getErrorRecorder",
    "IExecutionContext::destroy",
    "IExecutionContext::execute",
    "IExecutionContext::executeV2",
    "IExecutionContext::enqueueV2",
    "IExecutionContext::setAuxStreams",
    "IExecutionContext::setErrorRecorder",
    "INetworkDefinition::getErrorRecorder",
    "INetworkDefinition::setErrorRecorder",
    "IOnnxConfig::destroy",
    "IParser::setBuilderConfig",
    "IPluginRegistry::getErrorRecorder",
    "IPluginRegistry::getBuilderSafePluginRegistry",
    "IPluginRegistry::setParentSearchEnabled",
    "IPluginV2Ext::getTensorRTVersion",
    "IPluginV2IOExt::getTensorRTVersion",
    "IPluginV2DynamicExt::supportsFormatCombination",
    "IPluginV2DynamicExt::canBroadcastInputAcrossBatch",
    "IPluginV2DynamicExt::isOutputBroadcastAcrossBatch",
    "IPluginV2IOExt::supportsFormatCombination",
    "IPluginV3OneBuild::getAliasedInput",
    "IPluginV3OneBuild::getOutputDataTypes",
    "IPluginV3OneBuild::supportsFormatCombination",
    "IPluginV3OneRuntime::getFieldsToSerialize",
    "IVersionedInterface::getAPILanguage",
    "IVersionedInterface::getInterfaceInfo",
    "IErrorRecorder::getNbErrors",
    "IErrorRecorder::getErrorCode",
    "IErrorRecorder::getErrorDesc",
    "IErrorRecorder::getInterfaceInfo",
    "IErrorRecorder::hasOverflowed",
    "IErrorRecorder::EnumMax",
    "IBuilderConfig::getAvgTimingIterations",
    "IBuilderConfig::setAvgTimingIterations",
    "IBuilderConfig::getBuilderOptimizationLevel",
    "IBuilderConfig::setBuilderOptimizationLevel",
    "IBuilderConfig::getFlags",
    "IBuilderConfig::destroy",
    "IBuilderConfig::getPluginToSerialize",
    "IBuilderConfig::setPluginsToSerialize",
    "IRuntime::getPluginRegistry",
    "IRuntime::getLogger",
    "IRuntime::deserializeCudaEngine",
    "IRefitter::getLogger",
    "IParser::getNbErrors",
    "IParser::getNbSubgraphs",
    "IParser::getLayerOutputTensor",
    "IParser::getSubgraphNodes",
    "IParser::getUsedVCPluginLibraries",
    "IParser::parseWithWeightDescriptors",
    "IParserRefitter::getNbErrors",
    "IExecutionContext::getOutputAllocator",
    "IExecutionContext::setOutputAllocator",
    "IExecutionContext::getTemporaryStorageAllocator",
    "IExecutionContext::setTemporaryStorageAllocator",
    "IExecutionContext::getDebugListener",
    "IExecutionContext::setDebugListener",
    "IExecutionContext::setInputShapeBinding",
    "IExecutionContext::enqueueV3",
    "IBuilderConfig::getAlgorithmSelector",
    "IBuilderConfig::getInt8Calibrator",
    "IBuilderConfig::getProgressMonitor",
    "IBuilderConfig::setProgressMonitor",
    "IDebugListener::getInterfaceInfo",
    "IGpuAllocator::getInterfaceInfo",
    "IProfiler::reportLayerTime",
    "IOutputAllocator::getInterfaceInfo",
    "IProgressMonitor::getInterfaceInfo",
    "IProgressMonitor::phaseStart",
    "IProgressMonitor::stepComplete",
    "IProgressMonitor::phaseFinish",
    "IPluginCreator::getPluginName",
    "IPluginCreator::getPluginVersion",
    "IPluginCreator::getPluginNamespace",
    "IPluginCreator::getFieldNames",
    "IPluginCreator::getInterfaceInfo",
    "IPluginCreatorV3One::getPluginName",
    "IPluginCreatorV3One::getPluginVersion",
    "IPluginCreatorV3One::getPluginNamespace",
    "IPluginCreatorV3One::getFieldNames",
    "IPluginCreatorV3One::getInterfaceInfo",
    "IPluginRegistry::getPluginCreator",
    "IPluginRegistry::getPluginCreatorList",
    "IPluginRegistry::getAllCreators",
    "IPluginRegistry::getAllCreatorsRecursive")) {
    $explicitMatches = @(Find-ExplicitTensorRtInterfaceAliasApis $ManifestApis $Module $VersionLine $interfaceKey)
    if ($explicitMatches.Count -gt 0) {
      return $explicitMatches
    }

    if ($explicitOnlyOnMiss -contains $interfaceKey) {
      return @()
    }
  }

  foreach ($candidate in $methodCandidates) {
    $matches = New-Object System.Collections.Generic.List[object]
    foreach ($api in $ManifestApis) {
      if ($api.Module -ne $Module) { continue }
      if ($versionCandidates -notcontains $api.VersionLine) { continue }
      $text = $api.SearchText
      $classMatch = $false
      foreach ($alias in $classAliases) {
        if ($text.Contains($alias)) {
          $classMatch = $true
          break
        }
      }
      if (-not $classMatch) { continue }

      if ($text.Contains($candidate)) {
        $matches.Add($api) | Out-Null
      }
    }

    if ($matches.Count -gt 0) {
      return @($matches | Sort-Object EntryPoint -Unique)
    }
  }

  $explicitMatches = @(Find-ExplicitTensorRtInterfaceAliasApis $ManifestApis $Module $VersionLine $interfaceKey)
  if ($explicitMatches.Count -gt 0) {
    return $explicitMatches
  }

  return @()
}

function Get-TensorRtInterfaces {
  param([object]$Package)

  $results = New-Object System.Collections.Generic.List[object]
  $headers = Get-ChildItem -LiteralPath $Package.IncludeRoot -File -Include NvInfer*.h,NvOnnx*.h
  foreach ($header in $headers) {
    $raw = Get-Content -LiteralPath $header.FullName -Raw -Encoding utf8
    $text = Remove-CxxComments $raw
    $lines = $text -split "`r?`n"
    $className = $null
    $buffer = ""
    foreach ($line in $lines) {
      $trim = $line.Trim()
      if ([string]::IsNullOrWhiteSpace($trim) -or $trim.StartsWith("#")) {
        continue
      }

      if ($null -eq $className -and $trim -match '^class\s+(?:[A-Z_]+\s+)?([A-Za-z_][A-Za-z0-9_]*)\b') {
        $className = $matches[1]
        $buffer = ""
        continue
      }

      if ($null -ne $className) {
        if ($trim -match '^};') {
          $className = $null
          $buffer = ""
          continue
        }

        $buffer = "$buffer $trim"
        if ($trim.Contains(";")) {
          $statement = $buffer
          $buffer = ""
          if ($statement -match '\(' -and $statement -notmatch '\boperator\b' -and $statement -notmatch "^\s*(using|typedef)\b") {
            $matches = [regex]::Matches($statement, '(~?[A-Za-z_][A-Za-z0-9_]*)\s*\(')
            if ($matches.Count -gt 0) {
              $method = $matches[$matches.Count - 1].Groups[1].Value
              if ($className -like "I*" -and $method -ne $className -and -not $method.StartsWith("~") -and $method -notin @("if", "for", "while", "switch")) {
                $results.Add([pscustomobject]@{
                  Vendor = "TensorRT"
                  Package = $Package.Package
                  VersionLine = $Package.VersionLine
                  Version = $Package.Version
                  CudaVariant = $Package.CudaVariant
                  Header = $header.Name
                  Class = $className
                  Interface = "$className::$method"
                  Method = $method
                  Category = Get-TensorRtCategory $className $method $header.Name
                }) | Out-Null
              }
            }
          }
        }
      }
      elseif ($trim -match '\bTENSORRTAPI\b' -and $trim -match '\(') {
        $matches = [regex]::Matches($trim, '([A-Za-z_][A-Za-z0-9_]*)\s*\(')
        if ($matches.Count -gt 0) {
          $method = $matches[$matches.Count - 1].Groups[1].Value
          if ($method -notin @("if", "for", "while", "switch")) {
            $results.Add([pscustomobject]@{
              Vendor = "TensorRT"
              Package = $Package.Package
              VersionLine = $Package.VersionLine
              Version = $Package.Version
              CudaVariant = $Package.CudaVariant
              Header = $header.Name
              Class = "Global"
              Interface = "Global::$method"
              Method = $method
              Category = "global"
            }) | Out-Null
          }
        }
      }
    }
  }

  return $results
}

function Get-CudaToolkitInfo {
  param([string]$Root)

  $items = New-Object System.Collections.Generic.List[object]
  if ([string]::IsNullOrWhiteSpace($Root) -or -not (Test-Path -LiteralPath $Root -PathType Container)) {
    return $items
  }

  Get-ChildItem -LiteralPath $Root -Directory -Filter "v*" | ForEach-Object {
    $includeRoot = Join-Path $_.FullName "include"
    $header = Join-Path $includeRoot "cuda_runtime_api.h"
    if (Test-Path -LiteralPath $header -PathType Leaf) {
      $version = $_.Name.TrimStart('v')
      $items.Add([pscustomobject]@{
        Version = $version
        VersionLine = ($version -split '\.')[0]
        Root = $_.FullName
        Header = $header
      }) | Out-Null
    }
  }

  return $items
}

function Get-CudaCategory {
  param([string]$FunctionName)
  switch -Regex ($FunctionName) {
    '^cudaDevice|^cudaGetDevice|^cudaSetDevice|^cudaChooseDevice' { return "device" }
    '^cudaStream|^cudaThreadExchangeStreamCaptureMode' { return "stream" }
    '^cudaEvent' { return "event" }
    '^cudaGraph|^cudaUserObject' { return "graph" }
    '^cudaMemPool|^cudaMalloc|^cudaFree|^cudaMem(Get|Set)|^cudaMemcpy|^cudaMemset|^cudaHost|^cudaPointer|^cudaArray|^cudaMipmappedArray' { return "memory" }
    '^cudaFunc|^cudaLaunch|^cudaOccupancy|^cudaConfigure|^cudaSetup|^cudaLaunchKernel' { return "kernel-launch" }
    '^cudaGetLastError|^cudaPeekAtLastError|^cudaGetError' { return "error" }
    '^cudaProfiler|^cudaDeviceGetGraphMemAttribute|^cudaDeviceSetGraphMemAttribute' { return "diagnostics" }
    default { return "other" }
  }
}

function Get-CudaCandidates {
  param([string]$FunctionName)

  $candidates = New-Object System.Collections.Generic.List[string]
  $token = ConvertTo-ApiToken $FunctionName
  $candidates.Add($token) | Out-Null
  if ($FunctionName.StartsWith("cuda")) {
    $withoutPrefix = $FunctionName.Substring(4)
    $candidates.Add((ConvertTo-ApiToken $withoutPrefix)) | Out-Null
  }

  $alias = @{
    "cudaRuntimeGetVersion" = @("get-runtime-version", "runtime-version")
    "cudaDriverGetVersion" = @("get-driver-version", "driver-version")
    "cudaGetDeviceCount" = @("get-device-count", "device-count")
    "cudaGetDevice" = @("get-current-device", "current-device")
    "cudaSetDevice" = @("set-device")
    "cudaGetDeviceProperties" = @("get-device-info", "device-info", "device-properties")
    "cudaMemGetInfo" = @("get-memory-info", "memory-info")
    "cudaDeviceSynchronize" = @("device-synchronize")
    "cudaDeviceReset" = @("device-reset")
    "cudaDeviceGetLimit" = @("device-get-limit", "get-device-limit")
    "cudaDeviceSetLimit" = @("device-set-limit", "set-device-limit")
    "cudaDeviceGetAttribute" = @("device-get-attribute", "device-attribute")
    "cudaDeviceCanAccessPeer" = @("device-can-access-peer", "peer-access")
    "cudaDeviceEnablePeerAccess" = @("device-enable-peer-access", "enable-peer-access")
    "cudaDeviceDisablePeerAccess" = @("device-disable-peer-access", "disable-peer-access")
    "cudaChooseDevice" = @("choose-device")
    "cudaDeviceGetTexture1DLinearMaxWidth" = @("device-get-texture-1d-linear-max-width")
    "cudaDeviceGetGraphMemAttribute" = @("device-get-graph-memory-attribute")
    "cudaDeviceGraphMemTrim" = @("device-trim-graph-memory")
    "cudaDeviceSetGraphMemAttribute" = @("device-reset-graph-memory-high-watermark")
    "cudaDeviceGetHostAtomicCapabilities" = @("device-get-host-atomic-capabilities")
    "cudaDeviceGetP2PAtomicCapabilities" = @("device-get-p2p-atomic-capabilities")
    "cudaDeviceGetDevResource" = @("device-get-dev-resource-deferred")
    "cudaDeviceGetExecutionCtx" = @("device-get-execution-ctx-deferred")
    "cudaDeviceGetNvSciSyncAttributes" = @("device-get-nv-sci-sync-attributes-deferred")
    "cudaDeviceRegisterAsyncNotification" = @("device-register-async-notification-deferred")
    "cudaDeviceUnregisterAsyncNotification" = @("device-unregister-async-notification-deferred")
    "cudaStreamCreateWithFlags" = @("stream-create", "stream-create-with-flags")
    "cudaStreamCreateWithPriority" = @("stream-create-with-priority")
    "cudaStreamGetPriority" = @("stream-get-priority")
    "cudaStreamGetFlags" = @("stream-get-flags")
    "cudaStreamQuery" = @("stream-query")
    "cudaStreamSynchronize" = @("stream-synchronize")
    "cudaStreamWaitEvent" = @("stream-wait-event")
    "cudaStreamDestroy" = @("stream-destroy")
    "cudaStreamBeginCapture" = @("stream-begin-capture")
    "cudaStreamEndCapture" = @("stream-end-capture")
    "cudaStreamIsCapturing" = @("stream-is-capturing")
    "cudaStreamGetCaptureInfo" = @("stream-get-capture-info")
    "cudaStreamAddCallback" = @("stream-add-callback-deferred")
    "cudaStreamAttachMemAsync" = @("stream-attach-mem-async")
    "cudaStreamBeginCaptureToGraph" = @("stream-begin-capture-to-graph-deferred")
    "cudaStreamGetAttribute" = @("stream-get-attribute")
    "cudaStreamGetCaptureInfo_ptsz" = @("stream-get-capture-info-ptsz-deferred")
    "cudaStreamGetCaptureInfo_v3" = @("stream-get-capture-info-v3-deferred")
    "cudaStreamGetDevResource" = @("stream-get-dev-resource-deferred")
    "cudaStreamSetAttribute" = @("stream-set-attribute")
    "cudaStreamUpdateCaptureDependencies" = @("stream-update-capture-dependencies-deferred")
    "cudaStreamUpdateCaptureDependencies_ptsz" = @("stream-update-capture-dependencies-ptsz-deferred")
    "cudaStreamUpdateCaptureDependencies_v2" = @("stream-update-capture-dependencies-v2-deferred")
    "cudaFuncGetAttributes" = @("func-get-attributes")
    "cudaFuncGetName" = @("func-get-name")
    "cudaFuncGetParamCount" = @("func-get-param-count")
    "cudaFuncGetParamInfo" = @("func-get-param-info")
    "cudaFuncSetAttribute" = @("func-set-attribute")
    "cudaFuncSetCacheConfig" = @("func-set-cache-config")
    "cudaFuncSetSharedMemConfig" = @("func-set-shared-mem-config")
    "cudaLaunchCooperativeKernel" = @("launch-cooperative-kernel")
    "cudaLaunchHostFunc" = @("launch-host-func-deferred")
    "cudaLaunchHostFunc_v2" = @("launch-host-func-v2-deferred")
    "cudaLaunchKernel" = @("launch-kernel")
    "cudaLaunchKernelExC" = @("launch-kernel-ex-c")
    "cudaOccupancyAvailableDynamicSMemPerBlock" = @("occupancy-available-dynamic-smem-per-block")
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor" = @("occupancy-max-active-blocks-per-multiprocessor")
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags" = @("occupancy-max-active-blocks-per-multiprocessor-with-flags")
    "cudaOccupancyMaxActiveClusters" = @("occupancy-max-active-clusters")
    "cudaOccupancyMaxPotentialClusterSize" = @("occupancy-max-potential-cluster-size")
    "cudaEventCreateWithFlags" = @("event-create")
    "cudaEventRecord" = @("event-record")
    "cudaEventQuery" = @("event-query")
    "cudaEventSynchronize" = @("event-synchronize")
    "cudaEventElapsedTime" = @("event-elapsed-time")
    "cudaEventDestroy" = @("event-destroy")
    "cudaMalloc" = @("memory-allocate")
    "cudaFree" = @("memory-free")
    "cudaMemcpy" = @("memory-copy", "memcpy")
    "cudaMemcpyAsync" = @("memory-copy-async", "memcpy-async")
    "cudaMemcpyPeer" = @("memory-copy-peer", "peer-copy")
    "cudaMemcpyPeerAsync" = @("memory-copy-peer-async", "peer-copy-async")
    "cudaMallocHost" = @("malloc-host", "pinned-memory-allocate", "host-alloc")
    "cudaMalloc3D" = @("malloc-3d", "pitched-memory-allocate-3d")
    "cudaMalloc3DArray" = @("malloc-3d-array")
    "cudaMemcpy2D" = @("pitched-memory-copy", "copy-2d")
    "cudaMemcpy2DAsync" = @("memcpy-2d-async", "pitched-memory-copy-async", "copy-2d-async")
    "cudaMemcpy2DToArray" = @("memcpy-2d-to-array", "copy-2d-to-array")
    "cudaMemcpy2DToArrayAsync" = @("memcpy-2d-to-array-async", "copy-2d-to-array-async")
    "cudaMemcpy2DFromArray" = @("memcpy-2d-from-array", "copy-2d-from-array")
    "cudaMemcpy2DFromArrayAsync" = @("memcpy-2d-from-array-async", "copy-2d-from-array-async")
    "cudaMemcpy2DArrayToArray" = @("memcpy-2d-array-to-array", "copy-2d-array-to-array")
    "cudaMemcpy3D" = @("memcpy-3d", "pitched-memory-copy-3d", "copy-3d")
    "cudaMemcpy3DAsync" = @("memcpy-3d-async", "pitched-memory-copy-3d-async", "copy-3d-async")
    "cudaMemcpy3DPeer" = @("memcpy-3d-peer")
    "cudaMemcpy3DPeerAsync" = @("memcpy-3d-peer-async")
    "cudaMemcpy3DBatchAsync" = @("memcpy-3d-batch-async")
    "cudaMemcpy3DWithAttributesAsync" = @("memcpy-3d-with-attributes-async")
    "cudaMemcpy3DToArray" = @("memcpy-3d-to-array")
    "cudaMemcpy3DToArrayAsync" = @("memcpy-3d-to-array-async")
    "cudaMemcpy3DFromArray" = @("memcpy-3d-from-array")
    "cudaMemcpy3DFromArrayAsync" = @("memcpy-3d-from-array-async")
    "cudaMemcpy3DArrayToArray" = @("memcpy-3d-array-to-array")
    "cudaMemcpy3DArrayToArrayAsync" = @("memcpy-3d-array-to-array-async")
    "cudaMemcpyToSymbol" = @("memcpy-to-symbol")
    "cudaMemcpyFromSymbol" = @("memcpy-from-symbol")
    "cudaMemcpyToSymbolAsync" = @("memcpy-to-symbol-async")
    "cudaMemcpyFromSymbolAsync" = @("memcpy-from-symbol-async")
    "cudaMemset" = @("memory-memset", "memset")
    "cudaMemsetAsync" = @("memory-memset-async", "memset-async")
    "cudaMemset2D" = @("pitched-memory-memset-2d")
    "cudaMemset2DAsync" = @("pitched-memory-memset-2d-async")
    "cudaMemset3D" = @("pitched-memory-memset-3d")
    "cudaMemset3DAsync" = @("pitched-memory-memset-3d-async")
    "cudaMallocAsync" = @("memory-allocate-async")
    "cudaFreeAsync" = @("memory-free-async")
    "cudaMallocFromPoolAsync" = @("memory-pool-allocate-async", "selected-pool")
    "cudaMemPoolCreate" = @("memory-pool-create")
    "cudaMemPoolDestroy" = @("memory-pool-destroy")
    "cudaDeviceGetDefaultMemPool" = @("default-memory-pool")
    "cudaDeviceGetMemPool" = @("current-memory-pool")
    "cudaDeviceSetMemPool" = @("set-current-memory-pool")
    "cudaMemPoolTrimTo" = @("memory-pool-trim")
    "cudaMemPoolGetAttribute" = @("memory-pool-get-attribute")
    "cudaMemPoolSetAttribute" = @("memory-pool-set-attribute")
    "cudaMemPoolGetAccess" = @("memory-pool-get-access")
    "cudaMemPoolSetAccess" = @("memory-pool-set-access")
    "cudaMemGetDefaultMemPool" = @("mem-get-default-mem-pool")
    "cudaMemGetMemPool" = @("mem-get-mem-pool")
    "cudaMemSetMemPool" = @("mem-set-mem-pool")
    "cudaMemPoolExportPointer" = @("mem-pool-export-pointer")
    "cudaMemPoolExportToShareableHandle" = @("mem-pool-export-to-shareable-handle")
    "cudaMemPoolImportFromShareableHandle" = @("mem-pool-import-from-shareable-handle")
    "cudaMemPoolImportPointer" = @("mem-pool-import-pointer")
    "cudaMallocManaged" = @("managed-memory-allocate")
    "cudaMemPrefetchAsync" = @("managed-memory-prefetch", "memory-prefetch-range-async-safe")
    "cudaMemAdvise" = @("managed-memory-advise", "memory-advise-range-safe")
    "cudaHostAlloc" = @("pinned-memory-allocate", "host-alloc")
    "cudaHostRegister" = @("registered-host-memory-register", "host-register")
    "cudaHostUnregister" = @("registered-host-memory-free", "host-unregister")
    "cudaHostGetDevicePointer" = @("pinned-memory-get-device-pointer", "mapped-pinned")
    "cudaHostGetFlags" = @("pinned-memory-get-flags")
    "cudaMallocPitch" = @("pitched-memory-allocate")
    "cudaPointerGetAttributes" = @("pointer-get-attributes", "pointer-attributes")
    "cudaCreateSurfaceObject" = @("create-surface-object-deferred")
    "cudaCreateTextureObject" = @("create-texture-object-deferred")
    "cudaCreateTextureObject_v2" = @("create-texture-object-v2-deferred")
    "cudaDestroySurfaceObject" = @("destroy-surface-object-deferred")
    "cudaDestroyTextureObject" = @("destroy-texture-object-deferred")
    "cudaGetSurfaceObjectResourceDesc" = @("get-surface-object-resource-desc-deferred")
    "cudaGetTextureObjectResourceDesc" = @("get-texture-object-resource-desc-deferred")
    "cudaGetTextureObjectResourceViewDesc" = @("get-texture-object-resource-view-desc-deferred")
    "cudaGetTextureObjectTextureDesc" = @("get-texture-object-texture-desc-deferred")
    "cudaGetTextureObjectTextureDesc_v2" = @("get-texture-object-texture-desc-v2-deferred")
    "cudaDestroyExternalMemory" = @("destroy-external-memory-deferred")
    "cudaDestroyExternalSemaphore" = @("destroy-external-semaphore-deferred")
    "cudaExternalMemoryGetMappedBuffer" = @("external-memory-get-mapped-buffer-deferred")
    "cudaExternalMemoryGetMappedMipmappedArray" = @("external-memory-get-mapped-mipmapped-array-deferred")
    "cudaImportExternalMemory" = @("import-external-memory-deferred")
    "cudaImportExternalSemaphore" = @("import-external-semaphore-deferred")
    "cudaSignalExternalSemaphoresAsync" = @("signal-external-semaphores-async-deferred")
    "cudaSignalExternalSemaphoresAsync_ptsz" = @("signal-external-semaphores-async-ptsz-deferred")
    "cudaSignalExternalSemaphoresAsync_v2" = @("signal-external-semaphores-async-v2-deferred")
    "cudaWaitExternalSemaphoresAsync" = @("wait-external-semaphores-async-deferred")
    "cudaWaitExternalSemaphoresAsync_ptsz" = @("wait-external-semaphores-async-ptsz-deferred")
    "cudaWaitExternalSemaphoresAsync_v2" = @("wait-external-semaphores-async-v2-deferred")
    "cudaIpcCloseMemHandle" = @("ipc-close-mem-handle-deferred")
    "cudaIpcGetEventHandle" = @("ipc-get-event-handle-deferred")
    "cudaIpcGetMemHandle" = @("ipc-get-mem-handle-deferred")
    "cudaIpcOpenEventHandle" = @("ipc-open-event-handle-deferred")
    "cudaIpcOpenMemHandle" = @("ipc-open-mem-handle-deferred")
    "cudaGetDriverEntryPoint" = @("get-driver-entry-point-deferred")
    "cudaGetDriverEntryPointByVersion" = @("get-driver-entry-point-by-version-deferred")
    "cudaGetExportTable" = @("get-export-table-deferred")
    "cudaGetFuncBySymbol" = @("get-func-by-symbol-deferred")
    "cudaGetKernel" = @("get-kernel-deferred")
    "cudaGetSymbolAddress" = @("get-symbol-address-deferred")
    "cudaGetSymbolSize" = @("get-symbol-size-deferred")
    "cudaKernelSetAttributeForDevice" = @("kernel-set-attribute-for-device-deferred")
    "cudaLibraryEnumerateKernels" = @("library-enumerate-kernels-deferred")
    "cudaLibraryGetGlobal" = @("library-get-global-deferred")
    "cudaLibraryGetKernel" = @("library-get-kernel-deferred")
    "cudaLibraryGetKernelCount" = @("library-get-kernel-count-deferred")
    "cudaLibraryGetManaged" = @("library-get-managed-deferred")
    "cudaLibraryGetUnifiedFunction" = @("library-get-unified-function-deferred")
    "cudaLibraryLoadData" = @("library-load-data-deferred")
    "cudaLibraryLoadFromFile" = @("library-load-from-file-deferred")
    "cudaLibraryUnload" = @("library-unload-deferred")
    "cudaDevResourceGenerateDesc" = @("dev-resource-generate-desc-deferred")
    "cudaDevSmResourceSplit" = @("dev-sm-resource-split-deferred")
    "cudaDevSmResourceSplitByCount" = @("dev-sm-resource-split-by-count-deferred")
    "cudaExecutionCtxDestroy" = @("execution-ctx-destroy-deferred")
    "cudaExecutionCtxGetDevice" = @("execution-ctx-get-device-deferred")
    "cudaExecutionCtxGetDevResource" = @("execution-ctx-get-dev-resource-deferred")
    "cudaExecutionCtxGetId" = @("execution-ctx-get-id-deferred")
    "cudaExecutionCtxRecordEvent" = @("execution-ctx-record-event-deferred")
    "cudaExecutionCtxStreamCreate" = @("execution-ctx-stream-create-deferred")
    "cudaExecutionCtxSynchronize" = @("execution-ctx-synchronize-deferred")
    "cudaExecutionCtxWaitEvent" = @("execution-ctx-wait-event-deferred")
    "cudaGreenCtxCreate" = @("green-ctx-create-deferred")
    "cudaLogsCurrent" = @("logs-current-deferred")
    "cudaLogsDumpToFile" = @("logs-dump-to-file-deferred")
    "cudaLogsDumpToMemory" = @("logs-dump-to-memory-deferred")
    "cudaLogsRegisterCallback" = @("logs-register-callback-deferred")
    "cudaLogsUnregisterCallback" = @("logs-unregister-callback-deferred")
    "cudaInitDevice" = @("init-device")
    "cudaSetValidDevices" = @("set-valid-devices")
    "cudaMemAdvise_v2" = @("mem-advise-v2-deferred")
    "cudaMemDiscardAndPrefetchBatchAsync" = @("mem-discard-and-prefetch-batch-async-deferred")
    "cudaMemDiscardBatchAsync" = @("mem-discard-batch-async-deferred")
    "cudaMemPrefetchAsync_v2" = @("mem-prefetch-async-v2-deferred")
    "cudaMemPrefetchBatchAsync" = @("mem-prefetch-batch-async-deferred")
    "cudaMemRangeGetAttribute" = @("mem-range-get-attribute-deferred", "mem-range-get-attribute-scalar-safe", "mem-range-get-accessed-by-count-safe", "mem-range-copy-accessed-by-devices-safe")
    "cudaMemRangeGetAttributes" = @("mem-range-get-attributes-deferred", "mem-range-get-attributes-scalar-safe")
    "cudaGraphDebugDotPrint" = @("graph-debug-dot-print-deferred", "graph-debug-dot-print-safe")
    "cudaGraphAddEventRecordNode" = @("graph-add-event-record-node-deferred", "graph-add-event-record-node-safe")
    "cudaGraphAddEventWaitNode" = @("graph-add-event-wait-node-deferred", "graph-add-event-wait-node-safe")
    "cudaGraphAddMemcpyNode1D" = @("graph-add-memcpy-node1-d-deferred", "graph-add-memcpy-node-1d-device-to-device-safe", "graph-add-memcpy-node-1d-host-to-device-safe", "graph-add-memcpy-node-1d-device-to-host-safe")
    "cudaGraphEventRecordNodeSetEvent" = @("graph-event-record-node-set-event-deferred", "graph-event-record-node-set-event-safe")
    "cudaGraphEventRecordNodeGetEvent" = @("graph-event-record-node-get-event-deferred", "graph-event-record-node-has-event-safe")
    "cudaGraphEventWaitNodeSetEvent" = @("graph-event-wait-node-set-event-deferred", "graph-event-wait-node-set-event-safe")
    "cudaGraphEventWaitNodeGetEvent" = @("graph-event-wait-node-get-event-deferred", "graph-event-wait-node-has-event-safe")
    "cudaGraphExecEventRecordNodeSetEvent" = @("graph-exec-event-record-node-set-event-deferred", "graph-exec-event-record-node-set-event-safe")
    "cudaGraphExecEventWaitNodeSetEvent" = @("graph-exec-event-wait-node-set-event-deferred", "graph-exec-event-wait-node-set-event-safe")
    "cudaGraphExecMemcpyNodeSetParams1D" = @("graph-exec-memcpy-node-set-params1-d-deferred", "graph-exec-memcpy-node-set-params-1d-device-to-device-safe", "graph-exec-memcpy-node-set-params-1d-host-to-device-safe", "graph-exec-memcpy-node-set-params-1d-device-to-host-safe")
    "cudaGraphKernelNodeGetAttribute" = @("graph-kernel-node-get-attribute-deferred", "graph-kernel-node-get-attribute-scalar-safe")
    "cudaGraphKernelNodeGetParams" = @("graph-kernel-node-get-params-deferred")
    "cudaGraphKernelNodeSetAttribute" = @("graph-kernel-node-set-attribute-deferred", "graph-kernel-node-set-attribute-scalar-safe")
    "cudaGraphKernelNodeSetParams" = @("graph-kernel-node-set-params-deferred")
    "cudaGraphMemAllocNodeGetParams" = @("graph-mem-alloc-node-get-params-deferred")
    "cudaGraphMemcpyNodeGetParams" = @("graph-memcpy-node-get-params-deferred", "graph-memcpy-node-get-params-safe")
    "cudaGraphMemcpyNodeSetParams" = @("graph-memcpy-node-set-params-deferred")
    "cudaGraphMemcpyNodeSetParams1D" = @("graph-memcpy-node-set-params-1d-deferred", "graph-memcpy-node-set-params-1d-device-to-device-safe", "graph-memcpy-node-set-params-1d-host-to-device-safe", "graph-memcpy-node-set-params-1d-device-to-host-safe")
    "cudaGraphMemcpyNodeSetParamsFromSymbol" = @("graph-memcpy-node-set-params-from-symbol-deferred")
    "cudaGraphMemcpyNodeSetParamsToSymbol" = @("graph-memcpy-node-set-params-to-symbol-deferred")
    "cudaGraphMemFreeNodeGetParams" = @("graph-mem-free-node-get-params-deferred")
    "cudaGraphMemsetNodeGetParams" = @("graph-memset-node-get-params-deferred", "graph-memset-node-get-params-safe")
    "cudaGraphMemsetNodeSetParams" = @("graph-memset-node-set-params-deferred", "graph-memset-node-set-params-safe")
    "cudaGraphAddDependencies" = @("cuda-graph-add-dependencies-deferred", "graph-add-dependency-safe", "graph-add-dependency-v2-safe")
    "cudaGraphAddDependencies_v2" = @("cuda-graph-add-dependencies-v2-deferred", "graph-add-dependency-v2-safe")
    "cudaGraphAddNode" = @("cuda-cuda-graph-add-node-deferred", "cuda-graph-add-conditional-node-owner-safe")
    "cudaGraphConditionalHandleCreate" = @("cuda-cuda-graph-conditional-handle-create-deferred", "cuda-graph-conditional-handle-create-owner-safe")
    "cudaGraphConditionalHandleCreate_v2" = @("cuda-cuda-graph-conditional-handle-create-v2-deferred", "cuda-graph-conditional-handle-create-v2-owner-safe")
    "cudaGraphGetEdges" = @("graph-get-edges-v2-deferred", "graph-get-edges-v2-count-safe", "graph-get-edge-v2-safe")
    "cudaGraphGetEdges_v2" = @("graph-get-edges-v2-deferred", "graph-get-edges-v2-count-safe", "graph-get-edge-v2-safe")
    "cudaGraphNodeGetContainingGraph" = @("graph-node-get-containing-graph-deferred", "graph-node-is-in-graph-safe")
    "cudaGraphNodeGetDependencies" = @("graph-node-get-dependency-count-safe", "graph-node-get-dependency-safe", "graph-node-get-dependencies-v2-count-safe", "graph-node-get-dependency-v2-safe")
    "cudaGraphNodeGetDependencies_v2" = @("graph-node-get-dependencies-v2-deferred", "graph-node-get-dependencies-v2-count-safe", "graph-node-get-dependency-v2-safe")
    "cudaGraphNodeGetDependentNodes" = @("graph-node-get-dependent-count-safe", "graph-node-get-dependent-safe", "graph-node-get-dependent-nodes-v2-count-safe", "graph-node-get-dependent-node-v2-safe")
    "cudaGraphNodeGetDependentNodes_v2" = @("graph-node-get-dependent-nodes-v2-deferred", "graph-node-get-dependent-nodes-v2-count-safe", "graph-node-get-dependent-node-v2-safe")
    "cudaGraphNodeGetEnabled" = @("graph-exec-node-get-enabled-safe")
    "cudaGraphNodeGetParams" = @("graph-node-get-params-deferred")
    "cudaGraphNodeSetEnabled" = @("graph-exec-node-set-enabled-safe")
    "cudaGraphReleaseUserObject" = @("graph-release-user-object-deferred")
    "cudaGraphRemoveDependencies" = @("graph-remove-dependency-safe", "graph-remove-dependency-v2-safe")
    "cudaGraphRemoveDependencies_v2" = @("graph-remove-dependencies-v2-deferred", "graph-remove-dependency-v2-safe")
    "cudaGraphRetainUserObject" = @("graph-retain-user-object-deferred")
    "cudaUserObjectCreate" = @("user-object-create-deferred")
    "cudaUserObjectRelease" = @("user-object-release-deferred")
    "cudaUserObjectRetain" = @("user-object-retain-deferred")
    "cudaGraphInstantiate" = @("graph-instantiate")
    "cudaGraphLaunch" = @("graph-launch")
    "cudaGraphDestroy" = @("graph-destroy")
    "cudaGraphExecDestroy" = @("graph-exec-destroy")
    "cudaGetLastError" = @("get-last-error")
    "cudaPeekAtLastError" = @("peek-last-error")
    "cudaGetErrorName" = @("get-error-name")
    "cudaGetErrorString" = @("get-error-string")
  }

  if ($alias.ContainsKey($FunctionName)) {
    foreach ($item in $alias[$FunctionName]) {
      $candidates.Add($item) | Out-Null
    }
  }

  return @($candidates | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
}

function Find-ExplicitCudaManifestApis {
  param([object[]]$ManifestApis, [string]$FunctionName)

  $aliasMap = @{
    "cudaCreateSurfaceObject" = @("id:cuda-create-surface-object-array-owner-safe")
    "cudaDestroySurfaceObject" = @("id:cuda-destroy-surface-object-bridge-owned-safe")
    "cudaGetSurfaceObjectResourceDesc" = @("id:cuda-get-surface-object-resource-desc-copied-snapshot-safe")
    "cudaCreateTextureObject" = @("id:cuda-create-texture-object-array-owner-safe")
    "cudaCreateTextureObject_v2" = @("id:cuda-create-texture-object-v2-array-owner-safe")
    "cudaDestroyTextureObject" = @("id:cuda-destroy-texture-object-bridge-owned-safe")
    "cudaGetTextureObjectResourceDesc" = @("id:cuda-get-texture-object-resource-desc-copied-snapshot-safe")
    "cudaGetTextureObjectResourceViewDesc" = @("id:cuda-get-texture-object-resource-view-desc-copied-snapshot-safe")
    "cudaGetTextureObjectTextureDesc" = @("id:cuda-get-texture-object-texture-desc-copied-snapshot-safe")
    "cudaGetTextureObjectTextureDesc_v2" = @("id:cuda-get-texture-object-texture-desc-v2-copied-snapshot-safe")
    "cudaLibraryLoadData" = @("id:cuda-library-load-data-retained-copy-owner-safe")
    "cudaLibraryLoadFromFile" = @("id:cuda-library-load-from-file-bridge-owned-safe")
    "cudaLibraryUnload" = @("id:cuda-library-unload-bridge-owned-safe")
    "cudaLibraryGetKernelCount" = @("id:cuda-library-get-kernel-count-copied-scalar-safe")
    "cudaLibraryEnumerateKernels" = @("id:cuda-library-enumerate-kernels-copied-inventory-safe")
    "cudaLibraryGetKernel" = @("id:cuda-library-get-kernel-exists-by-name-safe")
    "cudaLibraryGetGlobal" = @("id:cuda-library-get-global-size-by-name-copied-scalar-safe")
    "cudaLibraryGetManaged" = @("id:cuda-library-get-managed-size-by-name-copied-scalar-safe")
    "cudaLibraryGetUnifiedFunction" = @("id:cuda-library-get-unified-function-exists-by-name-safe")
    "cudaKernelSetAttributeForDevice" = @("id:cuda-kernel-set-attribute-for-device-library-owner-name-safe")
    "cudaDeviceGetExecutionCtx" = @("id:cuda-device-get-primary-execution-context-owner-safe")
    "cudaDeviceGetDevResource" = @("id:cuda-device-get-dev-resource-copied-snapshot-safe")
    "cudaExecutionCtxGetDevice" = @("id:cuda-execution-context-get-device-copied-scalar-safe")
    "cudaExecutionCtxGetId" = @("id:cuda-execution-context-get-id-copied-scalar-safe")
    "cudaExecutionCtxSynchronize" = @("id:cuda-execution-context-synchronize-owner-safe")
    "cudaExecutionCtxStreamCreate" = @("id:cuda-execution-context-create-stream-owner-safe")
    "cudaExecutionCtxRecordEvent" = @("id:cuda-execution-context-record-event-owner-safe")
    "cudaExecutionCtxWaitEvent" = @("id:cuda-execution-context-wait-event-owner-safe")
    "cudaExecutionCtxGetDevResource" = @("id:cuda-execution-context-get-dev-resource-copied-snapshot-safe")
    "cudaGraphAddMemsetNode" = @("id:cuda-graph-add-memset-node*-safe")
    "cudaGraphAddMemAllocNode" = @("id:cuda-graph-add-mem-alloc-node-owner-safe")
    "cudaGraphAddMemFreeNode" = @("id:cuda-graph-add-mem-free-node-owner-safe")
    "cudaGraphAddChildGraphNode" = @("id:cuda-graph-add-child-graph-node*-safe")
    "cudaGraphAddNode" = @("id:cuda-graph-add-conditional-node-owner-safe")
    "cudaGraphConditionalHandleCreate" = @("id:cuda-graph-conditional-handle-create-owner-safe")
    "cudaGraphConditionalHandleCreate_v2" = @("id:cuda-graph-conditional-handle-create-v2-owner-safe")
    "cudaGraphChildGraphNodeGetGraph" = @("id:cuda-graph-child-graph-node-*-safe")
    "cudaGraphExecChildGraphNodeSetParams" = @("id:cuda-graph-exec-child-graph-node-set-params-safe")
    "cudaGraphExecUpdate" = @("id:cuda-graph-exec-update-copied-metadata-safe")
    "cudaGraphInstantiateWithParams" = @("id:cuda-graph-instantiate-with-params*-safe")
    "cudaGraphKernelNodeCopyAttributes" = @("id:cuda-graph-kernel-node-copy-attributes-safe")
    "cudaGraphDestroyNode" = @("id:cuda-graph-destroy-node-owner-scoped-safe")
    "cudaGraphExecMemsetNodeSetParams" = @("id:cuda-graph-exec-memset-node-set-params-owner-safe")
    "cudaGraphKernelNodeGetParams" = @("id:cuda-graph-kernel-node-get-params-copied-snapshot-safe")
    "cudaGraphHostNodeGetParams" = @("id:cuda-graph-host-node-get-params-copied-snapshot-safe")
    "cudaGraphMemAllocNodeGetParams" = @("id:cuda-graph-mem-alloc-node-get-params-copied-snapshot-safe")
    "cudaGraphMemFreeNodeGetParams" = @("id:cuda-graph-mem-free-node-get-params-copied-snapshot-safe")
    "cudaGraphExternalSemaphoresSignalNodeGetParams" = @("id:cuda-graph-external-semaphore-signal-node-get-params-copied-snapshot-safe")
    "cudaGraphExternalSemaphoresWaitNodeGetParams" = @("id:cuda-graph-external-semaphore-wait-node-get-params-copied-snapshot-safe")
    "cudaGraphNodeGetContainingGraph" = @("id:*graph-node-is-in-graph-safe")
    "cudaStreamGetCaptureInfo_ptsz" = @("id:cuda-stream-get-capture-info-ptsz-copied-scalars-safe")
    "cudaStreamUpdateCaptureDependencies_ptsz" = @("id:cuda-stream-update-capture-dependencies-ptsz-owner-token-array-safe")
    "cudaStreamUpdateCaptureDependencies_v2" = @("id:cuda-stream-update-capture-dependencies-v2-owner-token-edge-data-safe")
    "cudaStreamBeginCaptureToGraph" = @("id:cuda-stream-begin-capture-to-graph-owner-safe")
    "cudaStreamGetDevResource" = @("id:cuda-stream-get-dev-resource-copied-snapshot-safe")
    "cudaStreamGetCaptureInfo_v3" = @("id:cuda-stream-get-capture-info-copied-summary-safe")
    "cudaStreamUpdateCaptureDependencies" = @("id:cuda-stream-update-capture-dependencies-owner-token-array-safe")
    "cudaLogsCurrent" = @("id:cuda-logs-current-cursor-safe")
    "cudaLogsDumpToMemory" = @("id:cuda-logs-dump-to-memory-caller-buffer-safe")
    "cudaLogsDumpToFile" = @("id:cuda-logs-dump-to-file-safe")
    "cudaMemPrefetchBatchAsync" = @("id:cuda-managed-memory-prefetch-batch-owner-array-safe")
    "cudaMemDiscardBatchAsync" = @("id:cuda-managed-memory-discard-batch-owner-array-safe")
    "cudaMemDiscardAndPrefetchBatchAsync" = @("id:cuda-managed-memory-discard-and-prefetch-batch-owner-array-safe")
    "cudaMemPrefetchAsync_v2" = @("id:cuda-managed-memory-prefetch-location-range-async-safe")
    "cudaMemAdvise_v2" = @("id:cuda-managed-memory-advise-location-range-safe")
    "cudaIpcGetEventHandle" = @("id:cuda-ipc-get-event-handle-copied-export-token-safe")
    "cudaIpcGetMemHandle" = @("id:cuda-ipc-get-mem-handle-copied-export-token-safe")
  }
  $deferredHistoryAliasMap = @{
    "cudaCreateSurfaceObject" = @("id:cuda-create-surface-object-deferred")
    "cudaDestroySurfaceObject" = @("id:cuda-destroy-surface-object-deferred")
    "cudaGetSurfaceObjectResourceDesc" = @("id:cuda-get-surface-object-resource-desc-deferred")
    "cudaCreateTextureObject" = @("id:cuda-create-texture-object-deferred")
    "cudaCreateTextureObject_v2" = @("id:cuda-create-texture-object-v2-deferred")
    "cudaDestroyTextureObject" = @("id:cuda-destroy-texture-object-deferred")
    "cudaGetTextureObjectResourceDesc" = @("id:cuda-get-texture-object-resource-desc-deferred")
    "cudaGetTextureObjectResourceViewDesc" = @("id:cuda-get-texture-object-resource-view-desc-deferred")
    "cudaGetTextureObjectTextureDesc" = @("id:cuda-get-texture-object-texture-desc-deferred")
    "cudaGetTextureObjectTextureDesc_v2" = @("id:cuda-get-texture-object-texture-desc-v2-deferred")
    "cudaLibraryLoadData" = @("id:cuda-library-load-data-deferred")
    "cudaLibraryLoadFromFile" = @("id:cuda-library-load-from-file-deferred")
    "cudaLibraryUnload" = @("id:cuda-library-unload-deferred")
    "cudaLibraryGetKernelCount" = @("id:cuda-library-get-kernel-count-deferred")
    "cudaLibraryEnumerateKernels" = @("id:cuda-library-enumerate-kernels-deferred")
    "cudaLibraryGetKernel" = @("id:cuda-library-get-kernel-deferred")
    "cudaLibraryGetGlobal" = @("id:cuda-library-get-global-deferred")
    "cudaLibraryGetManaged" = @("id:cuda-library-get-managed-deferred")
    "cudaLibraryGetUnifiedFunction" = @("id:cuda-library-get-unified-function-deferred")
    "cudaKernelSetAttributeForDevice" = @("id:cuda-kernel-set-attribute-for-device-deferred")
    "cudaDeviceGetExecutionCtx" = @("id:cuda-device-get-execution-ctx-deferred")
    "cudaDeviceGetDevResource" = @("id:cuda-device-get-dev-resource-deferred")
    "cudaExecutionCtxGetDevice" = @("id:cuda-execution-ctx-get-device-deferred")
    "cudaExecutionCtxGetId" = @("id:cuda-execution-ctx-get-id-deferred")
    "cudaExecutionCtxSynchronize" = @("id:cuda-execution-ctx-synchronize-deferred")
    "cudaExecutionCtxStreamCreate" = @("id:cuda-execution-ctx-stream-create-deferred")
    "cudaExecutionCtxRecordEvent" = @("id:cuda-execution-ctx-record-event-deferred")
    "cudaExecutionCtxWaitEvent" = @("id:cuda-execution-ctx-wait-event-deferred")
    "cudaExecutionCtxGetDevResource" = @("id:cuda-execution-ctx-get-dev-resource-deferred")
    "cudaGraphAddMemsetNode" = @("id:*cuda-graph-add-memset-node-deferred")
    "cudaGraphAddMemAllocNode" = @("id:cuda-cuda-graph-add-mem-alloc-node-deferred")
    "cudaGraphAddMemFreeNode" = @("id:cuda-cuda-graph-add-mem-free-node-deferred")
    "cudaGraphAddChildGraphNode" = @("id:*cuda-graph-add-child-graph-node-deferred")
    "cudaGraphAddNode" = @("id:*cuda-cuda-graph-add-node-deferred")
    "cudaGraphConditionalHandleCreate" = @("id:cuda-cuda-graph-conditional-handle-create-deferred")
    "cudaGraphConditionalHandleCreate_v2" = @("id:cuda-cuda-graph-conditional-handle-create-v2-deferred")
    "cudaGraphChildGraphNodeGetGraph" = @("id:*cuda-graph-child-graph-node-get-graph-deferred")
    "cudaGraphExecChildGraphNodeSetParams" = @("id:*cuda-graph-exec-child-graph-node-set-params-deferred")
    "cudaGraphExecUpdate" = @("id:*cuda-graph-exec-update-deferred")
    "cudaGraphInstantiateWithParams" = @("id:*cuda-graph-instantiate-with-params-deferred")
    "cudaGraphKernelNodeCopyAttributes" = @("id:*cuda-graph-kernel-node-copy-attributes-deferred")
    "cudaGraphDestroyNode" = @("id:*cuda-graph-destroy-node-deferred")
    "cudaGraphExecMemsetNodeSetParams" = @("id:*cuda-graph-exec-memset-node-set-params-deferred")
    "cudaGraphKernelNodeGetParams" = @("id:*graph-kernel-node-get-params-deferred")
    "cudaGraphHostNodeGetParams" = @("id:*graph-host-node-get-params-deferred")
    "cudaGraphMemAllocNodeGetParams" = @("id:*graph-mem-alloc-node-get-params-deferred")
    "cudaGraphMemFreeNodeGetParams" = @("id:*graph-mem-free-node-get-params-deferred")
    "cudaGraphExternalSemaphoresSignalNodeGetParams" = @("id:*graph-external-semaphores-signal-node-get-params-deferred")
    "cudaGraphExternalSemaphoresWaitNodeGetParams" = @("id:*graph-external-semaphores-wait-node-get-params-deferred")
    "cudaGraphNodeGetContainingGraph" = @("id:*graph-node-get-containing-graph-deferred")
    "cudaStreamGetCaptureInfo_v3" = @("id:*stream-get-capture-info-v3-deferred")
    "cudaStreamUpdateCaptureDependencies" = @("id:*stream-update-capture-dependencies-deferred")
    "cudaStreamGetCaptureInfo_ptsz" = @("id:*stream-get-capture-info-ptsz-deferred")
    "cudaStreamUpdateCaptureDependencies_ptsz" = @("id:*stream-update-capture-dependencies-ptsz-deferred")
    "cudaStreamUpdateCaptureDependencies_v2" = @("id:*stream-update-capture-dependencies-v2-deferred")
    "cudaStreamBeginCaptureToGraph" = @("id:*stream-begin-capture-to-graph-deferred")
    "cudaStreamGetDevResource" = @("id:*stream-get-dev-resource-deferred")
    "cudaIpcGetEventHandle" = @("id:cuda-ipc-get-event-handle-deferred")
    "cudaIpcGetMemHandle" = @("id:cuda-ipc-get-mem-handle-deferred")
    "cudaLogsCurrent" = @("id:*logs-current-deferred")
    "cudaLogsDumpToMemory" = @("id:*logs-dump-to-memory-deferred")
    "cudaLogsDumpToFile" = @("id:*logs-dump-to-file-deferred")
    "cudaMemPrefetchBatchAsync" = @("id:cuda-mem-prefetch-batch-async-deferred")
    "cudaMemDiscardBatchAsync" = @("id:cuda-mem-discard-batch-async-deferred")
    "cudaMemDiscardAndPrefetchBatchAsync" = @("id:cuda-mem-discard-and-prefetch-batch-async-deferred")
    "cudaMemPrefetchAsync_v2" = @("id:cuda-mem-prefetch-async-v2-deferred")
    "cudaMemAdvise_v2" = @("id:cuda-mem-advise-v2-deferred")
  }

  if (-not $aliasMap.ContainsKey($FunctionName)) {
    return @()
  }

  $aliases = @($aliasMap[$FunctionName])
  if ($deferredHistoryAliasMap.ContainsKey($FunctionName)) {
    $aliases += @($deferredHistoryAliasMap[$FunctionName])
  }

  $matches = New-Object System.Collections.Generic.List[object]
  foreach ($api in $ManifestApis) {
    if ($api.Module -ne "cuda") { continue }
    foreach ($alias in $aliases) {
      $idPattern = $alias.Substring(3)
      if ($api.Id -like $idPattern) {
        $matches.Add($api) | Out-Null
        break
      }
    }
  }

  return @($matches | Sort-Object EntryPoint -Unique)
}

function Find-CudaManifestApis {
  param([object[]]$ManifestApis, [string]$FunctionName)

  $explicitMatches = @(Find-ExplicitCudaManifestApis $ManifestApis $FunctionName)
  if ($explicitMatches.Count -gt 0) {
    return $explicitMatches
  }

  $candidates = Get-CudaCandidates $FunctionName
  $matches = New-Object System.Collections.Generic.List[object]
  foreach ($api in $ManifestApis) {
    if ($api.Module -ne "cuda") { continue }
    foreach ($candidate in $candidates) {
      if ($api.SearchText.Contains($candidate)) {
        $matches.Add($api) | Out-Null
        break
      }
    }
  }

  return @($matches | Select-Object -Unique)
}

function Test-DeferredManifestId {
  param([string]$Id)

  return -not [string]::IsNullOrWhiteSpace($Id) -and $Id.EndsWith("-deferred", [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-ResolvedImplementationStatus {
  param(
    [object[]]$Matches,
    [System.Collections.Generic.HashSet[string]]$NativeExports
  )

  if (@($Matches).Count -eq 0) {
    return "missing"
  }

  $presentMatches = @($Matches | Where-Object {
      -not [string]::IsNullOrWhiteSpace([string]$_.EntryPoint) -and $NativeExports.Contains([string]$_.EntryPoint)
    })
  if ($presentMatches.Count -eq 0) {
    return "manifest-only"
  }

  $implementedMatches = @($presentMatches | Where-Object { -not (Test-DeferredManifestId -Id ([string]$_.Id)) })
  $deferredMatches = @($presentMatches | Where-Object { Test-DeferredManifestId -Id ([string]$_.Id) })

  if ($implementedMatches.Count -gt 0 -and $deferredMatches.Count -gt 0) {
    return "implemented-with-deferred-history"
  }

  if ($implementedMatches.Count -gt 0) {
    return "implemented"
  }

  return "deferred-only"
}

function Get-CudaRuntimeInterfaces {
  param([object]$Toolkit)

  $results = New-Object System.Collections.Generic.List[object]
  $raw = Get-Content -LiteralPath $Toolkit.Header -Raw -Encoding utf8
  $text = Remove-CxxComments $raw
  $matches = [regex]::Matches($text, 'extern\s+__host__\s+(?:__cudart_builtin__\s+)?cudaError_t\s+CUDARTAPI\s+(cuda[A-Za-z0-9_]+)\s*\(')
  foreach ($match in $matches) {
    $name = $match.Groups[1].Value
    $results.Add([pscustomobject]@{
      Vendor = "CUDA"
      ToolkitVersion = $Toolkit.Version
      VersionLine = $Toolkit.VersionLine
      Header = Split-Path -Leaf $Toolkit.Header
      Function = $name
      Category = Get-CudaCategory $name
    }) | Out-Null
  }

  return @($results | Sort-Object Function -Unique)
}

$manifestApis = @(Get-ManifestApis $RepositoryRoot)
$nativeExports = Get-NativeExportNames $RepositoryRoot
$tensorRtSourceText = Get-ManagedSourceText $RepositoryRoot "src\JYPPX.TensorRtSharp"
$cudaSourceText = Get-ManagedSourceText $RepositoryRoot "src\JYPPX.CudaSharp"

$tensorRtRows = New-Object System.Collections.Generic.List[object]
$tensorRtPackages = @(Get-TensorRtPackageInfo $TensorRtPackageRoot)
foreach ($package in $tensorRtPackages) {
  $interfaces = @(Get-TensorRtInterfaces $package)
  foreach ($item in $interfaces) {
    $matches = @(Find-MatchedManifestApis $manifestApis "tensorrt" $item.VersionLine $item.Class $item.Method)
    $entryPoints = @($matches | ForEach-Object { $_.EntryPoint })
    $ids = @($matches | ForEach-Object { $_.Id })
    $nativeStatus = if ($entryPoints.Count -eq 0) { "missing" } elseif (@($entryPoints | Where-Object { -not $nativeExports.Contains($_) }).Count -eq 0) { "present" } else { "manifest-only" }
    $implementationStatus = Get-ResolvedImplementationStatus -Matches $matches -NativeExports $nativeExports
    $managedInterop = if ($matches.Count -gt 0) { "generated-or-manual" } else { "missing" }
    $managedLikely = "unknown"
    foreach ($candidate in (Get-CSharpMethodCandidates $item.Method)) {
      if ($tensorRtSourceText.Contains($candidate)) {
        $managedLikely = "likely-present"
        break
      }
    }

    $tensorRtRows.Add([pscustomobject]@{
      Vendor = $item.Vendor
      Package = $item.Package
      VersionLine = $item.VersionLine
      Version = $item.Version
      CudaVariant = $item.CudaVariant
      Header = $item.Header
      Class = $item.Class
      Method = $item.Method
      Interface = $item.Interface
      Category = $item.Category
      ImplementationStatus = $implementationStatus
      NativeManifestStatus = if ($matches.Count -gt 0) { "present" } else { "missing" }
      NativeSourceStatus = $nativeStatus
      ManagedInteropStatus = $managedInterop
      ManagedHighLevelHeuristic = $managedLikely
      MatchedManifestIds = ($ids -join ';')
      MatchedEntryPoints = ($entryPoints -join ';')
      Notes = if ($matches.Count -eq 0) { "not found by manifest/token heuristic" } else { "" }
    }) | Out-Null
  }
}

$cudaRows = New-Object System.Collections.Generic.List[object]
$cudaToolkits = @(Get-CudaToolkitInfo $CudaToolkitRoot)
foreach ($toolkit in $cudaToolkits) {
  $interfaces = @(Get-CudaRuntimeInterfaces $toolkit)
  foreach ($item in $interfaces) {
    $matches = @(Find-CudaManifestApis $manifestApis $item.Function)
    $entryPoints = @($matches | ForEach-Object { $_.EntryPoint })
    $ids = @($matches | ForEach-Object { $_.Id })
    $nativeStatus = if ($entryPoints.Count -eq 0) { "missing" } elseif (@($entryPoints | Where-Object { -not $nativeExports.Contains($_) }).Count -eq 0) { "present" } else { "manifest-only" }
    $implementationStatus = Get-ResolvedImplementationStatus -Matches $matches -NativeExports $nativeExports
    $managedLikely = "unknown"
    foreach ($candidate in (Get-CSharpMethodCandidates $item.Function)) {
      if ($cudaSourceText.Contains($candidate)) {
        $managedLikely = "likely-present"
        break
      }
    }

    $cudaRows.Add([pscustomobject]@{
      Vendor = $item.Vendor
      ToolkitVersion = $item.ToolkitVersion
      VersionLine = $item.VersionLine
      Header = $item.Header
      Function = $item.Function
      Category = $item.Category
      ImplementationStatus = $implementationStatus
      NativeManifestStatus = if ($matches.Count -gt 0) { "present" } else { "missing" }
      NativeSourceStatus = $nativeStatus
      ManagedInteropStatus = if ($matches.Count -gt 0) { "generated-or-manual" } else { "missing" }
      ManagedHighLevelHeuristic = $managedLikely
      MatchedManifestIds = ($ids -join ';')
      MatchedEntryPoints = ($entryPoints -join ';')
      Notes = if ($matches.Count -eq 0) { "not found by manifest/token heuristic" } else { "" }
    }) | Out-Null
  }
}

$tensorRtCsv = Join-Path $OutputDirectory "tensorrt-interface-coverage.csv"
$cudaCsv = Join-Path $OutputDirectory "cuda-runtime-interface-coverage.csv"
$tensorRtComparisonCsv = Join-Path $OutputDirectory "tensorrt-interface-comparison.csv"
$cudaComparisonCsv = Join-Path $OutputDirectory "cuda-runtime-interface-comparison.csv"
$tensorRtJson = Join-Path $OutputDirectory "tensorrt-interface-coverage.json"
$cudaJson = Join-Path $OutputDirectory "cuda-runtime-interface-coverage.json"
$summaryPath = Join-Path $OutputDirectory "interface-coverage-summary.md"

$sortedTensorRtRows = @($tensorRtRows | Sort-Object Package,Class,Method)
$sortedCudaRows = @($cudaRows | Sort-Object ToolkitVersion,Function)

$sortedTensorRtRows | Export-Csv -LiteralPath $tensorRtCsv -NoTypeInformation -Encoding utf8
$sortedCudaRows | Export-Csv -LiteralPath $cudaCsv -NoTypeInformation -Encoding utf8
$sortedTensorRtRows | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $tensorRtJson -Encoding utf8
$sortedCudaRows | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $cudaJson -Encoding utf8

$sortedTensorRtRows |
  Select-Object Vendor,
    @{ Name = "TensorRtPackage"; Expression = { $_.Package } },
    @{ Name = "TensorRtVersion"; Expression = { $_.Version } },
    @{ Name = "TensorRtLine"; Expression = { $_.VersionLine } },
    CudaVariant,
    Header,
    Class,
    Method,
    Interface,
    Category,
    ImplementationStatus,
    NativeManifestStatus,
    NativeSourceStatus,
    ManagedInteropStatus,
    ManagedHighLevelHeuristic,
    MatchedManifestIds,
    MatchedEntryPoints,
    Notes |
  Export-Csv -LiteralPath $tensorRtComparisonCsv -NoTypeInformation -Encoding utf8

$sortedCudaRows |
  Select-Object Vendor,
    @{ Name = "CudaToolkitVersion"; Expression = { $_.ToolkitVersion } },
    @{ Name = "CudaLine"; Expression = { $_.VersionLine } },
    Header,
    Function,
    Category,
    ImplementationStatus,
    NativeManifestStatus,
    NativeSourceStatus,
    ManagedInteropStatus,
    ManagedHighLevelHeuristic,
    MatchedManifestIds,
    MatchedEntryPoints,
    Notes |
  Export-Csv -LiteralPath $cudaComparisonCsv -NoTypeInformation -Encoding utf8

$summary = [System.Text.StringBuilder]::new()
[void]$summary.AppendLine("# Interface Coverage Summary")
[void]$summary.AppendLine()
[void]$summary.AppendLine("Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
[void]$summary.AppendLine()
[void]$summary.AppendLine("This report is generated from local TensorRT/CUDA headers and the repository manifests. Coverage matching is heuristic because the bridge intentionally exposes C ABI methods with safe .NET-oriented names rather than NVIDIA C++ method names.")
[void]$summary.AppendLine()
[void]$summary.AppendLine("## Inputs")
[void]$summary.AppendLine()
[void]$summary.AppendLine("- TensorRT package root: ``$TensorRtPackageRoot``")
[void]$summary.AppendLine("- CUDA toolkit root: ``$CudaToolkitRoot``")
[void]$summary.AppendLine("- Manifest API count: $($manifestApis.Count)")
[void]$summary.AppendLine()
[void]$summary.AppendLine("## TensorRT Packages")
[void]$summary.AppendLine()
foreach ($group in $tensorRtRows | Group-Object Package) {
  $total = $group.Count
  $covered = @($group.Group | Where-Object { $_.NativeManifestStatus -eq "present" }).Count
  $source = @($group.Group | Where-Object { $_.NativeSourceStatus -eq "present" }).Count
  $implemented = @($group.Group | Where-Object { $_.ImplementationStatus -in @("implemented", "implemented-with-deferred-history") }).Count
  $deferredOnly = @($group.Group | Where-Object { $_.ImplementationStatus -eq "deferred-only" }).Count
  [void]$summary.AppendLine("- ``$($group.Name)``: official interfaces scanned=$total, manifest matched=$covered, native source present=$source, implemented=$implemented, deferred-only=$deferredOnly")
}
[void]$summary.AppendLine()
[void]$summary.AppendLine("## TensorRT Missing By Package / Category")
[void]$summary.AppendLine()
foreach ($group in $tensorRtRows | Where-Object { $_.NativeManifestStatus -ne "present" } | Group-Object Package,Category | Sort-Object Count -Descending | Select-Object -First 40) {
  [void]$summary.AppendLine("- $($group.Name): $($group.Count)")
}
[void]$summary.AppendLine()
[void]$summary.AppendLine("## CUDA Toolkits")
[void]$summary.AppendLine()
foreach ($group in $cudaRows | Group-Object ToolkitVersion) {
  $total = $group.Count
  $covered = @($group.Group | Where-Object { $_.NativeManifestStatus -eq "present" }).Count
  $source = @($group.Group | Where-Object { $_.NativeSourceStatus -eq "present" }).Count
  $implemented = @($group.Group | Where-Object { $_.ImplementationStatus -in @("implemented", "implemented-with-deferred-history") }).Count
  $deferredOnly = @($group.Group | Where-Object { $_.ImplementationStatus -eq "deferred-only" }).Count
  [void]$summary.AppendLine("- ``CUDA $($group.Name)``: runtime functions scanned=$total, manifest matched=$covered, native source present=$source, implemented=$implemented, deferred-only=$deferredOnly")
}
[void]$summary.AppendLine()
[void]$summary.AppendLine("## CUDA Missing By Toolkit / Category")
[void]$summary.AppendLine()
foreach ($group in $cudaRows | Where-Object { $_.NativeManifestStatus -ne "present" } | Group-Object ToolkitVersion,Category | Sort-Object Count -Descending | Select-Object -First 40) {
  [void]$summary.AppendLine("- $($group.Name): $($group.Count)")
}
[void]$summary.AppendLine()
[void]$summary.AppendLine("## Next Use")
[void]$summary.AppendLine()
[void]$summary.AppendLine('- Use `tensorrt-interface-coverage.csv` and `cuda-runtime-interface-coverage.csv` as the persistent interface checklist.')
[void]$summary.AppendLine('- `ImplementationStatus` separates `implemented`, `implemented-with-deferred-history`, `deferred-only`, `manifest-only`, and `missing`; do not treat deferred history as the active implementation when a real non-deferred export is present.')
[void]$summary.AppendLine('- Compatibility CSVs are also written to `tensorrt-interface-comparison.csv` and `cuda-runtime-interface-comparison.csv` for older review notes.')
[void]$summary.AppendLine('- After each API batch, rerun `eng/Export-InterfaceCoverageMatrix.ps1`; rows should move from `missing`, `manifest-only`, or `deferred-only` toward `implemented` or `implemented-with-deferred-history`.')
[void]$summary.AppendLine('- For release-quality decisions, manually review high-priority rows because token matching can produce false negatives or broad matches.')

Set-Content -LiteralPath $summaryPath -Value $summary.ToString() -Encoding utf8

Write-Host "TensorRT interface coverage written to $tensorRtCsv"
Write-Host "CUDA runtime interface coverage written to $cudaCsv"
Write-Host "TensorRT interface comparison written to $tensorRtComparisonCsv"
Write-Host "CUDA runtime interface comparison written to $cudaComparisonCsv"
Write-Host "Interface coverage summary written to $summaryPath"
