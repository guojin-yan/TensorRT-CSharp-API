[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
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

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Read-TextOrEmpty {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return "" }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8
}

function ConvertTo-Array {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Test-AnyToken {
  param([string]$Text, [string[]]$Tokens)

  foreach ($token in $Tokens) {
    if ($Text.Contains($token, [StringComparison]::Ordinal)) { return $true }
  }

  return $false
}

function New-ParityItem {
  param(
    [string]$OptionId,
    [string]$OfficialTrtexecOption,
    [string[]]$Tokens,
    [string[]]$WinFormsTokens,
    [string]$Status,
    [string]$ProofBoundary,
    [string]$NextImplementationPath,
    [string]$NextAction
  )

  $cliSupported = Test-AnyToken -Text $commandSource -Tokens $Tokens
  $optionsSupported = Test-AnyToken -Text $optionsSource -Tokens $Tokens
  $winFormsSupported = Test-AnyToken -Text $winFormsSource -Tokens $WinFormsTokens
  $readmeDocumented = Test-AnyToken -Text $readmeSource -Tokens $Tokens
  $commandPreviewSupported = $winFormsSource.Contains("ToArgumentLine", [StringComparison]::Ordinal) -and $winFormsSource.Contains("_commandPreview", [StringComparison]::Ordinal)

  [pscustomobject]@{
    optionId = $OptionId
    officialTrtexecOption = $OfficialTrtexecOption
    cliSupported = $cliSupported
    sharedOptionsSupported = $optionsSupported
    winFormsSupported = $winFormsSupported
    commandPreviewSupported = $commandPreviewSupported
    readmeDocumented = $readmeDocumented
    status = $Status
    proofBoundary = $ProofBoundary
    isRuntimeProof = $false
    isPackageConsumerRuntimeProof = $false
    performsPublish = $false
    nextImplementationPath = $NextImplementationPath
    nextAction = $NextAction
  }
}

$featureMatrix = Read-JsonOrNull "applications/TensorRtExec/tensor-rt-exec-feature-matrix.json"
$gapList = Read-JsonOrNull "applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json"
$parityMatrix = Read-JsonOrNull "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json"
$commandSource = Read-TextOrEmpty "applications/TensorRtExec/Console/TensorRtExecCommand.cs"
$optionsSource = Read-TextOrEmpty "applications/TensorRtExec/Core/TensorRtExecOptions.cs"
$winFormsSource = Read-TextOrEmpty "applications/TensorRtExec/WinForms/MainForm.cs"
$readmeSource = Read-TextOrEmpty "applications/TensorRtExec/README.md"

$items = @(
  New-ParityItem -OptionId "onnx" -OfficialTrtexecOption "--onnx" -Tokens @("--onnx") -WinFormsTokens @("_onnxPath", "OnBrowseOnnx") -Status "implemented" -ProofBoundary "ONNX build/report is build evidence only until real model sample proof exists." -NextImplementationPath "applications/TensorRtExec; src/JYPPX.TensorRtSharp.Tools" -NextAction "Keep cross-linking reports to owner real model records."
  New-ParityItem -OptionId "save-engine" -OfficialTrtexecOption "--saveEngine / --save-engine" -Tokens @("--saveEngine", "--save-engine") -WinFormsTokens @("_enginePath", "OnBrowseSaveEngine") -Status "implemented" -ProofBoundary "Serialized engine output is not inference proof." -NextImplementationPath "applications/TensorRtExec" -NextAction "Add owner engine SHA256 only after real builds."
  New-ParityItem -OptionId "load-engine" -OfficialTrtexecOption "--loadEngine / --load-engine" -Tokens @("--loadEngine", "--load-engine") -WinFormsTokens @("_loadEnginePath", "OnBrowseLoadEngine") -Status "readonly-diagnostics" -ProofBoundary "Load-engine path is diagnostics/preflight, not generic enqueue proof." -NextImplementationPath "applications/TensorRtExec; src/JYPPX.TensorRtSharp" -NextAction "Keep enqueue and output validation behind real runtime proof."
  New-ParityItem -OptionId "shape-profiles" -OfficialTrtexecOption "--minShapes / --optShapes / --maxShapes / --shapes / --inputShapes" -Tokens @("--minShapes", "--optShapes", "--maxShapes", "--shapes", "--inputShapes") -WinFormsTokens @("_minShapes", "_optShapes", "_maxShapes") -Status "implemented-report" -ProofBoundary "Shape profile settings do not prove every profile executed correctly." -NextImplementationPath "applications/TensorRtExec; samples/YoloVision" -NextAction "Tie real sample evidence to explicit profile metadata."
  New-ParityItem -OptionId "precision" -OfficialTrtexecOption "--fp16 / --bf16 / --noTF32" -Tokens @("--fp16", "--bf16", "--noTF32") -WinFormsTokens @("_fp16", "_bf16", "_tf32") -Status "wrapper-ready" -ProofBoundary "Precision switches require host/model proof before runtime claims." -NextImplementationPath "applications/TensorRtExec" -NextAction "Keep host-specific precision proof in owner records."
  New-ParityItem -OptionId "int8-calibration" -OfficialTrtexecOption "--int8 / --calib" -Tokens @("--int8", "--calib") -WinFormsTokens @("_int8", "_calibrationCachePath") -Status "diagnostic" -ProofBoundary "INT8 calibration/cache ownership remains deferred and cannot be proof." -NextImplementationPath "native/src/tensorrt; docs/articles/zh-cn/allocator-callback-owner-design.md" -NextAction "Design calibrator ownership before native implementation."
  New-ParityItem -OptionId "workspace-memory-pool" -OfficialTrtexecOption "--workspace / --memPoolSize" -Tokens @("--workspace", "--memPoolSize") -WinFormsTokens @("_workspace", "_memoryPoolSizes") -Status "implemented-report" -ProofBoundary "Memory settings are build/report evidence only." -NextImplementationPath "applications/TensorRtExec; src/JYPPX.TensorRtSharp.Tools" -NextAction "Keep builder config readback separate from runtime proof."
  New-ParityItem -OptionId "timing-cache" -OfficialTrtexecOption "--timingCacheFile / --timingCache / --exportTimingCache" -Tokens @("--timingCacheFile", "--timingCache", "--exportTimingCache") -WinFormsTokens @("_timingCachePath", "_exportTimingCachePath") -Status "implemented-build-cache-lifecycle" -ProofBoundary "Timing cache import/export and hashes are build-cache evidence, not runtime proof." -NextImplementationPath "applications/TensorRtExec; native/src/tensorrt" -NextAction "Keep owner cache hashes separate from runtime proof."
  New-ParityItem -OptionId "plugins" -OfficialTrtexecOption "--plugins" -Tokens @("--plugins") -WinFormsTokens @("_plugins") -Status "diagnostic" -ProofBoundary "Plugin paths/inventory diagnostics do not prove plugin library load/register/deregister." -NextImplementationPath "src/JYPPX.TensorRtSharp/TensorRtPluginRegistryInventory.cs" -NextAction "Keep copied inventory metadata read-only."
  New-ParityItem -OptionId "profiling" -OfficialTrtexecOption "--profilingVerbosity / --dumpProfile / --exportProfile / --saveProfile" -Tokens @("--profilingVerbosity", "--dumpProfile", "--exportProfile", "--saveProfile") -WinFormsTokens @("_profilingVerbosity", "_exportProfilePath", "_saveProfilePath") -Status "implemented-report" -ProofBoundary "Profiling artifacts are diagnostics unless real enqueue log proves runtime." -NextImplementationPath "applications/TensorRtExec" -NextAction "Hash profile artifacts in owner proof only after real run."
  New-ParityItem -OptionId "layer-info" -OfficialTrtexecOption "--dumpLayerInfo / --exportLayerInfo" -Tokens @("--dumpLayerInfo", "--exportLayerInfo") -WinFormsTokens @("_layerInfoPath", "OnBrowseLayerInfo") -Status "implemented-report" -ProofBoundary "Layer info is diagnostic metadata, not output correctness proof." -NextImplementationPath "src/JYPPX.TensorRtSharp/TensorRtEngineInspector.Trt11Diagnostics.cs" -NextAction "Use inspector output as diagnostics only."
  New-ParityItem -OptionId "report-export" -OfficialTrtexecOption "--exportReport / --report" -Tokens @("--exportReport", "--report") -WinFormsTokens @("_reportPath", "OnBrowseReport") -Status "implemented-report" -ProofBoundary "Report export aliases choose JSON/Markdown output paths only; reports are build/report evidence and not runtime proof." -NextImplementationPath "applications/TensorRtExec; samples/OnnxToEngine; src/JYPPX.TensorRtSharp.Tools" -NextAction "Keep --exportReport canonical while accepting --report for owner-facing commands."
  New-ParityItem -OptionId "runtime-benchmark" -OfficialTrtexecOption "--iterations / --warmUp / --duration / --streams / --infStreams / --avgRuns / --percentile / --threads / --useSpinWait / --useCudaGraph / --noDataTransfers" -Tokens @("--iterations", "--warmUp", "--duration", "--streams", "--infStreams", "--avgRuns", "--percentile", "--threads", "--useSpinWait", "--useCudaGraph", "--noDataTransfers") -WinFormsTokens @("_iterations", "_warmUp", "_duration", "_streams", "_infStreams", "_avgRuns", "_percentile", "_threads", "_useSpinWait", "_useCudaGraph", "_noDataTransfers") -Status "implemented-bounded-runtime" -ProofBoundary "Bounded scheduler execution is not external model correctness or package-consumer proof." -NextImplementationPath "applications/TensorRtExec" -NextAction "Require real model logs and hashes before proof promotion."
  New-ParityItem -OptionId "binding-output" -OfficialTrtexecOption "--loadInputs / --dumpOutput / --dumpRawBindingsToFile / --exportOutput / --exportTimes" -Tokens @("--loadInputs", "--dumpOutput", "--dumpRawBindingsToFile", "--exportOutput", "--exportTimes") -WinFormsTokens @("_loadInputs", "_dumpOutput", "_dumpRawBindingsPath", "_exportOutputPath", "_exportTimesPath") -Status "bounded" -ProofBoundary "Build-only artifacts write boundary JSON; external ONNX binding output requires real model evidence." -NextImplementationPath "samples/YoloVision; applications/TensorRtExec" -NextAction "Use YoloVision owner templates to capture tensor role metadata."
  New-ParityItem -OptionId "safety-cache-policy" -OfficialTrtexecOption "--safe / --consistency / --builderCache / --noBuilderCache" -Tokens @("--safe", "--consistency", "--builderCache", "--noBuilderCache") -WinFormsTokens @("_safe", "_consistency", "_builderCache", "_noBuilderCache") -Status "parse-report-only" -ProofBoundary "Safety and builder cache switches are intent/report fields, not runtime safety proof." -NextImplementationPath "applications/TensorRtExec" -NextAction "Keep parse-only until native behavior and model smoke prove effect."
  New-ParityItem -OptionId "device-dla" -OfficialTrtexecOption "--device / --useDLACore / --allowGPUFallback / --tacticSources / --directIO / --sparsity / --stronglyTyped" -Tokens @("--device", "--useDLACore", "--allowGPUFallback", "--tacticSources", "--directIO", "--sparsity", "--stronglyTyped") -WinFormsTokens @("_deviceOrdinal", "_dlaCore", "_allowGpuFallback", "_tacticSources", "_directIo", "_sparsity", "_stronglyTyped") -Status "implemented-build-readback-with-version-guards" -ProofBoundary "Device and builder-policy readback are host build evidence; not DLA/model/package runtime proof." -NextImplementationPath "applications/TensorRtExec; src/JYPPX.TensorRtSharp.Tools" -NextAction "Collect DLA-capable real-model proof separately."
)

$runtimeProofItems = @($items | Where-Object { [bool]$_.isRuntimeProof -or [bool]$_.isPackageConsumerRuntimeProof })
$cliSupportedCount = @($items | Where-Object { [bool]$_.cliSupported }).Count
$winFormsSupportedCount = @($items | Where-Object { [bool]$_.winFormsSupported }).Count
$previewSupportedCount = @($items | Where-Object { [bool]$_.commandPreviewSupported }).Count

$record = [pscustomobject]@{
  recordKind = "tensor-rt-exec-gui-cli-parity-checklist"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  checklistState = "release-candidate-gui-cli-parity-non-proof"
  application = "applications/TensorRtExec"
  featureMatrixPresent = ($null -ne $featureMatrix)
  gapListPresent = ($null -ne $gapList)
  parityMatrixPresent = ($null -ne $parityMatrix)
  itemCount = $items.Count
  cliSupportedCount = $cliSupportedCount
  winFormsSupportedCount = $winFormsSupportedCount
  commandPreviewSupportedCount = $previewSupportedCount
  runtimeProofItems = $runtimeProofItems.Count
  packageConsumerRuntimeProofItems = 0
  items = @($items)
  sourceArtifacts = @(
    "applications/TensorRtExec/tensor-rt-exec-feature-matrix.json",
    "applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json",
    "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json",
    "applications/TensorRtExec/README.md",
    "applications/TensorRtExec/Core/TensorRtExecOptions.cs",
    "applications/TensorRtExec/Console/TensorRtExecCommand.cs",
    "applications/TensorRtExec/WinForms/MainForm.cs"
  )
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  boundary = "TensorRtExec GUI/CLI parity checklist is implementation-readiness evidence only; CLI/WinForms command preview, build reports, dry-runs, screenshots, and diagnostics are not runtime proof and not package-consumer-runtime proof."
}

$jsonPath = Join-Path $OutputRoot "tensor-rt-exec-gui-cli-parity-checklist.json"
$markdownPath = Join-Path $OutputRoot "tensor-rt-exec-gui-cli-parity-checklist.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $items | ForEach-Object {
  "| ``$($_.optionId)`` | ``$($_.officialTrtexecOption)`` | ``$($_.cliSupported)`` | ``$($_.winFormsSupported)`` | ``$($_.commandPreviewSupported)`` | ``$($_.status)`` | $(ConvertTo-MarkdownCell $_.proofBoundary) |"
}

$markdown = @"
# TensorRtExec GUI/CLI Parity Checklist

| Field | Value |
| --- | --- |
| checklistState | ``$($record.checklistState)`` |
| itemCount | ``$($record.itemCount)`` |
| cliSupportedCount | ``$($record.cliSupportedCount)`` |
| winFormsSupportedCount | ``$($record.winFormsSupportedCount)`` |
| commandPreviewSupportedCount | ``$($record.commandPreviewSupportedCount)`` |
| runtimeProofItems | ``$($record.runtimeProofItems)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |

## Items

| ID | Trtexec Option | CLI | WinForms | Preview | Status | Boundary |
| --- | --- | ---: | ---: | ---: | --- | --- |
$($rows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "TensorRtExec GUI/CLI parity checklist written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ChecklistState=$($record.checklistState) Items=$($items.Count) RuntimeProofItems=$($runtimeProofItems.Count) CliSupported=$cliSupportedCount WinFormsSupported=$winFormsSupportedCount PreviewSupported=$previewSupportedCount"
