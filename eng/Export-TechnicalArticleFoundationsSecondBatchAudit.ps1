[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot = "docs\articles\zh-cn\publishing"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return [System.IO.Path]::GetFullPath($Path)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Test-RepositoryReference {
  param([string]$Reference)

  $fullPath = Resolve-RepositoryPath ($Reference.Replace('/', '\'))
  if ($Reference.IndexOfAny([char[]]'*?') -ge 0) {
    return @(Get-ChildItem -Path $fullPath -ErrorAction SilentlyContinue).Count -gt 0
  }

  return Test-Path -LiteralPath $fullPath
}

function Test-ArticleMarkdownLink {
  param(
    [string]$ArticlePath,
    [string]$Target
  )

  if ($Target -match '^(?:https?://|mailto:|#)') {
    return $true
  }

  $pathPart = ($Target -split '#', 2)[0]
  if ([string]::IsNullOrWhiteSpace($pathPart)) {
    return $true
  }

  $articleDirectory = [System.IO.Path]::GetDirectoryName($ArticlePath)
  $fullPath = [System.IO.Path]::GetFullPath((Join-Path $articleDirectory $pathPart))
  return Test-Path -LiteralPath $fullPath
}

$definitions = @(
  [ordered]@{
    roadmapEntryIds = @(28, 44)
    title = "Refit weights 使用场景与博客版"
    path = "docs/articles/zh-cn/blog-refit-weights-guide.md"
    beforeLineCount = 112
    beforeCharacterCount = 3173
    requiredMarkers = @("GetAllEntries", "SetWeights", "RefitCudaEngine", "OutputChanged=True")
    requiredAnchors = @(
      "src/JYPPX.TensorRtSharp/Refit/TensorRtRefitter.cs",
      "src/JYPPX.TensorRtSharp/Refit/TensorRtRefitWeightsBuffer.cs",
      "smoke/RefitWeightsSmokeRunner/Program.cs"
    )
  },
  [ordered]@{
    roadmapEntryIds = @(29)
    title = "TensorRT 11 modern layers"
    path = "docs/articles/zh-cn/trt11-modern-layers-guide.md"
    beforeLineCount = 31
    beforeCharacterCount = 736
    requiredMarkers = @("AddSqueeze", "TensorRtDims64", "Dims64Evidence", "JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11")
    requiredAnchors = @(
      "src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Trt11ModernLayers.cs",
      "native/manifests/tensorrt/v11/trt11-seventeenth-batch-dims64.manifest.json",
      "smoke/NetworkTrt11ModernLayersSmokeRunner/Program.cs",
      "smoke/NetworkTrt11ModernLayerMetadataRunner/Program.cs"
    )
  },
  [ordered]@{
    roadmapEntryIds = @(30, 45)
    title = "Network layer coverage 导览与博客版"
    path = "docs/articles/zh-cn/blog-network-layer-coverage-guide.md"
    beforeLineCount = 78
    beforeCharacterCount = 2684
    requiredMarkers = @("NetworkConvolutionScaleSmokeRunner", "OutputValidated", "TensorRtDims64", "real model + clean consumer proof")
    requiredAnchors = @(
      "artifacts/interface-coverage/tensorrt-interface-coverage.csv",
      "smoke/NetworkConvolutionScaleSmokeRunner",
      "smoke/NetworkTrt11ModernLayersSmokeRunner",
      "smoke/NetworkTrt11AdvancedLayersSmokeRunner"
    )
  },
  [ordered]@{
    roadmapEntryIds = @(31)
    title = "ErrorRecorder snapshot 与诊断"
    path = "docs/articles/zh-cn/error-recorder-diagnostics-design-gate.md"
    beforeLineCount = 61
    beforeCharacterCount = 3213
    requiredMarkers = @("TryGetErrorRecorderSnapshot", "CopiedDiagnosticsReady=True", "DirectRecorderOwnershipDeferred=True", "incRefCount")
    requiredAnchors = @(
      "src/JYPPX.TensorRtSharp/Callbacks/Monitoring/TensorRtErrorRecorderSnapshot.cs",
      "src/JYPPX.TensorRtSharp/Callbacks/Monitoring/TensorRtErrorRecorderDiagnosticsDesignGate.cs",
      "smoke/CallbackAllocatorSafeControlsSmokeRunner/Program.cs"
    )
  },
  [ordered]@{
    roadmapEntryIds = @(32)
    title = "Managed logger/profiler/progress monitor"
    path = "docs/articles/zh-cn/managed-logger-profiler-progress-monitor.md"
    beforeLineCount = 29
    beforeCharacterCount = 689
    requiredMarkers = @("EmitDiagnostic", "CallbackInvocationCount", "ManagedProgressMonitorAttach", "InvocationCount>0")
    requiredAnchors = @(
      "src/JYPPX.TensorRtSharp/Callbacks/Monitoring/TensorRtLogger.cs",
      "src/JYPPX.TensorRtSharp/Callbacks/Monitoring/TensorRtProfiler.cs",
      "src/JYPPX.TensorRtSharp/Callbacks/Monitoring/TensorRtProgressMonitor.cs",
      "smoke/ManagedLoggerCallbackSmokeRunner/Program.cs",
      "smoke/ManagedProfilerCallbackSmokeRunner/Program.cs",
      "smoke/ManagedProgressMonitorSmokeRunner/Program.cs"
    )
  },
  [ordered]@{
    roadmapEntryIds = @(38)
    title = "Dynamic Shape 博客版"
    path = "docs/articles/zh-cn/blog-dynamic-shape-optimization-profile.md"
    beforeLineCount = 70
    beforeCharacterCount = 2191
    requiredMarkers = @("SetShape", "SetInputShape", "GetReadiness", "DynamicShape Passed=True")
    requiredAnchors = @(
      "src/JYPPX.TensorRtSharp/Profiles/TensorRtOptimizationProfile.cs",
      "src/JYPPX.TensorRtSharp/Inference/TensorRtInferenceBindings.TensorGeometry.cs",
      "src/JYPPX.TensorRtSharp/Inference/TensorRtInferenceBindings.Execution.cs",
      "samples/DynamicShape/Program.cs"
    )
  },
  [ordered]@{
    roadmapEntryIds = @(39)
    title = "InferenceBindings Identity Network 博客版"
    path = "docs/articles/zh-cn/blog-inference-bindings-identity-network.md"
    beforeLineCount = 70
    beforeCharacterCount = 2156
    requiredMarkers = @("CopyInputFromHost", "AllocateDeviceBuffer", "BindAll", "InferenceBindings Passed=True")
    requiredAnchors = @(
      "src/JYPPX.TensorRtSharp/Inference/TensorRtInferenceBindings.Buffers.cs",
      "src/JYPPX.TensorRtSharp/Inference/TensorRtInferenceBindings.HostTransfers.cs",
      "src/JYPPX.TensorRtSharp/Inference/TensorRtInferenceBindings.AddressBinding.cs",
      "src/JYPPX.TensorRtSharp/Inference/TensorRtInferenceBindings.Execution.cs",
      "samples/InferenceBindings/Program.cs",
      "samples/InferenceBindings/README.md"
    )
  },
  [ordered]@{
    roadmapEntryIds = @(40)
    title = "ONNX Parser Engine RoundTrip 博客版"
    path = "docs/articles/zh-cn/blog-onnx-parser-engine-roundtrip.md"
    beforeLineCount = 71
    beforeCharacterCount = 2026
    requiredMarkers = @("--previewOnly", "--buildOnly", "EngineFileRoundTrip=True", "OutputMatch=True")
    requiredAnchors = @(
      "samples/OnnxToEngine/Program.cs",
      "samples/OnnxToEngine/trtexec-parity-matrix.json",
      "src/JYPPX.TensorRtSharp.Tools/Trtexec/TrtexecLikeParser.cs",
      "src/JYPPX.TensorRtSharp.Tools/Build/OnnxEngineBuildService.cs",
      "src/JYPPX.TensorRtSharp.Tools/Build/OnnxEngineBuildService.DeploymentConfiguration.cs",
      "src/JYPPX.TensorRtSharp.Tools/Build/OnnxEngineBuildService.RuntimeExecution.cs"
    )
  },
  [ordered]@{
    roadmapEntryIds = @(41)
    title = "MultiStream CUDA Stream/Event 博客版"
    path = "docs/articles/zh-cn/blog-multistream-cuda-stream-event.md"
    beforeLineCount = 87
    beforeCharacterCount = 2699
    requiredMarkers = @("IndependentStreams=True", "CrossStreamWait=True", "CudaStreamCreationFlags.NonBlocking", "WaitFor")
    requiredAnchors = @(
      "samples/MultiStream/Program.cs",
      "src/JYPPX.CudaSharp/Streams/CudaStream.cs",
      "src/JYPPX.CudaSharp/Events/CudaEvent.cs"
    )
  },
  [ordered]@{
    roadmapEntryIds = @(42)
    title = "Plugin Inventory 只读 API 博客版"
    path = "docs/articles/zh-cn/blog-plugin-inventory-readonly-api.md"
    beforeLineCount = 92
    beforeCharacterCount = 2881
    requiredMarkers = @("TryGetPluginRegistryInventory", "FindCreator", "CreatorFieldCollection", "IPluginCreator*")
    requiredAnchors = @(
      "src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginRegistryInventory.cs",
      "src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginCreatorInfo.cs",
      "src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginFieldInfo.cs",
      "src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginRegistryInventoryDiagnostics.cs",
      "src/JYPPX.TensorRtSharp/Runtime/TensorRtRuntime.PluginRegistryInventory.cs",
      "src/JYPPX.TensorRtSharp/Builder/TensorRtBuilder.PluginRegistryInventory.cs",
      "smoke/PluginRegistryInventorySmokeRunner/Program.cs"
    )
  },
  [ordered]@{
    roadmapEntryIds = @(43)
    title = "CUDA Memory Wrapper 博客版"
    path = "docs/articles/zh-cn/blog-cuda-memory-wrapper.md"
    beforeLineCount = 76
    beforeCharacterCount = 2569
    requiredMarkers = @("CudaPinnedMemory", "PinnedAsyncRoundTrip=True", "CudaManagedMemoryBatch", "IGpuAllocator")
    requiredAnchors = @(
      "src/JYPPX.CudaSharp/Memory/CudaMemory.cs",
      "src/JYPPX.CudaSharp/Memory/CudaPinnedMemory.cs",
      "src/JYPPX.CudaSharp/Memory/CudaManagedMemoryBatch.cs",
      "smoke/CudaSmokeRunner/Program.cs"
    )
  },
  [ordered]@{
    roadmapEntryIds = @(103)
    title = "Stream Capture To Graph owner-safe session"
    path = "docs/articles/zh-cn/cuda-stream-capture-to-graph-owner-safety.md"
    beforeLineCount = 56
    beforeCharacterCount = 1929
    requiredMarkers = @("CudaStreamCaptureToGraphSession", "CUDART_VERSION >= 12030", "implemented-with-deferred-history", "ToGraph=True")
    requiredAnchors = @(
      "src/JYPPX.CudaSharp/Streams/CudaStreamCaptureToGraphSession.cs",
      "src/JYPPX.CudaSharp/Streams/CudaStream.cs",
      "native/manifests/cuda/cuda-fifty-seventh-batch-stream-capture-to-graph.manifest.json",
      "artifacts/interface-coverage/cuda-stream-capture-to-graph-candidate-audit.md",
      "smoke/CudaGraphSmokeRunner/Program.cs"
    )
  }
)

$expectedRoadmapEntryIds = @(28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 43, 44, 45, 103)
$forbiddenMarkers = @(
  "canPublishPublicly=true",
  "canCloseReleaseIssue=true",
  "performsPublish=true",
  "dotnet nuget push",
  "YoloDet"
)
$referencePattern = '`((?:applications|artifacts|build-out|docs|eng|native|pack|samples|smoke|src|tests|tools)/[^`]+)`'
$markdownLinkPattern = '\[[^\]]+\]\((?<target>[^)]+)\)'
$canonicalResults = [System.Collections.Generic.List[object]]::new()

foreach ($definition in $definitions) {
  $fullPath = Resolve-RepositoryPath $definition.path
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    throw "Article not found: $($definition.path)"
  }

  $content = [System.IO.File]::ReadAllText($fullPath)
  $missingMarkers = @($definition.requiredMarkers | Where-Object {
      $content.IndexOf($_, [System.StringComparison]::OrdinalIgnoreCase) -lt 0
    })
  $missingAnchors = @($definition.requiredAnchors | Where-Object { -not (Test-RepositoryReference $_) })
  $references = @([System.Text.RegularExpressions.Regex]::Matches($content, $referencePattern) |
    ForEach-Object { $_.Groups[1].Value } |
    Select-Object -Unique)
  $missingReferences = @($references | Where-Object { -not (Test-RepositoryReference $_) })
  $markdownLinks = @([System.Text.RegularExpressions.Regex]::Matches($content, $markdownLinkPattern) |
    ForEach-Object { $_.Groups['target'].Value } |
    Select-Object -Unique)
  $missingMarkdownLinks = @($markdownLinks | Where-Object {
      -not (Test-ArticleMarkdownLink -ArticlePath $fullPath -Target $_)
    })
  $forbiddenFindings = @($forbiddenMarkers | Where-Object {
      $content.IndexOf($_, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    })
  $characterCount = $content.Length
  $contentState = if ($characterCount -ge 12000) {
    "complete-long-form"
  }
  elseif ($characterCount -ge 4500) {
    "complete-article"
  }
  else {
    "needs-expansion"
  }

  $canonicalResults.Add([pscustomobject][ordered]@{
      primaryArticleId = [int]$definition.roadmapEntryIds[0]
      roadmapEntryIds = @($definition.roadmapEntryIds)
      title = [string]$definition.title
      canonicalArticlePath = [string]$definition.path
      canonicalArticleExists = $true
      beforeLineCount = [int]$definition.beforeLineCount
      beforeCharacterCount = [int]$definition.beforeCharacterCount
      currentLineCount = ($content -split "`r?`n").Count
      currentCharacterCount = $characterCount
      characterGrowth = $characterCount - [int]$definition.beforeCharacterCount
      headingCount = [System.Text.RegularExpressions.Regex]::Matches($content, '(?m)^#{1,6} ').Count
      codeBlockCount = [int]([System.Text.RegularExpressions.Regex]::Matches($content, '(?m)^```').Count / 2)
      mermaidDiagramCount = [System.Text.RegularExpressions.Regex]::Matches($content, '```mermaid').Count
      contentState = $contentState
      contentComplete = $contentState -ne "needs-expansion"
      requiredMarkers = @($definition.requiredMarkers)
      missingMarkers = $missingMarkers
      requiredAnchors = @($definition.requiredAnchors)
      missingAnchors = $missingAnchors
      repositoryReferences = $references
      missingRepositoryReferences = $missingReferences
      markdownLinks = $markdownLinks
      missingMarkdownLinks = $missingMarkdownLinks
      forbiddenFindings = $forbiddenFindings
      proofState = "not-required-for-content-closure"
      performsPublish = $false
      canPublishPublicly = $false
      canCloseReleaseIssue = $false
    })
}

$canonicalArticles = @($canonicalResults | Sort-Object primaryArticleId)
$roadmapEntries = @($canonicalArticles | ForEach-Object {
    $article = $_
    @($article.roadmapEntryIds | ForEach-Object {
        [pscustomobject][ordered]@{
          articleId = [int]$_
          canonicalArticlePath = $article.canonicalArticlePath
          primaryArticleId = $article.primaryArticleId
          sharesCanonicalArticle = $article.roadmapEntryIds.Count -gt 1
          contentState = $article.contentState
          contentComplete = $article.contentComplete
        }
      })
  } | Sort-Object articleId)
$actualRoadmapEntryIds = @($roadmapEntries.articleId)
$missingRoadmapEntryIds = @($expectedRoadmapEntryIds | Where-Object { $_ -notin $actualRoadmapEntryIds })
$duplicateRoadmapEntryIds = @($actualRoadmapEntryIds | Group-Object | Where-Object Count -gt 1 | ForEach-Object { [int]$_.Name })
$sharedCanonicalMappings = @($canonicalArticles | Where-Object { $_.roadmapEntryIds.Count -gt 1 } | ForEach-Object {
    [pscustomobject][ordered]@{
      canonicalArticlePath = $_.canonicalArticlePath
      roadmapEntryIds = @($_.roadmapEntryIds)
    }
  })
$sharedMappingsValid =
  $sharedCanonicalMappings.Count -eq 2 -and
  @($sharedCanonicalMappings | Where-Object { $_.canonicalArticlePath -eq 'docs/articles/zh-cn/blog-refit-weights-guide.md' -and (@($_.roadmapEntryIds) -join ',') -eq '28,44' }).Count -eq 1 -and
  @($sharedCanonicalMappings | Where-Object { $_.canonicalArticlePath -eq 'docs/articles/zh-cn/blog-network-layer-coverage-guide.md' -and (@($_.roadmapEntryIds) -join ',') -eq '30,45' }).Count -eq 1

$roadmapPath = Resolve-RepositoryPath "docs/articles/zh-cn/technical-article-roadmap.md"
$roadmapContent = [System.IO.File]::ReadAllText($roadmapPath)
$roadmapStatusCompleteCount = @($expectedRoadmapEntryIds | Where-Object {
    $roadmapContent -match "(?m)^\|\s*$_\s*\|.*\|\s*完整教程已收口\s*\|$"
  }).Count
$missingMarkerCount = @($canonicalArticles | ForEach-Object { $_.missingMarkers }).Count
$missingAnchorCount = @($canonicalArticles | ForEach-Object { $_.missingAnchors }).Count
$missingReferenceCount = @($canonicalArticles | ForEach-Object { $_.missingRepositoryReferences }).Count
$missingMarkdownLinkCount = @($canonicalArticles | ForEach-Object { $_.missingMarkdownLinks }).Count
$forbiddenFindingCount = @($canonicalArticles | ForEach-Object { $_.forbiddenFindings }).Count
$contentCompleteCount = @($canonicalArticles | Where-Object contentComplete).Count
$auditValid =
  $roadmapEntries.Count -eq 14 -and
  $canonicalArticles.Count -eq 12 -and
  $missingRoadmapEntryIds.Count -eq 0 -and
  $duplicateRoadmapEntryIds.Count -eq 0 -and
  $sharedMappingsValid -and
  $roadmapStatusCompleteCount -eq 14 -and
  $contentCompleteCount -eq 12 -and
  $missingMarkerCount -eq 0 -and
  $missingAnchorCount -eq 0 -and
  $missingReferenceCount -eq 0 -and
  $missingMarkdownLinkCount -eq 0 -and
  $forbiddenFindingCount -eq 0

$ledgerPath = Resolve-RepositoryPath "docs/articles/zh-cn/publishing/technical-article-closure-ledger.json"
$record = [ordered]@{
  schemaVersion = 1
  recordKind = "technical-article-foundations-second-batch-audit"
  auditState = if ($auditValid) { "content-expanded-source-quality-audited" } else { "invalid-second-batch-audit" }
  sourceClosureLedger = "docs/articles/zh-cn/publishing/technical-article-closure-ledger.json"
  sourceClosureLedgerSha256 = (Get-FileHash -LiteralPath $ledgerPath -Algorithm SHA256).Hash.ToLowerInvariant()
  roadmapEntryIds = $expectedRoadmapEntryIds
  roadmapEntryCount = $roadmapEntries.Count
  uniqueCanonicalArticleCount = $canonicalArticles.Count
  sharedCanonicalMappingCount = $sharedCanonicalMappings.Count
  missingRoadmapEntryIds = $missingRoadmapEntryIds
  duplicateRoadmapEntryIds = $duplicateRoadmapEntryIds
  roadmapStatusCompleteCount = $roadmapStatusCompleteCount
  baselineRoadmapContentCompleteCount = 89
  baselineRoadmapNeedsExpansionCount = 14
  expectedPostExpansionContentCompleteCount = 103
  expectedPostExpansionNeedsExpansionCount = 0
  batchCanonicalContentCompleteCount = $contentCompleteCount
  completeLongFormCount = @($canonicalArticles | Where-Object contentState -eq "complete-long-form").Count
  completeArticleCount = @($canonicalArticles | Where-Object contentState -eq "complete-article").Count
  totalBeforeCharacterCount = [int](($canonicalArticles | Measure-Object beforeCharacterCount -Sum).Sum)
  totalCurrentCharacterCount = [int](($canonicalArticles | Measure-Object currentCharacterCount -Sum).Sum)
  totalCharacterGrowth = [int](($canonicalArticles | Measure-Object characterGrowth -Sum).Sum)
  totalHeadingCount = [int](($canonicalArticles | Measure-Object headingCount -Sum).Sum)
  totalCodeBlockCount = [int](($canonicalArticles | Measure-Object codeBlockCount -Sum).Sum)
  totalMermaidDiagramCount = [int](($canonicalArticles | Measure-Object mermaidDiagramCount -Sum).Sum)
  repositoryReferenceCount = @($canonicalArticles | ForEach-Object { $_.repositoryReferences }).Count
  markdownLinkCount = @($canonicalArticles | ForEach-Object { $_.markdownLinks }).Count
  missingMarkerCount = $missingMarkerCount
  missingAnchorCount = $missingAnchorCount
  missingRepositoryReferenceCount = $missingReferenceCount
  missingMarkdownLinkCount = $missingMarkdownLinkCount
  forbiddenFindingCount = $forbiddenFindingCount
  contentAndProofStateAreIndependent = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  proofBoundary = "This audit proves the final fourteen roadmap entries, twelve unique canonical article bodies, shared canonical mappings, required repository anchors, markers, and link resolution only. It is not runtime execution proof, package-consumer runtime proof, post-publish proof, publish approval, or release-close approval."
  sharedCanonicalMappings = $sharedCanonicalMappings
  roadmapEntries = $roadmapEntries
  canonicalArticles = $canonicalArticles
}

$outputDirectory = Resolve-RepositoryPath $OutputRoot
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$jsonPath = Join-Path $outputDirectory "technical-article-foundations-second-batch-audit.json"
$markdownPath = Join-Path $outputDirectory "technical-article-foundations-second-batch-audit.md"
$utf8 = [System.Text.UTF8Encoding]::new($false)
$json = $record | ConvertTo-Json -Depth 12
[System.IO.File]::WriteAllText($jsonPath, $json + "`n", $utf8)

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Technical Article Foundations Second Batch Audit")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- Audit state: ``$($record.auditState)``")
$lines.Add("- Roadmap entries: $($record.roadmapEntryCount)/14; unique canonical articles: $($record.uniqueCanonicalArticleCount)/12")
$lines.Add("- Batch canonical content complete: $($record.batchCanonicalContentCompleteCount)/12; roadmap statuses complete: $($record.roadmapStatusCompleteCount)/14")
$lines.Add("- Shared canonical mappings: $($record.sharedCanonicalMappingCount) (28/44 and 30/45)")
$lines.Add("- Before/current unique-canonical characters: $($record.totalBeforeCharacterCount) / $($record.totalCurrentCharacterCount); growth: $($record.totalCharacterGrowth)")
$lines.Add("- Complete long-form/article: $($record.completeLongFormCount) / $($record.completeArticleCount)")
$lines.Add("- Headings/code blocks/Mermaid diagrams: $($record.totalHeadingCount) / $($record.totalCodeBlockCount) / $($record.totalMermaidDiagramCount)")
$lines.Add("- Missing markers/anchors/repository references/Markdown links: $missingMarkerCount / $missingAnchorCount / $missingReferenceCount / $missingMarkdownLinkCount")
$lines.Add("- Forbidden findings: $forbiddenFindingCount")
$lines.Add("- Expected roadmap delta after closure-ledger export: content complete 89 -> 103; needs expansion 14 -> 0")
$lines.Add("")
$lines.Add("## Canonical Articles")
$lines.Add("")
$lines.Add("| Roadmap IDs | Canonical article | Before chars | Current chars | Growth | State | Headings | Code | Mermaid | References | Links | Missing |")
$lines.Add("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
foreach ($article in $canonicalArticles) {
  $missingCount = @($article.missingMarkers).Count + @($article.missingAnchors).Count + @($article.missingRepositoryReferences).Count + @($article.missingMarkdownLinks).Count + @($article.forbiddenFindings).Count
  $ids = @($article.roadmapEntryIds) -join ", "
  $lines.Add("| $ids | ``$($article.canonicalArticlePath)`` | $($article.beforeCharacterCount) | $($article.currentCharacterCount) | $($article.characterGrowth) | ``$($article.contentState)`` | $($article.headingCount) | $($article.codeBlockCount) | $($article.mermaidDiagramCount) | $(@($article.repositoryReferences).Count) | $(@($article.markdownLinks).Count) | $missingCount |")
}
$lines.Add("")
$lines.Add("## Shared Canonical Mappings")
$lines.Add("")
foreach ($mapping in $sharedCanonicalMappings) {
  $lines.Add("- $(@($mapping.roadmapEntryIds) -join '/') -> ``$($mapping.canonicalArticlePath)``")
}
$lines.Add("")
$lines.Add("## Shared Acceptance Rules")
$lines.Add("")
$lines.Add("1. All fourteen roadmap entries are explicit, while character and structure totals count the twelve unique canonical bodies only once.")
$lines.Add("2. Each canonical article stands alone with architecture, repository anchors, current E-drive commands, output interpretation, troubleshooting, proof boundary, and next reading.")
$lines.Add("3. Every backticked repository reference and Markdown relative link resolves; placeholders are not accepted as repository paths.")
$lines.Add("4. Shared canonical mappings are limited to 28/44 and 30/45 and do not create duplicate article bodies.")
$lines.Add("5. TRT/CUDA version guards, callback ownership, plugin borrowed pointers, and stream/graph owner safety remain explicit.")
$lines.Add("6. Content completion remains independent from external proof and publication state.")
$lines.Add("")
$lines.Add("## Proof Boundary")
$lines.Add("")
$lines.Add($record.proofBoundary)
$lines.Add("")
$lines.Add("- ``performsPublish=false``")
$lines.Add("- ``canPublishPublicly=false``")
$lines.Add("- ``canCloseReleaseIssue=false``")
[System.IO.File]::WriteAllText($markdownPath, ($lines -join "`n") + "`n", $utf8)

Write-Host "Technical article foundations second-batch audit written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "RoadmapEntries=$($record.roadmapEntryCount) UniqueCanonical=$($record.uniqueCanonicalArticleCount) ContentComplete=$($record.batchCanonicalContentCompleteCount) MissingReferences=$missingReferenceCount MissingLinks=$missingMarkdownLinkCount Forbidden=$forbiddenFindingCount"
Write-Host "PerformsPublish=$($record.performsPublish) CanPublishPublicly=$($record.canPublishPublicly) CanCloseReleaseIssue=$($record.canCloseReleaseIssue)"

if (-not $auditValid) {
  throw "Technical article foundations second-batch audit failed."
}
