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
    articleId = 2
    title = "为什么不是简单 P/Invoke"
    path = "docs/articles/zh-cn/why-not-plain-pinvoke.md"
    beforeLineCount = 78
    beforeCharacterCount = 2602
    requiredMarkers = @("Stable C ABI bridge", "SafeTensorRtObjectHandle", "caller-buffer", "Test-PackageConsumer.ps1")
    requiredAnchors = @(
      "native/manifests/bridge-api.schema.json",
      "native/include/jyppx/tensorrt/trt8.h",
      "native/src/tensorrt/common/object.cpp",
      "src/JYPPX.TensorRtSharp/Internal/Interop/NativeBridgeApi.cs",
      "src/JYPPX.TensorRtSharp/Internal/Handles/SafeTensorRtObjectHandle.cs"
    )
  },
  [ordered]@{
    articleId = 3
    title = "从接口清零到 deferred 边界提升"
    path = "docs/articles/zh-cn/interface-zero-to-deferred-boundary.md"
    beforeLineCount = 95
    beforeCharacterCount = 3905
    requiredMarkers = @("ImplementationStatus", "implemented-with-deferred-history", "Export-InterfaceCoverageMatrix.ps1", "Test-TensorRtNativeAbiSurface.ps1")
    requiredAnchors = @(
      "artifacts/interface-coverage/interface-coverage-summary.md",
      "artifacts/interface-coverage/tensorrt-interface-coverage.csv",
      "artifacts/interface-coverage/cuda-runtime-interface-coverage.csv",
      "eng/Export-InterfaceCoverageMatrix.ps1",
      "eng/Generate-Bindings.ps1"
    )
  },
  [ordered]@{
    articleId = 4
    title = "TRT8/TRT10/TRT11 跨版本策略"
    path = "docs/articles/zh-cn/trt-cross-version-strategy.md"
    beforeLineCount = 96
    beforeCharacterCount = 3040
    requiredMarkers = @("TensorRtApiLine", "CMakePresets.json", "runtime-packages.manifest.json", "report_vendor_mismatch")
    requiredAnchors = @(
      "native/include/jyppx/tensorrt/trt8.h",
      "native/include/jyppx/tensorrt/trt10.h",
      "native/include/jyppx/tensorrt/trt11.h",
      "src/JYPPX.Shared/Interop/TensorRtApiLine.cs",
      "pack/runtime/runtime-packages.manifest.json",
      "CMakePresets.json"
    )
  },
  [ordered]@{
    articleId = 5
    title = "Windows 本地开发环境准备"
    path = "docs/articles/zh-cn/windows-local-dev-environment.md"
    beforeLineCount = 116
    beforeCharacterCount = 3592
    requiredMarkers = @("Validate-WindowsRuntimeInputs.ps1", "Resolve-RuntimeRoots.ps1", "CMakePresets.json", "E:\TensorRtSharpAssets")
    requiredAnchors = @(
      "CMakePresets.json",
      "pack/runtime/runtime-packages.local.example.json",
      "eng/Resolve-RuntimeRoots.ps1",
      "eng/Validate-WindowsRuntimeInputs.ps1",
      "eng/Test-BindingGeneratorOutputs.ps1"
    )
  },
  [ordered]@{
    articleId = 6
    title = "Runtime Package 和 Split Package 怎么选"
    path = "docs/articles/zh-cn/runtime-package-selection.md"
    beforeLineCount = 108
    beforeCharacterCount = 2922
    requiredMarkers = @("18 个 key", "runtime-packages.manifest.json", "CudaCudnn", "Test-PackageConsumer.ps1")
    requiredAnchors = @(
      "pack/JYPPX.TensorRT.CSharp.API",
      "pack/runtime",
      "pack/runtime-split",
      "pack/runtime/runtime-packages.manifest.json",
      "eng/Test-PackageConsumer.ps1",
      "eng/Test-BridgePackageConsumer.ps1"
    )
  },
  [ordered]@{
    articleId = 10
    title = "TensorRT Builder/Runtime/Engine 对象模型"
    path = "docs/articles/zh-cn/tensorrt-object-model.md"
    beforeLineCount = 34
    beforeCharacterCount = 1031
    requiredMarkers = @("TensorRtLogger", "TensorRtBuilder", "TensorRtRuntime", "TensorRtEngine", "TensorRtExecutionContext", "TensorRtInferenceBindings")
    requiredAnchors = @(
      "src/JYPPX.TensorRtSharp/Builder/TensorRtBuilder.cs",
      "src/JYPPX.TensorRtSharp/Runtime/TensorRtRuntime.cs",
      "src/JYPPX.TensorRtSharp/Engine/TensorRtEngine.cs",
      "src/JYPPX.TensorRtSharp/Execution/TensorRtExecutionContext.cs",
      "src/JYPPX.TensorRtSharp/Inference/TensorRtInferenceBindings.cs",
      "samples/InferenceBindings/Program.cs"
    )
  },
  [ordered]@{
    articleId = 15
    title = "Plugin Serialization Paths"
    path = "docs/articles/zh-cn/plugin-serialization-paths.md"
    beforeLineCount = 77
    beforeCharacterCount = 2304
    requiredMarkers = @("SetPluginsToSerialize", "GetSerializedPluginSnapshot", "ClearPluginsToSerialize", "IPluginV2::serialize")
    requiredAnchors = @(
      "src/JYPPX.TensorRtSharp/Builder/TensorRtBuilderConfig.Trt11PluginSerialization.cs",
      "src/JYPPX.TensorRtSharp/Builder/TensorRtBuilderConfigSerializedPluginSnapshot.cs",
      "native/manifests/tensorrt/v10/trt10-runtime-serialization-plugin-paths.manifest.json",
      "smoke/PluginSerializationPathsSmokeRunner/Program.cs"
    )
  },
  [ordered]@{
    articleId = 16
    title = "CUDA memory wrapper 入门"
    path = "docs/articles/zh-cn/cuda-memory-wrapper.md"
    beforeLineCount = 53
    beforeCharacterCount = 1938
    requiredMarkers = @("CudaMemory", "CudaPinnedMemory", "CudaStream", "PinnedAsyncRoundTrip=True", "MultiStream")
    requiredAnchors = @(
      "src/JYPPX.CudaSharp/Memory/CudaMemory.cs",
      "src/JYPPX.CudaSharp/Memory/CudaPinnedMemory.cs",
      "src/JYPPX.CudaSharp/Internal/Handles/SafeCudaMemoryHandle.cs",
      "smoke/CudaSmokeRunner/Program.cs",
      "samples/MultiStream/Program.cs"
    )
  },
  [ordered]@{
    articleId = 19
    title = "CUDA memory range APIs"
    path = "docs/articles/zh-cn/cuda-memory-range-apis.md"
    beforeLineCount = 28
    beforeCharacterCount = 697
    requiredMarkers = @("CudaMemoryRangeAttribute", "GetRangeAccessedByDevices", "CudaMemoryLocation", "CudaManagedMemoryBatch", "CanPromoteRuntimeProof=false")
    requiredAnchors = @(
      "src/JYPPX.CudaSharp/Memory/CudaMemoryRangeAttribute.cs",
      "src/JYPPX.CudaSharp/Memory/CudaMemoryLocation.cs",
      "src/JYPPX.CudaSharp/Memory/CudaManagedMemoryBatch.cs",
      "src/JYPPX.CudaSharp/Internal/Interop/Memory/NativeCudaApi.MemoryRange.cs",
      "native/manifests/cuda/cuda-forty-second-batch-memory-range-attributes.manifest.json"
    )
  }
)

$forbiddenMarkers = @(
  "canPublishPublicly=true",
  "canCloseReleaseIssue=true",
  "performsPublish=true",
  "dotnet nuget push",
  "YoloDet"
)
$referencePattern = '`((?:applications|artifacts|build-out|docs|eng|native|pack|samples|smoke|src|tests|tools)/[^`]+)`'
$markdownLinkPattern = '\[[^\]]+\]\((?<target>[^)]+)\)'
$articleResults = [System.Collections.Generic.List[object]]::new()

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

  $articleResults.Add([pscustomobject][ordered]@{
      articleId = [int]$definition.articleId
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

$articles = @($articleResults | Sort-Object articleId)
$missingArticleIds = @(2, 3, 4, 5, 6, 10, 15, 16, 19 | Where-Object { $_ -notin @($articles.articleId) })
$missingMarkerCount = @($articles | ForEach-Object { $_.missingMarkers }).Count
$missingAnchorCount = @($articles | ForEach-Object { $_.missingAnchors }).Count
$missingReferenceCount = @($articles | ForEach-Object { $_.missingRepositoryReferences }).Count
$missingMarkdownLinkCount = @($articles | ForEach-Object { $_.missingMarkdownLinks }).Count
$forbiddenFindingCount = @($articles | ForEach-Object { $_.forbiddenFindings }).Count
$contentCompleteCount = @($articles | Where-Object contentComplete).Count
$auditValid =
  $articles.Count -eq 9 -and
  $missingArticleIds.Count -eq 0 -and
  $contentCompleteCount -eq 9 -and
  $missingMarkerCount -eq 0 -and
  $missingAnchorCount -eq 0 -and
  $missingReferenceCount -eq 0 -and
  $missingMarkdownLinkCount -eq 0 -and
  $forbiddenFindingCount -eq 0

$ledgerPath = Resolve-RepositoryPath "docs/articles/zh-cn/publishing/technical-article-closure-ledger.json"
$record = [ordered]@{
  schemaVersion = 1
  recordKind = "technical-article-foundations-first-batch-audit"
  auditState = if ($auditValid) { "content-expanded-source-quality-audited" } else { "invalid-first-batch-audit" }
  sourceClosureLedger = "docs/articles/zh-cn/publishing/technical-article-closure-ledger.json"
  sourceClosureLedgerSha256 = (Get-FileHash -LiteralPath $ledgerPath -Algorithm SHA256).Hash.ToLowerInvariant()
  articleIds = @(2, 3, 4, 5, 6, 10, 15, 16, 19)
  articleCount = $articles.Count
  missingArticleIds = $missingArticleIds
  baselineRoadmapContentCompleteCount = 80
  baselineRoadmapNeedsExpansionCount = 23
  expectedPostExpansionContentCompleteCount = 89
  expectedPostExpansionNeedsExpansionCount = 14
  batchContentCompleteCount = $contentCompleteCount
  completeLongFormCount = @($articles | Where-Object contentState -eq "complete-long-form").Count
  completeArticleCount = @($articles | Where-Object contentState -eq "complete-article").Count
  totalBeforeCharacterCount = [int](($articles | Measure-Object beforeCharacterCount -Sum).Sum)
  totalCurrentCharacterCount = [int](($articles | Measure-Object currentCharacterCount -Sum).Sum)
  totalCharacterGrowth = [int](($articles | Measure-Object characterGrowth -Sum).Sum)
  totalHeadingCount = [int](($articles | Measure-Object headingCount -Sum).Sum)
  totalCodeBlockCount = [int](($articles | Measure-Object codeBlockCount -Sum).Sum)
  totalMermaidDiagramCount = [int](($articles | Measure-Object mermaidDiagramCount -Sum).Sum)
  repositoryReferenceCount = @($articles | ForEach-Object { $_.repositoryReferences }).Count
  markdownLinkCount = @($articles | ForEach-Object { $_.markdownLinks }).Count
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
  proofBoundary = "This audit proves article structure, required repository anchors, markers, and repository-reference resolution only. It is not runtime execution proof, package-consumer runtime proof, post-publish proof, publish approval, or release-close approval."
  articles = $articles
}

$outputDirectory = Resolve-RepositoryPath $OutputRoot
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$jsonPath = Join-Path $outputDirectory "technical-article-foundations-first-batch-audit.json"
$markdownPath = Join-Path $outputDirectory "technical-article-foundations-first-batch-audit.md"
$utf8 = [System.Text.UTF8Encoding]::new($false)
$json = $record | ConvertTo-Json -Depth 12
[System.IO.File]::WriteAllText($jsonPath, $json + "`n", $utf8)

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Technical Article Foundations First Batch Audit")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- Audit state: ``$($record.auditState)``")
$lines.Add("- Articles: $($record.articleCount)/9; batch content complete: $($record.batchContentCompleteCount)/9")
$lines.Add("- Before/current characters: $($record.totalBeforeCharacterCount) / $($record.totalCurrentCharacterCount); growth: $($record.totalCharacterGrowth)")
$lines.Add("- Complete long-form/article: $($record.completeLongFormCount) / $($record.completeArticleCount)")
$lines.Add("- Headings/code blocks/Mermaid diagrams: $($record.totalHeadingCount) / $($record.totalCodeBlockCount) / $($record.totalMermaidDiagramCount)")
$lines.Add("- Missing markers/anchors/repository references/Markdown links: $missingMarkerCount / $missingAnchorCount / $missingReferenceCount / $missingMarkdownLinkCount")
$lines.Add("- Forbidden findings: $forbiddenFindingCount")
$lines.Add("- Expected roadmap delta after closure-ledger export: content complete 80 -> 89; needs expansion 23 -> 14")
$lines.Add("")
$lines.Add("## Articles")
$lines.Add("")
$lines.Add("| ID | Canonical article | Before chars | Current chars | Growth | State | Headings | Code | Mermaid | References | Links | Missing |")
$lines.Add("| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
foreach ($article in $articles) {
  $missingCount = @($article.missingMarkers).Count + @($article.missingAnchors).Count + @($article.missingRepositoryReferences).Count + @($article.missingMarkdownLinks).Count + @($article.forbiddenFindings).Count
  $lines.Add("| $($article.articleId) | ``$($article.canonicalArticlePath)`` | $($article.beforeCharacterCount) | $($article.currentCharacterCount) | $($article.characterGrowth) | ``$($article.contentState)`` | $($article.headingCount) | $($article.codeBlockCount) | $($article.mermaidDiagramCount) | $(@($article.repositoryReferences).Count) | $(@($article.markdownLinks).Count) | $missingCount |")
}
$lines.Add("")
$lines.Add("## Shared Acceptance Rules")
$lines.Add("")
$lines.Add("1. Each article stands alone with audience, problem statement, diagram, repository anchors, current commands, output interpretation, troubleshooting, proof boundary, and next reading.")
$lines.Add("2. Every backticked repository reference resolves under the repository; placeholders are not accepted as repository paths.")
$lines.Add("3. Public examples use owner-safe wrappers. Native pointers and safe handles remain internal implementation details.")
$lines.Add("4. TRT8/TRT10/TRT11 and CUDA version facts remain line-specific; evidence is not projected across runtime keys.")
$lines.Add("5. Content completion remains independent from external proof.")
$lines.Add("")
$lines.Add("## Proof Boundary")
$lines.Add("")
$lines.Add($record.proofBoundary)
$lines.Add("")
$lines.Add("- ``performsPublish=false``")
$lines.Add("- ``canPublishPublicly=false``")
$lines.Add("- ``canCloseReleaseIssue=false``")
[System.IO.File]::WriteAllText($markdownPath, ($lines -join "`n") + "`n", $utf8)

Write-Host "Technical article foundations first-batch audit written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ArticleCount=$($record.articleCount) ContentComplete=$($record.batchContentCompleteCount) MissingReferences=$missingReferenceCount MissingLinks=$missingMarkdownLinkCount Forbidden=$forbiddenFindingCount"
Write-Host "PerformsPublish=$($record.performsPublish) CanPublishPublicly=$($record.canPublishPublicly) CanCloseReleaseIssue=$($record.canCloseReleaseIssue)"

if (-not $auditValid) {
  throw "Technical article foundations first-batch audit failed."
}
