[CmdletBinding()]
param(
  [string]$RoadmapPath,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($RoadmapPath)) {
  $RoadmapPath = Join-Path $RepositoryRoot "docs\articles\zh-cn\publishing\article-roadmap-30plus.json"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8FileWithRetry {
  param(
    [string]$LiteralPath,
    [AllowNull()][object]$InputObject,
    [int]$MaxAttempts = 8,
    [int]$DelayMilliseconds = 250
  )

  $directory = Split-Path -Parent $LiteralPath
  if (-not [string]::IsNullOrWhiteSpace($directory)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
  }

  $content = @($InputObject) -join [Environment]::NewLine
  $tempPath = Join-Path $directory (".{0}.{1}.tmp" -f ([IO.Path]::GetFileName($LiteralPath)), [Guid]::NewGuid().ToString("N"))
  [IO.File]::WriteAllText($tempPath, $content + [Environment]::NewLine, $script:utf8)

  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Move-Item -LiteralPath $tempPath -Destination $LiteralPath -Force
      return
    }
    catch [System.IO.IOException] {
      if ($attempt -eq $MaxAttempts) { throw }
      Start-Sleep -Milliseconds $DelayMilliseconds
    }
    catch [System.UnauthorizedAccessException] {
      if ($attempt -eq $MaxAttempts) { throw }
      Start-Sleep -Milliseconds $DelayMilliseconds
    }
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return @()
  }

  return @($Value)
}

function New-RoadmapArticle {
  param(
    [int]$Id,
    [string]$Title,
    [string]$Audience,
    [string]$Type,
    [string[]]$Outline,
    [string]$SampleOrCodePath,
    [string[]]$VisualAssets,
    [string]$Status,
    [string]$TargetPath,
    [string[]]$SourceArtifacts
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    audience = $Audience
    type = $Type
    outline = @($Outline)
    sampleOrCodePath = $SampleOrCodePath
    visualAssets = @($VisualAssets)
    status = $Status
    targetPath = $TargetPath
    sourceArtifacts = @($SourceArtifacts)
    proofBoundary = "Article planning/content is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
    mustAvoidClaims = @()
  }
}

function Add-OrReplaceArticle {
  param(
    [object[]]$Articles,
    [object]$Article
  )

  $filtered = @($Articles | Where-Object { [int]$_.id -ne [int]$Article.id })
  return @($filtered + $Article | Sort-Object { [int]$_.id })
}

if (-not (Test-Path -LiteralPath $RoadmapPath -PathType Leaf)) {
  throw "Article roadmap not found: $RoadmapPath"
}

$roadmap = Get-Content -LiteralPath $RoadmapPath -Raw -Encoding utf8 | ConvertFrom-Json
$articles = @($roadmap.articles)

$articles = Add-OrReplaceArticle $articles (New-RoadmapArticle `
    -Id 37 `
    -Title "CUDA 初始化 Proof Scaffold：SetValidDevices、InitDevice 与 ChooseDevice 的本地 smoke 分层" `
    -Audience "CUDA wrapper 维护者、发布负责人" `
    -Type "发布证据说明" `
    -Outline @("pre-init call order", "local-smoke-not-external-proof", "Skipped=True forbidden substitute", "clean external validator promotion gate") `
    -SampleOrCodePath "smoke/CudaDeviceInitializationProofRunner/Program.cs" `
    -VisualAssets @("CUDA 初始化 proof ladder 图") `
    -Status "ready" `
    -TargetPath "docs/articles/zh-cn/publishing/cuda-初始化-proof-scaffold-本地-smoke-分层.md" `
    -SourceArtifacts @("smoke/CudaDeviceInitializationProofRunner/Program.cs", "artifacts/final-release/cuda-device-initialization-local-smoke-classification.json", "docs/articles/zh-cn/package-consumer-validation.md"))

$articles = Add-OrReplaceArticle $articles (New-RoadmapArticle `
    -Id 38 `
    -Title "CUDA Graph Event Node borrowed handle 安全边界：HasEvent 替代 GetEvent" `
    -Audience "CUDA Graph API 维护者、安全评审者" `
    -Type "安全边界说明" `
    -Outline @("borrowed cudaEvent_t risk", "EventRecordNodeHasEvent", "EventWaitNodeHasEvent", "no public borrowed handle") `
    -SampleOrCodePath "docs/articles/zh-cn/cuda-graph-borrowed-handle-safety-gate.md" `
    -VisualAssets @("Graph event node ownership 图") `
    -Status "ready" `
    -TargetPath "docs/articles/zh-cn/publishing/cuda-graph-event-node-borrowed-handle-安全边界.md" `
    -SourceArtifacts @("docs/articles/zh-cn/cuda-graph-borrowed-handle-safety-gate.md", "src/JYPPX.CudaSharp/Graphs/CudaGraph.cs", "smoke/CudaGraphSmokeRunner/Program.cs"))

$articles = Add-OrReplaceArticle $articles (New-RoadmapArticle `
    -Id 39 `
    -Title "Package Consumer Proof 分层边界：local smoke、package-feed substitute 与 clean external evidence" `
    -Audience "发布负责人、包验证维护者" `
    -Type "发布门禁说明" `
    -Outline @("local smoke classification", "local feed boundary", "clean consumer proof inputs", "strict validator promotion") `
    -SampleOrCodePath "docs/articles/zh-cn/package-consumer-validation.md" `
    -VisualAssets @("package consumer proof ladder 图") `
    -Status "ready" `
    -TargetPath "docs/articles/zh-cn/publishing/package-consumer-分层边界-local-smoke-package-feed-substitute-clean-external-evidence.md" `
    -SourceArtifacts @("docs/articles/zh-cn/package-consumer-validation.md", "docs/articles/zh-cn/package-consumer-runtime-proof-preflight-matrix.md", "artifacts/final-release/cuda-device-initialization-local-smoke-classification.json"))

$articles = Add-OrReplaceArticle $articles (New-RoadmapArticle `
    -Id 40 `
    -Title "YoloVision 真实资产证据链：从样例矩阵到 owner proof" `
    -Audience "视觉样例维护者、发布负责人" `
    -Type "案例教程" `
    -Outline @("YoloVision model matrix", "owner asset pack", "license and SHA256", "golden output validation") `
    -SampleOrCodePath "docs/articles/zh-cn/yolovision-owner-asset-evidence-guide.md" `
    -VisualAssets @("YOLO task matrix 图", "owner asset proof checklist") `
    -Status "near-ready-owner-proof-input" `
    -TargetPath "docs/articles/zh-cn/publishing/yolovision-真实资产证据链-从样例矩阵到-owner-proof.md" `
    -SourceArtifacts @("samples/YoloVision/yolo-model-matrix.json", "docs/articles/zh-cn/yolovision-owner-asset-evidence-guide.md"))

$articles = Add-OrReplaceArticle $articles (New-RoadmapArticle `
    -Id 41 `
    -Title "ONNX Parser 与 ParserRefitter 诊断：copied diagnostics 到 release gate" `
    -Audience "模型转换维护者、发布负责人" `
    -Type "发布证据说明" `
    -Outline @("parser error count", "copied diagnostic text", "used VC plugin library summary", "readiness wrapper group boundary") `
    -SampleOrCodePath "src/JYPPX.TensorRtSharp/Parsing/TensorRtOnnxParserDiagnosticSnapshot.cs" `
    -VisualAssets @("parser/refitter diagnostic evidence ladder 图") `
    -Status "ready" `
    -TargetPath "docs/articles/zh-cn/publishing/onnx-parser-parserrefitter-诊断-copied-diagnostics-release-gate.md" `
    -SourceArtifacts @("src/JYPPX.TensorRtSharp/Parsing/TensorRtOnnxParserDiagnosticSnapshot.cs", "eng/Test-BridgePackageConsumer.ps1", "eng/Test-RuntimePackageReadiness.ps1"))

$articles = Add-OrReplaceArticle $articles (New-RoadmapArticle `
    -Id 42 `
    -Title "CUDA Stream Capture To Graph：owner-safe session 与跨版本 guard" `
    -Audience "CUDA wrapper 维护者、C# 生命周期评审者、发布负责人" `
    -Type "安全边界说明" `
    -Outline @("CUDA 12.3+ vendor symbol audit", "stream and graph owner counts", "same-graph End validation", "deferred-history coverage", "compatible-host smoke boundary") `
    -SampleOrCodePath "docs/articles/zh-cn/cuda-stream-capture-to-graph-owner-safety.md" `
    -VisualAssets @("stream-to-existing-graph session 生命周期图") `
    -Status "ready" `
    -TargetPath "docs/articles/zh-cn/publishing/cuda-stream-capture-to-graph-owner-safe-session.md" `
    -SourceArtifacts @("artifacts/interface-coverage/cuda-stream-capture-to-graph-candidate-audit.md", "src/JYPPX.CudaSharp/Streams/CudaStreamCaptureToGraphSession.cs", "smoke/CudaGraphSmokeRunner/Program.cs"))

$enhancedArticles = New-Object System.Collections.Generic.List[object]

foreach ($article in $articles) {
  $sampleOrCodePath = [string]$article.sampleOrCodePath
  if ([string]::IsNullOrWhiteSpace($sampleOrCodePath)) {
    $sampleOrCodePath = "docs/articles/zh-cn"
  }

  $slug = ([string]$article.title).ToLowerInvariant()
  $slug = [Text.RegularExpressions.Regex]::Replace($slug, "[^\p{L}\p{Nd}]+", "-").Trim("-")
  if ([string]::IsNullOrWhiteSpace($slug)) {
    $slug = "article-$($article.id)"
  }

  $targetPath = if ($article.PSObject.Properties.Name -contains "targetPath" -and -not [string]::IsNullOrWhiteSpace([string]$article.targetPath)) {
    [string]$article.targetPath
  }
  else {
    "docs/articles/zh-cn/publishing/$slug.md"
  }

  $sourceArtifacts = Convert-ToArray $(if ($article.PSObject.Properties.Name -contains "sourceArtifacts") { $article.sourceArtifacts } else { @() })
  if ($sourceArtifacts.Count -eq 0) {
    $sourceArtifacts = @($sampleOrCodePath)
  }

  $proofBoundary = if ($article.PSObject.Properties.Name -contains "proofBoundary" -and -not [string]::IsNullOrWhiteSpace([string]$article.proofBoundary)) {
    [string]$article.proofBoundary
  }
  else {
    "Article planning/content is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }

  if ([string]$article.status -eq "near-ready-owner-proof-input" -and $proofBoundary.IndexOf("owner proof input", [StringComparison]::OrdinalIgnoreCase) -lt 0) {
    $proofBoundary = "Article body and owner proof input are near-ready guidance only: $proofBoundary Owner proof input still requires real assets, host/package metadata, stdout/stderr logs, hashes, and validators."
  }

  foreach ($requiredBoundaryMarker in @("not runtime proof", "not post-publish proof", "not publish approval", "not release close approval", "not package push")) {
    if ($proofBoundary.IndexOf($requiredBoundaryMarker, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $proofBoundary = "$proofBoundary $requiredBoundaryMarker."
    }
  }

  $mustAvoidClaims = @(
      "must not claim local feed is public package proof",
      "must not claim ProjectReference/direct nupkg is package-consumer-runtime proof",
      "must not claim build-only output is runtime proof",
      "must not claim failedBlockerCount=0 means ready",
      "must not claim dashboard/runbook/candidate/draft is publish approval",
      "must not claim CudaDeviceInitializationProofRunner local smoke is package-consumer-runtime proof",
      "must not treat Skipped=True as proof"
    )

  $enhancedArticles.Add([pscustomobject]@{
      id = [int]$article.id
      title = [string]$article.title
      audience = [string]$article.audience
      type = [string]$article.type
      outline = @(Convert-ToArray $article.outline)
      sampleOrCodePath = $sampleOrCodePath
      visualAssets = @(Convert-ToArray $article.visualAssets)
      status = [string]$article.status
      targetPath = $targetPath
      sourceArtifacts = @($sourceArtifacts)
      proofBoundary = $proofBoundary
      mustAvoidClaims = @($mustAvoidClaims)
    })
}

$enhanced = [pscustomobject]@{
  roadmapId = "article-roadmap-30plus"
  roadmapState = "release-readiness-planning"
  minimumArticleCount = 30
  articleCount = $enhancedArticles.Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "The 30+ article roadmap is publication planning only: not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. failedBlockerCount=0 is not ready."
  requiredFields = @("id", "title", "audience", "status", "targetPath", "sourceArtifacts", "proofBoundary", "mustAvoidClaims")
  articles = @($enhancedArticles.ToArray())
}

$enhanced | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $RoadmapPath -Encoding utf8

$markdownPath = [IO.Path]::ChangeExtension($RoadmapPath, ".md")
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# 30+ 篇技术与宣发文章规划")
$lines.Add("")
$lines.Add("``article-roadmap-30plus.json`` 是面向微信公众号、博客和项目文档的机器可读文章规划。它不是 proof，不批准公开发布，也不关闭 release issue。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| roadmapState | ``$($enhanced.roadmapState)`` |")
$lines.Add("| articleCount | ``$($enhanced.articleCount)`` |")
$lines.Add("| minimumArticleCount | ``$($enhanced.minimumArticleCount)`` |")
$lines.Add("| canPublishPublicly | ``$($enhanced.canPublishPublicly)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($enhanced.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($enhanced.boundary)
$lines.Add("")
$lines.Add("Rows marked ``near-ready-owner-proof-input`` are article-body and owner proof input guidance only. They still require Owner-provided real assets, public package metadata, stdout/stderr logs, SHA256 values, host metadata, and strict validators before any proof promotion.")
$lines.Add("")
$lines.Add("## 文章矩阵")
$lines.Add("")
$lines.Add("| Id | Title | Audience | Status | Target |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($article in $enhanced.articles) {
  $lines.Add("| ``$($article.id)`` | $(ConvertTo-MarkdownCell $article.title) | $(ConvertTo-MarkdownCell $article.audience) | ``$(ConvertTo-MarkdownCell $article.status)`` | $(ConvertTo-MarkdownCell $article.targetPath) |")
}
$lines.Add("")
$lines.Add("## Must Avoid Claims")
$lines.Add("")
$lines.Add("- 不把 local feed、ProjectReference、direct nupkg、dry-run、dashboard、runbook、candidate、draft 或 build-only 写成 proof。")
$lines.Add("- 不把 failedBlockerCount=0 写成 ready。")
$lines.Add("- 不恢复旧样例公开入口名。")

Write-Utf8FileWithRetry -LiteralPath $markdownPath -InputObject $lines

Write-Host "Article roadmap 30+ written: $RoadmapPath"
Write-Host "Article roadmap 30+ markdown written: $markdownPath"
