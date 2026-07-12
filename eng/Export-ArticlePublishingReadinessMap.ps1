[CmdletBinding()]
param(
  [string]$ArticleRoadmapPath = "docs/articles/zh-cn/publishing/article-roadmap-30plus.json",
  [string]$TechnicalArticleMatrixPath = "artifacts/final-release/technical-article-publication-matrix.json",
  [string]$ReleaseCloseStrictOrderPath = "artifacts/final-release/release-close-strict-proof-execution-order.json",
  [string]$FinalActionMapPath = "artifacts/final-release/final-publish-action-required-evidence-map.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  if ($Value -is [System.Collections.IEnumerable] -and $Value -isnot [string]) { return @($Value) }
  return @($Value)
}

function New-ReadinessArticle {
  param(
    [string]$Area,
    [string]$Title,
    [string]$Status,
    [string[]]$SourceArtifacts,
    [string[]]$RequiredProofBeforePublication,
    [string[]]$RequiredImagesOrTables,
    [string[]]$CodePaths,
    [string[]]$ForbiddenClaims
  )

  [pscustomobject]@{
    area = $Area
    title = $Title
    readinessState = $Status
    sourceArtifacts = @($SourceArtifacts)
    requiredProofBeforePublication = @($RequiredProofBeforePublication)
    requiredImagesOrTables = @($RequiredImagesOrTables)
    codePaths = @($CodePaths)
    forbiddenClaims = @($ForbiddenClaims)
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

$roadmap = Read-JsonOrNull -Path $ArticleRoadmapPath
$technicalMatrix = Read-JsonOrNull -Path $TechnicalArticleMatrixPath
$strictOrder = Read-JsonOrNull -Path $ReleaseCloseStrictOrderPath
$finalActionMap = Read-JsonOrNull -Path $FinalActionMapPath

$forbiddenClaims = @(
  "release-ready",
  "published",
  "post-publish verified",
  "issue-close ready",
  "template is proof",
  "dashboard is proof",
  "build report is proof",
  "local feed is package consumer proof",
  "ProjectReference is package consumer proof",
  "direct nupkg is package consumer proof"
)

$articles = @(
  New-ReadinessArticle -Area "YoloVision" -Title "YoloVision 全系列任务教程与真实资产证据链" -Status "blocked-real-model-owner-proof-required" -SourceArtifacts @("samples/YoloVision", "samples/YoloVision/yolo-model-matrix.json", "artifacts/user-acceptance/yolovision-owner-real-evidence-intake-dashboard.json") -RequiredProofBeforePublication @("real-model-runtime-owner-proof-required", "model source license", "labels/input/preprocessed data", "golden outputs", "log SHA256") -RequiredImagesOrTables @("YOLO family/task matrix", "det/seg/pose/obb/cls/sem evidence table", "sample output screenshot after real owner assets") -CodePaths @("samples/YoloVision") -ForbiddenClaims $forbiddenClaims
  New-ReadinessArticle -Area "OnnxToEngine" -Title "OnnxToEngine 与 trtexec 模型转换边界教程" -Status "ready-non-proof-technical-draft" -SourceArtifacts @("samples/OnnxToEngine", "docs/articles/zh-cn/onnx-to-engine-trtexec-conversion-guide.md") -RequiredProofBeforePublication @("conversion command examples", "official trtexec boundary statement", "not runtime proof marker") -RequiredImagesOrTables @("conversion option mapping table", "build report example") -CodePaths @("samples/OnnxToEngine") -ForbiddenClaims $forbiddenClaims
  New-ReadinessArticle -Area "TensorRtExec" -Title "TensorRtExec 控制台与 WinForms 应用教程" -Status "ready-non-proof-technical-draft" -SourceArtifacts @("applications/TensorRtExec", "docs/articles/zh-cn/onnxtoengine-and-tensorrtexec-boundary.md") -RequiredProofBeforePublication @("CLI option examples", "WinForms screenshots", "build-only/report-only boundary") -RequiredImagesOrTables @("WinForms main screen", "CLI command table", "report output screenshot") -CodePaths @("applications/TensorRtExec") -ForbiddenClaims $forbiddenClaims
  New-ReadinessArticle -Area "RuntimePackages" -Title "GitHub 全量依赖包与 NuGet 小包双路线发布说明" -Status "blocked-public-package-owner-proof-required" -SourceArtifacts @("artifacts/final-release/package-consumer-dual-route-proof-plan.json", "artifacts/final-release/clean-external-consumer-execution-kit.json") -RequiredProofBeforePublication @("package-consumer-runtime-owner-proof-required", "public package source", "managed/native/runtime SHA256", "clean external consumer smoke") -RequiredImagesOrTables @("dual route diagram", "public package source checklist", "clean external smoke evidence table") -CodePaths @("pack/runtime-split", "eng/Export-PackageConsumerDualRouteProofPlan.ps1") -ForbiddenClaims $forbiddenClaims
  New-ReadinessArticle -Area "CleanConsumer" -Title "Clean external package consumer 真实运行证明教程" -Status "blocked-clean-external-runtime-proof-required" -SourceArtifacts @("artifacts/final-release/clean-external-consumer-execution-kit.json", "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json") -RequiredProofBeforePublication @("package-consumer-runtime-owner-proof-required", "restore/build/smoke logs", "smoke log SHA256", "host metadata") -RequiredImagesOrTables @("clean root outside repository diagram", "restore/build/smoke transcript screenshot", "hash table") -CodePaths @("eng/New-PackageConsumerExternalSmokeScaffold.ps1") -ForbiddenClaims $forbiddenClaims
  New-ReadinessArticle -Area "PostPublish" -Title "PostPublish 公开渠道安装运行验证教程" -Status "blocked-post-publish-owner-proof-required" -SourceArtifacts @("artifacts/final-release/post-publish-verification-intake-map.json", "artifacts/final-release/post-publish-verification-record.template.json") -RequiredProofBeforePublication @("post-publish-verification-owner-proof-required", "public channel URL", "published version", "install/run logs", "owner review", "rollback/deprecation plan") -RequiredImagesOrTables @("post-publish intake checklist", "public channel install transcript", "rollback decision table") -CodePaths @("eng/Test-PostPublishVerificationRecord.ps1") -ForbiddenClaims $forbiddenClaims
  New-ReadinessArticle -Area "ReleaseClose" -Title "Release Close 严格证明与 Owner 授权流程" -Status "blocked-release-close-real-proof-required" -SourceArtifacts @("artifacts/final-release/release-close-strict-proof-execution-order.json", "artifacts/final-release/final-publish-action-required-evidence-map.json") -RequiredProofBeforePublication @("all six final action-required lanes", "strict validator pass", "owner final close decision", "rollback/deprecation review") -RequiredImagesOrTables @("strict proof execution order", "final action map table", "owner decision checklist") -CodePaths @("eng/Export-ReleaseCloseStrictProofExecutionOrder.ps1") -ForbiddenClaims $forbiddenClaims
)

$roadmapArticles = if ($null -ne $roadmap) { @(Convert-ToArray $roadmap.articles) } else { @() }
$technicalArticles = if ($null -ne $technicalMatrix) { @(Convert-ToArray $technicalMatrix.articles) } else { @() }
$minimumArticleCount = if ($null -ne $roadmap -and $roadmap.PSObject.Properties.Name -contains "minimumArticleCount") { [int]$roadmap.minimumArticleCount } else { 30 }
$roadmapArticleCount = if ($null -ne $roadmap -and $roadmap.PSObject.Properties.Name -contains "articleCount") { [int]$roadmap.articleCount } else { @($roadmapArticles).Count }
$technicalArticleCount = if ($null -ne $technicalMatrix -and $technicalMatrix.PSObject.Properties.Name -contains "articleCount") { [int]$technicalMatrix.articleCount } else { @($technicalArticles).Count }

$readinessMap = [pscustomobject]@{
  recordKind = "article-publishing-readiness-map"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  readinessState = "blocked-real-proof-required-before-public-publication"
  sourceArticleRoadmap = $ArticleRoadmapPath
  sourceTechnicalArticleMatrix = $TechnicalArticleMatrixPath
  sourceReleaseCloseStrictOrder = $ReleaseCloseStrictOrderPath
  sourceFinalActionMap = $FinalActionMapPath
  minimumArticleCount = $minimumArticleCount
  roadmapArticleCount = $roadmapArticleCount
  technicalArticleCount = $technicalArticleCount
  focusedReadinessArticleCount = @($articles).Count
  coveredAreas = @($articles | ForEach-Object { [string]$_.area })
  actionRequiredCount = if ($null -ne $finalActionMap) { [int]$finalActionMap.actionRequiredCount } else { 0 }
  strictExecutionStepCount = if ($null -ne $strictOrder) { [int]$strictOrder.executionStepCount } else { 0 }
  articles = @($articles)
  forbiddenClaims = @($forbiddenClaims)
  requiredPublicBoundaryMarkers = @("not runtime proof", "not post-publish proof", "not publish approval", "not release close approval", "not package push")
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This readiness map is article planning only. It does not publish articles, does not publish packages, does not close release issues, and does not promote runtime or post-publish proof."
}

$jsonPath = Join-Path $OutputRoot "article-publishing-readiness-map.json"
$markdownPath = Join-Path $OutputRoot "article-publishing-readiness-map.md"
$readinessMap | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($article in $articles) {
  "| ``$(ConvertTo-MarkdownCell $article.area)`` | $(ConvertTo-MarkdownCell $article.title) | ``$(ConvertTo-MarkdownCell $article.readinessState)`` | ``False`` |"
}

$markdown = @"
# Article Publishing Readiness Map

Generated at: ``$($readinessMap.generatedAtUtc)``

## Summary

- readinessState: ``$($readinessMap.readinessState)``
- minimumArticleCount: ``$($readinessMap.minimumArticleCount)``
- roadmapArticleCount: ``$($readinessMap.roadmapArticleCount)``
- technicalArticleCount: ``$($readinessMap.technicalArticleCount)``
- focusedReadinessArticleCount: ``$($readinessMap.focusedReadinessArticleCount)``
- actionRequiredCount: ``$($readinessMap.actionRequiredCount)``
- strictExecutionStepCount: ``$($readinessMap.strictExecutionStepCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Focus Areas

| Area | Article | State | Can Publish |
| --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($readinessMap.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Article publishing readiness map written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ReadinessState=$($readinessMap.readinessState) FocusedReadinessArticleCount=$($readinessMap.focusedReadinessArticleCount)"
