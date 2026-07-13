[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

function New-ArticleLane {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Category,
    [string]$PublicationState,
    [string[]]$AllowedClaims,
    [string[]]$BlockedClaims,
    [string[]]$RequiredProofBeforePublish,
    [string[]]$RequiredMedia,
    [string[]]$CodePaths
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    category = $Category
    publicationState = $PublicationState
    allowedClaims = @($AllowedClaims)
    blockedClaims = @($BlockedClaims)
    requiredProofBeforePublish = @($RequiredProofBeforePublish)
    requiredMedia = @($RequiredMedia)
    codePaths = @($CodePaths)
    mediaRequirementCount = @($RequiredMedia).Count
    codePathCount = @($CodePaths).Count
    canPublishNow = $false
    ownerProofRequired = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

$roadmap = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "docs\articles\zh-cn\publishing\article-roadmap-30plus.json"
$readinessMap = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\article-publishing-readiness-map-validation.json"
$strictClosure = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\release-close-strict-evidence-closure-validation.json"

$roadmapArticleCount = [int](Get-PropertyOrDefault -Object $roadmap -Name "articleCount" -DefaultValue 0)
$readinessState = [string](Get-PropertyOrDefault -Object $readinessMap -Name "validationState" -DefaultValue "missing-article-publishing-readiness-map-validation")
$closureState = [string](Get-PropertyOrDefault -Object $strictClosure -Name "validationState" -DefaultValue "missing-release-close-strict-evidence-closure-validation")

$blockedClaims = @(
  "已发布到 NuGet",
  "公开包已经可安装",
  "公开渠道 clean consumer 已通过",
  "release closed",
  "dry-run 证明真实发布",
  "dashboard 证明真实发布",
  "manual approval 证明真实发布",
  "queued workflow 证明真实发布",
  "TensorRtExec report 证明运行 proof"
)

$lanes = @(
  New-ArticleLane "project-overview" "TensorRT C# API 4.0 项目整体介绍" "public-ready-after-editorial-review" "blocked-until-boundary-review" @("项目目标", "API 分层", "跨版本设计", "当前 non-proof 边界") $blockedClaims @("proof boundary scan", "article body review") @("architecture diagram", "API layer table") @("src/JYPPX.TensorRtSharp", "src/JYPPX.CudaSharp")
  New-ArticleLane "api-interface-guide" "TensorRT/CUDA 接口体系与 deferred 提升路线" "technical-deep-dive" "blocked-until-boundary-review" @("manifest/source/wrapper 关系", "deferred 边界", "readonly API 优先级") $blockedClaims @("interface coverage matrix", "deferred boundary review") @("interface coverage table", "version guard diagram") @("artifacts/interface-coverage", "native/manifests")
  New-ArticleLane "source-build-cpp" "源码编译教程：C++ bridge、CUDA、TensorRT、cuDNN 环境" "build-guide" "blocked-until-build-proof-review" @("环境要求", "CMake preset", "native bridge build path") $blockedClaims @("source build command transcript", "environment matrix") @("build environment screenshot", "CMake preset table") @("native", "eng")
  New-ArticleLane "dual-package-strategy" "GitHub 全量依赖包与 NuGet 小包双路线发布说明" "package-guide" "blocked-public-package-owner-proof-required" @("双路线设计", "NuGet 小包边界", "GitHub runtime 包边界") $blockedClaims @("public package URL", "package hashes", "clean consumer proof") @("package route diagram", "package id table") @("pack/runtime-split", "eng/Export-DualPackagePublishPreflightMatrix.ps1")
  New-ArticleLane "yolovision-series" "YoloVision 全系列 det/cls/seg/obb/pose/sem 教程规划" "case-series" "blocked-real-model-owner-proof-required" @("YOLO family/task matrix", "模型获取流程", "任务输出结构") $blockedClaims @("model license", "real model inputs", "golden outputs", "runtime smoke logs") @("YOLO family task matrix", "sample output images") @("samples/YoloVision")
  New-ArticleLane "onnx-to-engine-trtexec" "OnnxToEngine 与官方 trtexec 转换能力对照" "case-series" "blocked-conversion-parity-proof-required" @("模型转换参数", "engine 输出", "trtexec 边界") $blockedClaims @("option mapping review", "conversion report examples") @("option mapping table", "conversion report screenshot") @("samples/OnnxToEngine")
  New-ArticleLane "tensorrtexec-application" "applications/TensorRtExec 控制台与 WinForms 使用教程" "application-guide" "blocked-gui-cli-proof-required" @("CLI/GUI 双入口", "trtexec-like 功能目标", "边界说明") $blockedClaims @("CLI/GUI parity checklist", "WinForms screenshots") @("main form screenshot", "CLI command table") @("applications/TensorRtExec")
  New-ArticleLane "post-publish-clean-consumer" "公开包发布后的 clean consumer 安装运行教程" "post-publish-proof-guide" "blocked-post-publish-owner-proof-required" @("公开源安装流程", "clean root outside repo", "hash verification method") $blockedClaims @("public package page URL", "downloaded SHA256", "restore/build/smoke logs") @("clean consumer workflow", "hash evidence table") @("eng/Export-PostPublishVerificationOwnerInputTemplate.ps1", "eng/Test-PostPublishVerificationRecord.ps1")
  New-ArticleLane "faq-troubleshooting" "常见问题排查：CUDA 35、DLL resolution、版本不匹配" "faq" "blocked-until-boundary-review" @("排查流程", "诊断脚本", "版本矩阵") $blockedClaims @("diagnostic examples", "known limitation review") @("troubleshooting flowchart", "version matrix table") @("docs/articles/zh-cn/cuda-error-35-troubleshooting.md", "eng/Export-Trt11RuntimeDllResolutionReport.ps1")
)

$record = [pscustomobject]@{
  recordKind = "public-article-readiness-matrix"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  matrixState = "blocked-public-article-proof-and-boundary-review-required"
  roadmapArticleCount = $roadmapArticleCount
  minimumArticleCount = 30
  laneCount = $lanes.Count
  blockedLaneCount = @($lanes | Where-Object { -not [bool]$_.canPublishNow }).Count
  sourceStates = [pscustomobject]@{
    articlePublishingReadinessMapValidationState = $readinessState
    releaseCloseStrictEvidenceClosureValidationState = $closureState
  }
  lanes = @($lanes)
  globalBlockedClaims = @($blockedClaims)
  ownerActionRequired = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Public article readiness matrix only. It plans public articles and claim boundaries, but does not publish articles, publish packages, prove public package installation, prove runtime execution, or close release issues."
}

$jsonPath = Join-Path $OutputRoot "public-article-readiness-matrix.json"
$mdPath = Join-Path $OutputRoot "public-article-readiness-matrix.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 14)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Public Article Readiness Matrix") | Out-Null
$md.Add("") | Out-Null
$md.Add("- matrixState: ``$($record.matrixState)``") | Out-Null
$md.Add("- roadmapArticleCount: ``$($record.roadmapArticleCount)``") | Out-Null
$md.Add("- laneCount: ``$($record.laneCount)``") | Out-Null
$md.Add("- blockedLaneCount: ``$($record.blockedLaneCount)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Lane | Title | State | Media | Code Paths |") | Out-Null
$md.Add("| --- | --- | --- | --- | --- |") | Out-Null
foreach ($lane in $lanes) {
  $md.Add("| ``$($lane.id)`` | $(ConvertTo-MarkdownCell $lane.title) | ``$($lane.publicationState)`` | $($lane.mediaRequirementCount) | $($lane.codePathCount) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "PublicArticleReadinessMatrixState=$($record.matrixState) Lanes=$($record.laneCount) Blocked=$($record.blockedLaneCount)"
