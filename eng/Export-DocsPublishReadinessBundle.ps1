[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function ConvertTo-RelativePath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return ""
  }

  $fullPath = [System.IO.Path]::GetFullPath($Path)
  $root = [System.IO.Path]::GetFullPath($RepositoryRoot)
  if ($fullPath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $fullPath.Substring($root.Length).TrimStart('\', '/')
  }

  return $Path
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-ArticleCategory {
  param(
    [string]$FileName,
    [string]$Content
  )

  if ($FileName.StartsWith("blog-", [System.StringComparison]::OrdinalIgnoreCase)) {
    return "blog"
  }

  if ($FileName.Contains("release", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("publish", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("package", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("nuget", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("linux", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("readiness", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("signing", [System.StringComparison]::OrdinalIgnoreCase)) {
    return "release-deployment"
  }

  if ($FileName.Contains("classification", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("yolo", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("sample", [System.StringComparison]::OrdinalIgnoreCase)) {
    return "sample"
  }

  if ($FileName.Contains("callback", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("allocator", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("debug-listener", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("output-", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("boundary", [System.StringComparison]::OrdinalIgnoreCase)) {
    return "boundary"
  }

  if ($FileName.Contains("cuda", [System.StringComparison]::OrdinalIgnoreCase)) {
    return "cuda"
  }

  if ($FileName.Contains("plugin", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("tensorrt", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("trt", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("network", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("refit", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("error-recorder", [System.StringComparison]::OrdinalIgnoreCase) -or
      $FileName.Contains("managed-", [System.StringComparison]::OrdinalIgnoreCase)) {
    return "api"
  }

  if ($Content.Contains("samples/", [System.StringComparison]::OrdinalIgnoreCase) -or
      $Content.Contains("smoke/", [System.StringComparison]::OrdinalIgnoreCase)) {
    return "tutorial"
  }

  return "project"
}

function New-SourceStatus {
  param(
    [string]$Id,
    [string]$Path,
    [bool]$Exists,
    [string]$State,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    path = $Path
    exists = $Exists
    state = $State
    boundary = $Boundary
  }
}

function New-CoreArticleRequirement {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Path,
    [string]$Area
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    path = $Path
    area = $Area
  }
}

function New-OwnerAction {
  param(
    [string]$Id,
    [string]$State,
    [string]$Action,
    [string]$RequiredEvidence,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    state = $State
    action = $Action
    requiredEvidence = $RequiredEvidence
    boundary = $Boundary
  }
}

function Test-ContentContainsPath {
  param(
    [string]$Content,
    [string]$Path
  )

  if ([string]::IsNullOrWhiteSpace($Content)) {
    return $false
  }

  $normalized = $Path.Replace('\', '/')
  $backslash = $Path.Replace('/', '\')
  $withoutDocsPrefix = if ($normalized.StartsWith("docs/", [System.StringComparison]::OrdinalIgnoreCase)) {
    $normalized.Substring(5)
  }
  else {
    $normalized
  }
  $withoutDocsBackslash = $withoutDocsPrefix.Replace('/', '\')
  return $Content.Contains($normalized, [System.StringComparison]::OrdinalIgnoreCase) -or
    $Content.Contains($backslash, [System.StringComparison]::OrdinalIgnoreCase) -or
    $Content.Contains($withoutDocsPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
    $Content.Contains($withoutDocsBackslash, [System.StringComparison]::OrdinalIgnoreCase)
}

$docsRoot = Join-Path $RepositoryRoot "docs"
$articlesRoot = Join-Path $docsRoot "articles\zh-cn"
$tocPath = Join-Path $docsRoot "toc.yml"
$indexPath = Join-Path $docsRoot "index.md"
$roadmapPath = Join-Path $articlesRoot "technical-article-roadmap.md"
$docfxSitePath = Join-Path $docsRoot "_site\index.html"
$sampleAssetsRoot = Join-Path $RepositoryRoot "samples\assets"
$tocContent = if (Test-Path -LiteralPath $tocPath -PathType Leaf) { Get-Content -LiteralPath $tocPath -Raw -Encoding utf8 } else { "" }
$indexContent = if (Test-Path -LiteralPath $indexPath -PathType Leaf) { Get-Content -LiteralPath $indexPath -Raw -Encoding utf8 } else { "" }

$articleFiles = @()
if (Test-Path -LiteralPath $articlesRoot -PathType Container) {
  $articleFiles = @(Get-ChildItem -LiteralPath $articlesRoot -File -Filter "*.md" | Sort-Object Name)
}

$articleRecords = New-Object System.Collections.Generic.List[object]
foreach ($article in $articleFiles) {
  $content = Get-Content -LiteralPath $article.FullName -Raw -Encoding utf8
  $category = Get-ArticleCategory -FileName $article.Name -Content $content
  $sampleBacked = $content.Contains("samples/", [System.StringComparison]::OrdinalIgnoreCase) -or
    $content.Contains("smoke/", [System.StringComparison]::OrdinalIgnoreCase) -or
    $content.Contains("README.md", [System.StringComparison]::OrdinalIgnoreCase)
  $hasEvidenceLanguage = $content.Contains("artifacts/", [System.StringComparison]::OrdinalIgnoreCase) -or
    $content.Contains("evidence", [System.StringComparison]::OrdinalIgnoreCase) -or
    $content.Contains("proof", [System.StringComparison]::OrdinalIgnoreCase) -or
    $content.Contains("验证", [System.StringComparison]::OrdinalIgnoreCase) -or
    $content.Contains("证据", [System.StringComparison]::OrdinalIgnoreCase)
  $hasCommand = $content.Contains("dotnet ", [System.StringComparison]::OrdinalIgnoreCase) -or
    $content.Contains("pwsh ", [System.StringComparison]::OrdinalIgnoreCase) -or
    $content.Contains("cmake ", [System.StringComparison]::OrdinalIgnoreCase)
  $highQualityCandidate = $content.Length -ge 800 -and ($hasEvidenceLanguage -or $hasCommand -or $sampleBacked)

  $titleLine = $content -split "`r?`n" | Where-Object { $_.StartsWith("# ", [System.StringComparison]::Ordinal) } | Select-Object -First 1
  if ([string]::IsNullOrWhiteSpace($titleLine)) {
    $titleLine = [System.IO.Path]::GetFileNameWithoutExtension($article.Name)
  }

  $articleRecords.Add([pscustomobject]@{
      file = ConvertTo-RelativePath -Path $article.FullName
      title = ($titleLine -replace "^#\s+", "")
      category = $category
      length = $content.Length
      sampleBacked = $sampleBacked
      hasEvidenceLanguage = $hasEvidenceLanguage
      hasCommand = $hasCommand
      highQualityCandidate = $highQualityCandidate
    })
}

$sampleReadmes = @()
$samplesRoot = Join-Path $RepositoryRoot "samples"
if (Test-Path -LiteralPath $samplesRoot -PathType Container) {
  $sampleReadmes = @(Get-ChildItem -LiteralPath $samplesRoot -Recurse -File -Filter "README.md" | Sort-Object FullName)
}

$smokeReadmes = @()
$smokeRoot = Join-Path $RepositoryRoot "smoke"
if (Test-Path -LiteralPath $smokeRoot -PathType Container) {
  $smokeReadmes = @(Get-ChildItem -LiteralPath $smokeRoot -Recurse -File -Filter "README.md" | Sort-Object FullName)
}

$sampleAssetTemplates = @()
if (Test-Path -LiteralPath $sampleAssetsRoot -PathType Container) {
  $sampleAssetTemplates = @(Get-ChildItem -LiteralPath $sampleAssetsRoot -File -Filter "*.json" | Sort-Object Name)
}

$articleCount = $articleRecords.Count
$highQualityArticleCount = @($articleRecords | Where-Object { $_.highQualityCandidate }).Count
$sampleBackedArticleCount = @($articleRecords | Where-Object { $_.sampleBacked }).Count
$hasArticleRoadmap = Test-Path -LiteralPath $roadmapPath -PathType Leaf
$hasToc = Test-Path -LiteralPath $tocPath -PathType Leaf
$hasIndex = Test-Path -LiteralPath $indexPath -PathType Leaf
$docfxValidationState = if (Test-Path -LiteralPath $docfxSitePath -PathType Leaf) { "ready" } else { "missing-docfx-site-output" }
$externalPublishingPlanState = "not-present"
$coverImagePlanState = "not-present"
$canPublishDocsExternally = $false

$coreArticleRequirements = @(
  New-CoreArticleRequirement -Id "project-overview" -Title "Project overview" -Path "docs/articles/zh-cn/project-overview.md" -Area "project"
  New-CoreArticleRequirement -Id "final-package-review-bundle" -Title "Final package review bundle" -Path "docs/articles/zh-cn/final-package-review-bundle.md" -Area "release"
  New-CoreArticleRequirement -Id "release-evidence-bundle" -Title "Release evidence bundle" -Path "docs/articles/zh-cn/release-evidence-bundle.md" -Area "release"
  New-CoreArticleRequirement -Id "release-candidate-full-acceptance-summary" -Title "Release candidate full acceptance summary" -Path "docs/articles/zh-cn/release-candidate-full-acceptance-summary.md" -Area "release"
  New-CoreArticleRequirement -Id "release-publish-execution-checklist" -Title "Release publish execution checklist" -Path "docs/articles/zh-cn/release-publish-execution-checklist.md" -Area "release"
  New-CoreArticleRequirement -Id "external-runtime-proof-record" -Title "External runtime proof record" -Path "docs/articles/zh-cn/external-runtime-proof-record.md" -Area "runtime-proof"
  New-CoreArticleRequirement -Id "post-publish-verification-record" -Title "Post publish verification record" -Path "docs/articles/zh-cn/post-publish-verification-record.md" -Area "post-publish"
  New-CoreArticleRequirement -Id "package-consumer-validation" -Title "Package consumer validation" -Path "docs/articles/zh-cn/package-consumer-validation.md" -Area "package"
  New-CoreArticleRequirement -Id "real-model-owner-backfill" -Title "Real model owner backfill" -Path "docs/articles/zh-cn/real-model-owner-backfill-checklist.md" -Area "sample-assets"
  New-CoreArticleRequirement -Id "classification-model-assets" -Title "Classification model assets" -Path "docs/articles/zh-cn/classification-model-assets.md" -Area "sample-assets"
  New-CoreArticleRequirement -Id "yolovision-model-assets" -Title "YoloVision model assets" -Path "docs/articles/zh-cn/yolovision-model-assets.md" -Area "sample-assets"
  New-CoreArticleRequirement -Id "tensorrtexec-tool-getting-started" -Title "TensorRtExec tool getting started" -Path "docs/articles/zh-cn/tensorrtexec-tool-getting-started.md" -Area "applications"
  New-CoreArticleRequirement -Id "known-limitations" -Title "Known limitations" -Path "docs/articles/zh-cn/known-limitations-4.0.0-rc.md" -Area "release"
)

$coreArticleCoverage = @($coreArticleRequirements | ForEach-Object {
    $required = $_
    $fullPath = Join-Path $RepositoryRoot ($required.path.Replace('/', '\'))
    $exists = Test-Path -LiteralPath $fullPath -PathType Leaf
    $articleRecord = $articleRecords | Where-Object { ($_.file.Replace('\', '/')) -eq $required.path } | Select-Object -First 1
    $includedInToc = Test-ContentContainsPath -Content $tocContent -Path $required.path
    $linkedFromIndex = Test-ContentContainsPath -Content $indexContent -Path $required.path
    $highQualityCandidate = $false
    $hasEvidenceLanguage = $false
    $hasCommand = $false
    $sampleBacked = $false
    if ($articleRecord) {
      $highQualityCandidate = [bool]$articleRecord.highQualityCandidate
      $hasEvidenceLanguage = [bool]$articleRecord.hasEvidenceLanguage
      $hasCommand = [bool]$articleRecord.hasCommand
      $sampleBacked = [bool]$articleRecord.sampleBacked
    }

    [pscustomobject]@{
      id = $required.id
      title = $required.title
      path = $required.path
      area = $required.area
      exists = $exists
      includedInToc = $includedInToc
      linkedFromIndex = $linkedFromIndex
      highQualityCandidate = $highQualityCandidate
      hasEvidenceLanguage = $hasEvidenceLanguage
      hasCommand = $hasCommand
      sampleBacked = $sampleBacked
      covered = $exists -and $includedInToc -and $linkedFromIndex -and $highQualityCandidate
      ownerAction = if ($exists -and $includedInToc -and $linkedFromIndex -and $highQualityCandidate) { "owner-review-before-external-publish" } else { "fix-core-article-coverage" }
    }
  })

$coreArticleCount = $coreArticleCoverage.Count
$coreArticleCoveredCount = @($coreArticleCoverage | Where-Object { $_.covered }).Count
$coreArticleMissingCount = @($coreArticleCoverage | Where-Object { -not $_.covered }).Count

$ownerActions = @(
  New-OwnerAction `
    -Id "external-publishing-plan" `
    -State "owner-action-required" `
    -Action "Choose and record the external documentation channels such as website, GitHub Pages, WeChat, blog, or community platform." `
    -RequiredEvidence "Owner-approved channel list, publication order, rollback/update plan, and target URLs after publication." `
    -Boundary "A local docs bundle is not external publication proof."
  New-OwnerAction `
    -Id "cover-media-assets" `
    -State "owner-action-required" `
    -Action "Prepare article covers, diagrams, screenshots, and model/sample images with license review." `
    -RequiredEvidence "Media inventory with source, license, intended article, and generated/edited asset paths." `
    -Boundary "Article text readiness does not approve image or media redistribution."
  New-OwnerAction `
    -Id "proof-claim-review" `
    -State "owner-action-required" `
    -Action "Review runtime proof, package consumer proof, post-publish proof, Linux proof, and sample proof claims before external publication." `
    -RequiredEvidence "Clean stale-claim audit plus owner sign-off for any release-facing claim." `
    -Boundary "blocked-by-cuda-driver, template-only, handoff-only, build-only, and dependency-probe-only remain non-proof."
  New-OwnerAction `
    -Id "sample-asset-claim-review" `
    -State "owner-action-required" `
    -Action "Review Classification/YoloVision/YOLOX-S asset-dependent articles before publication." `
    -RequiredEvidence "Real model owner handoff, sample asset manifest audit, sample run evidence validation, and license/hash notes." `
    -Boundary "Sample asset templates and owner handoff records are not sample smoke passes."
)
$ownerActionCount = @($ownerActions | Where-Object { $_.state -ne "ready" }).Count

if ($hasToc -and $hasIndex -and $hasArticleRoadmap -and $articleCount -ge 30 -and $highQualityArticleCount -ge 30 -and $coreArticleMissingCount -eq 0 -and $docfxValidationState -eq "ready") {
  $readinessState = "ready-for-owner-review"
}
else {
  $readinessState = "blocked-docs-incomplete"
}

$categoryCounts = @($articleRecords | Group-Object category | Sort-Object Name | ForEach-Object {
    [pscustomobject]@{
      category = $_.Name
      count = $_.Count
    }
  })

$sourceStatuses = @(
  New-SourceStatus -Id "docs-toc" -Path "docs/toc.yml" -Exists $hasToc -State ($(if ($hasToc) { "present" } else { "missing" })) -Boundary "TOC presence does not mean external publishing has happened."
  New-SourceStatus -Id "docs-index" -Path "docs/index.md" -Exists $hasIndex -State ($(if ($hasIndex) { "present" } else { "missing" })) -Boundary "Index presence does not mean external publishing has happened."
  New-SourceStatus -Id "technical-article-roadmap" -Path "docs/articles/zh-cn/technical-article-roadmap.md" -Exists $hasArticleRoadmap -State ($(if ($hasArticleRoadmap) { "present" } else { "missing" })) -Boundary "Roadmap rows are planning evidence and may include not-yet-run asset-dependent cases."
  New-SourceStatus -Id "docfx-site-output" -Path "docs/_site/index.html" -Exists ($docfxValidationState -eq "ready") -State $docfxValidationState -Boundary "DocFX local output is not external docs publication."
  New-SourceStatus -Id "sample-readmes" -Path "samples/**/README.md" -Exists ($sampleReadmes.Count -gt 0) -State ("count=" + $sampleReadmes.Count) -Boundary "Sample README files document local use; asset-dependent samples still need owner-provided assets."
  New-SourceStatus -Id "smoke-readmes" -Path "smoke/**/README.md" -Exists ($smokeReadmes.Count -gt 0) -State ("count=" + $smokeReadmes.Count) -Boundary "Smoke runner docs are validation guidance, not proof that all smokes passed on this machine."
  New-SourceStatus -Id "sample-asset-templates" -Path "samples/assets/*.json" -Exists ($sampleAssetTemplates.Count -gt 0) -State ("count=" + $sampleAssetTemplates.Count) -Boundary "Asset templates are not downloaded model assets and are not sample smoke passes."
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "docs-publish-readiness-bundle"
  readinessState = $readinessState
  canPublishDocsExternally = $canPublishDocsExternally
  articleCount = $articleCount
  highQualityArticleCount = $highQualityArticleCount
  sampleBackedArticleCount = $sampleBackedArticleCount
  coreArticleCount = $coreArticleCount
  coreArticleCoveredCount = $coreArticleCoveredCount
  coreArticleMissingCount = $coreArticleMissingCount
  ownerActionCount = $ownerActionCount
  categoryCounts = $categoryCounts
  hasArticleRoadmap = $hasArticleRoadmap
  hasToc = $hasToc
  hasIndex = $hasIndex
  docfxValidationState = $docfxValidationState
  externalPublishingPlanState = $externalPublishingPlanState
  coverImagePlanState = $coverImagePlanState
  sampleReadmeCount = $sampleReadmes.Count
  smokeReadmeCount = $smokeReadmes.Count
  sampleAssetTemplateCount = $sampleAssetTemplates.Count
  coreArticleCoverage = @($coreArticleCoverage)
  ownerActions = @($ownerActions)
  sourceStatuses = $sourceStatuses
  articles = @($articleRecords.ToArray())
  sourceEvidence = @(
    "docs/toc.yml",
    "docs/index.md",
    "docs/articles/zh-cn/technical-article-roadmap.md",
    "docs/articles/zh-cn",
    "docs/_site/index.html",
    "samples/README.md",
    "samples/**/README.md",
    "smoke/README.md",
    "samples/assets/*.json",
    "artifacts/final-release/stale-release-claims-audit.json",
    "artifacts/final-release/release-evidence-bundle.json"
  )
  safetyNotes = @(
    "This bundle does not publish documentation externally.",
    "canPublishDocsExternally=false until the owner approves an external channel, publishing plan, and cover/media assets.",
    "DocFX local output is local documentation validation, not external publication proof.",
    "Asset templates and sample README files are not downloaded model assets or sample smoke passes.",
    "Core article coverage means local owner-review readiness, not external publication.",
    "Article count and local completeness heuristics are owner-review inputs, not marketing approval."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "docs-publish-readiness-bundle.json"
$markdownPath = Join-Path $outputRoot "docs-publish-readiness-bundle.md"

$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Docs Publish Readiness Bundle")
$lines.Add("")
$lines.Add("Readiness state: ``$readinessState``")
$lines.Add("")
$lines.Add("This bundle counts Chinese technical and promotional articles, checks the roadmap and DocFX local output, and records sample-backed documentation evidence. It does not publish documentation externally.")
$lines.Add("")
$lines.Add("## Decision Fields")
$lines.Add("")
$lines.Add("- can publish docs externally: ``$canPublishDocsExternally``")
$lines.Add("- article count: $articleCount")
$lines.Add("- high-quality article count: $highQualityArticleCount")
$lines.Add("- sample-backed article count: $sampleBackedArticleCount")
$lines.Add("- core article covered count: $coreArticleCoveredCount / $coreArticleCount")
$lines.Add("- core article missing count: $coreArticleMissingCount")
$lines.Add("- owner action count: $ownerActionCount")
$lines.Add("- has article roadmap: ``$hasArticleRoadmap``")
$lines.Add("- DocFX validation state: ``$docfxValidationState``")
$lines.Add("- external publishing plan state: ``$externalPublishingPlanState``")
$lines.Add("- cover image plan state: ``$coverImagePlanState``")
$lines.Add("")
$lines.Add("## Category Counts")
$lines.Add("")
$lines.Add("| Category | Count |")
$lines.Add("| --- | ---: |")
foreach ($category in $categoryCounts) {
  $lines.Add("| ``$($category.category)`` | $($category.count) |")
}
$lines.Add("")
$lines.Add("## Core Article Coverage")
$lines.Add("")
$lines.Add("| ID | Covered | TOC | Index | High quality | Path | Owner action |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- |")
foreach ($article in $coreArticleCoverage) {
  $lines.Add("| ``$($article.id)`` | ``$($article.covered)`` | ``$($article.includedInToc)`` | ``$($article.linkedFromIndex)`` | ``$($article.highQualityCandidate)`` | ``$($article.path)`` | ``$($article.ownerAction)`` |")
}
$lines.Add("")
$lines.Add("## Owner Actions")
$lines.Add("")
$lines.Add("| ID | State | Action | Required evidence | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($action in $ownerActions) {
  $lines.Add("| ``$($action.id)`` | ``$($action.state)`` | $(ConvertTo-MarkdownCell $action.action) | $(ConvertTo-MarkdownCell $action.requiredEvidence) | $(ConvertTo-MarkdownCell $action.boundary) |")
}
$lines.Add("")
$lines.Add("## Source Statuses")
$lines.Add("")
$lines.Add("| ID | State | Exists | Path | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($status in $sourceStatuses) {
  $lines.Add("| ``$($status.id)`` | ``$($status.state)`` | ``$($status.exists)`` | ``$($status.path)`` | $(ConvertTo-MarkdownCell $status.boundary) |")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $record.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Docs publish readiness bundle written to $jsonPath"
Write-Host "Docs publish readiness bundle written to $markdownPath"
