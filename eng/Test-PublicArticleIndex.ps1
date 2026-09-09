[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
else {
  $RepositoryRoot = (Resolve-Path $RepositoryRoot).Path
}

function Assert-Condition {
  param(
    [bool]$Condition,
    [string]$Message
  )

  if (-not $Condition) {
    throw $Message
  }
}

function Resolve-RepositoryPath {
  param([string]$RelativePath)

  return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $RelativePath.Replace('/', [IO.Path]::DirectorySeparatorChar)))
}

function Get-MarkdownHeadings {
  param([string]$Content)

  $headings = [System.Collections.Generic.List[object]]::new()
  $inFence = $false
  foreach ($line in ($Content -split "`r?`n")) {
    if ($line -match '^\s*(```+|~~~+)') {
      $inFence = -not $inFence
      continue
    }

    if (-not $inFence -and $line -match '^(#{1,4})\s+(.+?)\s*$') {
      $headings.Add([pscustomobject]@{
        Level = $Matches[1].Length
        Text = $Matches[2]
      })
    }
  }

  return $headings.ToArray()
}

$indexPath = Resolve-RepositoryPath "docs/articles/zh-cn/article-index.json"
$index = Get-Content -LiteralPath $indexPath -Raw -Encoding UTF8 | ConvertFrom-Json

Assert-Condition ($index.schemaVersion -eq 1) "article-index.json schemaVersion must be 1."
Assert-Condition ($index.recordKind -eq "public-article-index") "article-index.json recordKind is invalid."
Assert-Condition ($index.sourceRoot -eq "docs/articles/zh-cn") "article-index.json sourceRoot is invalid."

$expectedStatuses = @("draft", "review", "ready", "published")
Assert-Condition (($index.statusValues -join '|') -eq ($expectedStatuses -join '|')) "article-index.json statusValues are invalid."

$requiredFields = @(
  "id",
  "module",
  "series",
  "title",
  "sourcePath",
  "sourceVersion",
  "status",
  "canonical",
  "immutable",
  "publishedAt",
  "csdnUrl",
  "supersedes",
  "publishedCommit",
  "publishedSha256"
)
$requiredIds = @(
  "REL-001",
  "MSC-001",
  "MSC-002",
  "MSC-003",
  "MSC-004",
  "MSC-005",
  "MSC-006",
  "MSC-007",
  "MSC-008",
  "MSC-009",
  "SMP-001",
  "SMP-002",
  "SMP-003",
  "SMP-004",
  "SMP-005",
  "SMP-006",
  "SMP-007",
  "SMP-008",
  "SMP-009",
  "APP-YV-001",
  "APP-YV-002",
  "APP-YV-003",
  "APP-YV-004",
  "APP-YV-005",
  "APP-YV-006",
  "APP-YV-007",
  "APP-YV-008",
  "APP-YV-009",
  "APP-ONNX-001",
  "APP-ONNX-002",
  "APP-ONNX-003",
  "APP-EXEC-001",
  "APP-EXEC-002",
  "APP-EXEC-003",
  "APP-EXEC-004",
  "APP-EXEC-005",
  "API-001",
  "API-002",
  "API-003",
  "API-004",
  "API-005",
  "API-006",
  "INS-001",
  "BLD-001",
  "INS-002",
  "INS-003",
  "INS-004",
  "BLD-002",
  "BLD-003",
  "BLD-004"
)
$seenIds = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$seenPaths = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$retiredBrand = 'TensorRT CSharp' + 'API'
$retiredBridgePackageListUrl = 'https://github.com/users/guojin-yan/packages' + '?repo_name=TensorRT-CSharp-API'

foreach ($article in $index.articles) {
  foreach ($field in $requiredFields) {
    Assert-Condition ($article.PSObject.Properties.Name -contains $field) "Article '$($article.id)' is missing '$field'."
  }

  Assert-Condition ($article.id -match '^(REL|SMP|MSC|API|INS|BLD)-[0-9]{3}$|^APP-(YV|ONNX|EXEC)-[0-9]{3}$') "Invalid article ID '$($article.id)'."
  Assert-Condition ($seenIds.Add([string]$article.id)) "Duplicate article ID '$($article.id)'."
  Assert-Condition ($seenPaths.Add([string]$article.sourcePath)) "Duplicate sourcePath '$($article.sourcePath)'."
  Assert-Condition ($expectedStatuses -contains $article.status) "Invalid status for '$($article.id)'."
  Assert-Condition ($article.canonical -eq $true) "Indexed article '$($article.id)' must be canonical."
  Assert-Condition (-not [string]::IsNullOrWhiteSpace($article.sourceVersion)) "Missing sourceVersion for '$($article.id)'."
  Assert-Condition ($article.sourcePath -notmatch '\\') "sourcePath must use forward slashes: '$($article.sourcePath)'."

  $sourcePath = Resolve-RepositoryPath $article.sourcePath
  Assert-Condition (Test-Path -LiteralPath $sourcePath -PathType Leaf) "Missing article source '$($article.sourcePath)'."
  $content = Get-Content -LiteralPath $sourcePath -Raw -Encoding UTF8
  $headings = @(Get-MarkdownHeadings -Content $content)
  Assert-Condition ($content -notmatch '(?i)\b(TODO|TBD)\b') "Unresolved placeholder in '$($article.sourcePath)'."
  Assert-Condition ($content -notmatch '(?i)[A-Z]:\\Users\\') "Machine-specific path in '$($article.sourcePath)'."
  Assert-Condition ($content -notmatch '(?<![A-Za-z0-9])TensorRtSharp4\.0\.sln(?![A-Za-z0-9])') "Canonical article '$($article.id)' uses the retired solution name 'TensorRtSharp4.0.sln'."
  $h1Headings = @($headings | Where-Object Level -eq 1)
  Assert-Condition ($h1Headings.Count -eq 1) "Canonical article '$($article.id)' must have exactly one H1."
  Assert-Condition ($h1Headings[0].Text -eq [string]$article.title) "H1 does not match article-index title for '$($article.id)'."
  Assert-Condition ([string]$article.title -notmatch 'TensorRtSharp4\.0') "Canonical article title '$($article.id)' uses the retired product title."
  Assert-Condition ([string]$article.title -notmatch [regex]::Escape($retiredBrand)) "Canonical article title '$($article.id)' omits the space in 'CSharp API'."
  Assert-Condition ($content -notmatch [regex]::Escape($retiredBrand)) "Canonical article '$($article.id)' omits the space in 'CSharp API'."
  Assert-Condition ($content -match 'TensorRT CSharp API') "Canonical article '$($article.id)' must use the spaced 'TensorRT CSharp API' branding."
  Assert-Condition ($content -match '(?m)^## 1\. 前言\r?$') "Canonical article '$($article.id)' must start with a numbered project preface."
  Assert-Condition ([regex]::Matches($content, '<!-- public-article-project-preface:start -->').Count -eq 1) "Canonical article '$($article.id)' must have one shared project preface."
  Assert-Condition ([regex]::Matches($content, '<!-- public-article-layout:start -->').Count -eq 1) "Canonical article '$($article.id)' must have one narrow diagram style block."
  Assert-Condition ([regex]::Matches($content, '<!-- public-article-declaration:start -->').Count -eq 1) "Canonical article '$($article.id)' must have one article declaration."
  Assert-Condition ($content.TrimEnd().EndsWith('<!-- public-article-declaration:end -->')) "Article declaration must be the final section in '$($article.sourcePath)'."
  Assert-Condition ($content.Contains('程序出处与输出说明')) "Canonical article '$($article.id)' is missing program provenance and output guidance."
  Assert-Condition ($content.Contains('personal-contact-banner-v6-zh.png')) "Canonical article '$($article.id)' is missing the contact banner."
  Assert-Condition ($content -notmatch '(?m)^##\s+\d+\.\s+(下一步|下一篇|后续文章)') "Canonical article '$($article.id)' links an unpublished next article."
  Assert-Condition ($content -notmatch '(?m)^\s*(flowchart|graph)\s+(LR|RL)\s*$') "Canonical article '$($article.id)' contains an overly wide horizontal flowchart."
  Assert-Condition ($content.Contains('.content article { min-width: 0; overflow-wrap: anywhere; }')) "Canonical article '$($article.id)' is missing the narrow-page overflow rule."
  Assert-Condition ($content.Contains('.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }')) "Canonical article '$($article.id)' is missing the long-link wrapping rule."
  Assert-Condition ($content.Contains('.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }')) "Canonical article '$($article.id)' is missing the local code/table scrolling rule."
  Assert-Condition ($content.Contains('pre.mermaid { max-width: 640px;')) "Canonical article '$($article.id)' is missing the narrow Mermaid width rule."
  Assert-Condition ($content -notmatch '(?<!!)\[[^\]]+\]\([^)]+\)') "Canonical article '$($article.id)' hides a URL behind Markdown link text."

  $prefaceStart = $content.IndexOf('<!-- public-article-project-preface:start -->', [StringComparison]::Ordinal)
  $prefaceEnd = $content.IndexOf('<!-- public-article-project-preface:end -->', [StringComparison]::Ordinal)
  Assert-Condition ($prefaceEnd -gt $prefaceStart -and ($prefaceEnd - $prefaceStart) -ge 1000) "Project preface is too short in '$($article.sourcePath)'."
  $preface = $content.Substring($prefaceStart, $prefaceEnd - $prefaceStart)
  $prefaceEntries = @(
    @{ Label = '项目主页'; Url = 'https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0' },
    @{ Label = '核心 NuGet'; Url = 'https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0' },
    @{ Label = 'Runtime Bridge 包列表'; Url = 'https://www.nuget.org/packages?q=+JYPPX.TensorRT.CSharp&includeComputedFrameworks=true&prerel=true&sortby=relevance' },
    @{ Label = '运行库清单'; Url = 'https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/pack/runtime/runtime-packages.manifest.json' }
  )
  foreach ($entry in $prefaceEntries) {
    $entryPattern = '(?m)^' + [regex]::Escape($entry.Label) + '：\r?\n\r?\n```text\r?\n' + [regex]::Escape($entry.Url) + '\r?\n```\r?$'
    Assert-Condition ($preface -match $entryPattern) "Project preface in '$($article.sourcePath)' must give '$($entry.Label)' its own text code block."
  }
  Assert-Condition ($content -notmatch [regex]::Escape($retiredBridgePackageListUrl)) "Canonical article '$($article.id)' uses the retired GitHub Packages listing URL."
  Assert-Condition ($content -match [regex]::Escape('https://www.nuget.org/packages?q=+JYPPX.TensorRT.CSharp&includeComputedFrameworks=true&prerel=true&sortby=relevance')) "Canonical article '$($article.id)' is missing the NuGet Runtime Bridge package search URL."

  foreach ($heading in ($headings | Where-Object Level -eq 2)) {
    Assert-Condition ($heading.Text -match '^\d+\.\s+\S') "Unnumbered H2 '$($heading.Text)' in '$($article.sourcePath)'."
  }
  foreach ($heading in ($headings | Where-Object Level -eq 3)) {
    Assert-Condition ($heading.Text -match '^\d+\.\d+\s+\S') "Unnumbered H3 '$($heading.Text)' in '$($article.sourcePath)'."
  }
  foreach ($heading in ($headings | Where-Object Level -eq 4)) {
    Assert-Condition ($heading.Text -match '^\d+\.\d+\.\d+\s+\S') "Unnumbered H4 '$($heading.Text)' in '$($article.sourcePath)'."
  }

  $declarationPhrases = @(
    '作者所有开源项目代码均遵循 Apache License 2.0 开源协议。',
    'AI 辅助开发：本代码在开发过程中使用了人工智能（AI）辅助生成与优化，并非完全由人工逐行编写。',
    '安全性承诺：作者郑重声明',
    '技术局限性：受限于作者个人的技术水平与能力',
    '测试范围：由于作者精力有限',
    '免责声明（重要）',
    '本项目承诺核心逻辑代码完全开源',
    '尽管存在上述不足，我们仍欢迎大家下载使用、提交 Issue 或参与测试'
  )
  foreach ($phrase in $declarationPhrases) {
    Assert-Condition ($content.Contains($phrase)) "Article declaration in '$($article.sourcePath)' is missing '$phrase'."
  }

  Assert-Condition ($content.Contains('https://github.com/guojin-yan/TensorRT-CSharp-API')) "Canonical article '$($article.id)' is missing the project source link."
  Assert-Condition ($content.Contains('https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0')) "Canonical article '$($article.id)' is missing the stable core package link."

  foreach ($match in [regex]::Matches($content, 'https://github\.com/guojin-yan/TensorRT-CSharp-API/(?:blob|tree)/TensorRtSharp4\.0/(?<path>[^<>\s)`"'']+)')) {
    $repositoryPath = [Uri]::UnescapeDataString($match.Groups['path'].Value)
    Assert-Condition (Test-Path -LiteralPath (Resolve-RepositoryPath $repositoryPath)) "Broken repository link '$repositoryPath' in '$($article.sourcePath)'."
  }

  if ($article.module -eq '02-samples') {
    Assert-Condition ($content.Contains('github.com/guojin-yan/TensorRT-CSharp-API') -and $content.Contains('/samples')) "Sample article '$($article.id)' is missing its GitHub sample source link."
  }

  if ($article.module -eq '03-applications') {
    Assert-Condition ($content.Contains('github.com/guojin-yan/TensorRT-CSharp-API') -and $content.Contains('/applications/')) "Application article '$($article.id)' is missing its GitHub application source link."
  }

  if ($null -ne $article.supersedes) {
    Assert-Condition (Test-Path -LiteralPath (Resolve-RepositoryPath $article.supersedes) -PathType Leaf) "Missing superseded source '$($article.supersedes)'."
  }

  if ($article.status -eq "published") {
    Assert-Condition ($article.immutable -eq $true) "Published article '$($article.id)' must be immutable."
    Assert-Condition ([string]$article.csdnUrl -match '^https://blog\.csdn\.net/') "Published article '$($article.id)' needs a CSDN URL."
    Assert-Condition (-not [string]::IsNullOrWhiteSpace($article.publishedAt)) "Published article '$($article.id)' needs publishedAt."
    Assert-Condition ([string]$article.publishedCommit -match '^[0-9a-f]{40}$') "Published article '$($article.id)' needs a commit SHA."
    Assert-Condition ([string]$article.publishedSha256 -match '^[0-9a-f]{64}$') "Published article '$($article.id)' needs a body SHA256."
  }
  else {
    Assert-Condition ($article.immutable -eq $false) "Unpublished article '$($article.id)' cannot be immutable."
    Assert-Condition ($null -eq $article.publishedAt) "Unpublished article '$($article.id)' cannot set publishedAt."
    Assert-Condition ($null -eq $article.csdnUrl) "Unpublished article '$($article.id)' cannot set csdnUrl."
    Assert-Condition ($null -eq $article.publishedCommit) "Unpublished article '$($article.id)' cannot set publishedCommit."
    Assert-Condition ($null -eq $article.publishedSha256) "Unpublished article '$($article.id)' cannot set publishedSha256."
  }

  foreach ($match in [regex]::Matches($content, '!?\[[^\]]*\]\(([^)]+)\)')) {
    $target = $match.Groups[1].Value.Trim().Trim('<', '>')
    if ($target -match '^(https?://|mailto:|#)') {
      continue
    }

    $target = $target.Split('#')[0]
    if ([string]::IsNullOrWhiteSpace($target)) {
      continue
    }

    $resolvedTarget = [IO.Path]::GetFullPath((Join-Path (Split-Path $sourcePath -Parent) $target))
    Assert-Condition (Test-Path -LiteralPath $resolvedTarget) "Broken link '$target' in '$($article.sourcePath)'."
  }

  foreach ($match in [regex]::Matches($content, '<img\b[^>]*\bsrc="([^"]+)"[^>]*>')) {
    $tag = $match.Value
    $target = $match.Groups[1].Value
    Assert-Condition ($tag -match '\bwidth="640"') "Image does not use the 640px public-article width in '$($article.sourcePath)'."
    Assert-Condition ($tag -match 'max-width:100%') "Image is not responsive in '$($article.sourcePath)'."
    if ($target -match '^(https?://|data:)') {
      continue
    }

    $target = $target.Split('#')[0]
    $resolvedTarget = [IO.Path]::GetFullPath((Join-Path (Split-Path $sourcePath -Parent) $target))
    Assert-Condition (Test-Path -LiteralPath $resolvedTarget -PathType Leaf) "Broken image '$target' in '$($article.sourcePath)'."
  }
}

foreach ($requiredId in $requiredIds) {
  Assert-Condition ($seenIds.Contains($requiredId)) "Missing required canonical article '$requiredId'."
}

$toc = Get-Content -LiteralPath (Resolve-RepositoryPath "docs/toc.yml") -Raw -Encoding UTF8
$docsIndex = Get-Content -LiteralPath (Resolve-RepositoryPath "docs/index.md") -Raw -Encoding UTF8
$articleReadme = Get-Content -LiteralPath (Resolve-RepositoryPath "docs/articles/zh-cn/README.md") -Raw -Encoding UTF8
$modules = @("01-release", "02-samples", "03-applications", "04-api", "05-installation", "06-source-build", "07-misc")

foreach ($module in $modules) {
  $modulePath = "articles/zh-cn/$module/README.md"
  Assert-Condition ($toc.Contains("href: $modulePath")) "TOC is missing '$modulePath'."
  Assert-Condition ($docsIndex.Contains("($modulePath)")) "docs/index.md is missing '$modulePath'."
  Assert-Condition ($articleReadme.Contains("($module/README.md)")) "Chinese article README is missing '$module'."

  $moduleRoot = Resolve-RepositoryPath "docs/articles/zh-cn/$module"
  foreach ($markdownFile in Get-ChildItem -LiteralPath $moduleRoot -Recurse -File -Filter '*.md') {
    $relativePath = [IO.Path]::GetRelativePath($RepositoryRoot, $markdownFile.FullName).Replace('\', '/')
    $moduleReadme = "docs/articles/zh-cn/$module/README.md"
    Assert-Condition ($relativePath -eq $moduleReadme -or $seenPaths.Contains($relativePath)) "Public module contains unindexed Markdown '$relativePath'."
  }
}

$writingSpecPath = Resolve-RepositoryPath 'docs/articles/zh-cn/publishing/public-article-writing-spec.md'
Assert-Condition (Test-Path -LiteralPath $writingSpecPath -PathType Leaf) 'Missing public article writing specification.'
Assert-Condition ($articleReadme.Contains('(publishing/public-article-writing-spec.md)')) 'Chinese article README is missing the writing specification.'

$publicEntrypoints = @(
  "README.md",
  "README.zh-CN.md",
  "samples/README.md",
  "samples/README.zh-CN.md",
  "docs/index.md"
)
$publicText = ($publicEntrypoints | ForEach-Object { Get-Content -LiteralPath (Resolve-RepositoryPath $_) -Raw -Encoding UTF8 }) -join [Environment]::NewLine
Assert-Condition ($publicText -notmatch 'dotnet add package[^\r\n]*4\.0\.0-\*') "Public install command still uses the preview wildcard."
Assert-Condition ($publicText -notmatch 'published-preview\.1') "Public entrypoint still uses the preview publication state."
Assert-Condition ($publicText.Contains('JYPPX.TensorRT.CSharp.API --version 4.0.0')) "Public entrypoints do not show the exact stable package version."

$sampleProps = Get-Content -LiteralPath (Resolve-RepositoryPath "build/JYPPX.PublicSamplePackages.props") -Raw -Encoding UTF8
Assert-Condition ($sampleProps.Contains('>4.0.0</JYPPXTensorRtSamplePackageVersion>')) "Samples do not consume the exact stable package version."

Write-Host "Public article index validation passed."
Write-Host "Articles=$($index.articles.Count) Modules=$($modules.Count) Statuses=$($expectedStatuses -join ',')"
