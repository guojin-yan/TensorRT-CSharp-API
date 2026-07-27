[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$RoadmapPath = "docs\articles\zh-cn\technical-article-roadmap.md",
  [string]$OutputRoot = "docs\articles\zh-cn\publishing"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

function Resolve-RootedPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return [System.IO.Path]::GetFullPath($Path)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
}

function ConvertTo-RepositoryPath {
  param([string]$Path)

  $fullPath = [System.IO.Path]::GetFullPath($Path)
  $rootWithSeparator = $RepositoryRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
  if (-not $fullPath.StartsWith($rootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $fullPath.Replace('\', '/')
  }

  return $fullPath.Substring($rootWithSeparator.Length).Replace('\', '/')
}

function Resolve-RepositoryReference {
  param([string]$Reference)

  $trimmed = $Reference.Trim()
  $normalized = $trimmed.Replace('/', '\')
  $candidates = New-Object System.Collections.Generic.List[string]
  $candidates.Add($normalized)
  if (-not $normalized.Contains('\') -and $normalized.EndsWith('.md', [System.StringComparison]::OrdinalIgnoreCase)) {
    $candidates.Add((Join-Path "docs\articles\zh-cn" $normalized))
  }

  foreach ($candidate in $candidates) {
    $fullCandidate = Resolve-RootedPath $candidate
    $containsWildcard = $candidate.IndexOfAny([char[]]'*?') -ge 0
    if ($containsWildcard) {
      $matches = @(Get-ChildItem -Path $fullCandidate -ErrorAction SilentlyContinue)
      if ($matches.Count -gt 0) {
        return [pscustomobject][ordered]@{
          declared = $trimmed
          repositoryPath = $candidate.Replace('\', '/')
          exists = $true
          referenceKind = "pattern"
          matchCount = $matches.Count
        }
      }
    }
    elseif (Test-Path -LiteralPath $fullCandidate) {
      return [pscustomobject][ordered]@{
        declared = $trimmed
        repositoryPath = ConvertTo-RepositoryPath $fullCandidate
        exists = $true
        referenceKind = if (Test-Path -LiteralPath $fullCandidate -PathType Container) { "directory" } else { "file" }
        matchCount = 1
      }
    }
  }

  return [pscustomobject][ordered]@{
    declared = $trimmed
    repositoryPath = $normalized.Replace('\', '/')
    exists = $false
    referenceKind = "unresolved"
    matchCount = 0
  }
}

function Get-MarkdownReferences {
  param([string]$Text)

  return @([System.Text.RegularExpressions.Regex]::Matches($Text, '`([^`]+)`') |
    ForEach-Object { $_.Groups[1].Value } |
    Where-Object {
      $_ -match '^(?:README(?:\.zh-CN)?\.md|(?:applications|artifacts|docs|eng|samples|smoke|src|tests)/)' -or
      ($_ -match '^[A-Za-z0-9_.-]+\.md$')
    } |
    Select-Object -Unique)
}

function Get-ArticleMetrics {
  param([string]$RepositoryPath)

  if ([string]::IsNullOrWhiteSpace($RepositoryPath)) {
    return [pscustomobject][ordered]@{
      exists = $false
      lineCount = 0
      characterCount = 0
      headingCount = 0
      codeBlockCount = 0
      validatorCommands = @()
      codeAndArtifactPaths = @()
    }
  }

  $fullPath = Resolve-RootedPath $RepositoryPath
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    return [pscustomobject][ordered]@{
      exists = $false
      lineCount = 0
      characterCount = 0
      headingCount = 0
      codeBlockCount = 0
      validatorCommands = @()
      codeAndArtifactPaths = @()
    }
  }

  $content = [System.IO.File]::ReadAllText($fullPath)
  $validators = @([System.Text.RegularExpressions.Regex]::Matches($content, 'Test-[A-Za-z0-9-]+\.ps1') |
    ForEach-Object { $_.Value } |
    Select-Object -Unique)
  $paths = @([System.Text.RegularExpressions.Regex]::Matches(
      $content,
      '`((?:applications|artifacts|docs|eng|samples|smoke|src|tests)/[^`]+)`') |
    ForEach-Object { $_.Groups[1].Value } |
    Select-Object -Unique)

  return [pscustomobject][ordered]@{
    exists = $true
    lineCount = ($content -split "`r?`n").Count
    characterCount = $content.Length
    headingCount = [System.Text.RegularExpressions.Regex]::Matches($content, '(?m)^#{1,6} ').Count
    codeBlockCount = [int]([System.Text.RegularExpressions.Regex]::Matches($content, '(?m)^```').Count / 2)
    validatorCommands = $validators
    codeAndArtifactPaths = $paths
  }
}

function Get-ProofDependencies {
  param([string]$Text)

  $dependencies = New-Object System.Collections.Generic.List[string]
  if ($Text -match '(?i)post[ -]publish') { $dependencies.Add('post-publish-verification') }
  if ($Text -match '(?i)package-consumer-runtime|package consumer runtime') { $dependencies.Add('package-consumer-runtime') }
  if ($Text -match '(?i)real callback runtime') { $dependencies.Add('real-callback-runtime') }
  if ($Text -match '(?i)Linux runner') { $dependencies.Add('linux-runner-proof') }
  if ($Text -match '(?i)real-model-runtime|真实模型|模型证据链|真实资产|sample-run-evidence|owner 提供真实模型') {
    $dependencies.Add('real-model-runtime')
  }
  elseif ($Text -match '(?i)用户自备.*(?:模型|ONNX|资产)|需.*(?:模型|ONNX|图片|资产)') {
    $dependencies.Add('external-model-asset')
  }
  if ($Text -match '(?i)owner authorization') { $dependencies.Add('owner-authorization') }
  if ($dependencies.Count -eq 0) { $dependencies.Add('source-quality-only') }
  return @($dependencies | Select-Object -Unique)
}

function Get-ProofState {
  param([string[]]$Dependencies)

  $requiredDependencies = @($Dependencies | Where-Object { $_ -ne 'source-quality-only' })
  if ($requiredDependencies.Count -gt 1) { return 'multiple-owner-or-runtime-proofs-required' }
  if ($Dependencies -contains 'post-publish-verification') { return 'post-publish-owner-proof-required' }
  if ($Dependencies -contains 'real-callback-runtime') { return 'callback-runtime-proof-required' }
  if ($Dependencies -contains 'linux-runner-proof') { return 'linux-runner-proof-required' }
  if ($Dependencies -contains 'package-consumer-runtime') { return 'package-consumer-owner-proof-required' }
  if ($Dependencies -contains 'real-model-runtime') { return 'real-model-owner-assets-required' }
  if ($Dependencies -contains 'external-model-asset') { return 'external-model-asset-required-not-runtime-proof' }
  if ($Dependencies -contains 'owner-authorization') { return 'owner-authorization-required' }
  return 'not-required-for-content-closure'
}

function Get-ContentState {
  param(
    [bool]$CanonicalCovered,
    [int]$CharacterCount,
    [string]$Series
  )

  if ($CanonicalCovered) { return 'canonical-covered' }
  if ($CharacterCount -ge 12000) { return 'complete-long-form' }
  if ($CharacterCount -ge 4500) { return 'complete-article' }
  if ($CharacterCount -ge 2500 -and $Series -match '发布|Owner|README|审计|索引|最终|边界') {
    return 'complete-operational-guide'
  }
  if ($CharacterCount -gt 0) { return 'needs-expansion' }
  return 'planned-no-canonical-body'
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return '' }
  return ([string]$Value).Replace('|', '\|').Replace("`r", ' ').Replace("`n", ' ')
}

$roadmapFullPath = Resolve-RootedPath $RoadmapPath
$outputFullPath = Resolve-RootedPath $OutputRoot
if (-not (Test-Path -LiteralPath $roadmapFullPath -PathType Leaf)) {
  throw "Roadmap not found: $roadmapFullPath"
}
New-Item -ItemType Directory -Force -Path $outputFullPath | Out-Null

$canonicalMappings = @{
  '25' = [ordered]@{ articleId = 74; path = 'docs/articles/zh-cn/yolovision-detection-tutorial.md'; relatedArticleIds = @(74); reason = 'The later YoloVision detection tutorial is the canonical deployment body.' }
  '26' = [ordered]@{ articleId = 74; path = 'docs/articles/zh-cn/yolovision-detection-tutorial.md'; relatedArticleIds = @(74); reason = 'The later YoloVision detection tutorial owns output-layout troubleshooting.' }
  '34' = [ordered]@{ articleId = 79; path = 'docs/articles/zh-cn/callback-allocator-safety-bridge-roadmap.md'; relatedArticleIds = @(79); reason = 'The callback and allocator safety roadmap is the canonical OutputAllocator/DebugListener boundary body.' }
  '63' = [ordered]@{ articleId = 81; path = 'docs/articles/zh-cn/project-release-story-and-boundaries.md'; relatedArticleIds = @(81); reason = 'The later project capability and release-boundary story is the canonical long-form body.' }
  '64' = [ordered]@{ articleId = 72; path = 'docs/articles/zh-cn/tensorrtexec-option-layering-deep-dive.md'; relatedArticleIds = @(72); reason = 'The later option-layering deep dive is the canonical long-form body.' }
  '65' = [ordered]@{ articleId = 73; path = 'docs/articles/zh-cn/yolovision-all-task-overview.md'; relatedArticleIds = @(73, 74, 75, 76, 77, 78); reason = 'The all-task overview and six task tutorials form the canonical YoloVision series.' }
  '66' = [ordered]@{ articleId = 69; path = 'docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md'; relatedArticleIds = @(69); reason = 'The package consumer runtime proof playbook is the canonical execution article.' }
  '67' = [ordered]@{ articleId = 70; path = 'docs/articles/zh-cn/post-publish-verification-proof-playbook.md'; relatedArticleIds = @(70); reason = 'The post-publish proof playbook is the canonical execution article.' }
  '68' = [ordered]@{ articleId = 79; path = 'docs/articles/zh-cn/callback-allocator-safety-bridge-roadmap.md'; relatedArticleIds = @(79); reason = 'The later callback and allocator safety roadmap is the canonical boundary article.' }
  '71' = [ordered]@{ articleId = 80; path = 'docs/articles/zh-cn/external-model-evidence-case-study.md'; relatedArticleIds = @(80); reason = 'The external-model evidence case study is the canonical long-form owner backfill tutorial.' }
}
$preferredCanonicalPaths = @{
  '69' = 'docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md'
  '70' = 'docs/articles/zh-cn/post-publish-verification-proof-playbook.md'
  '79' = 'docs/articles/zh-cn/callback-allocator-safety-bridge-roadmap.md'
}

$mainPattern = '^\|\s*(?<id>\d+)\s*\|\s*(?<series>.*?)\s*\|\s*(?<title>.*?)\s*\|\s*(?<content>.*?)\s*\|\s*(?<evidence>.*?)\s*\|\s*(?<assets>.*?)\s*\|\s*(?<status>.*?)\s*\|$'
$supplementalPattern = '^\|\s*(?<id>\d+\.\d+)\s*\|\s*(?<series>.*?)\s*\|\s*(?<title>.*?)\s*\|'
$entries = New-Object System.Collections.Generic.List[object]
$supplementalEntries = New-Object System.Collections.Generic.List[object]

foreach ($line in [System.IO.File]::ReadAllLines($roadmapFullPath)) {
  $mainMatch = [System.Text.RegularExpressions.Regex]::Match($line, $mainPattern)
  if (-not $mainMatch.Success) {
    $supplementalMatch = [System.Text.RegularExpressions.Regex]::Match($line, $supplementalPattern)
    if ($supplementalMatch.Success) {
      $supplementalEntries.Add([pscustomobject][ordered]@{
        articleId = $supplementalMatch.Groups['id'].Value
        series = $supplementalMatch.Groups['series'].Value.Trim()
        title = $supplementalMatch.Groups['title'].Value.Trim()
      })
    }
    continue
  }

  $articleId = [int]$mainMatch.Groups['id'].Value
  $series = $mainMatch.Groups['series'].Value.Trim()
  $title = $mainMatch.Groups['title'].Value.Trim()
  $content = $mainMatch.Groups['content'].Value.Trim()
  $evidence = $mainMatch.Groups['evidence'].Value.Trim()
  $assets = $mainMatch.Groups['assets'].Value.Trim()
  $sourceStatus = $mainMatch.Groups['status'].Value.Trim()
  $combined = "$title $content $evidence $assets $sourceStatus"

  $resolvedReferences = @(Get-MarkdownReferences $evidence | ForEach-Object { Resolve-RepositoryReference $_ })
  $mapping = $canonicalMappings[[string]$articleId]
  $canonicalCovered = $null -ne $mapping
  $canonicalArticleId = if ($canonicalCovered) { [int]$mapping.articleId } else { $articleId }
  $canonicalPath = if ($canonicalCovered) {
    [string]$mapping.path
  }
  elseif ($preferredCanonicalPaths.ContainsKey([string]$articleId)) {
    [string]$preferredCanonicalPaths[[string]$articleId]
  }
  else {
    ''
  }

  if ([string]::IsNullOrWhiteSpace($canonicalPath)) {
    $articleCandidates = @($resolvedReferences |
      Where-Object {
        $_.exists -and $_.referenceKind -eq 'file' -and
        $_.repositoryPath -like 'docs/articles/zh-cn/*.md'
      } |
      ForEach-Object {
        $metrics = Get-ArticleMetrics $_.repositoryPath
        [pscustomobject]@{ path = $_.repositoryPath; characters = $metrics.characterCount }
      } |
      Sort-Object characters -Descending)
    if ($articleCandidates.Count -gt 0) {
      $canonicalPath = [string]$articleCandidates[0].path
    }
  }

  $metrics = Get-ArticleMetrics $canonicalPath
  $proofDependencies = @(Get-ProofDependencies $combined)
  $proofState = Get-ProofState $proofDependencies
  $proofRequired = $proofState -notin @('not-required-for-content-closure', 'external-model-asset-required-not-runtime-proof')
  $contentState = Get-ContentState $canonicalCovered $metrics.characterCount $series

  $entries.Add([pscustomobject][ordered]@{
    articleId = $articleId
    series = $series
    title = $title
    sourceStatus = $sourceStatus
    contentState = $contentState
    contentComplete = $contentState -in @('canonical-covered', 'complete-long-form', 'complete-article', 'complete-operational-guide')
    proofState = $proofState
    proofRequired = $proofRequired
    proofComplete = -not $proofRequired
    externalAssetRequired = $proofDependencies -contains 'external-model-asset'
    proofDependencies = $proofDependencies
    canonicalArticleId = $canonicalArticleId
    canonicalArticlePath = $canonicalPath
    canonicalArticleExists = [bool]$metrics.exists
    canonicalLineCount = [int]$metrics.lineCount
    canonicalCharacterCount = [int]$metrics.characterCount
    canonicalHeadingCount = [int]$metrics.headingCount
    canonicalCodeBlockCount = [int]$metrics.codeBlockCount
    validatorCommands = @($metrics.validatorCommands)
    codeAndArtifactPaths = @($metrics.codeAndArtifactPaths)
    declaredReferences = $resolvedReferences
    canonicalCoverageReason = if ($canonicalCovered) { [string]$mapping.reason } else { 'This roadmap entry owns its selected canonical body.' }
    relatedArticleIds = if ($canonicalCovered) { @($mapping.relatedArticleIds) } else { @($articleId) }
    assetRequirement = $assets
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  })
}

$articleArray = @($entries | Sort-Object articleId)
$expectedIds = @(1..103)
$actualIds = @($articleArray | ForEach-Object { $_.articleId })
$missingIds = @($expectedIds | Where-Object { $_ -notin $actualIds })
$duplicateIds = @($actualIds | Group-Object | Where-Object Count -gt 1 | ForEach-Object { [int]$_.Name })
$targetArticles = @($articleArray | Where-Object articleId -in @(69, 70, 79))
$forbiddenMarkers = @('canPublishPublicly=true', 'canCloseReleaseIssue=true', 'performsPublish=true', 'YoloDet')
$targetForbiddenFindings = New-Object System.Collections.Generic.List[object]
foreach ($article in $targetArticles) {
  $articleText = if ($article.canonicalArticleExists) { [System.IO.File]::ReadAllText((Resolve-RootedPath $article.canonicalArticlePath)) } else { '' }
  foreach ($marker in $forbiddenMarkers) {
    if ($articleText.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
      $targetForbiddenFindings.Add([pscustomobject][ordered]@{
        articleId = $article.articleId
        path = $article.canonicalArticlePath
        marker = $marker
      })
    }
  }
}

$record = [ordered]@{
  schemaVersion = 1
  recordKind = 'technical-article-closure-ledger'
  ledgerState = if ($missingIds.Count -eq 0 -and $duplicateIds.Count -eq 0 -and $targetForbiddenFindings.Count -eq 0) { 'content-closure-audited-release-frozen' } else { 'invalid-closure-ledger' }
  sourceRoadmap = ConvertTo-RepositoryPath $roadmapFullPath
  sourceRoadmapSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $roadmapFullPath).Hash.ToLowerInvariant()
  articleCount = $articleArray.Count
  supplementalArticleCount = $supplementalEntries.Count
  expectedArticleCount = 103
  missingArticleIds = $missingIds
  duplicateArticleIds = $duplicateIds
  canonicalMappingCount = $canonicalMappings.Count
  contentCompleteCount = @($articleArray | Where-Object contentComplete).Count
  completeLongFormCount = @($articleArray | Where-Object contentState -eq 'complete-long-form').Count
  canonicalCoveredCount = @($articleArray | Where-Object contentState -eq 'canonical-covered').Count
  needsExpansionCount = @($articleArray | Where-Object contentState -in @('needs-expansion', 'planned-no-canonical-body')).Count
  externalDependencyRequiredCount = @($articleArray | Where-Object { $_.externalAssetRequired -or $_.proofRequired }).Count
  ownerOrRuntimeProofRequiredCount = @($articleArray | Where-Object proofRequired).Count
  targetLongFormArticleIds = @(69, 70, 79)
  targetForbiddenMarkerCount = $targetForbiddenFindings.Count
  targetForbiddenFindings = @($targetForbiddenFindings.ToArray())
  contentAndProofStateAreIndependent = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  proofBoundary = 'Content closure, canonical mapping, article length, code-path references, and validator links are documentation evidence only. They are not runtime proof, not post-publish proof, not publish approval, not package push, and not release close approval.'
  supplementalArticles = @($supplementalEntries.ToArray())
  articles = $articleArray
}

$jsonPath = Join-Path $outputFullPath 'technical-article-closure-ledger.json'
$markdownPath = Join-Path $outputFullPath 'technical-article-closure-ledger.md'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$json = $record | ConvertTo-Json -Depth 16
[System.IO.File]::WriteAllText($jsonPath, $json + [Environment]::NewLine, $utf8)

$rows = @($articleArray | ForEach-Object {
  $canonical = if ([string]::IsNullOrWhiteSpace($_.canonicalArticlePath)) { 'none' } else { $_.canonicalArticlePath }
  $dependencies = ($_.proofDependencies -join ', ')
  "| $($_.articleId) | $(ConvertTo-MarkdownCell $_.series) | $(ConvertTo-MarkdownCell $_.title) | ``$($_.contentState)`` | ``$($_.proofState)`` | $($_.canonicalArticleId) | ``$canonical`` | $($_.canonicalCharacterCount) | $(ConvertTo-MarkdownCell $dependencies) |"
})

$markdown = @"
# Technical Article Closure Ledger

## Summary

- record kind: ``$($record.recordKind)``
- ledger state: ``$($record.ledgerState)``
- source roadmap: ``$($record.sourceRoadmap)``
- source roadmap SHA256: ``$($record.sourceRoadmapSha256)``
- article count: ``$($record.articleCount)``
- supplemental article count: ``$($record.supplementalArticleCount)``
- content complete count: ``$($record.contentCompleteCount)``
- complete long-form count: ``$($record.completeLongFormCount)``
- canonical covered count: ``$($record.canonicalCoveredCount)``
- needs expansion count: ``$($record.needsExpansionCount)``
- owner/runtime proof required count: ``$($record.ownerOrRuntimeProofRequiredCount)``
- external dependency required count: ``$($record.externalDependencyRequiredCount)``
- target forbidden marker count: ``$($record.targetForbiddenMarkerCount)``
- performsPublish=false
- canPublishPublicly=false
- canCloseReleaseIssue=false

## State Model

``contentState`` 只回答文章正文是否完整或是否由后续 canonical 文章覆盖；``proofState`` 只回答真实 runtime、owner 或发布后证明是否仍待外部输入。两者互不替代。

## Proof Boundary

$($record.proofBoundary)

## Articles

| ID | Series | Title | Content state | Proof state | Canonical ID | Canonical article | Characters | Proof dependencies |
|---:|---|---|---|---|---:|---|---:|---|
$($rows -join [Environment]::NewLine)
"@
[System.IO.File]::WriteAllText($markdownPath, $markdown + [Environment]::NewLine, $utf8)

Write-Host "Technical article closure ledger written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ArticleCount=$($record.articleCount) Supplemental=$($record.supplementalArticleCount) ContentComplete=$($record.contentCompleteCount) NeedsExpansion=$($record.needsExpansionCount)"
Write-Host "PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"
