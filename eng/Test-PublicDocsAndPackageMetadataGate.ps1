[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
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

function ConvertTo-RelativePath {
  param([string]$Path)

  return $Path.Substring($RepositoryRoot.Length).TrimStart('\', '/')
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-GateItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Test-AllowedBoundaryContext {
  param([string]$Line)

  $lower = $Line.ToLowerInvariant()
  foreach ($marker in @(
      "not proof",
      "not runtime proof",
      "not package-consumer",
      "not post-publish",
      "does not",
      "cannot",
      "must not",
      "no ",
      "owner-action-required",
      "action-required",
      "expected evidence",
      "blocked",
      "requires",
      "required",
      "forbidden",
      "non-proof",
      "不是",
      "不能",
      "不得",
      "不可",
      "不会",
      "不要",
      "不代表",
      "不使用",
      "不把",
      "不允许",
      "无 ",
      "避免",
      "禁止",
      "需要",
      "必须",
      "阻断",
      "边界",
      "占位",
      "模板",
      "误写成",
      "当前不能说",
      "不能说",
      "non-substitute",
      "non-promotable",
      "free package proof",
      "when needed",
      "not suitable"
    )) {
    if ($lower.Contains($marker)) {
      return $true
    }
  }

  return $false
}

$scanRoots = @(
  "README.md",
  "README.zh-CN.md",
  "docs",
  "samples",
  "applications",
  "src",
  "pack",
  ".github"
)

$files = New-Object System.Collections.Generic.List[string]
foreach ($root in $scanRoots) {
  $path = Join-Path $RepositoryRoot $root
  if (-not (Test-Path -LiteralPath $path)) {
    continue
  }

  if (Test-Path -LiteralPath $path -PathType Leaf) {
    $files.Add((Resolve-Path -LiteralPath $path).Path) | Out-Null
    continue
  }

  Get-ChildItem -LiteralPath $path -Recurse -File |
    Where-Object {
      $relative = (ConvertTo-RelativePath -Path $_.FullName).Replace('/', '\')
      $relative -notmatch "(^|\\)(bin|obj)(\\|$)" -and
      $relative -notlike "docs\_site\*" -and
      $_.Extension -in @(".md", ".yml", ".yaml", ".json", ".props", ".targets", ".csproj", ".cs", ".nuspec", ".ps1", ".xml", ".txt")
    } |
    ForEach-Object { $files.Add($_.FullName) | Out-Null }
}

$rules = @(
  [pscustomobject]@{ id = "yolodet-live-name"; pattern = "\bYoloDet\b"; description = "YoloDet must not return to live docs/source/samples/applications." }
  [pscustomobject]@{ id = "tensorrt-layer-tensor-info-stale"; pattern = "TensorRtLayerTensorInfo"; description = "TensorRtLayerTensorInfo stale claim must not return to live docs/source/samples/applications." }
  [pscustomobject]@{ id = "published-to-nuget"; pattern = "published to NuGet|NuGet published|已经发布到\s*nuget|已发布到\s*nuget|发布到\s*NuGet"; description = "Public package publication must not be claimed before real owner channel proof." }
  [pscustomobject]@{ id = "post-publish-verified"; pattern = "post-publish verified|post-publish 已验证|post-publish verification passed|post-publish verification 已通过"; description = "Post-publish verification must not be claimed before real public channel install/run evidence." }
  [pscustomobject]@{ id = "package-consumer-runtime-passed"; pattern = "package-consumer-runtime passed|package-consumer-runtime 已通过|package consumer runtime proof complete|package-consumer runtime proof complete"; description = "Package-consumer runtime proof must not be claimed from template, local feed, ProjectReference, or direct nupkg evidence." }
  [pscustomobject]@{ id = "release-ready-to-publish"; pattern = "ready to publish|ready-to-publish|可以发布|发布就绪"; description = "Release readiness must not be claimed before real owner/public/post-publish proof gates pass." }
  [pscustomobject]@{ id = "build-only-as-proof"; pattern = "build-only means release proof|build-only 是 proof|build-only 表示 proof|build-only 作为 proof"; description = "Build-only evidence is not release proof." }
  [pscustomobject]@{ id = "dry-run-as-proof"; pattern = "dry-run means proof|dry-run 是 proof|dry-run 表示 proof|dry-run 作为 proof"; description = "Dry-run evidence is not release proof." }
  [pscustomobject]@{ id = "template-as-proof"; pattern = "template means proof|template 是 proof|template 表示 proof|模板.*proof|模板.*证明"; description = "Templates are not proof." }
  [pscustomobject]@{ id = "local-feed-as-proof"; pattern = "local feed.*proof|local feed.*证明|本地源.*proof|本地源.*证明"; description = "Local feeds are forbidden substitutes." }
  [pscustomobject]@{ id = "project-reference-as-proof"; pattern = "ProjectReference.*proof|ProjectReference.*证明"; description = "ProjectReference is a forbidden substitute." }
  [pscustomobject]@{ id = "direct-nupkg-as-proof"; pattern = "direct \.nupkg.*proof|direct \.nupkg.*证明|直接.*\.nupkg.*proof|直接.*\.nupkg.*证明"; description = "Direct nupkg references are forbidden substitutes." }
)

$claimMatches = New-Object System.Collections.Generic.List[object]
$blockedMatches = New-Object System.Collections.Generic.List[object]
$allowedBoundaryMatches = New-Object System.Collections.Generic.List[object]

foreach ($file in @($files | Sort-Object -Unique)) {
  $relative = ConvertTo-RelativePath -Path $file
  $lines = Get-Content -LiteralPath $file -Encoding utf8
  for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = [string]$lines[$i]
    foreach ($rule in $rules) {
      if ($line -match $rule.pattern) {
        $allowed = Test-AllowedBoundaryContext -Line $line
        $record = [pscustomobject]@{
          ruleId = $rule.id
          file = $relative
          line = $i + 1
          allowedBoundaryContext = $allowed
          text = $line.Trim()
        }

        $claimMatches.Add($record) | Out-Null
        if ($allowed) {
          $allowedBoundaryMatches.Add($record) | Out-Null
        }
        else {
          $blockedMatches.Add($record) | Out-Null
        }
      }
    }
  }
}

$boundaryText = ""
foreach ($path in @("README.md", "README.zh-CN.md", "docs")) {
  $candidate = Join-Path $RepositoryRoot $path
  if (-not (Test-Path -LiteralPath $candidate)) {
    continue
  }

  if (Test-Path -LiteralPath $candidate -PathType Leaf) {
    $boundaryText += "`n" + (Get-Content -LiteralPath $candidate -Raw -Encoding utf8)
  }
  else {
    Get-ChildItem -LiteralPath $candidate -Recurse -File -Include *.md |
      Where-Object { (ConvertTo-RelativePath -Path $_.FullName) -notlike "docs\_site\*" } |
      Select-Object -First 20 |
      ForEach-Object { $boundaryText += "`n" + (Get-Content -LiteralPath $_.FullName -Raw -Encoding utf8) }
  }
}

$boundaryLower = $boundaryText.ToLowerInvariant()
$requiredBoundaryMarkers = @("owner", "post-publish", "package-consumer", "not proof")
$missingBoundaryMarkers = @($requiredBoundaryMarkers | Where-Object { -not $boundaryLower.Contains($_) })

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-GateItem "no-yolodet-live" (@($blockedMatches | Where-Object { $_.ruleId -eq "yolodet-live-name" }).Count -eq 0) "blocker" "YoloDet must not appear in live docs/source/samples/applications unless it is clearly a regression guard.")) | Out-Null
$items.Add((New-GateItem "no-tensorrt-layer-tensor-info-live" (@($blockedMatches | Where-Object { $_.ruleId -eq "tensorrt-layer-tensor-info-stale" }).Count -eq 0) "blocker" "TensorRtLayerTensorInfo stale claim must not appear in live docs/source/samples/applications.")) | Out-Null
$items.Add((New-GateItem "no-publication-overclaim" (@($blockedMatches | Where-Object { $_.ruleId -in @("published-to-nuget", "release-ready-to-publish") }).Count -eq 0) "blocker" "Docs/package metadata must not claim real publication or ready-to-publish state before owner proof.")) | Out-Null
$items.Add((New-GateItem "no-runtime-proof-overclaim" (@($blockedMatches | Where-Object { $_.ruleId -in @("post-publish-verified", "package-consumer-runtime-passed") }).Count -eq 0) "blocker" "Docs/package metadata must not claim post-publish or package-consumer runtime proof before real evidence.")) | Out-Null
$items.Add((New-GateItem "no-forbidden-substitute-as-proof" (@($blockedMatches | Where-Object { $_.ruleId -in @("build-only-as-proof", "dry-run-as-proof", "template-as-proof", "local-feed-as-proof", "project-reference-as-proof", "direct-nupkg-as-proof") }).Count -eq 0) "blocker" "Docs/package metadata must not describe forbidden substitutes as proof.")) | Out-Null
$items.Add((New-GateItem "public-boundary-markers-present" ($missingBoundaryMarkers.Count -eq 0) "blocker" "Public docs should contain owner, post-publish, package-consumer, and not-proof boundary markers.")) | Out-Null

$failedBlockerCount = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" }).Count
$gateState = if ($failedBlockerCount -eq 0) { "blocked-owner-public-postpublish-proof-required" } else { "failed-public-docs-package-metadata-gate" }

$report = [ordered]@{
  recordKind = "public-docs-package-metadata-gate"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  repositoryRoot = $RepositoryRoot
  gateState = $gateState
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  failedBlockerCount = [int]$failedBlockerCount
  scannedPathCount = @($files | Sort-Object -Unique).Count
  matchCount = $claimMatches.Count
  allowedBoundaryMatchCount = $allowedBoundaryMatches.Count
  blockedMatchCount = $blockedMatches.Count
  blockedMatches = @($blockedMatches.ToArray())
  allowedBoundaryMatches = @($allowedBoundaryMatches.ToArray())
  validationItems = @($items.ToArray())
  scanRoots = @($scanRoots)
  rules = @($rules)
  requiredBoundaryMarkers = @($requiredBoundaryMarkers)
  missingBoundaryMarkers = @($missingBoundaryMarkers)
  boundary = "Public docs and package metadata gate is a non-publishing claim-safety gate. It does not promote proof, does not close release issues, and only prevents stale or over-claimed public release statements."
}

$jsonPath = Join-Path $OutputRoot "public-docs-package-metadata-gate.json"
$markdownPath = Join-Path $OutputRoot "public-docs-package-metadata-gate.md"
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$itemRows = foreach ($item in $items) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$blockedRows = foreach ($match in $blockedMatches) {
  "| ``$(ConvertTo-MarkdownCell $match.ruleId)`` | ``$(ConvertTo-MarkdownCell $match.file)`` | ``$($match.line)`` | $(ConvertTo-MarkdownCell $match.text) |"
}

$markdown = @"
# Public Docs And Package Metadata Gate

Generated at: ``$($report.generatedAtUtc)``

## Summary

- recordKind: ``$($report.recordKind)``
- gateState: ``$($report.gateState)``
- failedBlockerCount: ``$($report.failedBlockerCount)``
- scannedPathCount: ``$($report.scannedPathCount)``
- matchCount: ``$($report.matchCount)``
- allowedBoundaryMatchCount: ``$($report.allowedBoundaryMatchCount)``
- blockedMatchCount: ``$($report.blockedMatchCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromoteRuntimeProof: ``False``

## Validation Items

| Id | Passed | Severity | Detail |
| --- | --- | --- | --- |
$($itemRows -join "`r`n")

## Blocked Matches

| Rule | File | Line | Text |
| --- | --- | ---: | --- |
$(if ($blockedMatches.Count -eq 0) { "| none |  |  |  |" } else { $blockedRows -join "`r`n" })

## Boundary

$($report.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Public docs package metadata gate written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "GateState=$($report.gateState) FailedBlockers=$($report.failedBlockerCount) ScannedPaths=$($report.scannedPathCount) BlockedMatches=$($report.blockedMatchCount)"

if ($Strict.IsPresent -and $failedBlockerCount -gt 0) {
  exit 1
}
