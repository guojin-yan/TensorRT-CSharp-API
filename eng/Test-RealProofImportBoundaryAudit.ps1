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

function ConvertTo-FlatStringLines {
  param([AllowNull()][object]$Value)

  foreach ($item in @($Value)) {
    if ($null -eq $item) {
      ""
    }
    elseif ($item -is [string]) {
      $item
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { [string]$child }
    }
    else {
      [string]$item
    }
  }
}
function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $lines = @(ConvertTo-FlatStringLines -Value $InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$scanRoots = @(
  "artifacts\final-release",
  "README.md",
  "README.zh-CN.md",
  "docs",
  "samples",
  "applications",
  "src"
)

$excludedPathFragments = @(
  "/docs/_site/",
  "/docs/articles/zh-cn/publishing/article-roadmap-30plus.json",
  "/docs/articles/zh-cn/publishing/article-roadmap-30plus.md",
  "/artifacts/final-release/article-roadmap-30plus-validation.json",
  "/artifacts/final-release/article-roadmap-30plus-validation.md",
  "/artifacts/final-release/real-proof-import-boundary-audit.json",
  "/artifacts/final-release/real-proof-import-boundary-audit.md",
  "/artifacts/final-release/public-article-draft-boundary-scan.json",
  "/artifacts/final-release/public-article-draft-boundary-scan.md",
  "/artifacts/final-release/public-docs-package-metadata-gate.json",
  "/artifacts/final-release/public-docs-package-metadata-gate.md"
)

$files = New-Object System.Collections.Generic.List[string]
foreach ($root in $scanRoots) {
  $path = Join-Path $RepositoryRoot $root
  if (Test-Path -LiteralPath $path -PathType Leaf) {
    $files.Add($path) | Out-Null
    continue
  }

  if (Test-Path -LiteralPath $path -PathType Container) {
    Get-ChildItem -LiteralPath $path -Recurse -File -Include *.json,*.md,*.yml,*.yaml,*.cs,*.ps1,*.csproj | ForEach-Object {
      $files.Add($_.FullName) | Out-Null
    }
  }
}

$blockedClaimPatterns = @(
  [pscustomobject]@{ id = "local-feed-public-package-proof"; regex = "(?i)local feed.{0,80}(is|as|=|作为|等于).{0,80}(public package proof|公开包 proof|发布 proof)" },
  [pscustomobject]@{ id = "project-reference-package-consumer-proof"; regex = "(?i)(ProjectReference|direct nupkg|direct \\.nupkg).{0,80}(is|as|=|作为|等于).{0,80}(package-consumer-runtime proof|runtime proof|运行 proof)" },
  [pscustomobject]@{ id = "build-only-runtime-proof"; regex = "(?i)(build-only|dry-run|parse-only|sidecar-only).{0,80}(is|as|=|作为|等于).{0,80}(runtime proof|runtime execution proof|运行 proof)" },
  [pscustomobject]@{ id = "dashboard-owner-approval"; regex = "(?i)(dashboard|runbook|candidate|draft).{0,80}(is|as|=|作为|等于).{0,80}(Owner approval|publish approval|发布批准|owner authorization)" },
  [pscustomobject]@{ id = "failed-blocker-count-ready"; regex = "(?i)failedBlockerCount\\s*=\\s*0.{0,80}(ready|proof ready|publish ready|可发布|完成)" },
  [pscustomobject]@{ id = "quality-freeze-publish-approval"; regex = "(?i)final quality freeze dashboard.{0,80}(is|as|=|作为|等于).{0,80}(publish approval|publication approval|发布批准)" },
  [pscustomobject]@{ id = "article-roadmap-publish-ready-proof"; regex = "(?i)article roadmap.{0,80}(is|as|=|作为|等于).{0,80}(publish-ready proof|release proof|发布 proof)" }
)

$allowedNegationMarkers = @(
  "not",
  "is not",
  "are not",
  "cannot",
  "must not",
  "do not",
  "不是",
  "都不是",
  "并不是",
  "不要",
  "不要让",
  "避免",
  "防止",
  "不会",
  "不会被",
  "不能",
  "不能晋级",
  "不能提升",
  "不可",
  "不可晋级",
  "不可提升",
  "不得",
  "不等于",
  "不应",
  "不可替代",
  "不能替代",
  "误判",
  "误晋级",
  "误提升",
  "被误晋级",
  "被误提升",
  "当作",
  "写成",
  "替代",
  "标注为非",
  "严格标注为非",
  "仍需被严格标注",
  "禁止",
  "非 proof",
  "非 runtime proof",
  "非 Runtime proof",
  "非 package consumer proof",
  "非 post-publish proof",
  "forbidden",
  "mustAvoidClaims",
  "must avoid",
  "avoid claims",
  "blocked",
  "non-proof",
  "non proof",
  "not proof",
  "is not proof",
  "cannot substitute",
  "cannot replace"
)

$findings = New-Object System.Collections.Generic.List[object]
$scannedFileCount = 0
foreach ($file in $files | Sort-Object -Unique) {
  $relative = [System.IO.Path]::GetRelativePath($RepositoryRoot, $file).Replace("\", "/")
  $relativeForMatch = "/$relative"
  $excluded = $false
  foreach ($fragment in $excludedPathFragments) {
    if ($relativeForMatch.IndexOf($fragment, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
      $excluded = $true
      break
    }
  }

  if ($excluded) {
    continue
  }

  $text = Get-Content -LiteralPath $file -Raw -Encoding utf8 -ErrorAction SilentlyContinue
  if ([string]::IsNullOrWhiteSpace($text)) {
    continue
  }

  $scannedFileCount++
  foreach ($pattern in $blockedClaimPatterns) {
    $matches = [regex]::Matches($text, $pattern.regex)
    foreach ($match in $matches) {
      $snippetStart = [Math]::Max(0, $match.Index - 120)
      $snippetLength = [Math]::Min($text.Length - $snippetStart, $match.Length + 240)
      $snippet = $text.Substring($snippetStart, $snippetLength)
      $isNegated = $false
      foreach ($marker in $allowedNegationMarkers) {
        if ($snippet.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
          $isNegated = $true
          break
        }
      }

      $lineStart = $text.LastIndexOf("`n", [Math]::Max(0, $match.Index - 1))
      $lineEnd = $text.IndexOf("`n", $match.Index)
      if ($lineStart -lt 0) { $lineStart = 0 } else { $lineStart++ }
      if ($lineEnd -lt 0) { $lineEnd = $text.Length }
      $line = $text.Substring($lineStart, $lineEnd - $lineStart)
      if ($line.IndexOf("not", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("cannot", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("do not", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("不是", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("都不是", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("不要", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("避免", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("防止", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("不会", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("不能", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("不可", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("误判", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("误晋级", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("误提升", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("标注为非", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("非 runtime proof", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("非 proof", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("当作", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("写成", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("与真实", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("要求", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("owner-facing", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("最后一层", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("stale", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("audit", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("审计", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("风险", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $line.IndexOf("|", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
        $isNegated = $true
      }

      if (-not $isNegated) {
        $findings.Add([pscustomobject]@{
            id = $pattern.id
            severity = "blocker"
            file = $relative
            match = $match.Value
            message = "Potential real proof import boundary violation: forbidden substitute may be described as proof or approval."
          }) | Out-Null
      }
    }
  }
}

$findingCount = $findings.Count
$auditState = if ($findingCount -eq 0) { "real-proof-import-boundary-audit-passed" } else { "real-proof-import-boundary-audit-blocked" }

$record = [ordered]@{
  recordKind = "real-proof-import-boundary-audit"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  auditState = $auditState
  scannedFileCount = $scannedFileCount
  scannedRoots = @($scanRoots)
  findingCount = $findingCount
  blockedFindingCount = $findingCount
  findings = @($findings.ToArray())
  blockedClaimPatterns = @($blockedClaimPatterns)
  performsPublish = $false
  notExecutedByAutomation = $true
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Real proof import boundary audit scans public and final-release surfaces for forbidden substitute proof claims. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "real-proof-import-boundary-audit.json"
$markdownPath = Join-Path $OutputRoot "real-proof-import-boundary-audit.md"
$record | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = if ($findings.Count -eq 0) {
  @("| none | none | none |")
}
else {
  foreach ($finding in $findings) {
    "| ``$(ConvertTo-MarkdownCell $finding.id)`` | ``$(ConvertTo-MarkdownCell $finding.file)`` | $(ConvertTo-MarkdownCell $finding.match) |"
  }
}

Write-Utf8File -LiteralPath $markdownPath -InputObject @(
  "# Real Proof Import Boundary Audit",
  "",
  "- auditState: $auditState",
  "- scannedFileCount: $scannedFileCount",
  "- findingCount: $findingCount",
  "- canPublishPublicly: False",
  "- canCloseReleaseIssue: False",
  "- boundary: $($record.boundary)",
  "",
  "| Finding | File | Match |",
  "|---|---|---|",
  @($rows)
)

Write-Host "AuditState=$auditState FindingCount=$findingCount"
if ($Strict -and $findingCount -gt 0) {
  throw "Real proof import boundary audit failed with $findingCount finding(s)."
}
