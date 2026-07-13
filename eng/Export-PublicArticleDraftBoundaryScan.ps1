[CmdletBinding()]
param(
  [string]$ArticlesRoot = "docs\articles\zh-cn",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($ArticlesRoot)) { $ArticlesRoot = Join-Path $RepositoryRoot $ArticlesRoot }

function New-Pattern {
  param([string]$Id, [string]$Regex, [string]$Claim, [string]$RequiredProof)
  [pscustomobject]@{ id = $Id; regex = $Regex; claim = $Claim; requiredProof = $RequiredProof }
}

function Test-BoundaryContext {
  param([string]$Line)
  return $Line -match "禁止|不得|不能|不是|不代表|未证明|未完成|缺少|blocked|non-proof|owner proof|required|不能宣称|不要宣称|no proof"
}

$patterns = @(
  New-Pattern "nuget-published" "已发布到\s*NuGet|NuGet\s*已发布|published\s+to\s+NuGet" "NuGet package is publicly published." "Owner public package page/download URL and publish transcript."
  New-Pattern "public-install-verified" "公开包已经可安装|公开渠道.*安装.*通过|public package.*install.*verified" "Public package install is verified." "PostPublish owner record and clean consumer logs."
  New-Pattern "clean-consumer-passed" "clean consumer.*已通过|clean consumer.*passed|公开渠道 clean consumer 已通过" "Clean consumer from public channel passed." "Repository-external restore/build/smoke proof."
  New-Pattern "release-closed" "release closed|release.*已关闭|最终关闭 release" "Release issue is closed." "Owner close decision and strict closure acceptance."
  New-Pattern "runtime-proof-complete" "runtime proof.*已完成|真实 GPU.*proof.*完成|runtime proof.*complete" "Runtime proof is complete." "Real runtime smoke report and strict validator transcript."
  New-Pattern "dry-run-as-proof" "dry-run.*证明|dry run.*proof|dashboard.*证明|manual approval.*证明|queued workflow.*proof" "Dry-run/dashboard/manual approval is treated as proof." "Replace with real Owner evidence."
  New-Pattern "local-feed-proof" "local feed.*proof|ProjectReference.*proof|direct .?nupkg.*proof|本地源.*证明" "Local substitute is treated as package consumer proof." "Use public package source and no-substitute scan."
  New-Pattern "tensorrtexec-proof" "TensorRtExec report.*proof|TensorRtExec.*证明运行" "TensorRtExec sidecar/report is treated as runtime proof." "Use real runtime smoke and Owner evidence."
)

$claimMatches = New-Object System.Collections.Generic.List[object]
if (Test-Path -LiteralPath $ArticlesRoot -PathType Container) {
  $files = Get-ChildItem -LiteralPath $ArticlesRoot -Recurse -Filter "*.md" -File
  foreach ($file in $files) {
    $relativePath = [System.IO.Path]::GetRelativePath($RepositoryRoot, $file.FullName).Replace("\", "/")
    $lineNumber = 0
    foreach ($line in Get-Content -LiteralPath $file.FullName -Encoding utf8) {
      $lineNumber++
      foreach ($pattern in $patterns) {
        if ($line -match $pattern.regex) {
          $hasBoundaryContext = Test-BoundaryContext -Line $line
          $claimMatches.Add([pscustomobject]@{
            patternId = $pattern.id
            path = $relativePath
            line = $lineNumber
            matchedText = $Matches[0]
            hasBoundaryContext = $hasBoundaryContext
            severity = if ($hasBoundaryContext) { "boundary-mentioned" } else { "blocked-claim-review-required" }
            claim = $pattern.claim
            requiredProof = $pattern.requiredProof
            ownerActionRequired = -not $hasBoundaryContext
          }) | Out-Null
        }
      }
    }
  }
}

$blockedMatches = @($claimMatches.ToArray() | Where-Object { [string]$_.severity -eq "blocked-claim-review-required" })
$record = [pscustomobject]@{
  recordKind = "public-article-draft-boundary-scan"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  scanState = if ($blockedMatches.Count -gt 0) { "blocked-public-article-draft-boundary-review-required" } else { "public-article-draft-boundary-scan-clean-non-proof" }
  articlesRoot = [System.IO.Path]::GetRelativePath($RepositoryRoot, $ArticlesRoot).Replace("\", "/")
  scanPatternCount = $patterns.Count
  scannedFileCount = if (Test-Path -LiteralPath $ArticlesRoot -PathType Container) { @(Get-ChildItem -LiteralPath $ArticlesRoot -Recurse -Filter "*.md" -File).Count } else { 0 }
  matchCount = $claimMatches.Count
  blockedClaimMatchCount = $blockedMatches.Count
  matches = @($claimMatches.ToArray())
  patterns = @($patterns)
  ownerActionRequired = $blockedMatches.Count -gt 0
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Public article draft boundary scan only. It scans markdown claim language and never publishes articles, publishes packages, promotes runtime/PostPublish proof, or closes release issues."
}

$jsonPath = Join-Path $OutputRoot "public-article-draft-boundary-scan.json"
$mdPath = Join-Path $OutputRoot "public-article-draft-boundary-scan.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Public Article Draft Boundary Scan") | Out-Null
$md.Add("") | Out-Null
$md.Add("- scanState: ``$($record.scanState)``") | Out-Null
$md.Add("- scannedFileCount: ``$($record.scannedFileCount)``") | Out-Null
$md.Add("- scanPatternCount: ``$($record.scanPatternCount)``") | Out-Null
$md.Add("- matchCount: ``$($record.matchCount)``") | Out-Null
$md.Add("- blockedClaimMatchCount: ``$($record.blockedClaimMatchCount)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Pattern | File | Line | Severity |") | Out-Null
$md.Add("| --- | --- | --- | --- |") | Out-Null
foreach ($match in @($claimMatches.ToArray() | Select-Object -First 80)) {
  $md.Add("| ``$($match.patternId)`` | ``$(ConvertTo-MarkdownCell $match.path)`` | $($match.line) | ``$($match.severity)`` |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "PublicArticleDraftBoundaryScanState=$($record.scanState) Files=$($record.scannedFileCount) Matches=$($record.matchCount) Blocked=$($record.blockedClaimMatchCount)"
