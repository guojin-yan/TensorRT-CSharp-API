[CmdletBinding()]
param(
  [string]$RoadmapPath,
  [string]$OutputDirectory,
  [switch]$Strict,
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

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\final-release"
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

function Add-Finding {
  param(
    [System.Collections.Generic.List[object]]$Findings,
    [string]$Id,
    [string]$Message,
    [AllowNull()][object]$ArticleId = $null
  )

  $Findings.Add([pscustomobject]@{
      id = $Id
      severity = "blocker"
      articleId = $ArticleId
      message = $Message
    })
}

if (-not (Test-Path -LiteralPath $RoadmapPath -PathType Leaf)) {
  throw "Article roadmap not found: $RoadmapPath"
}

$roadmap = Get-Content -LiteralPath $RoadmapPath -Raw -Encoding utf8 | ConvertFrom-Json
$articles = @($roadmap.articles)
$findings = New-Object System.Collections.Generic.List[object]

if ([string]$roadmap.roadmapId -ne "article-roadmap-30plus") {
  Add-Finding $findings "roadmap-id" "roadmapId must be article-roadmap-30plus."
}

if ($articles.Count -lt 30) {
  Add-Finding $findings "article-count" "Article roadmap must contain at least 30 articles."
}

if ([bool]$roadmap.canPublishPublicly -or [bool]$roadmap.canCloseReleaseIssue -or [bool]$roadmap.isRuntimeExecutionProof -or [bool]$roadmap.isPostPublishProof -or [bool]$roadmap.isReleaseCloseProof) {
  Add-Finding $findings "roadmap-proof-flags" "Roadmap-level proof/publish/close flags must remain false."
}

$ids = @{}
foreach ($article in $articles) {
  $id = [int]$article.id
  if ($ids.ContainsKey($id)) {
    Add-Finding $findings "duplicate-article-id" "Article id must be unique." $id
  }
  $ids[$id] = $true

  foreach ($field in @("id", "title", "audience", "status", "targetPath", "proofBoundary")) {
    if (-not ($article.PSObject.Properties.Name -contains $field) -or [string]::IsNullOrWhiteSpace([string]$article.$field)) {
      Add-Finding $findings "missing-required-field" "Article is missing required field: $field" $id
    }
  }

  foreach ($arrayField in @("sourceArtifacts", "mustAvoidClaims")) {
    if (-not ($article.PSObject.Properties.Name -contains $arrayField) -or @($article.$arrayField).Count -eq 0) {
      Add-Finding $findings "missing-required-array-field" "Article is missing required non-empty array field: $arrayField" $id
    }
  }

  $boundary = [string]$article.proofBoundary
  foreach ($marker in @("not runtime proof", "not post-publish proof", "not publish approval", "not release close approval", "not package push")) {
    if ($boundary.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
      Add-Finding $findings "article-boundary-marker" "Article proofBoundary is missing marker: $marker" $id
    }
  }

  $combined = (($article | ConvertTo-Json -Depth 8) -join " ")
  if ($combined -match "YoloDet|samples[/\\]YoloDet") {
    Add-Finding $findings "old-yolo-det-reference" "Article roadmap must not expose old YoloDet public entry." $id
  }
}

$requiredMarkers = @(
  "TensorRtExec",
  "OnnxToEngine",
  "YoloVision",
  "Plugin",
  "Owner",
  "StrictClose",
  "ReleaseClose",
  "CUDA",
  "TensorRT",
  "NuGet"
)
$roadmapText = Get-Content -LiteralPath $RoadmapPath -Raw -Encoding utf8
foreach ($marker in $requiredMarkers) {
  if ($roadmapText.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
    Add-Finding $findings "missing-topic-marker" "Roadmap is missing topic marker: $marker"
  }
}

$validationState = if ($findings.Count -eq 0) { "article-roadmap-30plus-validation-passed-non-proof-planning" } else { "article-roadmap-30plus-validation-blocked" }
$record = [pscustomobject]@{
  recordKind = "article-roadmap-30plus-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  roadmapPath = $RoadmapPath
  articleCount = $articles.Count
  minimumArticleCount = 30
  findingCount = $findings.Count
  blockedFindingCount = $findings.Count
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  performsPublish = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  findings = @($findings.ToArray())
  boundary = "The 30+ article roadmap validation is content planning only: not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputDirectory "article-roadmap-30plus-validation.json"
$markdownPath = Join-Path $OutputDirectory "article-roadmap-30plus-validation.md"
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Article Roadmap 30+ Validation")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| validationState | ``$validationState`` |")
$lines.Add("| articleCount | ``$($record.articleCount)`` |")
$lines.Add("| findingCount | ``$($record.findingCount)`` |")
$lines.Add("| canPublishPublicly | ``$($record.canPublishPublicly)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Findings")
$lines.Add("")
if ($findings.Count -eq 0) {
  $lines.Add("- No findings. The article roadmap remains a non-proof planning matrix.")
}
else {
  foreach ($finding in $findings) {
    $lines.Add("- ``$($finding.id)`` article=$($finding.articleId) $($finding.message)")
  }
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)

Write-Utf8FileWithRetry -LiteralPath $markdownPath -InputObject $lines

Write-Host "Article roadmap 30+ validation written: $jsonPath"
Write-Host "Article roadmap 30+ validation markdown written: $markdownPath"
Write-Host "ValidationState=$validationState FindingCount=$($findings.Count)"

if ($Strict -and $findings.Count -ne 0) {
  throw "Article roadmap 30+ validation failed with $($findings.Count) finding(s)."
}
