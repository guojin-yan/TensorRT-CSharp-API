[CmdletBinding()]
param(
  [string]$OutputDirectory,
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Add-Finding {
  param(
    [System.Collections.Generic.List[object]]$Findings,
    [string]$Id,
    [string]$Severity,
    [string]$Path,
    [int]$Line,
    [string]$Snippet,
    [string]$Message
  )

  $Findings.Add([pscustomobject]@{
      id = $Id
      severity = $Severity
      path = $Path
      line = $Line
      snippet = $Snippet.Trim()
      message = $Message
    })
}

function Test-NegatedContext {
  param([string]$Text)

  return $Text -match "(not|cannot|can't|must not|blocked|non-proof|not proof|不是|不能|不可|不得|不把|仍需|需要|未|没有|保持|false|owner-action|required|template|draft|不表示|不等于|避免|误写|不能替代|不是 proof|不证明|只表示|only|does not|cannot replace|is not|publishability|可发布性)"
}

function Test-FileContainsAll {
  param(
    [string]$Text,
    [string[]]$Markers
  )

  foreach ($marker in @($Markers)) {
    if ([string]::IsNullOrWhiteSpace($marker)) {
      continue
    }

    if ($Text.IndexOf([string]$marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      return $false
    }
  }

  return $true
}

function Add-RequiredBoundaryFinding {
  param(
    [System.Collections.Generic.List[object]]$Findings,
    [string]$Id,
    [string]$Path,
    [string]$Message
  )

  Add-Finding $Findings $Id "blocker" $Path 0 "" $Message
}

$scanRoots = @(
  "README.md",
  "README.zh-CN.md",
  "docs",
  "samples",
  "applications",
  "src"
)

$allowedExtensions = @(".md", ".cs", ".csproj", ".json", ".yml", ".yaml", ".ps1", ".txt")
$files = New-Object System.Collections.Generic.List[string]
foreach ($root in $scanRoots) {
  $path = Join-Path $RepositoryRoot $root
  if (Test-Path -LiteralPath $path -PathType Leaf) {
    $files.Add($path)
  }
  elseif (Test-Path -LiteralPath $path -PathType Container) {
    Get-ChildItem -LiteralPath $path -Recurse -File | Where-Object {
      $allowedExtensions -contains $_.Extension -and
      $_.FullName -notmatch "\\(bin|obj|artifacts|TestResults)\\"
    } | ForEach-Object { $files.Add($_.FullName) }
  }
}

$findings = New-Object System.Collections.Generic.List[object]
$proofClaimPattern = "(runbook|dashboard|candidate|draft|dry-run|local feed|ProjectReference|direct \.nupkg|build-only|local nupkg|direct nupkg).{0,80}(proof passed|proof-ready|ready to publish|published|release closed|can close release issue|can publish|可发布|已发布|可以关闭|发布完成|证明已通过)"
$failedBlockerPattern = "failedBlockerCount\s*=\s*0.{0,80}(ready|proof-ready|can publish|可发布|可关闭|ready to publish)"
$blockedPublishPattern = "blocked-final-publish-real-proof-required.{0,80}(ready|can publish|可发布|approved|发布批准|可以发布)"
$oldSamplePattern = "(samples[/\\]YoloDet|YoloDet\.csproj)"

foreach ($file in $files) {
  $relativePath = [IO.Path]::GetRelativePath($RepositoryRoot, $file).Replace("\", "/")
  $lines = Get-Content -LiteralPath $file -Encoding utf8 -ErrorAction SilentlyContinue
  for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = [string]$lines[$i]
    if ([string]::IsNullOrWhiteSpace($line)) {
      continue
    }

    if ($line -match $oldSamplePattern -and -not (Test-NegatedContext $line)) {
      Add-Finding $findings "old-yolo-det-public-entry" "blocker" $relativePath ($i + 1) $line "公开入口不应恢复旧 YOLO 检测样例名。"
    }

    if ($line -match $proofClaimPattern -and -not (Test-NegatedContext $line)) {
      Add-Finding $findings "non-proof-artifact-promoted" "blocker" $relativePath ($i + 1) $line "疑似把 runbook/dashboard/candidate/draft/dry-run/local feed/ProjectReference/direct nupkg/build-only 晋级为 proof 或发布完成。"
    }

    if ($line -match $failedBlockerPattern -and -not (Test-NegatedContext $line)) {
      Add-Finding $findings "failed-blocker-zero-promoted" "blocker" $relativePath ($i + 1) $line "疑似把 failedBlockerCount=0 解读为 ready。"
    }

    if ($line -match $blockedPublishPattern -and -not (Test-NegatedContext $line)) {
      Add-Finding $findings "blocked-final-publish-promoted" "blocker" $relativePath ($i + 1) $line "疑似把 blocked final publish proof gate 描述为可发布。"
    }
  }
}

$requiredPublicFreezeChecks = @(
  [pscustomobject]@{
    path = "README.md"
    markers = @(
      "blocked",
      "non-proof",
      "owner-action",
      "clean-consumer-proof-execution-bundle",
      "clean-consumer-external-proof-closure-pack",
      "not runtime proof",
      "not post-publish proof"
    )
    message = "README.md must keep the release proof boundary visible: blocked/non-proof/owner-action and clean consumer closure packs cannot be promoted."
  },
  [pscustomobject]@{
    path = "README.zh-CN.md"
    markers = @(
      "blocked",
      "non-proof",
      "Owner",
      "clean-consumer-proof-execution-bundle",
      "clean-consumer-external-proof-closure-pack",
      "不是 runtime proof",
      "post-publish proof"
    )
    message = "README.zh-CN.md 必须保留 blocked/non-proof/Owner 行动口径，并明确 clean consumer closure packs 不能晋级 proof。"
  },
  [pscustomobject]@{
    path = "docs/articles/zh-cn/package-consumer-validation.md"
    markers = @(
      "package-consumer",
      "runtime proof",
      "non-proof",
      "ProjectReference",
      "local feed"
    )
    message = "package-consumer-validation.md 必须说明本地 feed、ProjectReference 和 package-consumer runtime proof 的边界。"
  },
  [pscustomobject]@{
    path = "docs/articles/zh-cn/release-candidate-gate.md"
    markers = @(
      "owner-action",
      "blocked",
      "runtime proof",
      "post-publish"
    )
    message = "release-candidate-gate.md 必须保持 release candidate 仍 blocked/owner-action-required 的公开口径。"
  },
  [pscustomobject]@{
    path = "docs/articles/zh-cn/runtime-distribution-strategy.md"
    markers = @(
      "runtime",
      "package",
      "proof",
      "local feed"
    )
    message = "runtime-distribution-strategy.md 必须保留 runtime package 与 local feed / proof 的边界描述。"
  }
)

foreach ($check in $requiredPublicFreezeChecks) {
  $path = Join-Path $RepositoryRoot ([string]$check.path)
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    Add-RequiredBoundaryFinding $findings "missing-public-freeze-file" ([string]$check.path) $check.message
    continue
  }

  $text = Get-Content -LiteralPath $path -Raw -Encoding utf8
  if (-not (Test-FileContainsAll -Text $text -Markers @($check.markers))) {
    Add-RequiredBoundaryFinding $findings "missing-public-freeze-boundary" ([string]$check.path) $check.message
  }
}

$publicFreezeRequiredCount = $requiredPublicFreezeChecks.Count
$publicFreezeFindingCount = @($findings.ToArray() | Where-Object { $_.id -in @("missing-public-freeze-file", "missing-public-freeze-boundary") }).Count
$publicFreezeState = if ($publicFreezeFindingCount -eq 0) { "public-docs-proof-boundary-freeze-passed" } else { "public-docs-proof-boundary-freeze-blocked" }
$blockedFindingCount = @($findings.ToArray() | Where-Object { $_.severity -eq "blocker" }).Count
$auditState = if ($blockedFindingCount -eq 0) { "public-proof-claim-boundary-audit-passed" } else { "public-proof-claim-boundary-audit-blocked" }
$boundary = "This audit scans public-facing docs/source/sample/application text for stale proof claims and required proof-boundary freeze markers. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. failedBlockerCount=0 is not ready."

$record = [pscustomobject]@{
  recordKind = "public-proof-claim-boundary-audit"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditState = $auditState
  publicFreezeState = $publicFreezeState
  scannedFileCount = $files.Count
  publicFreezeRequiredCount = $publicFreezeRequiredCount
  publicFreezeFindingCount = $publicFreezeFindingCount
  findingCount = $findings.Count
  blockedFindingCount = $blockedFindingCount
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  performsPublish = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  notExecutedByAutomation = $true
  scanRoots = @($scanRoots)
  nonProofBoundary = @(
    "not runtime proof",
    "not post-publish proof",
    "not publish approval",
    "not release close approval",
    "not package push",
    "failedBlockerCount=0 is not ready"
  )
  findings = @($findings.ToArray())
  boundary = $boundary
}

$jsonPath = Join-Path $OutputDirectory "public-proof-claim-boundary-audit.json"
$markdownPath = Join-Path $OutputDirectory "public-proof-claim-boundary-audit.md"
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Public Proof Claim Boundary Audit")
$md.Add("")
$md.Add("`public-proof-claim-boundary-audit` 扫描 README、docs、samples、applications 和 src 中的公开 claim，防止把 non-proof 产物误写成发布完成或 proof 通过。")
$md.Add("")
$md.Add("## Summary")
$md.Add("")
$md.Add("| Field | Value |")
$md.Add("| --- | --- |")
$md.Add("| auditState | ``$auditState`` |")
$md.Add("| publicFreezeState | ``$publicFreezeState`` |")
$md.Add("| scannedFileCount | ``$($record.scannedFileCount)`` |")
$md.Add("| publicFreezeRequiredCount | ``$($record.publicFreezeRequiredCount)`` |")
$md.Add("| publicFreezeFindingCount | ``$($record.publicFreezeFindingCount)`` |")
$md.Add("| findingCount | ``$($record.findingCount)`` |")
$md.Add("| blockedFindingCount | ``$($record.blockedFindingCount)`` |")
$md.Add("| canPublishPublicly | ``$($record.canPublishPublicly)`` |")
$md.Add("| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |")
$md.Add("")
$md.Add("## Findings")
$md.Add("")
if ($findings.Count -eq 0) {
  $md.Add("- No stale public proof claim findings.")
}
else {
  $md.Add("| Id | Severity | Path | Line | Message |")
  $md.Add("| --- | --- | --- | --- | --- |")
  foreach ($finding in $findings) {
    $md.Add("| $(ConvertTo-MarkdownCell $finding.id) | $(ConvertTo-MarkdownCell $finding.severity) | $(ConvertTo-MarkdownCell $finding.path) | ``$($finding.line)`` | $(ConvertTo-MarkdownCell $finding.message) |")
  }
}
$md.Add("")
$md.Add("## Boundary")
$md.Add("")
$md.Add($boundary)

Write-Utf8FileWithRetry -LiteralPath $markdownPath -InputObject $md

Write-Host "Public proof claim boundary audit written: $jsonPath"
Write-Host "Public proof claim boundary audit markdown written: $markdownPath"
Write-Host "AuditState=$auditState FindingCount=$($findings.Count)"

if ($Strict -and $blockedFindingCount -ne 0) {
  throw "Public proof claim boundary audit failed with $blockedFindingCount blocker finding(s)."
}
