[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputPath = "artifacts\final-release\worktree-staging-audit.json",
  [string]$MarkdownOutputPath = "artifacts\final-release\worktree-staging-audit.md"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepoPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Invoke-GitLines {
  param([Parameter(Mandatory = $true)][string[]]$Arguments)

  $errorPath = [System.IO.Path]::GetTempFileName()
  $previousLocation = (Get-Location).Path
  try {
    Set-Location -LiteralPath $RepositoryRoot
    $output = @(& git @Arguments 2> $errorPath)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
      $errorText = if (Test-Path -LiteralPath $errorPath -PathType Leaf) {
        Get-Content -LiteralPath $errorPath -Raw
      }
      else {
        ""
      }

      throw "git $($Arguments -join ' ') failed with exit code ${exitCode}: $($errorText.Trim())"
    }

    if ($output.Count -eq 0) {
      return @()
    }

    return @($output | ForEach-Object { [string]$_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  }
  finally {
    Set-Location -LiteralPath $previousLocation
    Remove-Item -LiteralPath $errorPath -Force -ErrorAction SilentlyContinue
  }
}

function Get-Root {
  param([Parameter(Mandatory = $true)][string]$Path)

  $normalized = $Path -replace "\\", "/"
  return ($normalized -split "/", 2)[0]
}

function Get-StagingBucket {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Source
  )

  $p = $Path -replace "\\", "/"

  if ($p -match '(^|/)(bin|obj)/' -or
      $p -match '(^|/)TestResults/' -or
      $p -match '^artifacts/.*/test-results/' -or
      $p -match '^artifacts/github-actions-runs/' -or
      $p -match '^-Strict(/|$)' -or
      $p -match '\.(trx|log|tmp|cache)$') {
    return "ignore-output"
  }

  if ($p -match '\.(nupkg|snupkg|zip|7z|tar|gz|dll|pdb|exe|lib|obj)$') {
    return "binary-review"
  }

  if ($p -match '^artifacts/(final-release|interface-coverage|package-consumer)/.+\.(json|md|csv|txt)$') {
    return "evidence-review"
  }

  if ($p -match '^\.github/workflows/') {
    return "workflow"
  }

  if ($p -match '^eng/') {
    return "engineering-script"
  }

  if ($p -match '^tests/') {
    return "quality-test"
  }

  if ($p -match '^src/') {
    return "managed-source"
  }

  if ($p -match '^native/') {
    if ($p -match '^native/generated/') {
      return "native-generated"
    }

    if ($p -match '^native/manifests/') {
      return "native-manifest"
    }

    return "native-source"
  }

  if ($p -match '^samples/') {
    return "sample"
  }

  if ($p -match '^applications/') {
    return "application"
  }

  if ($p -match '^smoke/') {
    return "smoke"
  }

  if ($p -match '^docs/') {
    if ($p -match '^docs/articles/zh-cn/blog-' -or $p -match '^docs/articles/zh-cn/.+tutorial' -or $p -match '^docs/articles/zh-cn/.+guide') {
      return "article-content"
    }

    return "documentation"
  }

  if ($p -match '^pack/') {
    return "packaging"
  }

  if ($p -match '^tools/') {
    return "tooling"
  }

  if ($p -match '^README' -or $p -eq 'TensorRtSharp.sln' -or $p -eq '.gitignore') {
    return "root-project-file"
  }

  return "manual-review"
}

function Get-PublicPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  $normalized = $Path -replace "\\", "/"
  if ($normalized -match '^samples/YoloDet/' -or $normalized -match 'YoloDet\.csproj') {
    return ($normalized -replace '^samples/YoloDet/', 'samples/legacy-yolo-sample-removed/' -replace 'YoloDet\.csproj', 'legacy-yolo-sample.csproj')
  }

  return $Path
}

function New-ChangeItem {
  param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Status,
    [Parameter(Mandatory = $true)][string]$Path
  )

  $bucket = Get-StagingBucket -Path $Path -Source $Source
  return [pscustomobject]@{
    source = $Source
    status = $Status
    path = Get-PublicPath -Path $Path
    root = Get-Root -Path $Path
    bucket = $bucket
  }
}

$tracked = @(Invoke-GitLines -Arguments @("diff", "--name-status") | ForEach-Object {
    $parts = $_ -split "`t"
    New-ChangeItem -Source "tracked" -Status $parts[0] -Path $parts[-1]
  })

$untracked = @(Invoke-GitLines -Arguments @("ls-files", "--others", "--exclude-standard") | ForEach-Object {
    New-ChangeItem -Source "untracked" -Status "??" -Path $_
  })

$items = @($tracked + $untracked)
$bucketSummary = @($items |
  Group-Object bucket |
  Sort-Object Name |
  ForEach-Object { [pscustomobject]@{ bucket = $_.Name; count = $_.Count } })
$rootSummary = @($items |
  Group-Object root, bucket |
  Sort-Object Count -Descending |
  ForEach-Object {
    $parts = $_.Name -split ", "
    [pscustomobject]@{ root = $parts[0]; bucket = $parts[1]; count = $_.Count }
  })
$statusSummary = @($items |
  Group-Object source, status |
  Sort-Object Name |
  ForEach-Object {
    $parts = $_.Name -split ", "
    [pscustomobject]@{ source = $parts[0]; status = $parts[1]; count = $_.Count }
  })

$safeStageBuckets = @(
  "workflow",
  "engineering-script",
  "quality-test",
  "managed-source",
  "native-source",
  "native-manifest",
  "native-generated",
  "sample",
  "application",
  "smoke",
  "documentation",
  "article-content",
  "packaging",
  "tooling",
  "root-project-file"
)
$reviewBuckets = @("evidence-review", "binary-review", "manual-review")
$ignoreBuckets = @("ignore-output")

$record = [pscustomobject]@{
  recordKind = "worktree-staging-audit"
  generatedAt = (Get-Date).ToString("o")
  repositoryRoot = $RepositoryRoot
  branch = (Invoke-GitLines -Arguments @("branch", "--show-current") | Select-Object -First 1)
  headSha = (Invoke-GitLines -Arguments @("rev-parse", "HEAD") | Select-Object -First 1)
  counts = [pscustomobject]@{
    trackedDirty = @($tracked).Count
    untracked = @($untracked).Count
    total = @($items).Count
    safeStageCandidate = @($items | Where-Object { $safeStageBuckets -contains $_.bucket }).Count
    reviewCandidate = @($items | Where-Object { $reviewBuckets -contains $_.bucket }).Count
    ignoreCandidate = @($items | Where-Object { $ignoreBuckets -contains $_.bucket }).Count
  }
  bucketSummary = $bucketSummary
  rootSummary = $rootSummary
  statusSummary = $statusSummary
  safeStageBuckets = $safeStageBuckets
  reviewBuckets = $reviewBuckets
  ignoreBuckets = $ignoreBuckets
  safeStageCandidates = @($items | Where-Object { $safeStageBuckets -contains $_.bucket } | Select-Object source, status, path, bucket)
  reviewCandidates = @($items | Where-Object { $reviewBuckets -contains $_.bucket } | Select-Object source, status, path, bucket)
  ignoreCandidates = @($items | Where-Object { $ignoreBuckets -contains $_.bucket } | Select-Object source, status, path, bucket)
  pathspecs = [pscustomobject]@{
    safeStage = @($items | Where-Object { $safeStageBuckets -contains $_.bucket } | Select-Object -ExpandProperty path | Sort-Object -Unique)
    reviewHold = @($items | Where-Object { $reviewBuckets -contains $_.bucket } | Select-Object -ExpandProperty path | Sort-Object -Unique)
    ignoreHold = @($items | Where-Object { $ignoreBuckets -contains $_.bucket } | Select-Object -ExpandProperty path | Sort-Object -Unique)
  }
  recommendedStageCommand = "Use explicit git add pathspecs from safeStageBuckets after reviewing this report; do not use git add ."
  boundary = "This audit is classification guidance only. It does not stage, commit, push, delete, or publish files. Review untracked and generated files before staging."
}

$jsonPath = Resolve-RepoPath -Path $OutputPath
$markdownPath = Resolve-RepoPath -Path $MarkdownOutputPath
New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($jsonPath)) -Force | Out-Null
New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($markdownPath)) -Force | Out-Null
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$pathspecRoot = Resolve-RepoPath -Path "artifacts\final-release\staging-pathspecs"
New-Item -ItemType Directory -Path $pathspecRoot -Force | Out-Null
$safeStagePath = Join-Path $pathspecRoot "safe-stage-pathspecs.txt"
$reviewHoldPath = Join-Path $pathspecRoot "review-hold-pathspecs.txt"
$ignoreHoldPath = Join-Path $pathspecRoot "ignore-hold-pathspecs.txt"
Set-Content -LiteralPath $safeStagePath -Value @($record.pathspecs.safeStage) -Encoding utf8
Set-Content -LiteralPath $reviewHoldPath -Value @($record.pathspecs.reviewHold) -Encoding utf8
Set-Content -LiteralPath $ignoreHoldPath -Value @($record.pathspecs.ignoreHold) -Encoding utf8

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Worktree Staging Audit")
$lines.Add("")
$lines.Add("- Branch: ``$($record.branch)``")
$lines.Add("- HEAD: ``$($record.headSha)``")
$lines.Add("- Tracked dirty: ``$($record.counts.trackedDirty)``")
$lines.Add("- Untracked: ``$($record.counts.untracked)``")
$lines.Add("- Safe stage candidates: ``$($record.counts.safeStageCandidate)``")
$lines.Add("- Review candidates: ``$($record.counts.reviewCandidate)``")
$lines.Add("- Ignore candidates: ``$($record.counts.ignoreCandidate)``")
$lines.Add("- Safe stage pathspec file: ``artifacts/final-release/staging-pathspecs/safe-stage-pathspecs.txt``")
$lines.Add("- Review hold pathspec file: ``artifacts/final-release/staging-pathspecs/review-hold-pathspecs.txt``")
$lines.Add("- Ignore hold pathspec file: ``artifacts/final-release/staging-pathspecs/ignore-hold-pathspecs.txt``")
$lines.Add("")
$lines.Add("## Bucket Summary")
$lines.Add("")
$lines.Add("| Bucket | Count |")
$lines.Add("| --- | ---: |")
foreach ($item in $bucketSummary) {
  $lines.Add("| $($item.bucket) | $($item.count) |")
}
$lines.Add("")
$lines.Add("## Top Root Summary")
$lines.Add("")
$lines.Add("| Root | Bucket | Count |")
$lines.Add("| --- | --- | ---: |")
foreach ($item in @($rootSummary | Select-Object -First 80)) {
  $lines.Add("| $($item.root) | $($item.bucket) | $($item.count) |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Worktree staging audit written to $jsonPath"
Write-Host "Worktree staging audit written to $markdownPath"
Write-Host "Safe stage pathspecs written to $safeStagePath"
Write-Host "Review hold pathspecs written to $reviewHoldPath"
Write-Host "Ignore hold pathspecs written to $ignoreHoldPath"
Write-Host "trackedDirty=$($record.counts.trackedDirty)"
Write-Host "untracked=$($record.counts.untracked)"
Write-Host "safeStageCandidate=$($record.counts.safeStageCandidate)"
Write-Host "reviewCandidate=$($record.counts.reviewCandidate)"
Write-Host "ignoreCandidate=$($record.counts.ignoreCandidate)"
