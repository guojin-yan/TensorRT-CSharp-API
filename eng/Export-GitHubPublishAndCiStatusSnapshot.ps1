[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Invoke-GitLines {
  param([string[]]$Arguments)
  $output = & git -C $RepositoryRoot @Arguments 2>$null
  if ($LASTEXITCODE -ne 0) { return @() }
  return @($output)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$branch = (Invoke-GitLines @("branch", "--show-current") | Select-Object -First 1)
$head = (Invoke-GitLines @("rev-parse", "HEAD") | Select-Object -First 1)
$headShort = (Invoke-GitLines @("log", "-1", "--oneline") | Select-Object -First 1)
$remotes = @(Invoke-GitLines @("remote", "-v"))
$upstream = (Invoke-GitLines @("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}") | Select-Object -First 1)
$remoteContainsHead = @(Invoke-GitLines @("branch", "-r", "--contains", "HEAD"))
$hasRemoteContainingHead = $remoteContainsHead.Count -gt 0

$localValidationCommands = @(
  "git diff --check",
  "git diff --cached --check",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseScriptReferenceIndex.ps1",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseScriptReferenceIndex.ps1 -Strict",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict",
  "dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter <focused-release-filter>"
)

$statusItems = @(
  [pscustomobject]@{
    id = "source-code-pushed"
    state = if ($hasRemoteContainingHead) { "source-head-present-on-remote-branch" } else { "missing-remote-head-proof" }
    proofAvailable = $hasRemoteContainingHead
    detail = "git branch -r --contains HEAD reports remote branch containment only; it is not CI proof and not package publish proof."
  }
  [pscustomobject]@{
    id = "local-validation"
    state = "local-validation-recorded-non-proof"
    proofAvailable = $true
    detail = "Local validation commands are recorded as local checks only; they do not prove GitHub Actions, NuGet publish, or post-publish public consumer execution."
  }
  [pscustomobject]@{
    id = "github-actions-proof"
    state = "missing-github-actions-proof"
    proofAvailable = $false
    detail = "No verified GitHub Actions run URL, run id, log hash, artifact hash, or package publish workflow result is imported by this snapshot."
  }
  [pscustomobject]@{
    id = "nuget-package-publish-on-github"
    state = "blocked-real-github-actions-package-publish-proof-required"
    proofAvailable = $false
    detail = "No GitHub-hosted package publish run proof is present. Local build/test/package output cannot substitute a GitHub Actions package publish result."
  }
  [pscustomobject]@{
    id = "github-packages-publish"
    state = "blocked-real-github-packages-publish-proof-required"
    proofAvailable = $false
    detail = "No real GitHub Packages publish result URL/log/hash is imported. This snapshot does not trigger workflow dispatch."
  }
)

$record = [pscustomobject]@{
  recordKind = "github-publish-and-ci-status-snapshot"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  snapshotState = "blocked-github-actions-and-public-publish-proof-required"
  branch = [string]$branch
  headCommit = [string]$head
  headCommitSummary = [string]$headShort
  upstream = [string]$upstream
  remoteContainsHead = @($remoteContainsHead)
  remoteContainsHeadCount = $remoteContainsHead.Count
  sourceHeadPresentOnRemote = $hasRemoteContainingHead
  remotes = @($remotes)
  localValidationCommands = @($localValidationCommands)
  statusItems = @($statusItems)
  githubActionsProofState = "missing-github-actions-proof"
  packagePublishOnGitHubState = "blocked-real-github-actions-package-publish-proof-required"
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isGitHubActionsProof = $false
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  safetyBoundary = "GitHub publish and CI status snapshot is read-only local repository metadata plus explicit missing-proof declarations. It does not trigger GitHub Actions, does not run publish workflows, does not push packages, and is not runtime proof, post-publish proof, release close approval, package publish proof, or GitHub Actions proof."
}

$jsonPath = Join-Path $OutputRoot "github-publish-and-ci-status-snapshot.json"
$markdownPath = Join-Path $OutputRoot "github-publish-and-ci-status-snapshot.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $statusItems | ForEach-Object { "| $(ConvertTo-MarkdownCell $_.id) | $(ConvertTo-MarkdownCell $_.state) | ``$($_.proofAvailable)`` | $(ConvertTo-MarkdownCell $_.detail) |" }
$markdown = @"
# GitHub Publish And CI Status Snapshot

| Item | Value |
|---|---|
| snapshotState | ``$($record.snapshotState)`` |
| branch | ``$($record.branch)`` |
| headCommit | ``$($record.headCommit)`` |
| sourceHeadPresentOnRemote | ``$($record.sourceHeadPresentOnRemote)`` |
| githubActionsProofState | ``$($record.githubActionsProofState)`` |
| packagePublishOnGitHubState | ``$($record.packagePublishOnGitHubState)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Status Items

| ID | State | Proof Available | Detail |
|---|---|---:|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "GitHub publish and CI status snapshot written: $jsonPath"
Write-Host "SnapshotState=$($record.snapshotState) SourceHeadPresentOnRemote=$($record.sourceHeadPresentOnRemote)"
