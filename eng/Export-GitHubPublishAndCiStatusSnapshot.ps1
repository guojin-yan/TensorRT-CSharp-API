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

function Get-RemoteHeadMatches {
  param(
    [string]$RemoteName,
    [string]$BranchName,
    [string]$HeadSha
  )

  if ([string]::IsNullOrWhiteSpace($RemoteName) -or [string]::IsNullOrWhiteSpace($BranchName) -or [string]::IsNullOrWhiteSpace($HeadSha)) {
    return [pscustomobject]@{
      remoteName = $RemoteName
      branchName = $BranchName
      remoteHeadSha = ""
      remoteHeadMatchesCurrentHead = $false
      querySucceeded = $false
      detail = "Remote name, branch name, or local head SHA is missing."
    }
  }

  $remoteHeadLine = (Invoke-GitLines @("ls-remote", "--heads", $RemoteName, $BranchName) | Select-Object -First 1)
  if ([string]::IsNullOrWhiteSpace($remoteHeadLine)) {
    return [pscustomobject]@{
      remoteName = $RemoteName
      branchName = $BranchName
      remoteHeadSha = ""
      remoteHeadMatchesCurrentHead = $false
      querySucceeded = $false
      detail = "git ls-remote did not return a head for the remote branch."
    }
  }

  $remoteHeadSha = ($remoteHeadLine -split "\s+")[0]
  $matches = $remoteHeadSha.Equals($HeadSha, [StringComparison]::OrdinalIgnoreCase)
  return [pscustomobject]@{
    remoteName = $RemoteName
    branchName = $BranchName
    remoteHeadSha = $remoteHeadSha
    remoteHeadMatchesCurrentHead = $matches
    querySucceeded = $true
    detail = if ($matches) { "Read-only ls-remote confirms current HEAD is present on the remote branch." } else { "Read-only ls-remote confirms the remote branch points at a different SHA." }
  }
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
$remoteName = if (-not [string]::IsNullOrWhiteSpace($upstream) -and $upstream.Contains("/")) { $upstream.Split("/", 2)[0] } else { "origin" }
$remoteBranchName = if (-not [string]::IsNullOrWhiteSpace($upstream) -and $upstream.Contains("/")) { $upstream.Split("/", 2)[1] } else { [string]$branch }
$remoteHeadMatch = Get-RemoteHeadMatches -RemoteName $remoteName -BranchName $remoteBranchName -HeadSha ([string]$head)
$hasRemoteContainingHead = $remoteContainsHead.Count -gt 0 -or [bool]$remoteHeadMatch.remoteHeadMatchesCurrentHead

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
    detail = "Read-only git branch -r --contains HEAD and git ls-remote remote/branch checks report source presence only; they are not CI proof and not package publish proof."
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
  remoteName = [string]$remoteName
  remoteBranchName = [string]$remoteBranchName
  remoteContainsHead = @($remoteContainsHead)
  remoteContainsHeadCount = $remoteContainsHead.Count
  remoteHeadSha = [string]$remoteHeadMatch.remoteHeadSha
  remoteHeadQuerySucceeded = [bool]$remoteHeadMatch.querySucceeded
  remoteHeadMatchesCurrentHead = [bool]$remoteHeadMatch.remoteHeadMatchesCurrentHead
  remoteHeadMatchDetail = [string]$remoteHeadMatch.detail
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
| remoteName | ``$($record.remoteName)`` |
| remoteBranchName | ``$($record.remoteBranchName)`` |
| remoteHeadSha | ``$($record.remoteHeadSha)`` |
| remoteHeadMatchesCurrentHead | ``$($record.remoteHeadMatchesCurrentHead)`` |
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
