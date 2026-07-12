[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-release-close-owner-approval-contract.json",
  [string]$PreflightPath = "artifacts\final-release\final-release-close-owner-approval-preflight.json",
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

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

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

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$input = Read-JsonOrNull -Path $InputPath
$preflight = Read-JsonOrNull -Path $PreflightPath
$approvalLanes = @((Get-PropertyOrDefault -Object $input -Name "approvalLanes" -DefaultValue @()))
$approvalResults = @((Get-PropertyOrDefault -Object $preflight -Name "approvalResults" -DefaultValue @()))
$readyIds = @($approvalResults | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "readyForCloseCandidate" -DefaultValue $false) } | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })

$candidateItems = @(
  foreach ($lane in $approvalLanes) {
    $id = [string](Get-PropertyOrDefault -Object $lane -Name "id" -DefaultValue "")
    if ($readyIds -notcontains $id) { continue }

    [pscustomobject]@{
      id = $id
      candidateState = "final-release-close-owner-approval-candidate"
      ownerReviewer = Get-PropertyOrDefault -Object $lane -Name "ownerReviewer" -DefaultValue ""
      ownerReviewTimestampUtc = Get-PropertyOrDefault -Object $lane -Name "ownerReviewTimestampUtc" -DefaultValue ""
      ownerApprovalDecision = Get-PropertyOrDefault -Object $lane -Name "ownerApprovalDecision" -DefaultValue ""
      ownerApprovalRationale = Get-PropertyOrDefault -Object $lane -Name "ownerApprovalRationale" -DefaultValue ""
      releaseIssueUrl = Get-PropertyOrDefault -Object $lane -Name "releaseIssueUrl" -DefaultValue ""
      releaseIssueCloseDecision = Get-PropertyOrDefault -Object $lane -Name "releaseIssueCloseDecision" -DefaultValue ""
      rollbackDecision = Get-PropertyOrDefault -Object $lane -Name "rollbackDecision" -DefaultValue ""
      rollbackRationale = Get-PropertyOrDefault -Object $lane -Name "rollbackRationale" -DefaultValue ""
      releaseNotesPath = Get-PropertyOrDefault -Object $lane -Name "releaseNotesPath" -DefaultValue ""
      releaseNotesSha256 = Get-PropertyOrDefault -Object $lane -Name "releaseNotesSha256" -DefaultValue ""
      finalPublicPackageUrl = Get-PropertyOrDefault -Object $lane -Name "finalPublicPackageUrl" -DefaultValue ""
      finalPublicPackageUrlReviewDecision = Get-PropertyOrDefault -Object $lane -Name "finalPublicPackageUrlReviewDecision" -DefaultValue ""
      finalPublicPackageSha256 = Get-PropertyOrDefault -Object $lane -Name "finalPublicPackageSha256" -DefaultValue ""
      finalPackageIdentity = Get-PropertyOrDefault -Object $lane -Name "finalPackageIdentity" -DefaultValue $null
      postPublishCleanConsumerProofCandidateId = Get-PropertyOrDefault -Object $lane -Name "postPublishCleanConsumerProofCandidateId" -DefaultValue ""
      whyNotReleaseCloseProof = @(
        "Owner approval candidate still must be consumed by the final release close validator.",
        "Candidate import does not perform package push.",
        "Candidate import does not close the release issue by itself."
      )
      performsPublish = $false
      canPublishPublicly = $false
      canCloseReleaseIssue = $false
      canPromoteRuntimeProof = $false
      isRuntimeExecutionProof = $false
      isPostPublishProof = $false
      isReleaseCloseProof = $false
      boundary = "Final release close Owner approval candidate only. It is not package push, not publish execution, and cannot close the release by itself."
    }
  }
)

$candidateState = if ($candidateItems.Count -gt 0) { "final-release-close-owner-approval-candidate-created" } else { "blocked-no-final-release-close-owner-approval-candidate-ready" }

$record = [ordered]@{
  recordKind = "final-release-close-owner-approval-candidate"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  candidateState = $candidateState
  sourceInputPath = $InputPath
  sourcePreflightPath = $PreflightPath
  preflightState = [string](Get-PropertyOrDefault -Object $preflight -Name "preflightState" -DefaultValue "missing-final-release-close-owner-approval-preflight")
  approvalLaneCount = $approvalLanes.Count
  closeCandidateItemCount = $candidateItems.Count
  blockedCloseCandidateItemCount = $approvalLanes.Count - $candidateItems.Count
  closeCandidateItems = @($candidateItems)
  whyNotReleaseClose = @(
    "No real Owner final release close approval has passed preflight.",
    "No final release close validator has accepted a complete close record.",
    "No package push is performed by this importer.",
    "No release issue is closed by this importer."
  )
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    "artifacts/final-release/final-release-close-owner-approval-contract.json",
    "artifacts/final-release/final-release-close-owner-approval-preflight.json"
  )
  boundary = "Final release close Owner approval candidate only. It is not publish approval, not package push, not release close proof, and cannot close the release."
}

$jsonPath = Join-Path $OutputRoot "final-release-close-owner-approval-candidate.json"
$markdownPath = Join-Path $OutputRoot "final-release-close-owner-approval-candidate.md"
$record | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $candidateItems) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$(ConvertTo-MarkdownCell $item.candidateState)`` | ``$($item.canCloseReleaseIssue)`` |"
}

$markdown = @(
  "# Final Release Close Owner Approval Candidate",
  "",
  "- candidateState: ``$candidateState``",
  "- approvalLaneCount: ``$($approvalLanes.Count)``",
  "- closeCandidateItemCount: ``$($candidateItems.Count)``",
  "- blockedCloseCandidateItemCount: ``$($record.blockedCloseCandidateItemCount)``",
  "- boundary: $($record.boundary)",
  "",
  "| Close Candidate | State | Can Close Release |",
  "|---|---|---|",
  @($rows)
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
