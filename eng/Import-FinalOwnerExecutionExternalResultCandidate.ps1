[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-external-result-input-contract.json",
  [string]$PreflightPath = "artifacts\final-release\final-owner-execution-external-result-input-preflight.json",
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
$contractItems = @((Get-PropertyOrDefault -Object $input -Name "contractItems" -DefaultValue @()))
$laneResults = @((Get-PropertyOrDefault -Object $preflight -Name "laneResults" -DefaultValue @()))
$readyLaneIds = @($laneResults | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "readyForCandidate" -DefaultValue $false) } | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "sourceDraftItemId" -DefaultValue "") })

$candidateItems = @(
  foreach ($item in $contractItems) {
    $sourceDraftItemId = [string](Get-PropertyOrDefault -Object $item -Name "sourceDraftItemId" -DefaultValue "")
    if ($readyLaneIds -notcontains $sourceDraftItemId) { continue }
    [pscustomobject]@{
      sourceDraftItemId = $sourceDraftItemId
      sourceExecutionStepId = [string](Get-PropertyOrDefault -Object $item -Name "sourceExecutionStepId" -DefaultValue "")
      laneId = [string](Get-PropertyOrDefault -Object $item -Name "laneId" -DefaultValue "")
      candidateState = "owner-external-result-candidate"
      realExecutionRoot = Get-PropertyOrDefault -Object $item -Name "realExecutionRoot" -DefaultValue ""
      stdoutPath = Get-PropertyOrDefault -Object $item -Name "stdoutPath" -DefaultValue ""
      stderrPath = Get-PropertyOrDefault -Object $item -Name "stderrPath" -DefaultValue ""
      mergedTranscriptPath = Get-PropertyOrDefault -Object $item -Name "mergedTranscriptPath" -DefaultValue ""
      validatorOutputPath = Get-PropertyOrDefault -Object $item -Name "validatorOutputPath" -DefaultValue ""
      stdoutSha256 = Get-PropertyOrDefault -Object $item -Name "stdoutSha256" -DefaultValue ""
      stderrSha256 = Get-PropertyOrDefault -Object $item -Name "stderrSha256" -DefaultValue ""
      mergedTranscriptSha256 = Get-PropertyOrDefault -Object $item -Name "mergedTranscriptSha256" -DefaultValue ""
      validatorOutputSha256 = Get-PropertyOrDefault -Object $item -Name "validatorOutputSha256" -DefaultValue ""
      exitCode = Get-PropertyOrDefault -Object $item -Name "exitCode" -DefaultValue ""
      executedCommand = Get-PropertyOrDefault -Object $item -Name "executedCommand" -DefaultValue ""
      executedAtUtc = Get-PropertyOrDefault -Object $item -Name "executedAtUtc" -DefaultValue ""
      hostIdentity = Get-PropertyOrDefault -Object $item -Name "hostIdentity" -DefaultValue $null
      packageIdentity = Get-PropertyOrDefault -Object $item -Name "packageIdentity" -DefaultValue $null
      ownerReviewer = Get-PropertyOrDefault -Object $item -Name "ownerReviewer" -DefaultValue ""
      ownerReviewTimestampUtc = Get-PropertyOrDefault -Object $item -Name "ownerReviewTimestampUtc" -DefaultValue ""
      ownerProvidedNonSubstituteConfirmations = Get-PropertyOrDefault -Object $item -Name "ownerProvidedNonSubstituteConfirmations" -DefaultValue @()
      whyNotProof = @(
        "Candidate is not runtime proof until release validators consume it as a real proof record.",
        "Candidate is not post-publish proof until a clean external consumer validates a public package.",
        "Candidate is not release close proof until Owner approves final release close with post-publish evidence.",
        "Candidate is not package push and does not perform publish."
      )
      performsPublish = $false
      canPublishPublicly = $false
      canCloseReleaseIssue = $false
      canPromoteRuntimeProof = $false
      isRuntimeExecutionProof = $false
      isPostPublishProof = $false
      isReleaseCloseProof = $false
      boundary = "Owner external result candidate only. It preserves real external execution evidence metadata, but is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
    }
  }
)

$candidateState = if ($candidateItems.Count -gt 0) { "owner-external-result-candidate-created" } else { "blocked-no-owner-external-result-candidate-ready" }

$record = [ordered]@{
  recordKind = "final-owner-execution-external-result-candidate"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  candidateState = $candidateState
  sourceInputPath = $InputPath
  sourcePreflightPath = $PreflightPath
  preflightState = [string](Get-PropertyOrDefault -Object $preflight -Name "preflightState" -DefaultValue "missing-final-owner-execution-external-result-input-preflight")
  contractItemCount = $contractItems.Count
  candidateItemCount = $candidateItems.Count
  blockedCandidateItemCount = $contractItems.Count - $candidateItems.Count
  candidateItems = @($candidateItems)
  whyNotProof = @(
    "No public package post-publish clean consumer proof has been imported.",
    "No release close Owner approval can be inferred from candidate records.",
    "No package push is performed by this importer.",
    "Candidate records cannot substitute runtime proof, post-publish proof, or release close proof."
  )
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    "artifacts/final-release/final-owner-execution-external-result-input-contract.json",
    "artifacts/final-release/final-owner-execution-external-result-input-preflight.json"
  )
  boundary = "Final Owner external result candidate only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-external-result-candidate.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-external-result-candidate.md"
$record | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $candidateItems) {
  "| ``$(ConvertTo-MarkdownCell $item.sourceExecutionStepId)`` | ``$(ConvertTo-MarkdownCell $item.laneId)`` | ``$(ConvertTo-MarkdownCell $item.candidateState)`` |"
}

$markdown = @(
  "# Final Owner Execution External Result Candidate",
  "",
  "- candidateState: ``$candidateState``",
  "- contractItemCount: ``$($contractItems.Count)``",
  "- candidateItemCount: ``$($candidateItems.Count)``",
  "- blockedCandidateItemCount: ``$($record.blockedCandidateItemCount)``",
  "- boundary: $($record.boundary)",
  "",
  "| Execution Step | Lane | Candidate State |",
  "|---|---|---|",
  @($rows)
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
