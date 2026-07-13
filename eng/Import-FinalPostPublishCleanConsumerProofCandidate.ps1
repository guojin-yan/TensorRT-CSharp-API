[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-post-publish-clean-consumer-proof-record-contract.json",
  [string]$PreflightPath = "artifacts\final-release\final-post-publish-clean-consumer-proof-preflight.json",
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
$proofResults = @((Get-PropertyOrDefault -Object $preflight -Name "proofResults" -DefaultValue @()))
$readyIds = @($proofResults | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "readyForProofCandidate" -DefaultValue $false) } | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })

$candidateItems = @(
  foreach ($item in $contractItems) {
    $id = [string](Get-PropertyOrDefault -Object $item -Name "id" -DefaultValue "")
    if ($readyIds -notcontains $id) { continue }

    [pscustomobject]@{
      id = $id
      candidateState = "post-publish-clean-consumer-proof-candidate"
      packageIdentity = Get-PropertyOrDefault -Object $item -Name "packageIdentity" -DefaultValue $null
      cleanConsumerProjectRoot = Get-PropertyOrDefault -Object $item -Name "cleanConsumerProjectRoot" -DefaultValue ""
      cleanConsumerProjectSha256Manifest = Get-PropertyOrDefault -Object $item -Name "cleanConsumerProjectSha256Manifest" -DefaultValue ""
      cleanConsumerRestoreLogPath = Get-PropertyOrDefault -Object $item -Name "cleanConsumerRestoreLogPath" -DefaultValue ""
      cleanConsumerRestoreLogSha256 = Get-PropertyOrDefault -Object $item -Name "cleanConsumerRestoreLogSha256" -DefaultValue ""
      cleanConsumerBuildLogPath = Get-PropertyOrDefault -Object $item -Name "cleanConsumerBuildLogPath" -DefaultValue ""
      cleanConsumerBuildLogSha256 = Get-PropertyOrDefault -Object $item -Name "cleanConsumerBuildLogSha256" -DefaultValue ""
      cleanConsumerRunLogPath = Get-PropertyOrDefault -Object $item -Name "cleanConsumerRunLogPath" -DefaultValue ""
      cleanConsumerRunLogSha256 = Get-PropertyOrDefault -Object $item -Name "cleanConsumerRunLogSha256" -DefaultValue ""
      cleanConsumerMergedTranscriptPath = Get-PropertyOrDefault -Object $item -Name "cleanConsumerMergedTranscriptPath" -DefaultValue ""
      cleanConsumerMergedTranscriptSha256 = Get-PropertyOrDefault -Object $item -Name "cleanConsumerMergedTranscriptSha256" -DefaultValue ""
      cleanConsumerValidatorOutputPath = Get-PropertyOrDefault -Object $item -Name "cleanConsumerValidatorOutputPath" -DefaultValue ""
      cleanConsumerValidatorOutputSha256 = Get-PropertyOrDefault -Object $item -Name "cleanConsumerValidatorOutputSha256" -DefaultValue ""
      executedCommand = Get-PropertyOrDefault -Object $item -Name "executedCommand" -DefaultValue ""
      exitCode = Get-PropertyOrDefault -Object $item -Name "exitCode" -DefaultValue ""
      executedAtUtc = Get-PropertyOrDefault -Object $item -Name "executedAtUtc" -DefaultValue ""
      hostIdentity = Get-PropertyOrDefault -Object $item -Name "hostIdentity" -DefaultValue $null
      ownerReviewer = Get-PropertyOrDefault -Object $item -Name "ownerReviewer" -DefaultValue ""
      ownerReviewTimestampUtc = Get-PropertyOrDefault -Object $item -Name "ownerReviewTimestampUtc" -DefaultValue ""
      noProjectReferenceConfirmation = [bool](Get-PropertyOrDefault -Object $item -Name "noProjectReferenceConfirmation" -DefaultValue $false)
      noLocalFeedConfirmation = [bool](Get-PropertyOrDefault -Object $item -Name "noLocalFeedConfirmation" -DefaultValue $false)
      noDirectNupkgConfirmation = [bool](Get-PropertyOrDefault -Object $item -Name "noDirectNupkgConfirmation" -DefaultValue $false)
      noSourceCheckoutReferenceConfirmation = [bool](Get-PropertyOrDefault -Object $item -Name "noSourceCheckoutReferenceConfirmation" -DefaultValue $false)
      whyNotReleaseClose = @(
        "Post-publish clean consumer proof candidate still needs Owner final close decision.",
        "Rollback/no-rollback Owner decision has not been imported.",
        "Release notes and final public package URL review have not been approved.",
        "Candidate import does not perform package push and cannot close the release issue."
      )
      performsPublish = $false
      canPublishPublicly = $false
      canCloseReleaseIssue = $false
      canPromoteRuntimeProof = $false
      isRuntimeExecutionProof = $false
      isPostPublishProof = $false
      isReleaseCloseProof = $false
      boundary = "Final post-publish clean consumer proof candidate only. It is not publish approval, not release close approval, not package push, and cannot close the release."
    }
  }
)

$candidateState = if ($candidateItems.Count -gt 0) { "post-publish-clean-consumer-proof-candidate-created" } else { "blocked-no-post-publish-clean-consumer-proof-candidate-ready" }

$record = [ordered]@{
  recordKind = "final-post-publish-clean-consumer-proof-candidate"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  candidateState = $candidateState
  sourceInputPath = $InputPath
  sourcePreflightPath = $PreflightPath
  preflightState = [string](Get-PropertyOrDefault -Object $preflight -Name "preflightState" -DefaultValue "missing-final-post-publish-clean-consumer-proof-preflight")
  contractItemCount = $contractItems.Count
  proofCandidateItemCount = $candidateItems.Count
  blockedProofCandidateItemCount = $contractItems.Count - $candidateItems.Count
  proofCandidateItems = @($candidateItems)
  whyNotReleaseClose = @(
    "No Owner final release close decision has been imported.",
    "No rollback/no-rollback Owner decision has been imported.",
    "No release notes and final public package URL Owner approval has been imported.",
    "Post-publish clean consumer candidate cannot substitute release close proof, publish approval, or package push."
  )
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    "artifacts/final-release/final-post-publish-clean-consumer-proof-record-contract.json",
    "artifacts/final-release/final-post-publish-clean-consumer-proof-preflight.json"
  )
  boundary = "Final post-publish clean consumer proof candidate only. It is not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $OutputRoot "final-post-publish-clean-consumer-proof-candidate.json"
$markdownPath = Join-Path $OutputRoot "final-post-publish-clean-consumer-proof-candidate.md"
$record | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $candidateItems) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$(ConvertTo-MarkdownCell $item.candidateState)`` | ``$($item.canCloseReleaseIssue)`` |"
}

$markdown = @(
  "# Final Post-Publish Clean Consumer Proof Candidate",
  "",
  "- candidateState: ``$candidateState``",
  "- contractItemCount: ``$($contractItems.Count)``",
  "- proofCandidateItemCount: ``$($candidateItems.Count)``",
  "- blockedProofCandidateItemCount: ``$($record.blockedProofCandidateItemCount)``",
  "- boundary: $($record.boundary)",
  "",
  "| Proof Candidate | State | Can Close Release |",
  "|---|---|---|",
  @($rows)
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
