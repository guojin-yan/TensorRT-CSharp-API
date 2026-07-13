[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", "<br>")
}

$dashboard = Read-JsonOrNull "artifacts\final-release\release-rc-proof-dashboard.json"
if ($null -eq $dashboard) {
  & (Join-Path $PSScriptRoot "Export-ReleaseRcProofDashboard.ps1") -RepositoryRoot $RepositoryRoot | Out-Null
  $dashboard = Read-JsonOrNull "artifacts\final-release\release-rc-proof-dashboard.json"
}

$actions = @($dashboard.proofBlockers | ForEach-Object {
  [pscustomobject]@{
    id = $_.id
    proofClass = $_.proofClass
    currentState = $_.state
    ownerAction = $_.nextCommand
    validator = $_.requiredValidator
    requiredOwnerInputs = @($_.requiredOwnerInputs)
    sourceArtifacts = @($_.sourceArtifacts)
    cannotUse = @($_.cannotUse)
    ready = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
})

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "release-rc-owner-handoff"
  handoffState = "blocked-real-proof-required"
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  dashboardSource = "artifacts/final-release/release-rc-proof-dashboard.json"
  actionCount = $actions.Count
  readyActionCount = 0
  blockedActionCount = $actions.Count
  ownerActions = $actions
  nonSubstituteProofKinds = @($dashboard.nonSubstituteProofKinds)
  sourceArtifacts = @(
    "artifacts/final-release/release-rc-proof-dashboard.json",
    "artifacts/final-release/release-rc-proof-dashboard-validation.json",
    "artifacts/final-release/release-owner-proof-input-record-template.json",
    "artifacts/final-release/release-owner-proof-input-record-validation.json",
    "artifacts/final-release/release-issue-close-record-template.json",
    "artifacts/final-release/release-issue-close-record-validation.json",
    "artifacts/final-release/owner-release-execution-package.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-close-preflight.json"
  )
  finalCloseOwnerAction = [ordered]@{
    id = "release-issue-close-record"
    ownerAction = "Fill release-issue-close-record.json only after all real proof gates pass, then run Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady before closing the release issue."
    validator = "Test-ReleaseIssueCloseRecord.ps1"
    requiredOwnerInputs = @(
      "release issue id and URL",
      "owner identity and approval timestamp",
      "selected channel",
      "managed/runtime package URL and SHA256",
      "post-publish verification proof",
      "release close preflight",
      "stale release claims audit findingCount=0",
      "release evidence bundle SHA256",
      "rollback/yank/deprecate plan"
    )
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    boundary = "This action is owner guidance only until a real non-template close record validates."
  }
  boundary = "Release RC owner handoff is an execution checklist only. It does not publish packages, close release issues, or convert guidance into proof."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "release-rc-owner-handoff.json"
$markdownPath = Join-Path $artifactRoot "release-rc-owner-handoff.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $actions | ForEach-Object {
  $inputs = (@($_.requiredOwnerInputs) -join "<br>").Replace("|", "\|")
  "| ``$($_.id)`` | ``$($_.proofClass)`` | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.currentState) | $(ConvertTo-MarkdownCell $_.ownerAction) | $(ConvertTo-MarkdownCell $_.validator) | $inputs |"
}
$nonSubstituteLines = $record.nonSubstituteProofKinds | ForEach-Object { "- ``$_``" }

$markdown = @"
# Release RC Owner Handoff

生成时间：$($record.generatedAtUtc)

## Summary

- handoff state: ``$($record.handoffState)``
- performs publish: ``False``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- action count: ``$($record.actionCount)``
- blocked action count: ``$($record.blockedActionCount)``

## Final Close Owner Action

- id: ``$($record.finalCloseOwnerAction.id)``
- validator: ``$($record.finalCloseOwnerAction.validator)``
- canCloseReleaseIssue=false
- boundary: $($record.finalCloseOwnerAction.boundary)

## Owner Actions

| ID | Proof class | Ready | Current state | Owner next command | Validator | Required owner inputs |
| --- | --- | --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Boundary

$($record.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release RC owner handoff written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "HandoffState=$($record.handoffState)"
Write-Output "ActionCount=$($record.actionCount)"
Write-Output "CanPublishPublicly=False"
