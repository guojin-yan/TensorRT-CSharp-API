[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory
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

function New-Action {
  param(
    [string]$Id,
    [string]$Title,
    [string]$OwnerMustSupply,
    [string]$ExpectedFileHash,
    [string]$ValidatorCommand,
    [string]$SourceArtifact
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    ownerMustSupply = $OwnerMustSupply
    expectedFileHash = $ExpectedFileHash
    validatorCommand = $ValidatorCommand
    sourceArtifact = $SourceArtifact
    whyNotProof = "Owner action worklist is execution guidance only; until real external files, hashes, stdout/stderr, package identity, host identity, approvals, and strict validators pass, this item is not proof."
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

$freezePath = Join-Path $OutputDirectory "final-public-publish-pre-execution-freeze.json"
if (-not (Test-Path -LiteralPath $freezePath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Export-FinalPublicPublishPreExecutionFreeze.ps1") -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
}

$freeze = Get-Content -LiteralPath $freezePath -Raw -Encoding utf8 | ConvertFrom-Json

$actions = @(
  New-Action "public-package-identity-url-sha256" "回填 public package identity / URL / SHA256" "Owner supplies final public package id, version, URL, SHA256, source channel and reviewer." "package URL plus 64-char SHA256" "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPackageHashCrossCheckGate.ps1 -Strict" "artifacts/final-release/final-release-close-owner-approval-candidate-validation.json"
  New-Action "external-clean-consumer-restore" "干净外部 consumer restore" "Owner runs restore from public package source in a repository-external project." "restore stdout/stderr/merged transcript and SHA256" "dotnet restore <external-clean-consumer.csproj> --source <public-package-source>" "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json"
  New-Action "external-clean-consumer-build" "干净外部 consumer build" "Owner runs build in the same external project without ProjectReference, local feed, or direct .nupkg." "build stdout/stderr/merged transcript and SHA256" "dotnet build <external-clean-consumer.csproj> -c Release --no-restore" "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json"
  New-Action "external-clean-consumer-runtime-smoke" "干净外部 consumer runtime smoke run" "Owner runs runtime smoke against installed public packages on a compatible host." "runtime stdout/stderr/merged transcript/validator output and SHA256" "dotnet run --project <external-clean-consumer.csproj> -c Release --no-build" "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json"
  New-Action "clean-consumer-log-and-validator-hashes" "提供 stdout/stderr/merged transcript/validator output 和 SHA256" "Owner supplies every required log path and hash used by strict validators." "stdoutSha256/stderrSha256/mergedTranscriptSha256/validatorOutputSha256" "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalPostPublishCleanConsumerProofCandidate.ps1 -Strict" "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json"
  New-Action "release-notes-path-sha256" "提供 release notes path/SHA256" "Owner supplies reviewed final release notes path and matching SHA256." "release notes file path plus 64-char SHA256" "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseCloseOwnerApprovalCandidate.ps1 -Strict" "artifacts/final-release/final-release-close-owner-approval-candidate-validation.json"
  New-Action "rollback-decision" "提供 rollback/no-rollback 决策" "Owner supplies rollback decision, rationale, reviewer, timestamp, and related artifact hashes." "rollback decision record and SHA256" "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseCloseOwnerApprovalPreflight.ps1 -Strict" "artifacts/final-release/final-release-close-owner-approval-preflight.json"
  New-Action "release-issue-close-decision" "提供 release issue close/keep-open 决策" "Owner supplies final close/keep-open decision after real proof and approval gates pass." "release issue decision record and SHA256" "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseCloseOwnerApprovalCandidate.ps1 -Strict" "artifacts/final-release/final-release-close-owner-approval-candidate-validation.json"
  New-Action "final-package-url-hash-approval" "提供 final public package URL/hash 审批" "Owner explicitly approves final public package URL and SHA256 as release-close input." "approval record, package URL and SHA256" "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseCloseOwnerApprovalCandidate.ps1 -Strict" "artifacts/final-release/final-release-close-owner-approval-candidate-validation.json"
  New-Action "refresh-bundle-and-classification-audit" "运行 release evidence bundle 和 classification audit" "Owner or maintainer refreshes bundle/audit after all real records are imported." "release-evidence-bundle.json and release-evidence-classification-audit.json hashes" "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict" "artifacts/final-release/release-evidence-classification-audit.json"
  New-Action "post-publish-proof-candidate-promotion" "导入 post-publish clean consumer proof candidate" "Owner imports only validator-passing public channel clean consumer proof." "candidate json/md plus validator output hash" "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-FinalPostPublishCleanConsumerProofCandidate.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalPostPublishCleanConsumerProofCandidate.ps1 -Strict" "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json"
  New-Action "final-close-approval-candidate-promotion" "导入 final release close approval candidate" "Owner imports final approval only after public package proof and close approval are real." "close candidate json/md plus validator output hash" "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-FinalReleaseCloseOwnerApprovalCandidate.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseCloseOwnerApprovalCandidate.ps1 -Strict" "artifacts/final-release/final-release-close-owner-approval-candidate-validation.json"
)

$record = [pscustomobject]@{
  recordKind = "final-public-publish-owner-action-worklist"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("O")
  worklistState = "blocked-public-publish-owner-action-required"
  freezeState = [string]$freeze.freezeState
  ownerActionCount = $actions.Count
  blockedOwnerActionCount = $actions.Count
  readyOwnerActionCount = 0
  failedBlockerCount = 0
  failedActionRequiredCount = $actions.Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseReady = $false
  actionItems = @($actions)
  sourceArtifacts = @(
    "artifacts/final-release/final-public-publish-pre-execution-freeze.json",
    "artifacts/final-release/final-public-publish-pre-execution-freeze-validation.json",
    "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json",
    "artifacts/final-release/final-release-close-owner-approval-candidate-validation.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-evidence-classification-audit.json"
  )
  boundary = "Final public publish owner action worklist is Owner handoff only. It does not execute dotnet nuget push, does not publish, does not close release issue, and is not runtime proof, post-publish proof, release close proof, or package push."
}

$jsonPath = Join-Path $OutputDirectory "final-public-publish-owner-action-worklist.json"
$mdPath = Join-Path $OutputDirectory "final-public-publish-owner-action-worklist.md"
Write-Utf8FileWithRetry -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Final Public Publish Owner Action Worklist")
$lines.Add("")
$lines.Add("- worklistState: ``$($record.worklistState)``")
$lines.Add("- ownerActionCount: ``$($record.ownerActionCount)``")
$lines.Add("- failedActionRequiredCount: ``$($record.failedActionRequiredCount)``")
$lines.Add("")
$lines.Add("| ID | Title | Validator |")
$lines.Add("| --- | --- | --- |")
foreach ($action in $actions) {
  $lines.Add("| $($action.id) | $(ConvertTo-MarkdownCell $action.title) | ``$(ConvertTo-MarkdownCell $action.validatorCommand)`` |")
}
$lines.Add("")
$lines.Add("> $($record.boundary)")
Write-Utf8FileWithRetry -LiteralPath $mdPath -InputObject $lines

Write-Host "WorklistState=$($record.worklistState)"
Write-Host "OwnerActionCount=$($record.ownerActionCount)"
