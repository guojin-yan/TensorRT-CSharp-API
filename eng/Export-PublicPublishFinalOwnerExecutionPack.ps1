[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ExecutionLane {
  param(
    [string]$Id,
    [string]$Title,
    [string]$RequiredInput,
    [string]$Validator,
    [string]$OwnerAction
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    requiredInput = $RequiredInput
    validator = $Validator
    ownerAction = $OwnerAction
    status = "owner-action-required"
    ready = $false
    ownerExecutionOnly = $true
    notExecutedByAutomation = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    boundary = "Final owner execution lane only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$ownerManualCommandHandoffValidation = Read-JsonOrNull "artifacts\final-release\public-publish-owner-manual-command-handoff-validation.json"
$publicPublishResultOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\public-publish-result-owner-input-validation.json"
$publicPublishResultImportValidation = Read-JsonOrNull "artifacts\final-release\public-publish-result-import-validation.json"
$cleanExternalRunbookValidation = Read-JsonOrNull "artifacts\final-release\clean-external-package-consumer-owner-runbook-validation.json"
$postPublishRunbookValidation = Read-JsonOrNull "artifacts\final-release\post-publish-owner-verification-runbook-validation.json"
$postPublishCleanConsumerConvergenceValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-result-convergence-validation.json"
$strictCloseReadyValidation = Read-JsonOrNull "artifacts\final-release\strict-close-ready-convergence-dashboard-validation.json"
$releaseEvidenceClassificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"

$executionLanes = @(
  New-ExecutionLane -Id "owner-command-review" -Title "Owner reviews final public publish commands" -RequiredInput "Owner reviewed command plan, package identity, runtime key, rollback plan, and credential handling acknowledgement." -Validator "Test-PublicPublishCommandCrossCheck.ps1 -Strict" -OwnerAction "Review generated command placeholders and confirm no automation has executed a publish command."
  New-ExecutionLane -Id "managed-package-public-publish" -Title "Owner manually publishes managed package" -RequiredInput "Managed package id, version, selected channel, public package URL, nupkg SHA256, and publish transcript hash." -Validator "Test-PublicPublishResultOwnerInput.ps1 -Strict" -OwnerAction "Run the owner-approved public package publish command outside automation and capture transcript/log/hash."
  New-ExecutionLane -Id "runtime-package-public-publish" -Title "Owner manually publishes runtime package" -RequiredInput "Runtime package id, runtime package key, selected channel, public package URL, nupkg SHA256, and publish transcript hash." -Validator "Test-PublicPublishResultOwnerInput.ps1 -Strict" -OwnerAction "Publish the matching runtime package for the selected runtime key and capture public URL/hash evidence."
  New-ExecutionLane -Id "public-publish-result-import" -Title "Import owner public publish result" -RequiredInput "Owner-filled public publish result record." -Validator "Test-PublicPublishResultImport.ps1 -Strict" -OwnerAction "Import the owner-filled result only after real public publish evidence exists."
  New-ExecutionLane -Id "clean-external-package-consumer-owner-runbook" -Title "Execute clean external package consumer runbook" -RequiredInput "Repository-external consumer, selected package source, stdoutPath, stderrPath, mergedTranscriptPath, package/log SHA256 values, owner review, and nonSubstituteConfirmations." -Validator "Test-CleanExternalPackageConsumerOwnerRunbook.ps1 -Strict" -OwnerAction "Use the runbook to collect package-consumer runtime proof inputs without treating the runbook itself as proof."
  New-ExecutionLane -Id "post-publish-owner-verification-runbook" -Title "Execute post-publish owner verification runbook" -RequiredInput "Real public package URL, public package source URL, downloaded nupkg SHA256, repository-external post-publish clean consumer, logs, hashes, host metadata, and owner review." -Validator "Test-PostPublishOwnerVerificationRunbook.ps1 -Strict" -OwnerAction "Run post-publish verification only after real publication; this lane does not run dotnet nuget push."
  New-ExecutionLane -Id "post-publish-clean-consumer" -Title "Run clean consumer from public package source" -RequiredInput "Clean consumer outside repo, public package sources, no ProjectReference, restore/build/smoke logs, and SHA256 values." -Validator "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" -OwnerAction "Run clean consumer restore/build/runtime smoke against the public package source on a compatible host."
  New-ExecutionLane -Id "post-publish-proof-owner-input" -Title "Backfill post-publish clean consumer proof" -RequiredInput "Published package URLs, downloaded package hashes, host metadata, commands, logs, and reviewed stdout/stderr summaries." -Validator "Test-PostPublishVerificationOwnerInput.ps1 -Strict" -OwnerAction "Fill owner post-publish proof input from real clean consumer execution results."
  New-ExecutionLane -Id "strict-close-record" -Title "Validate strict release issue close record" -RequiredInput "Real public package proof, post-publish proof, evidence bundle hash, rollback plan, stale claim audit, and owner final close decision." -Validator "Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady" -OwnerAction "Run strict close validator only after all real proof gates have passed."
  New-ExecutionLane -Id "final-owner-decision" -Title "Owner makes final close decision" -RequiredInput "Release issue URL, final owner decision, known limitation acknowledgement, rollback owner, and evidence bundle hash." -Validator "Test-ReleaseIssueCloseOwnerDecisionInput.ps1 -Strict" -OwnerAction "Fill the final close decision input after strict close readiness is proven."
)

$sourceStates = @(
  [pscustomobject]@{ id = "public-publish-owner-manual-command-handoff"; state = [string](Get-PropertyOrDefault -Object $ownerManualCommandHandoffValidation -Name "validationState" -DefaultValue "missing-public-publish-owner-manual-command-handoff-validation") }
  [pscustomobject]@{ id = "public-publish-result-owner-input"; state = [string](Get-PropertyOrDefault -Object $publicPublishResultOwnerInputValidation -Name "validationState" -DefaultValue "missing-public-publish-result-owner-input-validation") }
  [pscustomobject]@{ id = "public-publish-result-import"; state = [string](Get-PropertyOrDefault -Object $publicPublishResultImportValidation -Name "validationState" -DefaultValue "missing-public-publish-result-import-validation") }
  [pscustomobject]@{ id = "clean-external-package-consumer-owner-runbook"; state = [string](Get-PropertyOrDefault -Object $cleanExternalRunbookValidation -Name "validationState" -DefaultValue "missing-clean-external-package-consumer-owner-runbook-validation") }
  [pscustomobject]@{ id = "post-publish-owner-verification-runbook"; state = [string](Get-PropertyOrDefault -Object $postPublishRunbookValidation -Name "validationState" -DefaultValue "missing-post-publish-owner-verification-runbook-validation") }
  [pscustomobject]@{ id = "post-publish-clean-consumer-result-convergence"; state = [string](Get-PropertyOrDefault -Object $postPublishCleanConsumerConvergenceValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-result-convergence-validation") }
  [pscustomobject]@{ id = "strict-close-ready-convergence-dashboard"; state = [string](Get-PropertyOrDefault -Object $strictCloseReadyValidation -Name "validationState" -DefaultValue "missing-strict-close-ready-convergence-dashboard-validation") }
  [pscustomobject]@{ id = "release-evidence-classification-audit"; state = [string](Get-PropertyOrDefault -Object $releaseEvidenceClassificationAudit -Name "auditState" -DefaultValue "missing-release-evidence-classification-audit") }
)

$record = [pscustomobject]@{
  recordKind = "public-publish-final-owner-execution-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  executionPackState = "blocked-public-publish-final-owner-execution-required"
  runtimePackageKey = $RuntimePackageKey
  executionLaneCount = $executionLanes.Count
  blockedExecutionLaneCount = @($executionLanes | Where-Object { -not [bool]$_.ready }).Count
  executionLanes = $executionLanes
  sourceStates = $sourceStates
  manualCommandPolicy = [pscustomobject]@{
    notExecutedByAutomation = $true
    ownerExecutionOnly = $true
    materializedExecutableCommand = ""
    commandTemplateBoundary = "Commands are represented as owner-reviewed placeholders only. Automation must not run package push or release close commands."
  }
  requiredOwnerInputs = @(
    "ownerName",
    "selectedChannel",
    "managedPackageId",
    "managedPackageVersion",
    "runtimePackageId",
    "runtimePackageVersion",
    "runtimePackageKey",
    "publishedPackageUrl",
    "managedPackageSha256",
    "runtimePackageSha256",
    "publishTranscriptPath",
    "publishTranscriptSha256",
    "cleanConsumerRootOutsideRepository",
    "repositoryExternalConsumerRoot",
    "publicPackageSourceUrl",
    "downloadedNupkgSha256",
    "stdoutPath",
    "stderrPath",
    "mergedTranscriptPath",
    "nonSubstituteConfirmations",
    "postPublishSmokeLogSha256",
    "releaseIssueUrl",
    "rollbackPlan",
    "ownerFinalCloseDecision"
  )
  sourceArtifacts = @(
    "artifacts/final-release/public-publish-owner-manual-command-handoff-validation.json",
    "artifacts/final-release/public-publish-result-owner-input-validation.json",
    "artifacts/final-release/public-publish-result-import-validation.json",
    "artifacts/final-release/clean-external-package-consumer-owner-runbook.json",
    "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json",
    "artifacts/final-release/post-publish-owner-verification-runbook.json",
    "artifacts/final-release/post-publish-owner-verification-runbook-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-result-convergence-validation.json",
    "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json",
    "artifacts/final-release/release-evidence-classification-audit.json"
  )
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Public publish final owner execution pack is a blocked handoff package only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "public-publish-final-owner-execution-pack.json"
$markdownPath = Join-Path $artifactRoot "public-publish-final-owner-execution-pack.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$laneRows = $record.executionLanes | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | $(ConvertTo-MarkdownCell $_.status) | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.validator) |"
}

$stateRows = $record.sourceStates | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | $(ConvertTo-MarkdownCell $_.state) |"
}

$markdown = @"
# Public Publish Final Owner Execution Pack

生成时间：$($record.generatedAtUtc)

该执行包只整理 Owner 公开发布前后的最终人工动作、输入字段和 validator 顺序。它不执行真实发布、不上传包、不关闭 release issue。

| 项目 | 当前值 |
|---|---|
| executionPackState | ``$($record.executionPackState)`` |
| runtimePackageKey | ``$($record.runtimePackageKey)`` |
| executionLaneCount | ``$($record.executionLaneCount)`` |
| blockedExecutionLaneCount | ``$($record.blockedExecutionLaneCount)`` |
| notExecutedByAutomation | ``$($record.notExecutedByAutomation)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Source States

| Source | State |
|---|---|
$($stateRows -join "`r`n")

## Execution Lanes

| Lane | Status | Ready | Validator |
|---|---|---:|---|
$($laneRows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public publish final owner execution pack written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ExecutionPackState=$($record.executionPackState) Lanes=$($record.executionLaneCount) Blocked=$($record.blockedExecutionLaneCount)"
