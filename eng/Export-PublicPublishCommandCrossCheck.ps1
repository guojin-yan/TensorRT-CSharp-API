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

function New-CrossCheck {
  param([string]$Id, [string]$Expected, [string]$Observed, [string]$OwnerAction)
  [pscustomobject]@{
    id = $Id
    expected = $Expected
    observed = $Observed
    status = "owner-action-required"
    passed = $false
    ownerAction = $OwnerAction
    boundary = "Cross-check item only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$ownerInput = Read-JsonOrNull "artifacts\final-release\public-publish-result-owner-input.template.json"
$manualHandoff = Read-JsonOrNull "artifacts\final-release\public-publish-owner-manual-command-handoff.json"
$finalPack = Read-JsonOrNull "artifacts\final-release\public-publish-final-owner-execution-pack.json"
$cleanExternalRunbookValidation = Read-JsonOrNull "artifacts\final-release\clean-external-package-consumer-owner-runbook-validation.json"
$postPublishRunbookValidation = Read-JsonOrNull "artifacts\final-release\post-publish-owner-verification-runbook-validation.json"

$packageId = [string](Get-PropertyOrDefault -Object $ownerInput -Name "packageId" -DefaultValue "JYPPX.TensorRT.CSharp.API")
$packageVersion = [string](Get-PropertyOrDefault -Object $ownerInput -Name "packageVersion" -DefaultValue "<owner-fill-managed-package-version>")
$selectedChannel = [string](Get-PropertyOrDefault -Object $ownerInput -Name "selectedChannel" -DefaultValue "<owner-fill-selected-channel>")

$crossChecks = @(
  New-CrossCheck -Id "managed-package-id" -Expected "JYPPX.TensorRT.CSharp.API" -Observed $packageId -OwnerAction "Confirm the managed package id before any owner manual publish command is used."
  New-CrossCheck -Id "managed-package-version" -Expected "<owner-approved-public-version>" -Observed $packageVersion -OwnerAction "Fill the exact owner-approved public version."
  New-CrossCheck -Id "runtime-package-key" -Expected $RuntimePackageKey -Observed ([string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimePackageKey" -DefaultValue $RuntimePackageKey)) -OwnerAction "Confirm the runtime package key matches the selected package lane."
  New-CrossCheck -Id "selected-channel" -Expected "nuget.org or GitHub Packages selected by owner" -Observed $selectedChannel -OwnerAction "Select the public package channel and record the source URL."
  New-CrossCheck -Id "managed-package-url" -Expected "absolute public package URL" -Observed ([string](Get-PropertyOrDefault -Object $ownerInput -Name "nugetPackageUrl" -DefaultValue "<owner-fill-public-url>")) -OwnerAction "Fill the public package URL after owner manual publish."
  New-CrossCheck -Id "managed-package-sha256" -Expected "64-character SHA256 from downloaded public package" -Observed ([string](Get-PropertyOrDefault -Object $ownerInput -Name "managedNupkgSha256" -DefaultValue "<owner-fill-managed-sha256>")) -OwnerAction "Download from selected channel and record SHA256."
  New-CrossCheck -Id "publish-transcript-sha256" -Expected "64-character SHA256 from owner command transcript" -Observed ([string](Get-PropertyOrDefault -Object $ownerInput -Name "publishCommandTranscriptSha256" -DefaultValue "<owner-fill-transcript-sha256>")) -OwnerAction "Capture owner manual command transcript and hash it."
  New-CrossCheck -Id "clean-external-package-consumer-runbook" -Expected "validated runbook plus real repository-external execution result import" -Observed ([string](Get-PropertyOrDefault -Object $cleanExternalRunbookValidation -Name "validationState" -DefaultValue "missing-clean-external-package-consumer-owner-runbook-validation")) -OwnerAction "Use the clean external package consumer runbook only as guidance; real stdoutPath, stderrPath, mergedTranscriptPath, SHA256 values, and nonSubstituteConfirmations must still be imported."
  New-CrossCheck -Id "post-publish-owner-verification-runbook" -Expected "validated post-publish runbook plus real public package source proof" -Observed ([string](Get-PropertyOrDefault -Object $postPublishRunbookValidation -Name "validationState" -DefaultValue "missing-post-publish-owner-verification-runbook-validation")) -OwnerAction "Run post-publish verification only after owner manual publish; record public package source URL, downloaded nupkg SHA256, public package URL, logs, hashes, and owner review."
  New-CrossCheck -Id "rollback-plan" -Expected "owner-reviewed rollback plan" -Observed "<owner-fill-rollback-plan>" -OwnerAction "Confirm rollback owner, trigger, and recovery channel before publish."
  New-CrossCheck -Id "credential-handling" -Expected "owner acknowledges credentials are handled outside repository and automation" -Observed "<owner-fill-credential-handling-acknowledgement>" -OwnerAction "Confirm no credentials are stored in artifacts, logs, or generated docs."
)

$record = [pscustomobject]@{
  recordKind = "public-publish-command-cross-check"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  crossCheckState = "blocked-public-publish-command-cross-check-owner-action-required"
  crossCheckCount = $crossChecks.Count
  blockedCrossCheckCount = @($crossChecks | Where-Object { -not [bool]$_.passed }).Count
  crossChecks = $crossChecks
  sourceStates = @(
    [pscustomobject]@{ id = "public-publish-result-owner-input"; state = [string](Get-PropertyOrDefault -Object $ownerInput -Name "templateState" -DefaultValue "missing-public-publish-result-owner-input") }
    [pscustomobject]@{ id = "public-publish-owner-manual-command-handoff"; state = [string](Get-PropertyOrDefault -Object $manualHandoff -Name "handoffState" -DefaultValue "missing-public-publish-owner-manual-command-handoff") }
    [pscustomobject]@{ id = "public-publish-final-owner-execution-pack"; state = [string](Get-PropertyOrDefault -Object $finalPack -Name "executionPackState" -DefaultValue "missing-public-publish-final-owner-execution-pack") }
    [pscustomobject]@{ id = "clean-external-package-consumer-owner-runbook"; state = [string](Get-PropertyOrDefault -Object $cleanExternalRunbookValidation -Name "validationState" -DefaultValue "missing-clean-external-package-consumer-owner-runbook-validation") }
    [pscustomobject]@{ id = "post-publish-owner-verification-runbook"; state = [string](Get-PropertyOrDefault -Object $postPublishRunbookValidation -Name "validationState" -DefaultValue "missing-post-publish-owner-verification-runbook-validation") }
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
  safetyBoundary = "Public publish command cross-check is a blocked owner review artifact only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. The post-publish owner verification runbook does not run dotnet nuget push and cannot replace real public package source URL, downloaded nupkg SHA256, logs, hashes, or owner review."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "public-publish-command-cross-check.json"
$markdownPath = Join-Path $artifactRoot "public-publish-command-cross-check.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $record.crossChecks | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | $(ConvertTo-MarkdownCell $_.status) | $(ConvertTo-MarkdownCell $_.observed) | $(ConvertTo-MarkdownCell $_.ownerAction) |"
}

$markdown = @"
# Public Publish Command Cross Check

| 项目 | 当前值 |
|---|---|
| crossCheckState | ``$($record.crossCheckState)`` |
| crossCheckCount | ``$($record.crossCheckCount)`` |
| blockedCrossCheckCount | ``$($record.blockedCrossCheckCount)`` |
| notExecutedByAutomation | ``$($record.notExecutedByAutomation)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Cross Checks

| ID | Status | Observed | Owner Action |
|---|---|---|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public publish command cross-check written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "CrossCheckState=$($record.crossCheckState) CrossChecks=$($record.crossCheckCount) Blocked=$($record.blockedCrossCheckCount)"
