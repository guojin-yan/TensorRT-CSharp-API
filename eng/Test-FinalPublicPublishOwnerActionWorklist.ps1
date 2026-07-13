[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$InputPath,
  [string]$OutputDirectory,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { $OutputDirectory = Join-Path $RepositoryRoot "artifacts\final-release" }
if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $OutputDirectory "final-public-publish-owner-action-worklist.json" }

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8FileWithRetry {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $directory = Split-Path -Parent $LiteralPath
  if (-not [string]::IsNullOrWhiteSpace($directory)) { New-Item -ItemType Directory -Path $directory -Force | Out-Null }
  [IO.File]::WriteAllText($LiteralPath, (@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine, $script:utf8)
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Export-FinalPublicPublishOwnerActionWorklist.ps1") -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$actions = @($record.actionItems)
$failedBlockerCount = 0
$failedActionRequiredCount = 0

if ($record.recordKind -ne "final-public-publish-owner-action-worklist") { $failedBlockerCount++ }
if ($record.worklistState -ne "blocked-public-publish-owner-action-required") { $failedBlockerCount++ }
if ([int]$record.ownerActionCount -lt 10) { $failedBlockerCount++ }
if ([bool]$record.performsPublish -or [bool]$record.canPublishPublicly -or [bool]$record.canCloseReleaseIssue -or [bool]$record.isReleaseReady) { $failedBlockerCount++ }

foreach ($action in $actions) {
  if ([string]::IsNullOrWhiteSpace([string]$action.ownerMustSupply) -or
      [string]::IsNullOrWhiteSpace([string]$action.expectedFileHash) -or
      [string]::IsNullOrWhiteSpace([string]$action.validatorCommand) -or
      [string]::IsNullOrWhiteSpace([string]$action.sourceArtifact) -or
      [string]::IsNullOrWhiteSpace([string]$action.whyNotProof)) {
    $failedBlockerCount++
  }

  if ([bool]$action.performsPublish -or [bool]$action.canPublishPublicly -or [bool]$action.canCloseReleaseIssue) {
    $failedBlockerCount++
  }

  $failedActionRequiredCount++
}

$validation = [pscustomobject]@{
  recordKind = "final-public-publish-owner-action-worklist-validation"
  validationState = if ($failedBlockerCount -eq 0) { "blocked-public-publish-owner-action-required" } else { "failed-final-public-publish-owner-action-worklist" }
  ownerActionCount = [int]$record.ownerActionCount
  blockedOwnerActionCount = [int]$record.blockedOwnerActionCount
  failedBlockerCount = $failedBlockerCount
  failedActionRequiredCount = $failedActionRequiredCount
  performsPublish = [bool]$record.performsPublish
  canPublishPublicly = [bool]$record.canPublishPublicly
  canCloseReleaseIssue = [bool]$record.canCloseReleaseIssue
  isReleaseReady = [bool]$record.isReleaseReady
  boundary = "Owner action worklist validation keeps all actions blocked until real Owner evidence and approvals are supplied."
}

Write-Utf8FileWithRetry -LiteralPath (Join-Path $OutputDirectory "final-public-publish-owner-action-worklist-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8FileWithRetry -LiteralPath (Join-Path $OutputDirectory "final-public-publish-owner-action-worklist-validation.md") -InputObject @(
  "# Final Public Publish Owner Action Worklist Validation",
  "",
  "- validationState: ``$($validation.validationState)``",
  "- ownerActionCount: ``$($validation.ownerActionCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- failedActionRequiredCount: ``$($validation.failedActionRequiredCount)``",
  "",
  "> $($validation.boundary)"
)

Write-Host "ValidationState=$($validation.validationState)"
Write-Host "OwnerActionCount=$($validation.ownerActionCount)"
if ($Strict -and $failedBlockerCount -gt 0) { throw "Final public publish owner action worklist validation failed." }
