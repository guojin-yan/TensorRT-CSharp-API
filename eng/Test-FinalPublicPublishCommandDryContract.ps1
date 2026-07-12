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
if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $OutputDirectory "final-public-publish-command-dry-contract.json" }

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
  & (Join-Path $PSScriptRoot "Export-FinalPublicPublishCommandDryContract.ps1") -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = $record | ConvertTo-Json -Depth 12
$failedBlockerCount = 0

if ($record.recordKind -ne "final-public-publish-command-dry-contract") { $failedBlockerCount++ }
if (-not [bool]$record.isDryContract) { $failedBlockerCount++ }
if ([bool]$record.performsPublish -or [bool]$record.canPublishPublicly -or [bool]$record.canCloseReleaseIssue -or [bool]$record.isReleaseReady) { $failedBlockerCount++ }
if ([bool]$record.storesToken -or [bool]$record.executesDotnetNugetPush -or [bool]$record.executesGitHubReleaseUpload -or [bool]$record.closesReleaseIssue) { $failedBlockerCount++ }
if ($raw -notmatch "dotnet nuget push" -or $raw -notmatch "gh release upload") { $failedBlockerCount++ }
if ($raw -notmatch "<OWNER_SUPPLIED_TOKEN>" -or $raw -match "(?i)api[_-]?key\s*=\s*[A-Za-z0-9]") { $failedBlockerCount++ }

foreach ($group in @($record.commandGroups)) {
  if ([bool]$group.performsPublish -or [bool]$group.canExecuteInAutomation) { $failedBlockerCount++ }
}

$validation = [pscustomobject]@{
  recordKind = "final-public-publish-command-dry-contract-validation"
  validationState = if ($failedBlockerCount -eq 0) { "blocked-public-publish-owner-manual-execution-required" } else { "failed-final-public-publish-command-dry-contract" }
  contractState = [string]$record.contractState
  isDryContract = [bool]$record.isDryContract
  commandGroupCount = [int]$record.commandGroupCount
  failedBlockerCount = $failedBlockerCount
  failedActionRequiredCount = [int]$record.commandGroupCount
  performsPublish = [bool]$record.performsPublish
  canPublishPublicly = [bool]$record.canPublishPublicly
  canCloseReleaseIssue = [bool]$record.canCloseReleaseIssue
  isReleaseReady = [bool]$record.isReleaseReady
  executesDotnetNugetPush = [bool]$record.executesDotnetNugetPush
  storesToken = [bool]$record.storesToken
  boundary = "Dry contract validation confirms command placeholders are non-executing and non-proof."
}

Write-Utf8FileWithRetry -LiteralPath (Join-Path $OutputDirectory "final-public-publish-command-dry-contract-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8FileWithRetry -LiteralPath (Join-Path $OutputDirectory "final-public-publish-command-dry-contract-validation.md") -InputObject @(
  "# Final Public Publish Command Dry Contract Validation",
  "",
  "- validationState: ``$($validation.validationState)``",
  "- isDryContract: ``$($validation.isDryContract)``",
  "- commandGroupCount: ``$($validation.commandGroupCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- failedActionRequiredCount: ``$($validation.failedActionRequiredCount)``",
  "",
  "> $($validation.boundary)"
)

Write-Host "ValidationState=$($validation.validationState)"
Write-Host "FailedBlockerCount=$($validation.failedBlockerCount)"
if ($Strict -and $failedBlockerCount -gt 0) { throw "Final public publish command dry contract validation failed." }
