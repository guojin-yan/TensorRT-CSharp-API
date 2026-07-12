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

$commandGroups = @(
  [pscustomobject]@{
    id = "nuget-push-placeholder"
    title = "NuGet push placeholder command"
    command = "dotnet nuget push <PACKAGE_PATH> --api-key <OWNER_SUPPLIED_TOKEN> --source <PUBLIC_NUGET_SOURCE> --skip-duplicate"
    manualOnly = $true
    containsSecretPlaceholder = $true
    performsPublish = $false
    canExecuteInAutomation = $false
    requiredBeforeExecution = @("Owner approval", "public package path", "package SHA256", "token supplied outside repository", "dry contract replaced by manual execution record")
    boundary = "Placeholder only; this script never executes dotnet nuget push and does not store tokens."
  }
  [pscustomobject]@{
    id = "github-release-upload-placeholder"
    title = "GitHub release upload placeholder command"
    command = "gh release upload <TAG> <PACKAGE_PATH> --clobber --repo <OWNER_REPO>"
    manualOnly = $true
    containsSecretPlaceholder = $true
    performsPublish = $false
    canExecuteInAutomation = $false
    requiredBeforeExecution = @("Owner approval", "tag", "asset path", "asset SHA256", "GitHub token outside repository")
    boundary = "Placeholder only; it is not an upload proof and is not executed by automation."
  }
  [pscustomobject]@{
    id = "post-publish-clean-consumer-validation-sequence"
    title = "Post-publish clean consumer validation sequence"
    command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalPostPublishCleanConsumerProofCandidate.ps1 -Strict"
    manualOnly = $true
    containsSecretPlaceholder = $false
    performsPublish = $false
    canExecuteInAutomation = $false
    requiredBeforeExecution = @("real public package URL", "external clean consumer restore/build/run logs", "matching SHA256", "host identity", "package identity")
    boundary = "Validator command sequence only; it cannot substitute real public-channel post-publish proof."
  }
  [pscustomobject]@{
    id = "final-close-validation-sequence"
    title = "Final close validation sequence"
    command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseCloseOwnerApprovalCandidate.ps1 -Strict; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict"
    manualOnly = $true
    containsSecretPlaceholder = $false
    performsPublish = $false
    canExecuteInAutomation = $false
    requiredBeforeExecution = @("Owner release close approval", "release notes SHA256", "final public package URL/SHA256 approval", "post-publish proof candidate ready")
    boundary = "Final close validation sequence only; it does not close release issue."
  }
)

$record = [pscustomobject]@{
  recordKind = "final-public-publish-command-dry-contract"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("O")
  contractState = "blocked-public-publish-owner-manual-execution-required"
  isDryContract = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseReady = $false
  storesToken = $false
  executesDotnetNugetPush = $false
  executesGitHubReleaseUpload = $false
  closesReleaseIssue = $false
  commandGroupCount = $commandGroups.Count
  failedBlockerCount = 0
  failedActionRequiredCount = $commandGroups.Count
  commandGroups = @($commandGroups)
  sourceArtifacts = @(
    "artifacts/final-release/final-public-publish-pre-execution-freeze.json",
    "artifacts/final-release/final-public-publish-owner-action-worklist.json",
    "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json",
    "artifacts/final-release/final-release-close-owner-approval-candidate-validation.json"
  )
  boundary = "Final public publish command dry contract is a manual command contract only. It contains placeholders for future Owner execution, but performsPublish=false, canPublishPublicly=false, canCloseReleaseIssue=false, does not write tokens, does not execute dotnet nuget push, does not upload GitHub release assets, does not fake package URL/SHA256, and does not close release issue."
}

$jsonPath = Join-Path $OutputDirectory "final-public-publish-command-dry-contract.json"
$mdPath = Join-Path $OutputDirectory "final-public-publish-command-dry-contract.md"
Write-Utf8FileWithRetry -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Final Public Publish Command Dry Contract")
$lines.Add("")
$lines.Add("- contractState: ``$($record.contractState)``")
$lines.Add("- isDryContract: ``$($record.isDryContract)``")
$lines.Add("- performsPublish: ``$($record.performsPublish)``")
$lines.Add("- canPublishPublicly: ``$($record.canPublishPublicly)``")
$lines.Add("- canCloseReleaseIssue: ``$($record.canCloseReleaseIssue)``")
$lines.Add("- executesDotnetNugetPush: ``$($record.executesDotnetNugetPush)``")
$lines.Add("")
$lines.Add("| ID | Command | Boundary |")
$lines.Add("| --- | --- | --- |")
foreach ($group in $commandGroups) {
  $lines.Add("| $($group.id) | ``$(ConvertTo-MarkdownCell $group.command)`` | $(ConvertTo-MarkdownCell $group.boundary) |")
}
$lines.Add("")
$lines.Add("> $($record.boundary)")
Write-Utf8FileWithRetry -LiteralPath $mdPath -InputObject $lines

Write-Host "ContractState=$($record.contractState)"
Write-Host "IsDryContract=$($record.isDryContract)"
Write-Host "PerformsPublish=$($record.performsPublish)"
