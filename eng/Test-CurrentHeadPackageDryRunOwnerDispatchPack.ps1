[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\current-head-package-dry-run-owner-dispatch-pack.json",
  [string]$OutputPath = "artifacts\final-release\current-head-package-dry-run-owner-dispatch-pack-validation.json",
  [string]$MarkdownOutputPath = "artifacts\final-release\current-head-package-dry-run-owner-dispatch-pack-validation.md",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepoPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Get-BoolPropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [bool]$DefaultValue
  )

  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue $DefaultValue
  if ($value -is [bool]) {
    return [bool]$value
  }

  $parsed = $false
  if ([bool]::TryParse(([string]$value).Trim(), [ref]$parsed)) {
    return $parsed
  }

  return $DefaultValue
}

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$resolvedInputPath = Resolve-RepoPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner dispatch pack not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$command = [string](Get-PropertyOrDefault -Object $record -Name "dispatchCommand" -DefaultValue "")
$effectiveInputs = Get-PropertyOrDefault -Object $record -Name "effectivePackageManagedInputs" -DefaultValue $null

$items = @(
  New-ValidationItem -Id "record-kind" -Passed (([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")) -eq "current-head-package-dry-run-owner-dispatch-pack") -Severity "blocker" -Detail "recordKind must be current-head-package-dry-run-owner-dispatch-pack."
  New-ValidationItem -Id "command-targets-release-quality-gate" -Passed ($command.Contains("gh workflow run release-quality-gate.yml", [StringComparison]::Ordinal) -and $command.Contains("--ref TensorRtSharp4.0", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "dispatchCommand must target release-quality-gate.yml on TensorRtSharp4.0."
  New-ValidationItem -Id "dry-run-enabled-only" -Passed ($command.Contains("-f run_package_managed_dry_run=true", [StringComparison]::Ordinal) -and $command.Contains("-f run_release_artifact_audit=false", [StringComparison]::Ordinal) -and $command.Contains("-f run_split_package_build=false", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "Command must enable only package-managed dry-run."
  New-ValidationItem -Id "publish-flags-disabled" -Passed ((-not (Get-BoolPropertyOrDefault -Object $effectiveInputs -Name "publish_to_nuget" -DefaultValue $true)) -and (-not (Get-BoolPropertyOrDefault -Object $effectiveInputs -Name "publish_to_github_packages" -DefaultValue $true)) -and (-not (Get-BoolPropertyOrDefault -Object $effectiveInputs -Name "attach_to_github_release" -DefaultValue $true))) -Severity "blocker" -Detail "Effective package-managed publish flags must be false."
  New-ValidationItem -Id "no-command-execution" -Passed (-not (Get-BoolPropertyOrDefault -Object $record -Name "workflowDispatchExecuted" -DefaultValue $true)) -Severity "blocker" -Detail "Pack must only generate command text, not execute workflow_dispatch."
  New-ValidationItem -Id "owner-authorization-required" -Passed (Get-BoolPropertyOrDefault -Object $record -Name "requiresOwnerAuthorization" -DefaultValue $false) -Severity "action-required" -Detail "Owner must explicitly authorize workflow_dispatch before any dry-run execution."
  New-ValidationItem -Id "no-publish-close-or-proof-promotion" -Passed ((-not (Get-BoolPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not (Get-BoolPropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true)) -and (-not (Get-BoolPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not (Get-BoolPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not (Get-BoolPropertyOrDefault -Object $record -Name "canPromoteProof" -DefaultValue $true)) -and (-not (Get-BoolPropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true)) -and (-not (Get-BoolPropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true))) -Severity "blocker" -Detail "Dispatch pack cannot publish, close, or promote proof."
  New-ValidationItem -Id "does-not-embed-publish-command" -Passed ((-not $command.Contains("dotnet nuget push", [StringComparison]::OrdinalIgnoreCase)) -and (-not $command.Contains("Push-NuGetPackages", [StringComparison]::Ordinal)) -and (-not $command.Contains("gh release upload", [StringComparison]::OrdinalIgnoreCase))) -Severity "blocker" -Detail "Dispatch command must not include any publish or release upload command."
)

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "current-head-package-dry-run-dispatch-pack-ready-for-owner"
}
elseif ($failedBlockers.Count -eq 0) {
  "blocked-owner-authorization-required"
}
else {
  "blocked-invalid-dispatch-pack"
}

$validation = [pscustomobject]@{
  recordKind = "current-head-package-dry-run-owner-dispatch-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  workflowDispatchExecuted = $false
  requiresOwnerAuthorization = Get-BoolPropertyOrDefault -Object $record -Name "requiresOwnerAuthorization" -DefaultValue $false
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  validationItems = @($items)
  safetyBoundary = "Validation confirms dispatch pack shape only. It does not run workflow_dispatch and cannot promote package, runtime, post-publish, or close proof."
}

$resolvedOutputPath = Resolve-RepoPath -Path $OutputPath
$resolvedMarkdownOutputPath = Resolve-RepoPath -Path $MarkdownOutputPath
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($resolvedOutputPath)) | Out-Null

$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $resolvedOutputPath -Encoding utf8
$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}
$markdown = @"
# Current HEAD Package Dry-Run Owner Dispatch Pack Validation

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| workflowDispatchExecuted | ``$($validation.workflowDispatchExecuted)`` |
| requiresOwnerAuthorization | ``$($validation.requiresOwnerAuthorization)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $resolvedMarkdownOutputPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Current HEAD package dry-run owner dispatch pack validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Current HEAD package dry-run owner dispatch pack validation written:"
Write-Host "  Json=$resolvedOutputPath"
Write-Host "  Markdown=$resolvedMarkdownOutputPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"
