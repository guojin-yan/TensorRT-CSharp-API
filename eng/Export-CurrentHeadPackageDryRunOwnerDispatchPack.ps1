[CmdletBinding()]
param(
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [string]$Ref = "TensorRtSharp4.0",
  [string]$CurrentHeadPackageDryRunPreflightPath = "artifacts\final-release\current-head-package-dry-run-preflight.json",
  [string]$OutputPath = "artifacts\final-release\current-head-package-dry-run-owner-dispatch-pack.json",
  [string]$MarkdownOutputPath = "artifacts\final-release\current-head-package-dry-run-owner-dispatch-pack.md",
  [string]$RepositoryRoot
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

function Resolve-RepoPath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return $Path
  }

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)

  $resolvedPath = Resolve-RepoPath -Path $Path
  if ([string]::IsNullOrWhiteSpace($resolvedPath) -or -not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$preflight = Read-JsonOrNull -Path $CurrentHeadPackageDryRunPreflightPath
$preflightPresent = $null -ne $preflight -and
  ([string](Get-PropertyOrDefault -Object $preflight -Name "recordKind" -DefaultValue "")).Equals("current-head-package-dry-run-preflight", [StringComparison]::Ordinal)
$preflightState = if ($preflightPresent) { [string](Get-PropertyOrDefault -Object $preflight -Name "state" -DefaultValue "missing-state") } else { "missing-current-head-package-dry-run-preflight" }
$currentHead = if ($preflightPresent) { [string](Get-PropertyOrDefault -Object $preflight -Name "currentHead" -DefaultValue "") } else { "" }
$sourceQualityRunId = if ($preflightPresent) { [string](Get-PropertyOrDefault -Object $preflight -Name "sourceQualityRunId" -DefaultValue "") } else { "" }
$packageDryRunRunId = if ($preflightPresent) { [string](Get-PropertyOrDefault -Object $preflight -Name "packageDryRunRunId" -DefaultValue "") } else { "" }
$currentHeadPackageDryRunReady = $preflightPresent -and (Get-BoolPropertyOrDefault -Object $preflight -Name "canClaimGitHubActionsPackageDryRunPackForCurrentHead" -DefaultValue $false)
$packageDryRunRequiresOwnerAuthorization = -not $currentHeadPackageDryRunReady
$blockedReason = if ($preflightPresent) {
  [string](Get-PropertyOrDefault -Object $preflight -Name "blockedReason" -DefaultValue "Current HEAD package dry-run proof is not ready.")
}
else {
  "Current HEAD package dry-run preflight has not been generated."
}

$dispatchInputs = [ordered]@{
  run_package_managed_dry_run = "true"
  run_release_artifact_audit = "false"
  run_split_package_build = "false"
}

$effectivePackageManagedInputs = [ordered]@{
  publish_to_nuget = $false
  publish_to_github_packages = $false
  attach_to_github_release = $false
  artifact_name = "package-managed-dry-run"
  release_tag = ""
}

$dispatchCommandParts = @(
  "gh",
  "workflow",
  "run",
  "release-quality-gate.yml",
  "--repo",
  $Repository,
  "--ref",
  $Ref
)
foreach ($entry in $dispatchInputs.GetEnumerator()) {
  $dispatchCommandParts += @("-f", "$($entry.Key)=$($entry.Value)")
}
$dispatchCommand = $dispatchCommandParts -join " "

$ownerChecklist = @(
  [pscustomobject]@{
    id = "confirm-current-head"
    required = $true
    detail = "Confirm currentHead equals the commit that should receive package-managed dry-run evidence."
  },
  [pscustomobject]@{
    id = "confirm-non-publish-flags"
    required = $true
    detail = "Confirm release-quality-gate dispatch only enables run_package_managed_dry_run and keeps publish_to_nuget, publish_to_github_packages, and attach_to_github_release false through the reusable package-managed workflow."
  },
  [pscustomobject]@{
    id = "execute-command-manually"
    required = $true
    detail = "Owner may copy and run dispatchCommand manually; this pack never executes it."
  },
  [pscustomobject]@{
    id = "import-result-after-completion"
    required = $true
    detail = "After the run completes, download artifacts and re-run GitHub Actions evidence import for the new run."
  }
)

$record = [pscustomobject]@{
  recordKind = "current-head-package-dry-run-owner-dispatch-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  packState = if ($currentHeadPackageDryRunReady) { "current-head-package-dry-run-already-ready" } else { "owner-authorization-required-before-workflow-dispatch" }
  repository = $Repository
  ref = $Ref
  workflow = "release-quality-gate.yml"
  currentHeadPackageDryRunPreflightPath = $CurrentHeadPackageDryRunPreflightPath
  currentHeadPackageDryRunPreflightPresent = $preflightPresent
  currentHeadPackageDryRunPreflightState = $preflightState
  currentHead = $currentHead
  sourceQualityRunId = $sourceQualityRunId
  packageDryRunRunId = $packageDryRunRunId
  currentHeadPackageDryRunReady = $currentHeadPackageDryRunReady
  canClaimGitHubActionsPackageDryRunPackForCurrentHead = $currentHeadPackageDryRunReady
  requiresOwnerAuthorization = $packageDryRunRequiresOwnerAuthorization
  workflowDispatchExecuted = $false
  dispatchCommand = $dispatchCommand
  dispatchInputs = [pscustomobject]$dispatchInputs
  effectivePackageManagedInputs = [pscustomobject]$effectivePackageManagedInputs
  blockedReason = $blockedReason
  ownerChecklist = @($ownerChecklist)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteProof = $false
  isPackageDryRunProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  forbiddenClaims = @(
    "dispatch pack is package proof",
    "dispatch command was executed",
    "queued workflow is proof",
    "source-only run is package dry-run proof",
    "old package dry-run run is current-head proof",
    "dry-run nupkg is public package proof"
  )
  sourceArtifacts = @(
    $CurrentHeadPackageDryRunPreflightPath,
    "artifacts/final-release/github-actions-source-quality-run-evidence-import.json",
    "artifacts/final-release/github-actions-source-quality-run-evidence-validation/github-actions-run-evidence-import-validation.json",
    "artifacts/final-release/github-actions-run-evidence-import.json"
  )
  safetyBoundary = "This owner dispatch pack is command text and checklist only. It never calls gh workflow run, never publishes NuGet, never publishes GitHub Packages, never uploads release assets, never closes a release issue, and cannot promote package dry-run, package-consumer runtime, or post-publish proof."
}

$resolvedOutputPath = Resolve-RepoPath -Path $OutputPath
$resolvedMarkdownOutputPath = Resolve-RepoPath -Path $MarkdownOutputPath
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($resolvedOutputPath)) | Out-Null
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($resolvedMarkdownOutputPath)) | Out-Null

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $resolvedOutputPath -Encoding utf8

$inputRows = $dispatchInputs.GetEnumerator() | ForEach-Object {
  "| ``$(ConvertTo-MarkdownCell $_.Key)`` | ``$(ConvertTo-MarkdownCell $_.Value)`` |"
}
$effectiveRows = $effectivePackageManagedInputs.GetEnumerator() | ForEach-Object {
  "| ``$(ConvertTo-MarkdownCell $_.Key)`` | ``$(ConvertTo-MarkdownCell $_.Value)`` |"
}
$checkRows = $ownerChecklist | ForEach-Object {
  "| ``$(ConvertTo-MarkdownCell $_.id)`` | ``$($_.required)`` | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# Current HEAD Package Dry-Run Owner Dispatch Pack

生成时间：$($record.generatedAtUtc)

## Summary

| 项目 | 当前值 |
|---|---|
| packState | ``$($record.packState)`` |
| currentHead | ``$($record.currentHead)`` |
| sourceQualityRunId | ``$($record.sourceQualityRunId)`` |
| oldPackageDryRunRunId | ``$($record.packageDryRunRunId)`` |
| currentHeadPackageDryRunReady | ``$($record.currentHeadPackageDryRunReady)`` |
| requiresOwnerAuthorization | ``$($record.requiresOwnerAuthorization)`` |
| workflowDispatchExecuted | ``$($record.workflowDispatchExecuted)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |
| isPackageConsumerRuntimeProof | ``$($record.isPackageConsumerRuntimeProof)`` |
| isPostPublishProof | ``$($record.isPostPublishProof)`` |

## Dispatch Command

```powershell
$($record.dispatchCommand)
```

## Dispatch Inputs

| Input | Value |
|---|---|
$($inputRows -join "`r`n")

## Effective Package-Managed Inputs

These values are fixed by `.github/workflows/release-quality-gate.yml` when it calls `.github/workflows/package-managed.yml`.

| Input | Value |
|---|---|
$($effectiveRows -join "`r`n")

## Owner Checklist

| ID | Required | Detail |
|---|---|---|
$($checkRows -join "`r`n")

## Blocked Reason

$($record.blockedReason)

## Safety Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $resolvedMarkdownOutputPath -Encoding utf8

Write-Host "Current HEAD package dry-run owner dispatch pack written:"
Write-Host "  Json=$resolvedOutputPath"
Write-Host "  Markdown=$resolvedMarkdownOutputPath"
Write-Host "PackState=$($record.packState) WorkflowDispatchExecuted=False PerformsPublish=False"
