[CmdletBinding()]
param(
  [string]$SourceQualityRunEvidenceImportPath = "artifacts\final-release\github-actions-source-quality-run-evidence-import.json",
  [string]$PackageDryRunEvidenceImportPath = "artifacts\final-release\github-actions-run-evidence-import.json",
  [string]$OutputPath = "artifacts\final-release\current-head-package-dry-run-preflight.json",
  [string]$MarkdownOutputPath = "artifacts\final-release\current-head-package-dry-run-preflight.md",
  [string]$CurrentHead,
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

  if ([System.IO.Path]::IsPathRooted($Path)) {
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

function Get-GitHeadOrEmpty {
  if (-not [string]::IsNullOrWhiteSpace($CurrentHead)) {
    return $CurrentHead.Trim()
  }

  try {
    $head = (& git -C $RepositoryRoot rev-parse HEAD 2>$null)
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($head)) {
      return ([string]$head).Trim()
    }
  }
  catch {
  }

  return ""
}

function Test-SameSha {
  param(
    [string]$Left,
    [string]$Right
  )

  return -not [string]::IsNullOrWhiteSpace($Left) -and
    -not [string]::IsNullOrWhiteSpace($Right) -and
    $Left.Trim().Equals($Right.Trim(), [StringComparison]::OrdinalIgnoreCase)
}

function New-Check {
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

$currentHeadValue = Get-GitHeadOrEmpty
$sourceQualityEvidence = Read-JsonOrNull -Path $SourceQualityRunEvidenceImportPath
$packageDryRunEvidence = Read-JsonOrNull -Path $PackageDryRunEvidenceImportPath

$sourceQualityEvidencePresent = $null -ne $sourceQualityEvidence -and
  ([string](Get-PropertyOrDefault -Object $sourceQualityEvidence -Name "recordKind" -DefaultValue "")).Equals("github-actions-run-evidence-import", [StringComparison]::Ordinal)
$sourceQualityRunId = if ($sourceQualityEvidencePresent) { [string](Get-PropertyOrDefault -Object $sourceQualityEvidence -Name "runId" -DefaultValue "") } else { "" }
$sourceQualityRunUrl = if ($sourceQualityEvidencePresent) { [string](Get-PropertyOrDefault -Object $sourceQualityEvidence -Name "runUrl" -DefaultValue "") } else { "" }
$sourceQualityHeadSha = if ($sourceQualityEvidencePresent) { [string](Get-PropertyOrDefault -Object $sourceQualityEvidence -Name "headSha" -DefaultValue "") } else { "" }
$sourceQualityRunEvidenceReady = $sourceQualityEvidencePresent -and (
  (Get-BoolPropertyOrDefault -Object $sourceQualityEvidence -Name "canClaimGitHubActionsSourceQualityForRun" -DefaultValue $false) -or
  ([string](Get-PropertyOrDefault -Object $sourceQualityEvidence -Name "evidenceState" -DefaultValue "")).Equals("source-quality-run-evidence-ready", [StringComparison]::OrdinalIgnoreCase)
)
$sourceQualityHeadMatchesCurrentHead = Test-SameSha -Left $sourceQualityHeadSha -Right $currentHeadValue

$packageDryRunEvidencePresent = $null -ne $packageDryRunEvidence -and
  ([string](Get-PropertyOrDefault -Object $packageDryRunEvidence -Name "recordKind" -DefaultValue "")).Equals("github-actions-run-evidence-import", [StringComparison]::Ordinal)
$packageDryRunRunId = if ($packageDryRunEvidencePresent) { [string](Get-PropertyOrDefault -Object $packageDryRunEvidence -Name "runId" -DefaultValue "") } else { "" }
$packageDryRunRunUrl = if ($packageDryRunEvidencePresent) { [string](Get-PropertyOrDefault -Object $packageDryRunEvidence -Name "runUrl" -DefaultValue "") } else { "" }
$packageDryRunHeadSha = if ($packageDryRunEvidencePresent) { [string](Get-PropertyOrDefault -Object $packageDryRunEvidence -Name "headSha" -DefaultValue "") } else { "" }
$packageDryRunCanClaimPackForRun = $packageDryRunEvidencePresent -and
  (Get-BoolPropertyOrDefault -Object $packageDryRunEvidence -Name "canClaimGitHubActionsPackageDryRunPackForRun" -DefaultValue $false)
$packageDryRunHeadMatchesCurrentHead = Test-SameSha -Left $packageDryRunHeadSha -Right $currentHeadValue
$canClaimGitHubActionsPackageDryRunPackForCurrentHead = $packageDryRunEvidencePresent -and
  $packageDryRunCanClaimPackForRun -and
  $packageDryRunHeadMatchesCurrentHead
$packageDryRunRequiresOwnerAuthorization = -not $canClaimGitHubActionsPackageDryRunPackForCurrentHead

$state = if ($canClaimGitHubActionsPackageDryRunPackForCurrentHead) {
  "current-head-package-dry-run-ready"
}
elseif (-not $packageDryRunEvidencePresent) {
  "missing-current-head-package-dry-run-run"
}
elseif (-not $packageDryRunCanClaimPackForRun) {
  "package-dry-run-evidence-not-claimable"
}
else {
  "blocked-owner-authorization-required"
}

$blockedReason = switch ($state) {
  "current-head-package-dry-run-ready" { "none" }
  "missing-current-head-package-dry-run-run" { "No package dry-run evidence import is present for the current HEAD." }
  "package-dry-run-evidence-not-claimable" { "The package dry-run evidence cannot claim package-managed dry-run pack success." }
  default { "Existing package dry-run evidence is not for the current HEAD; Owner must explicitly authorize a new workflow_dispatch dry-run with publishing disabled." }
}

$checks = @(
  New-Check -Id "source-quality-run-evidence-present" -Passed $sourceQualityEvidencePresent -Severity "info" -Detail $SourceQualityRunEvidenceImportPath
  New-Check -Id "source-quality-run-evidence-ready" -Passed $sourceQualityRunEvidenceReady -Severity "info" -Detail "sourceQualityRunId=$sourceQualityRunId"
  New-Check -Id "source-quality-head-current" -Passed $sourceQualityHeadMatchesCurrentHead -Severity "info" -Detail "sourceHeadSha=$sourceQualityHeadSha; currentHead=$currentHeadValue"
  New-Check -Id "package-dry-run-evidence-present" -Passed $packageDryRunEvidencePresent -Severity "action-required" -Detail $PackageDryRunEvidenceImportPath
  New-Check -Id "package-dry-run-pack-claim-for-run" -Passed $packageDryRunCanClaimPackForRun -Severity "action-required" -Detail "packageDryRunRunId=$packageDryRunRunId"
  New-Check -Id "package-dry-run-head-current" -Passed $packageDryRunHeadMatchesCurrentHead -Severity "action-required" -Detail "packageDryRunHeadSha=$packageDryRunHeadSha; currentHead=$currentHeadValue"
  New-Check -Id "package-dry-run-current-head-claim-ready" -Passed $canClaimGitHubActionsPackageDryRunPackForCurrentHead -Severity "action-required" -Detail "Only true when dry-run pack claim and run head both match current HEAD."
  New-Check -Id "no-publish-side-effects" -Passed $true -Severity "blocker" -Detail "This preflight does not trigger workflow_dispatch, publish NuGet, publish GitHub Packages, or close the release issue."
)

$record = [pscustomobject]@{
  recordKind = "current-head-package-dry-run-preflight"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  state = $state
  currentHead = $currentHeadValue
  sourceQualityRunEvidenceImportPath = $SourceQualityRunEvidenceImportPath
  sourceQualityRunId = $sourceQualityRunId
  sourceQualityRunUrl = $sourceQualityRunUrl
  sourceQualityHeadSha = $sourceQualityHeadSha
  sourceQualityRunEvidencePresent = $sourceQualityEvidencePresent
  sourceQualityRunEvidenceReady = $sourceQualityRunEvidenceReady
  sourceQualityHeadMatchesCurrentHead = $sourceQualityHeadMatchesCurrentHead
  packageDryRunEvidenceImportPath = $PackageDryRunEvidenceImportPath
  packageDryRunRunId = $packageDryRunRunId
  packageDryRunRunUrl = $packageDryRunRunUrl
  packageDryRunHeadSha = $packageDryRunHeadSha
  packageDryRunEvidencePresent = $packageDryRunEvidencePresent
  packageDryRunCanClaimPackForRun = $packageDryRunCanClaimPackForRun
  packageDryRunHeadMatchesCurrentHead = $packageDryRunHeadMatchesCurrentHead
  canClaimGitHubActionsPackageDryRunPackForCurrentHead = $canClaimGitHubActionsPackageDryRunPackForCurrentHead
  packageDryRunRequiresOwnerAuthorization = $packageDryRunRequiresOwnerAuthorization
  manualWorkflowDispatchNotPerformed = $true
  ownerAuthorizationState = if ($packageDryRunRequiresOwnerAuthorization) { "owner-authorization-required-before-workflow-dispatch" } else { "owner-authorization-not-required-for-existing-current-head-evidence" }
  blockedReason = $blockedReason
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  proofBoundary = "This preflight classifies whether existing imported package-managed dry-run evidence belongs to the current HEAD. It does not run workflow_dispatch, publish packages, validate package-consumer runtime, create post-publish proof, or close the release issue."
  checks = @($checks)
}

$resolvedOutputPath = Resolve-RepoPath -Path $OutputPath
$resolvedMarkdownPath = Resolve-RepoPath -Path $MarkdownOutputPath
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($resolvedOutputPath)) | Out-Null
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($resolvedMarkdownPath)) | Out-Null

$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $resolvedOutputPath -Encoding utf8

$checkRows = $record.checks | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Current HEAD Package Dry-Run Preflight

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| state | ``$($record.state)`` |
| currentHead | ``$($record.currentHead)`` |
| sourceQualityRunId | ``$($record.sourceQualityRunId)`` |
| sourceQualityHeadSha | ``$($record.sourceQualityHeadSha)`` |
| sourceQualityRunEvidenceReady | ``$($record.sourceQualityRunEvidenceReady)`` |
| packageDryRunRunId | ``$($record.packageDryRunRunId)`` |
| packageDryRunHeadSha | ``$($record.packageDryRunHeadSha)`` |
| packageDryRunCanClaimPackForRun | ``$($record.packageDryRunCanClaimPackForRun)`` |
| packageDryRunHeadMatchesCurrentHead | ``$($record.packageDryRunHeadMatchesCurrentHead)`` |
| canClaimGitHubActionsPackageDryRunPackForCurrentHead | ``$($record.canClaimGitHubActionsPackageDryRunPackForCurrentHead)`` |
| packageDryRunRequiresOwnerAuthorization | ``$($record.packageDryRunRequiresOwnerAuthorization)`` |
| manualWorkflowDispatchNotPerformed | ``$($record.manualWorkflowDispatchNotPerformed)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Blocked Reason

$($record.blockedReason)

## Checks

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($checkRows -join "`r`n")

## Safety Boundary

$($record.proofBoundary)
"@

$markdown | Set-Content -LiteralPath $resolvedMarkdownPath -Encoding utf8

Write-Host "Current HEAD package dry-run preflight written:"
Write-Host "  Json=$resolvedOutputPath"
Write-Host "  Markdown=$resolvedMarkdownPath"
Write-Host "State=$($record.state) CanClaimCurrentHeadPackageDryRun=$($record.canClaimGitHubActionsPackageDryRunPackForCurrentHead) OwnerAuthorizationRequired=$($record.packageDryRunRequiresOwnerAuthorization)"
