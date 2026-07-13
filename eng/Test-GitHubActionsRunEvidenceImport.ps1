[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\github-actions-run-evidence-import.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
$record = $null
if (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf) {
  $record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
}

$runId = [string](Get-PropertyOrDefault -Object $record -Name "runId" -DefaultValue "")
$runUrl = [string](Get-PropertyOrDefault -Object $record -Name "runUrl" -DefaultValue "")
$headSha = [string](Get-PropertyOrDefault -Object $record -Name "headSha" -DefaultValue "")
$runConclusion = [string](Get-PropertyOrDefault -Object $record -Name "runConclusion" -DefaultValue "")
$sourceQualityConclusion = [string](Get-PropertyOrDefault -Object $record -Name "sourceQualityConclusion" -DefaultValue "")
$packagePackConclusion = [string](Get-PropertyOrDefault -Object $record -Name "packageManagedDryRunPackConclusion" -DefaultValue "")
$publishNugetConclusion = [string](Get-PropertyOrDefault -Object $record -Name "publishNugetConclusion" -DefaultValue "")
$publishGitHubPackagesConclusion = [string](Get-PropertyOrDefault -Object $record -Name "publishGitHubPackagesConclusion" -DefaultValue "")
$nupkgPackages = @((Get-PropertyOrDefault -Object $record -Name "nupkgPackages" -DefaultValue @()))
$importFailedBlockerCount = [int](Get-PropertyOrDefault -Object $record -Name "failedBlockerCount" -DefaultValue 999)
$canClaimDryRunPack = [bool](Get-PropertyOrDefault -Object $record -Name "canClaimGitHubActionsPackageDryRunPackForRun" -DefaultValue $false)
$workflowRunLogSha256 = [string](Get-PropertyOrDefault -Object $record -Name "workflowRunLogSha256" -DefaultValue "")
$artifactManifestSha256 = [string](Get-PropertyOrDefault -Object $record -Name "artifactManifestSha256" -DefaultValue "")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "input-present" -Passed ($null -ne $record) -Severity "action-required" -Detail "Owner must import a real GitHub Actions run evidence artifact before this lane can be ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "record-kind" -Passed ($null -eq $record -or [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "github-actions-run-evidence-import") -Severity "blocker" -Detail "recordKind must be github-actions-run-evidence-import when input is present.")) | Out-Null
$items.Add((New-ValidationItem -Id "run-id-present" -Passed (-not [string]::IsNullOrWhiteSpace($runId)) -Severity "action-required" -Detail "runId must identify the GitHub Actions workflow run.")) | Out-Null
$items.Add((New-ValidationItem -Id "run-url-public" -Passed ($runUrl.StartsWith("https://github.com/", [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "runUrl must be a GitHub workflow run URL.")) | Out-Null
$items.Add((New-ValidationItem -Id "head-sha-format" -Passed ($headSha -match "^[0-9a-fA-F]{40}$") -Severity "action-required" -Detail "headSha must be a 40-character git commit SHA.")) | Out-Null
$items.Add((New-ValidationItem -Id "run-conclusion-success" -Passed ($runConclusion.Equals("success", [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "GitHub Actions run conclusion must be success.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-quality-success" -Passed ($sourceQualityConclusion.Equals("success", [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "source-quality job must succeed.")) | Out-Null
$items.Add((New-ValidationItem -Id "package-dry-run-pack-success" -Passed ($packagePackConclusion.Equals("success", [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "package-managed-dry-run / pack job must succeed.")) | Out-Null
$items.Add((New-ValidationItem -Id "publish-jobs-skipped" -Passed ($null -eq $record -or ($publishNugetConclusion.Equals("skipped", [StringComparison]::OrdinalIgnoreCase) -and $publishGitHubPackagesConclusion.Equals("skipped", [StringComparison]::OrdinalIgnoreCase))) -Severity "blocker" -Detail "Publish jobs must be skipped for this import; it is CI/package dry-run evidence, not publish proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "nupkg-packages-present" -Passed ($nupkgPackages.Count -gt 0) -Severity "action-required" -Detail "At least one package dry-run nupkg artifact must be present.")) | Out-Null
$items.Add((New-ValidationItem -Id "import-blockers-zero" -Passed ($null -eq $record -or $importFailedBlockerCount -eq 0) -Severity "blocker" -Detail "Import failedBlockerCount must be zero.")) | Out-Null
$items.Add((New-ValidationItem -Id "dry-run-pack-claim-ready" -Passed $canClaimDryRunPack -Severity "action-required" -Detail "Import must be internally ready to claim GitHub Actions package dry-run pack evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "workflow-run-log-sha256-present" -Passed ($workflowRunLogSha256 -match "^[0-9a-fA-F]{64}$") -Severity "action-required" -Detail "A real GitHub Actions proof lane needs the workflow run log SHA256; dry-run package evidence alone is not enough.")) | Out-Null
$items.Add((New-ValidationItem -Id "artifact-manifest-sha256-present" -Passed ($artifactManifestSha256 -match "^[0-9a-fA-F]{64}$") -Severity "action-required" -Detail "A real GitHub Actions proof lane needs an artifact manifest SHA256 tying downloaded artifacts to the run.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ($null -eq $record -or (
      [bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $false) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $false) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isGitHubActionsProof" -DefaultValue $false)
    )) -Severity "blocker" -Detail "Validator must not classify the import as publish, runtime, post-publish, release-close, or GitHub Actions proof by itself.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$ready = $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-github-actions-run-evidence-import"
}
elseif ($ready) {
  "github-actions-run-evidence-ready"
}
else {
  "blocked-github-actions-run-evidence-required"
}

$validation = [pscustomobject]@{
  recordKind = "github-actions-run-evidence-import-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  githubActionsRunEvidenceReady = $ready
  runId = $runId
  runUrl = $runUrl
  headSha = $headSha
  runConclusion = $runConclusion
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isGitHubActionsProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "GitHub Actions run evidence import validation is read-only. It can make the remote CI lane structurally ready only after real imported run artifacts pass, but it is not package publish proof, not runtime proof, not post-publish proof, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "github-actions-run-evidence-import-validation.json"
$markdownPath = Join-Path $OutputRoot "github-actions-run-evidence-import-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.passed)`` | $(ConvertTo-MarkdownCell $_.severity) | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# GitHub Actions Run Evidence Import Validation

| Item | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| githubActionsRunEvidenceReady | ``$($validation.githubActionsRunEvidenceReady)`` |
| runId | ``$($validation.runId)`` |
| headSha | ``$($validation.headSha)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| isPostPublishProof | ``$($validation.isPostPublishProof)`` |
| isGitHubActionsProof | ``$($validation.isGitHubActionsProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "GitHub Actions run evidence import validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "GitHub Actions run evidence import validation failed with $($failedBlockers.Count) blocker(s)."
}
