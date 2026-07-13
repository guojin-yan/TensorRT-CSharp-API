[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-input-contract-convergence.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
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

function ConvertTo-StringArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  return @($Value | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
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
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner input contract convergence not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]

$surfaces = @((Get-PropertyOrDefault -Object $record -Name "contractSurfaces" -DefaultValue @()))
$runbooks = @((Get-PropertyOrDefault -Object $record -Name "runbookInputs" -DefaultValue @()))
$canonicalFields = @((Get-PropertyOrDefault -Object $record -Name "canonicalFields" -DefaultValue @()))
$fieldCoverage = @((Get-PropertyOrDefault -Object $record -Name "fieldCoverage" -DefaultValue @()))
$sourceArtifacts = ConvertTo-StringArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())

$surfaceIds = @($surfaces | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$runbookIds = @($runbooks | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$canonicalNames = @($canonicalFields | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "") })

$requiredSurfaceIds = @(
  "owner-external-proof-execution-result-input",
  "public-publish-real-result-owner-input-contract",
  "public-package-proof-owner-input",
  "post-publish-verification-owner-input",
  "release-issue-close-owner-decision-input"
)

$requiredCanonicalNames = @(
  "publicPackageUrl",
  "publicPackageSourceUrl",
  "downloadedNupkgSha256",
  "publishedTimestampUtc",
  "stdoutPath",
  "stderrPath",
  "mergedTranscriptPath",
  "stdoutSha256",
  "stderrSha256",
  "mergedTranscriptSha256",
  "hostMetadata",
  "ownerReviewer",
  "ownerReviewTimestampUtc",
  "nonSubstituteConfirmations"
)

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-input-contract-convergence") -Severity "blocker" -Detail "recordKind must be owner-input-contract-convergence.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "convergenceState" -DefaultValue "") -eq "blocked-owner-input-contract-convergence-real-owner-input-required") -Severity "blocker" -Detail "Convergence must stay blocked until real owner input is imported and strict validators pass.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-surfaces-present" -Passed (@($requiredSurfaceIds | Where-Object { $surfaceIds -notcontains $_ }).Count -eq 0 -and $surfaces.Count -ge 5) -Severity "blocker" -Detail "Convergence must cover owner external result, public publish, public package proof, post-publish verification, and close decision inputs.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-runbooks-present" -Passed (($runbookIds -contains "clean-external-package-consumer-owner-runbook") -and ($runbookIds -contains "post-publish-owner-verification-runbook")) -Severity "blocker" -Detail "Convergence must include both owner runbooks that drive real owner input backfill.")) | Out-Null
$items.Add((New-ValidationItem -Id "canonical-fields-present" -Passed (@($requiredCanonicalNames | Where-Object { $canonicalNames -notcontains $_ }).Count -eq 0 -and $canonicalFields.Count -ge 14) -Severity "blocker" -Detail "Convergence must define canonical public package, log/hash, host metadata, owner review, and non-substitute fields.")) | Out-Null

foreach ($name in $requiredCanonicalNames) {
  $coverage = @($fieldCoverage | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "canonicalName" -DefaultValue "") -eq $name })
  $coveredSurfaceCount = if ($coverage.Count -gt 0) { [int](Get-PropertyOrDefault -Object $coverage[0] -Name "coveredSurfaceCount" -DefaultValue 0) } else { 0 }
  $items.Add((New-ValidationItem -Id "canonical-$name-covered" -Passed ($coverage.Count -eq 1 -and $coveredSurfaceCount -gt 0) -Severity "blocker" -Detail "Canonical field $name must map to at least one current owner input surface.")) | Out-Null
}

$allSurfacesSafe = $true
foreach ($surface in $surfaces) {
  $allSurfacesSafe = $allSurfacesSafe -and
    -not [bool](Get-PropertyOrDefault -Object $surface -Name "ready" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $surface -Name "performsPublish" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $surface -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $surface -Name "canPublishPublicly" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $surface -Name "canCloseReleaseIssue" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $surface -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $surface -Name "isPostPublishProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $surface -Name "isReleaseCloseProof" -DefaultValue $true)
}

$items.Add((New-ValidationItem -Id "surfaces-non-proof" -Passed $allSurfacesSafe -Severity "blocker" -Detail "Every owner input surface must remain blocked/non-proof/non-publishing.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Convergence artifact must not publish, promote proof, prove post-publish, or close release issue.")) | Out-Null

foreach ($artifact in @(
  "owner-external-proof-execution-result.input.template.json",
  "public-publish-real-result-owner-input-contract.json",
  "public-package-proof-owner-input.template.json",
  "post-publish-verification-owner-input.template.json",
  "release-issue-close-owner-decision-input.template.json",
  "clean-external-package-consumer-owner-runbook.json",
  "post-publish-owner-verification-runbook.json"
)) {
  $items.Add((New-ValidationItem -Id "source-$($artifact.Replace('.', '-'))-present" -Passed (($sourceArtifacts -join "`n").Contains($artifact, [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Source artifact $artifact must be listed.")) | Out-Null
}

foreach ($marker in @("local feed", "ProjectReference", "direct nupkg", "direct .nupkg", "template", "draft", "candidate", "dashboard", "dry-run", "build-only", "runbook as proof", "manual handoff as proof")) {
  $items.Add((New-ValidationItem -Id "forbidden-$($marker.Replace(' ', '-').Replace('.', 'dot'))-visible" -Passed ($raw.Contains($marker, [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Forbidden substitute marker '$marker' must remain visible.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-owner-input-contract-convergence" } else { "blocked-owner-input-contract-convergence-real-owner-input-required" }

$validation = [pscustomobject]@{
  recordKind = "owner-input-contract-convergence-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  contractSurfaceCount = $surfaces.Count
  canonicalFieldCount = $canonicalFields.Count
  runbookInputCount = $runbooks.Count
  failedBlockerCount = $failedBlockers.Count
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "Owner input contract convergence validation checks schema and terminology only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-input-contract-convergence-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-input-contract-convergence-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 14)
$rows = foreach ($item in $items) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Owner Input Contract Convergence Validation

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| contractSurfaceCount | ``$($validation.contractSurfaceCount)`` |
| canonicalFieldCount | ``$($validation.canonicalFieldCount)`` |
| runbookInputCount | ``$($validation.runbookInputCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner input contract convergence validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$($validation.validationState) Surfaces=$($validation.contractSurfaceCount) CanonicalFields=$($validation.canonicalFieldCount) Runbooks=$($validation.runbookInputCount) FailedBlockers=$($validation.failedBlockerCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Owner input contract convergence validation failed with $($failedBlockers.Count) blocker(s)."
}
