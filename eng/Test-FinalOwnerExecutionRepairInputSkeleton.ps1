[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-repair-input-skeleton.json",
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

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)
  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function Test-ArrayTextContains {
  param([AllowNull()][object]$Values, [string]$Needle)
  return ((Convert-ToStringArray $Values) -join "`n").IndexOf($Needle, [StringComparison]::OrdinalIgnoreCase) -ge 0
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final owner execution repair input skeleton not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$skeletonItems = @((Get-PropertyOrDefault -Object $record -Name "skeletonItems" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-execution-repair-input-skeleton") -Severity "blocker" -Detail "recordKind must be final-owner-execution-repair-input-skeleton.")) | Out-Null
$items.Add((New-ValidationItem -Id "skeleton-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "skeletonState" -DefaultValue "") -eq "blocked-owner-real-evidence-input-required") -Severity "blocker" -Detail "Skeleton must remain blocked until Owner fills real evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "eight-skeleton-items" -Passed ($skeletonItems.Count -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "skeletonItemCount" -DefaultValue 0) -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedSkeletonItemCount" -DefaultValue 0) -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "readySkeletonItemCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Skeleton must mirror all eight repair checklist items.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "Skeleton must stay non-proof and non-publish.") ) | Out-Null

foreach ($summaryField in @("fileInputCount", "sha256InputCount", "identityInputCount", "nonSubstituteConfirmationCount")) {
  $items.Add((New-ValidationItem -Id "$summaryField-positive" -Passed ([int](Get-PropertyOrDefault -Object $record -Name $summaryField -DefaultValue 0) -gt 0) -Severity "blocker" -Detail "$summaryField must be positive.")) | Out-Null
}

$sourceArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())
foreach ($artifact in @(
  "artifacts/final-release/final-owner-execution-repair-checklist.json",
  "artifacts/final-release/final-owner-execution-repair-checklist-validation.json",
  "artifacts/final-release/final-owner-execution-package.json",
  "artifacts/final-release/final-owner-execution-package-validation.json"
)) {
  $items.Add((New-ValidationItem -Id "source-$($artifact.Replace('/', '-').Replace('.', '-'))" -Passed ($sourceArtifacts -contains $artifact) -Severity "blocker" -Detail "Source artifact $artifact must be listed.")) | Out-Null
}

foreach ($marker in @("stdoutPath", "stderrPath", "mergedTranscriptPath", "validatorOutputPath", "stdoutSha256", "stderrSha256", "mergedTranscriptSha256", "validatorOutputSha256", "exitCode", "hostIdentity", "packageIdentity", "ownerReviewer", "local feed", "ProjectReference", "direct .nupkg", "template", "draft", "dry-run", "dashboard", "candidate", "build-only", "dependency-probe-only", "blocked-by-cuda-driver")) {
  $items.Add((New-ValidationItem -Id "raw-marker-$($marker.Replace(' ', '-').Replace('.', 'dot'))" -Passed ($raw.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Skeleton raw JSON must contain marker $marker.")) | Out-Null
}

foreach ($item in $skeletonItems) {
  $id = [string](Get-PropertyOrDefault -Object $item -Name "sourceRepairItemId" -DefaultValue "")
  $fileInputs = @((Get-PropertyOrDefault -Object $item -Name "fileInputs" -DefaultValue @()))
  $shaInputs = @((Get-PropertyOrDefault -Object $item -Name "sha256Inputs" -DefaultValue @()))
  $identityInputs = @((Get-PropertyOrDefault -Object $item -Name "identityInputs" -DefaultValue @()))
  $confirmations = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "nonSubstituteConfirmations" -DefaultValue @())
  $validatorCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "expectedValidatorCommands" -DefaultValue @())
  $expectedArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "expectedResultArtifacts" -DefaultValue @())
  $ownerMustFill = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "ownerMustFill" -DefaultValue @())
  $cannotUseMarkers = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "cannotUseMarkers" -DefaultValue @())
  $policy = Get-PropertyOrDefault -Object $item -Name "evidenceRootPolicy" -DefaultValue $null
  $preflight = Get-PropertyOrDefault -Object $item -Name "importPreflight" -DefaultValue $null
  $boundary = [string](Get-PropertyOrDefault -Object $item -Name "boundary" -DefaultValue "")

  $items.Add((New-ValidationItem -Id "$id-state" -Passed ([string](Get-PropertyOrDefault -Object $item -Name "inputState" -DefaultValue "") -eq "owner-real-evidence-required") -Severity "blocker" -Detail "$id must require owner real evidence.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-owner-fill" -Passed ($ownerMustFill.Count -ge 5) -Severity "blocker" -Detail "$id must list owner-fill instructions.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-file-inputs" -Passed ($fileInputs.Count -ge 3) -Severity "blocker" -Detail "$id must expose file inputs.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-sha-inputs" -Passed ($shaInputs.Count -ge 3) -Severity "blocker" -Detail "$id must expose SHA256 inputs.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-identity-inputs" -Passed ($identityInputs.Count -ge 3) -Severity "blocker" -Detail "$id must expose identity inputs.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-confirmations" -Passed ($confirmations.Count -ge 8) -Severity "blocker" -Detail "$id must expose non-substitute confirmations.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-validator-commands" -Passed ($validatorCommands.Count -ge 1) -Severity "blocker" -Detail "$id must include expected validator commands.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-result-artifacts" -Passed ($expectedArtifacts.Count -ge 1) -Severity "blocker" -Detail "$id must include expected result artifacts.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-evidence-policy" -Passed ($null -ne $policy -and [bool](Get-PropertyOrDefault -Object $policy -Name "mustExist" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $policy -Name "mustStayUnderAllowedEvidenceRoot" -DefaultValue $false)) -Severity "blocker" -Detail "$id must include strict evidence root policy.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-import-preflight" -Passed ($null -ne $preflight -and [string](Get-PropertyOrDefault -Object $preflight -Name "importer" -DefaultValue "") -eq "eng/Import-OwnerExternalProofExecutionResult.ps1" -and [bool](Get-PropertyOrDefault -Object $preflight -Name "requiresSha256Match" -DefaultValue $false)) -Severity "blocker" -Detail "$id must point to owner result importer and SHA256 matching.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-cannot-use" -Passed ((Test-ArrayTextContains -Values $cannotUseMarkers -Needle "local feed") -and (Test-ArrayTextContains -Values $cannotUseMarkers -Needle "ProjectReference") -and (Test-ArrayTextContains -Values $cannotUseMarkers -Needle "direct .nupkg") -and (Test-ArrayTextContains -Values $cannotUseMarkers -Needle "blocked-by-cuda-driver")) -Severity "blocker" -Detail "$id must preserve cannot-use markers.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $item -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "$id must remain non-proof guidance.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-boundary" -Passed ($boundary.Contains("input skeleton only") -and $boundary.Contains("not runtime proof") -and $boundary.Contains("not package push")) -Severity "blocker" -Detail "$id must state input skeleton non-proof boundary.")) | Out-Null
}

$failedItems = @($items | Where-Object { -not $_.passed })
$failedBlockers = @($failedItems | Where-Object { $_.severity -eq "blocker" }).Count
$failedActionRequired = @($skeletonItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "inputState" -DefaultValue "") -eq "owner-real-evidence-required" }).Count
$validationState = if ($failedBlockers -eq 0) { "blocked-owner-real-evidence-input-required" } else { "invalid-final-owner-execution-repair-input-skeleton" }

$validation = [ordered]@{
  recordKind = "final-owner-execution-repair-input-skeleton-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  skeletonState = [string](Get-PropertyOrDefault -Object $record -Name "skeletonState" -DefaultValue "")
  skeletonItemCount = $skeletonItems.Count
  blockedSkeletonItemCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedSkeletonItemCount" -DefaultValue 0)
  readySkeletonItemCount = [int](Get-PropertyOrDefault -Object $record -Name "readySkeletonItemCount" -DefaultValue 0)
  fileInputCount = [int](Get-PropertyOrDefault -Object $record -Name "fileInputCount" -DefaultValue 0)
  sha256InputCount = [int](Get-PropertyOrDefault -Object $record -Name "sha256InputCount" -DefaultValue 0)
  identityInputCount = [int](Get-PropertyOrDefault -Object $record -Name "identityInputCount" -DefaultValue 0)
  nonSubstituteConfirmationCount = [int](Get-PropertyOrDefault -Object $record -Name "nonSubstituteConfirmationCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers
  failedActionRequiredCount = $failedActionRequired
  findingCount = $failedItems.Count
  performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $false)
  canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $false)
  canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $false)
  canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $false)
  isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $false)
  isPostPublishProof = [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $false)
  isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $false)
  validationItems = @($items.ToArray())
  failedItems = @($failedItems)
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-repair-input-skeleton-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-repair-input-skeleton-validation.md"
$validation | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Final Owner Execution Repair Input Skeleton Validation",
  "",
  "- validationState: ``$validationState``",
  "- failedBlockerCount: ``$failedBlockers``",
  "- failedActionRequiredCount: ``$failedActionRequired``",
  "- skeletonItemCount: ``$($skeletonItems.Count)``",
  "- fileInputCount: ``$($validation.fileInputCount)``",
  "- sha256InputCount: ``$($validation.sha256InputCount)``",
  "- identityInputCount: ``$($validation.identityInputCount)``"
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"

if ($Strict -and $failedBlockers -gt 0) {
  throw "Final owner execution repair input skeleton validation failed with $failedBlockers blocker(s)."
}
