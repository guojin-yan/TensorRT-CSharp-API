[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-external-proof-execution-result-import.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

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

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner external proof execution result import not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$importState = [string](Get-PropertyOrDefault -Object $record -Name "importState" -DefaultValue "")
$importItems = @(Get-PropertyOrDefault -Object $record -Name "resultImportItems" -DefaultValue @())
$blockedItems = @($importItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "importItemState" -DefaultValue "") -eq "blocked-owner-external-proof-execution-result-required" })
$readyItems = @($importItems | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "readyForRealProofRecordImport" -DefaultValue $false) })
$promotableItems = @($importItems | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteLaneResult" -DefaultValue $false) })
$laneSummaries = @((Get-PropertyOrDefault -Object $record -Name "laneSummaries" -DefaultValue @()))
$laneReadinessSummary = @((Get-PropertyOrDefault -Object $record -Name "laneReadinessSummary" -DefaultValue @()))
$summary = Get-PropertyOrDefault -Object $record -Name "summary" -DefaultValue $null
$forbiddenSubstituteMarkers = @((Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteMarkers" -DefaultValue @()))
$fileMissingCount = ($importItems | ForEach-Object { [int](Get-PropertyOrDefault -Object $_ -Name "fileMissingCount" -DefaultValue 0) } | Measure-Object -Sum).Sum
if ($null -eq $fileMissingCount) { $fileMissingCount = 0 }
$invalidSha256Count = ($importItems | ForEach-Object { [int](Get-PropertyOrDefault -Object $_ -Name "invalidSha256Count" -DefaultValue 0) } | Measure-Object -Sum).Sum
if ($null -eq $invalidSha256Count) { $invalidSha256Count = 0 }
$hashMismatchCount = ($importItems | ForEach-Object { [int](Get-PropertyOrDefault -Object $_ -Name "hashMismatchCount" -DefaultValue 0) } | Measure-Object -Sum).Sum
if ($null -eq $hashMismatchCount) { $hashMismatchCount = 0 }
$outsideAllowedEvidenceRootCount = ($importItems | ForEach-Object { [int](Get-PropertyOrDefault -Object $_ -Name "outsideAllowedEvidenceRootCount" -DefaultValue 0) } | Measure-Object -Sum).Sum
if ($null -eq $outsideAllowedEvidenceRootCount) { $outsideAllowedEvidenceRootCount = 0 }
$forbiddenSubstituteFindingCount = ($importItems | ForEach-Object { [int](Get-PropertyOrDefault -Object $_ -Name "forbiddenSubstituteFindingCount" -DefaultValue 0) } | Measure-Object -Sum).Sum
if ($null -eq $forbiddenSubstituteFindingCount) { $forbiddenSubstituteFindingCount = 0 }

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-external-proof-execution-result-import") -Severity "blocker" -Detail "recordKind must be owner-external-proof-execution-result-import.")) | Out-Null
$items.Add((New-ValidationItem -Id "import-state" -Passed ($importState -eq "blocked-owner-external-proof-execution-result-required" -or $importState -eq "owner-external-proof-execution-result-ready") -Severity "blocker" -Detail "Import state must be blocked or ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "import-item-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "resultImportItemCount" -DefaultValue 0) -eq 6 -and $importItems.Count -eq 6) -Severity "blocker" -Detail "Import must cover 6 result lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-count-consistent" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedResultImportItemCount" -DefaultValue -1) -eq $blockedItems.Count) -Severity "blocker" -Detail "Blocked import count must match items.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-count-consistent" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "readyResultImportItemCount" -DefaultValue -1) -eq $readyItems.Count) -Severity "blocker" -Detail "Ready import count must match items.")) | Out-Null
$items.Add((New-ValidationItem -Id "promotable-count-consistent" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "promotableResultImportItemCount" -DefaultValue -1) -eq $promotableItems.Count -and $promotableItems.Count -eq 0) -Severity "blocker" -Detail "Imported owner execution results remain non-promotable until strict real proof validators consume them.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-summary-projected" -Passed ($laneSummaries.Count -eq 6 -and [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultLaneCount" -DefaultValue 0) -eq 6 -and [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultBlockedLaneCount" -DefaultValue -1) -eq $blockedItems.Count -and [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultReadyLaneCount" -DefaultValue -1) -eq $readyItems.Count -and [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultPromotableLaneCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Import must project lane-level result counts for release close dashboards and gates.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-validator-summary-projected" -Passed ($null -ne $summary -and $laneReadinessSummary.Count -eq 6 -and [int](Get-PropertyOrDefault -Object $summary -Name "readyForStrictValidatorLaneCount" -DefaultValue -1) -eq $readyItems.Count -and [int](Get-PropertyOrDefault -Object $summary -Name "blockedLaneCount" -DefaultValue -1) -eq $blockedItems.Count -and [int](Get-PropertyOrDefault -Object $summary -Name "promotableLaneCount" -DefaultValue -1) -eq 0 -and [bool](Get-PropertyOrDefault -Object $summary -Name "strictValidatorInputOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $summary -Name "proofPromotionAllowed" -DefaultValue $true)) -Severity "blocker" -Detail "Import must explicitly summarize ready vs blocked strict-validator lanes without proof promotion.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitute-markers-projected" -Passed ((@("local feed", "ProjectReference", "direct .nupkg", "build-only", "dry-run", "candidate", "dashboard", "blocked-by-driver") | Where-Object { $forbiddenSubstituteMarkers -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Import must carry forbidden substitute markers for downstream readiness/evidence gates.")) | Out-Null
$items.Add((New-ValidationItem -Id "evidence-failure-counts-projected" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "fileMissingCount" -DefaultValue -1) -eq $fileMissingCount -and [int](Get-PropertyOrDefault -Object $record -Name "invalidSha256Count" -DefaultValue -1) -eq $invalidSha256Count -and [int](Get-PropertyOrDefault -Object $record -Name "hashMismatchCount" -DefaultValue -1) -eq $hashMismatchCount -and [int](Get-PropertyOrDefault -Object $record -Name "outsideAllowedEvidenceRootCount" -DefaultValue -1) -eq $outsideAllowedEvidenceRootCount -and [int](Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteFindingCount" -DefaultValue -1) -eq $forbiddenSubstituteFindingCount) -Severity "blocker" -Detail "Import must project classified evidence failures: missing files, invalid hashes, mismatches, disallowed roots, and forbidden substitutes.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Import must not promote proof, post-publish proof, or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Import must not publish.")) | Out-Null

foreach ($importItem in $importItems) {
  $itemId = [string](Get-PropertyOrDefault -Object $importItem -Name "resultImportItemId" -DefaultValue "unknown-import-item")
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $importItem -Name "resultInputId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $importItem -Name "proofLane" -DefaultValue "")) -and
    @((Get-PropertyOrDefault -Object $importItem -Name "requiredResultFields" -DefaultValue @())).Count -ge 20 -and
    @((Get-PropertyOrDefault -Object $importItem -Name "fileEvidenceChecks" -DefaultValue @())).Count -ge 5 -and
    ($importItem.PSObject.Properties.Name -contains "forbiddenSubstituteFindingCount") -and
    ($importItem.PSObject.Properties.Name -contains "fileMissingCount") -and
    ($importItem.PSObject.Properties.Name -contains "invalidSha256Count") -and
    ($importItem.PSObject.Properties.Name -contains "hashMismatchCount") -and
    ($importItem.PSObject.Properties.Name -contains "outsideAllowedEvidenceRootCount") -and
    ($importItem.PSObject.Properties.Name -contains "nonSubstituteConfirmationsReady") -and
    ($importItem.PSObject.Properties.Name -contains "exitCodeZero") -and
    ($importItem.PSObject.Properties.Name -contains "ownerReviewReady")
  $items.Add((New-ValidationItem -Id "$itemId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each import item must include lane identity, required fields, file evidence checks, non-substitute guard, exit code, and owner review state.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$itemId-non-promotable" -Passed (-not [bool](Get-PropertyOrDefault -Object $importItem -Name "canPromoteLaneResult" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $importItem -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $importItem -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $importItem -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Result import items must remain non-proof until downstream strict validators promote real records.")) | Out-Null
  if (-not [bool](Get-PropertyOrDefault -Object $importItem -Name "readyForRealProofRecordImport" -DefaultValue $false)) {
    $items.Add((New-ValidationItem -Id "$itemId-real-evidence-required" -Passed $false -Severity "action-required" -Detail "Owner must provide real existing files, hashes, metadata, validator output, and review fields before proof import.")) | Out-Null
  }
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-owner-external-proof-execution-result-import"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-owner-external-proof-execution-result-required"
}
else {
  "owner-external-proof-execution-result-ready"
}

$validation = [pscustomobject]@{
  recordKind = "owner-external-proof-execution-result-import-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  resultImportItemCount = $importItems.Count
  blockedResultImportItemCount = $blockedItems.Count
  readyResultImportItemCount = $readyItems.Count
  promotableResultImportItemCount = $promotableItems.Count
  ownerExternalProofResultLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultLaneCount" -DefaultValue 0)
  ownerExternalProofResultBlockedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultBlockedLaneCount" -DefaultValue 0)
  ownerExternalProofResultReadyLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultReadyLaneCount" -DefaultValue 0)
  ownerExternalProofResultPromotableLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultPromotableLaneCount" -DefaultValue 0)
  missingRealEvidenceCount = [int](Get-PropertyOrDefault -Object $record -Name "missingRealEvidenceCount" -DefaultValue 0)
  fileMissingCount = [int]$fileMissingCount
  invalidSha256Count = [int]$invalidSha256Count
  hashMismatchCount = [int]$hashMismatchCount
  outsideAllowedEvidenceRootCount = [int]$outsideAllowedEvidenceRootCount
  forbiddenSubstituteFindingCount = [int]$forbiddenSubstituteFindingCount
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates imported owner result evidence only. It is not runtime proof, publication approval, post-publish proof, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-external-proof-execution-result-import-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-external-proof-execution-result-import-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
$markdown = @"
# Owner External Proof Execution Result Import Validation

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| resultImportItemCount | ``$($validation.resultImportItemCount)`` |
| promotableResultImportItemCount | ``$($validation.promotableResultImportItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| missingRealEvidenceCount | ``$($validation.missingRealEvidenceCount)`` |
| fileMissingCount | ``$($validation.fileMissingCount)`` |
| invalidSha256Count | ``$($validation.invalidSha256Count)`` |
| hashMismatchCount | ``$($validation.hashMismatchCount)`` |
| outsideAllowedEvidenceRootCount | ``$($validation.outsideAllowedEvidenceRootCount)`` |
| forbiddenSubstituteFindingCount | ``$($validation.forbiddenSubstituteFindingCount)`` |
| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner external proof execution result import validation written to $jsonPath"
Write-Host "Owner external proof execution result import validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) MissingRealEvidence=$($validation.missingRealEvidenceCount)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Owner external proof execution result import validation failed with $($failedBlockers.Count) blocker(s)."
}
