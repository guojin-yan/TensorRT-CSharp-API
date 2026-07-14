[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-real-publish-evidence-availability-ledger.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

foreach ($pathName in @("InputPath", "OutputRoot")) {
  if (-not [System.IO.Path]::IsPathRooted((Get-Variable $pathName).Value)) {
    Set-Variable -Name $pathName -Value (Join-Path $RepositoryRoot (Get-Variable $pathName).Value)
  }
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)

  $directory = Split-Path -Parent $LiteralPath
  if ([string]::IsNullOrWhiteSpace($directory)) { $directory = "." }
  New-Item -ItemType Directory -Path $directory -Force | Out-Null
  $lines = New-Object System.Collections.Generic.List[string]
  foreach ($item in @($InputObject)) {
    if ($null -eq $item) {
      $lines.Add("") | Out-Null
    }
    elseif ($item -is [string]) {
      $lines.Add($item) | Out-Null
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { $lines.Add([string]$child) | Out-Null }
    }
    else {
      $lines.Add([string]$item) | Out-Null
    }
  }

  [System.IO.File]::WriteAllText($LiteralPath, (($lines.ToArray() -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerRealPublishEvidenceAvailabilityLedger.ps1") -RepositoryRoot $RepositoryRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$slots = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "slots" -DefaultValue @()))
$slotIds = @($slots | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$sourceArtifacts = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @()))
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$requiredSlotIds = @(
  "owner-public-publish-result",
  "github-actions-run-proof",
  "public-package-download-proof",
  "repository-external-clean-consumer-proof",
  "post-publish-clean-consumer-proof",
  "post-publish-user-verification",
  "release-issue-close-owner-decision",
  "release-issue-close-record",
  "strict-close-final-convergence"
)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-real-publish-evidence-availability-ledger") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-ValidationItem "blocked-ledger-state" ([string](Get-PropertyOrDefault -Object $record -Name "ledgerState" -DefaultValue "") -eq "blocked-owner-real-publish-evidence-required") "blocker" "Ledger must stay blocked until all real Owner proof inputs are available.")) | Out-Null
$items.Add((New-ValidationItem "slot-count" ($slots.Count -ge 9 -and [int](Get-PropertyOrDefault -Object $record -Name "slotCount" -DefaultValue 0) -eq $slots.Count) "blocker" "Ledger must include all publish/download/consumer/post-publish/close slots.")) | Out-Null
$items.Add((New-ValidationItem "blocked-slot-count" ([int](Get-PropertyOrDefault -Object $record -Name "blockedSlotCount" -DefaultValue 0) -eq $slots.Count) "blocker" "All slots should remain blocked in this repo without committed real Owner proof files.")) | Out-Null
$items.Add((New-ValidationItem "proof-ready-zero" ([int](Get-PropertyOrDefault -Object $record -Name "proofReadySlotCount" -DefaultValue -1) -eq 0 -and -not [bool](Get-PropertyOrDefault -Object $record -Name "allRequiredOwnerProofAvailable" -DefaultValue $true)) "blocker" "No slot can be proof-ready without real Owner evidence files and ready validators.")) | Out-Null
$items.Add((New-ValidationItem "owner-action-required" ([bool](Get-PropertyOrDefault -Object $record -Name "ownerActionRequired" -DefaultValue $false)) "blocker" "Ledger must require Owner action.")) | Out-Null
$items.Add((New-ValidationItem "blocked-validation-items" ([int](Get-PropertyOrDefault -Object $record -Name "blockedValidationItemCount" -DefaultValue 0) -ge 20) "blocker" "Ledger should aggregate blocked field/action validation items.")) | Out-Null
$items.Add((New-ValidationItem "template-files-present-but-not-proof" ([int](Get-PropertyOrDefault -Object $record -Name "templateOrGuidanceFileCount" -DefaultValue 0) -gt 0 -and [int](Get-PropertyOrDefault -Object $record -Name "generatedOutputFileCount" -DefaultValue 0) -gt 0) "blocker" "Templates and generated outputs should be visible but not promoted.")) | Out-Null
$items.Add((New-ValidationItem "non-proof-flags" ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsRuntimeExecution" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) "blocker" "Ledger must not publish, run proof, promote proof, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem "boundary" ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must preserve all non-proof classifications.")) | Out-Null

foreach ($requiredSlotId in $requiredSlotIds) {
  $items.Add((New-ValidationItem "slot-present-$requiredSlotId" ($slotIds -contains $requiredSlotId) "blocker" "Required slot '$requiredSlotId' must be present.")) | Out-Null
}

foreach ($source in @(
    "artifacts/final-release/public-publish-result-owner-input-validation.json",
    "artifacts/final-release/public-package-download-proof-input-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json",
    "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
    "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json"
  )) {
  $items.Add((New-ValidationItem "source-$([System.IO.Path]::GetFileNameWithoutExtension($source))" ($sourceArtifacts -contains $source) "blocker" "sourceArtifacts must include '$source'.")) | Out-Null
}

foreach ($slot in $slots) {
  $slotId = [string](Get-PropertyOrDefault -Object $slot -Name "id" -DefaultValue "")
  $slotBoundary = [string](Get-PropertyOrDefault -Object $slot -Name "boundary" -DefaultValue "")
  $items.Add((New-ValidationItem "slot-$slotId-blocked" ([bool](Get-PropertyOrDefault -Object $slot -Name "blocked" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $slot -Name "proofReady" -DefaultValue $true)) "blocker" "Slot '$slotId' must remain blocked without real Owner proof.")) | Out-Null
  $items.Add((New-ValidationItem "slot-$slotId-non-proof-flags" ((-not [bool](Get-PropertyOrDefault -Object $slot -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $slot -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $slot -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $slot -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $slot -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $slot -Name "isReleaseCloseProof" -DefaultValue $true))) "blocker" "Slot '$slotId' has unsafe publish/proof flags.")) | Out-Null
  $items.Add((New-ValidationItem "slot-$slotId-boundary" ($slotBoundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $slotBoundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $slotBoundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Slot '$slotId' boundary must be explicit.")) | Out-Null
}

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "owner-real-publish-evidence-availability-ledger-ready-non-proof" } else { "blocked-owner-real-publish-evidence-availability-ledger-invalid" }

$validation = [pscustomobject]@{
  recordKind = "owner-real-publish-evidence-availability-ledger-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  isValidOwnerRealPublishEvidenceAvailabilityLedger = $failedBlockers.Count -eq 0
  validationItemCount = $validationItems.Count
  failedBlockerCount = $failedBlockers.Count
  slotCount = $slots.Count
  proofReadySlotCount = [int](Get-PropertyOrDefault -Object $record -Name "proofReadySlotCount" -DefaultValue 0)
  blockedSlotCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedSlotCount" -DefaultValue 0)
  blockedValidationItemCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedValidationItemCount" -DefaultValue 0)
  ownerActionRequired = $true
  passed = $false
  notExecutedByAutomation = $true
  performsPublish = $false
  usesPublishToken = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($validationItems)
  boundary = "Owner real publish evidence availability ledger validation checks inventory structure and non-proof boundaries only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-real-publish-evidence-availability-ledger-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-real-publish-evidence-availability-ledger-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 16)

$rows = foreach ($item in $validationItems) {
  "| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

Write-Utf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Real Publish Evidence Availability Ledger Validation",
  "",
  "- validationState: ``$validationState``",
  "- failedBlockerCount: ``$($failedBlockers.Count)``",
  "- slotCount: ``$($slots.Count)``",
  "- proofReadySlotCount: ``$($validation.proofReadySlotCount)``",
  "- blockedSlotCount: ``$($validation.blockedSlotCount)``",
  "- canPublishPublicly: ``False``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "| ID | Passed | Severity | Detail |",
  "|---|---:|---|---|",
  @($rows),
  "",
  "## Boundary",
  "",
  $validation.boundary
)

Write-Host "OwnerRealPublishEvidenceAvailabilityLedgerValidationState=$validationState FailedBlockers=$($failedBlockers.Count) Slots=$($slots.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Owner real publish evidence availability ledger validation failed."
}
