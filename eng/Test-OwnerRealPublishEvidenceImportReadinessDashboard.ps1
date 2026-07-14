[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-real-publish-evidence-import-readiness-dashboard.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerRealPublishEvidenceImportReadinessDashboard.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$slots = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "slots" -DefaultValue @()))
$slotIds = @($slots | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
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
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-real-publish-evidence-import-readiness-dashboard") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "dashboard-state" ([string](Get-PropertyOrDefault -Object $record -Name "dashboardState" -DefaultValue "") -eq "blocked-owner-real-publish-evidence-import-required") "blocker" "Dashboard must remain blocked until real Owner evidence exists.")) | Out-Null
$items.Add((New-OwnerValidationItem "slot-count" ($slots.Count -ge 9 -and [int](Get-PropertyOrDefault -Object $record -Name "slotCount" -DefaultValue 0) -eq $slots.Count) "blocker" "Dashboard must include all Owner evidence slots.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-slots" ([int](Get-PropertyOrDefault -Object $record -Name "blockedSlotCount" -DefaultValue 0) -eq $slots.Count) "blocker" "All slots must remain blocked in repo without real Owner inputs.")) | Out-Null
$items.Add((New-OwnerValidationItem "proof-ready-zero" ([int](Get-PropertyOrDefault -Object $record -Name "proofReadySlotCount" -DefaultValue -1) -eq 0) "blocker" "No slot may be proof-ready by default.")) | Out-Null
$items.Add((New-OwnerValidationItem "expected-fields" ([int](Get-PropertyOrDefault -Object $record -Name "expectedEvidenceFieldCount" -DefaultValue 0) -ge 70) "blocker" "Dashboard must expose a broad field-level import contract.")) | Out-Null
$items.Add((New-OwnerValidationItem "validators" ([int](Get-PropertyOrDefault -Object $record -Name "validatorScriptCount" -DefaultValue 0) -ge 15) "blocker" "Dashboard must link validator scripts for Owner evidence import.")) | Out-Null
$items.Add((New-OwnerValidationItem "non-proof-flags" ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) "blocker" "Dashboard must not publish, close, or promote proof.")) | Out-Null

foreach ($id in $requiredSlotIds) {
  $items.Add((New-OwnerValidationItem "slot-$id" ($slotIds -contains $id) "blocker" "Missing required slot: $id")) | Out-Null
}

foreach ($slot in $slots) {
  $id = [string](Get-PropertyOrDefault -Object $slot -Name "id" -DefaultValue "")
  $boundary = [string](Get-PropertyOrDefault -Object $slot -Name "boundary" -DefaultValue "")
  $items.Add((New-OwnerValidationItem "slot-$id-blocked" ([bool](Get-PropertyOrDefault -Object $slot -Name "blocked" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $slot -Name "proofReady" -DefaultValue $true)) "blocker" "Slot must stay blocked and non-proof.")) | Out-Null
  $items.Add((New-OwnerValidationItem "slot-$id-fields" ([int](Get-PropertyOrDefault -Object $slot -Name "expectedEvidenceFieldCount" -DefaultValue 0) -ge 7 -and [int](Get-PropertyOrDefault -Object $slot -Name "validatorScriptCount" -DefaultValue 0) -ge 1) "blocker" "Slot must include expected fields and validators.")) | Out-Null
  $items.Add((New-OwnerValidationItem "slot-$id-forbidden" ([int](Get-PropertyOrDefault -Object $slot -Name "forbiddenSubstituteCount" -DefaultValue 0) -ge 8) "blocker" "Slot must list forbidden substitutes.")) | Out-Null
  $items.Add((New-OwnerValidationItem "slot-$id-boundary" ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Slot boundary must be explicit.")) | Out-Null
}

$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$items.Add((New-OwnerValidationItem "boundary" ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must preserve all non-proof classifications.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "owner-real-publish-evidence-import-readiness-dashboard-ready-non-proof" } else { "blocked-owner-real-publish-evidence-import-readiness-dashboard-invalid" }
$validation = [pscustomobject]@{
  recordKind = "owner-real-publish-evidence-import-readiness-dashboard-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = [int]$failedBlockers.Count
  validationItemCount = [int]$validationItems.Count
  slotCount = [int]$slots.Count
  blockedSlotCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedSlotCount" -DefaultValue 0)
  proofReadySlotCount = [int](Get-PropertyOrDefault -Object $record -Name "proofReadySlotCount" -DefaultValue 0)
  expectedEvidenceFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "expectedEvidenceFieldCount" -DefaultValue 0)
  validatorScriptCount = [int](Get-PropertyOrDefault -Object $record -Name "validatorScriptCount" -DefaultValue 0)
  validationItems = @($validationItems)
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner real publish evidence import readiness dashboard validation is non-proof structure validation only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-real-publish-evidence-import-readiness-dashboard-validation.json"
$mdPath = Join-Path $OutputRoot "owner-real-publish-evidence-import-readiness-dashboard-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Real Publish Evidence Import Readiness Dashboard Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$($failedBlockers.Count)``",
  "- slotCount: ``$($validation.slotCount)``",
  "- proofReadySlotCount: ``$($validation.proofReadySlotCount)``",
  "",
  $validation.boundary
)
Write-Host "OwnerRealPublishEvidenceImportReadinessDashboardValidationState=$state FailedBlockers=$($failedBlockers.Count) Slots=$($validation.slotCount)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Owner real publish evidence import readiness dashboard validation failed." }
