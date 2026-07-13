[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-missing-real-publish-evidence-repair-pack.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-OwnerMissingRealPublishEvidenceRepairPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$groups = @(Get-PropertyOrDefault -Object $record -Name "repairGroups" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12
$requiredIds = @("owner-public-publish-contract", "post-publish-owner-input", "post-publish-strict-cross-checks", "release-close-blocked-lanes")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-missing-real-publish-evidence-repair-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-by-default" (([string](Get-PropertyOrDefault -Object $record -Name "repairState" -DefaultValue "")).Contains("blocked") -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerActionRequired" -DefaultValue $false)) "blocker" "Repair pack must remain Owner-action blocked.")) | Out-Null
$items.Add((New-OwnerValidationItem "contract-count" ([int](Get-PropertyOrDefault -Object $record -Name "contractRequiredFieldCount" -DefaultValue 0) -ge 178 -and [int](Get-PropertyOrDefault -Object $record -Name "ownerPublishMissingFieldCount" -DefaultValue 0) -ge 178) "blocker" "Repair pack must preserve the 178-field Owner public publish contract gap.")) | Out-Null
$items.Add((New-OwnerValidationItem "post-publish-gaps" ([int](Get-PropertyOrDefault -Object $record -Name "postPublishMissingFieldCount" -DefaultValue 0) -ge 10 -and [int](Get-PropertyOrDefault -Object $record -Name "failedPostPublishCrossCheckCount" -DefaultValue 0) -ge 1) "blocker" "Repair pack must expose PostPublish placeholder/cross-check gaps.")) | Out-Null
$items.Add((New-OwnerValidationItem "release-close-gaps" ([int](Get-PropertyOrDefault -Object $record -Name "releaseCloseBlockedLaneCount" -DefaultValue 0) -ge 1) "blocker" "Repair pack must expose release close blocked lanes.")) | Out-Null
foreach ($id in $requiredIds) {
  $items.Add((New-OwnerValidationItem "group-$id" (@($groups | Where-Object { [string]$_.id -eq $id -and [bool]$_.ownerActionRequired -and [bool]$_.blocked -and [int]$_.missingItemCount -ge 1 -and -not [bool]$_.performsPublish -and -not [bool]$_.isProof }).Count -eq 1) "blocker" "Missing or unsafe repair group: $id")) | Out-Null
}
$items.Add((New-OwnerValidationItem "field-scope" ($text.Contains("publicPackageUrl") -and $text.Contains("downloadedManagedPackageSha256") -and $text.Contains("noProjectReference") -and $text.Contains("release-close")) "blocker" "Repair pack must include publish, PostPublish, no-substitute, and release close fields.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Repair pack must not publish or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "owner-missing-real-publish-evidence-repair-pack-validation-ready-non-proof" } else { "blocked-owner-missing-real-publish-evidence-repair-pack-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "owner-missing-real-publish-evidence-repair-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  repairGroupCount = [int]$groups.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner missing real publish evidence repair pack validation only; not proof, not package publication, not public availability validation, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-missing-real-publish-evidence-repair-pack-validation.json"
$mdPath = Join-Path $OutputRoot "owner-missing-real-publish-evidence-repair-pack-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Missing Real Publish Evidence Repair Pack Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- repairGroupCount: ``$($validation.repairGroupCount)``",
  "",
  $validation.boundary
)
Write-Host "OwnerMissingRealPublishEvidenceRepairPackValidationState=$state FailedBlockers=$failedBlockerCount Groups=$($validation.repairGroupCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Owner missing real publish evidence repair pack validation failed." }
