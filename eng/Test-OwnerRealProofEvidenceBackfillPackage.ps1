[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-real-proof-evidence-backfill-package.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-OwnerRealProofEvidenceBackfillPackage.ps1") -RepositoryRoot $RepositoryRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "backfillItems" -DefaultValue @()))
$groups = @($items | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "group" -DefaultValue "") } | Sort-Object -Unique)
$text = $record | ConvertTo-Json -Depth 16
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$validationItems = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault $record "recordKind" "") -eq "owner-real-proof-evidence-backfill-package") "blocker" "recordKind must match."
  New-OwnerValidationItem "blocked-non-proof" ([string](Get-PropertyOrDefault $record "packageState" "") -eq "blocked-owner-real-proof-evidence-backfill-required" -and -not [bool](Get-PropertyOrDefault $record "passed" $true) -and [bool](Get-PropertyOrDefault $record "ownerActionRequired" $false)) "blocker" "Backfill package must remain blocked/non-proof."
  New-OwnerValidationItem "required-groups" (@("missing-field", "missing-file", "missing-sha256", "missing-host-metadata", "missing-owner-confirmation" | Where-Object { $groups -notcontains $_ }).Count -eq 0) "blocker" "Backfill groups must cover field/file/SHA256/host/confirmation."
  New-OwnerValidationItem "strict-validators" ($text.Contains("Import-ExternalCleanConsumerExecutionResult.ps1") -and $text.Contains("Import-PostPublishCleanConsumerProofResult.ps1")) "blocker" "Backfill items must point to strict import validators."
  New-OwnerValidationItem "non-proof-flags" (-not [bool](Get-PropertyOrDefault $record "canCloseReleaseIssue" $true) -and -not [bool](Get-PropertyOrDefault $record "isRuntimeExecutionProof" $true) -and -not [bool](Get-PropertyOrDefault $record "isPostPublishProof" $true)) "blocker" "Backfill package cannot claim proof."
  New-OwnerValidationItem "boundary" ($boundary.Contains("not runtime proof") -and $boundary.Contains("not post-publish proof") -and $boundary.Contains("not package push")) "blocker" "Boundary must preserve non-proof status."
)

$failed = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failed.Count -eq 0) { "owner-real-proof-evidence-backfill-package-ready-non-proof" } else { "blocked-owner-real-proof-evidence-backfill-package-invalid" }
$validation = [pscustomobject]@{
  recordKind = "owner-real-proof-evidence-backfill-package-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failed.Count
  backfillItemCount = $items.Count
  validationItems = @($validationItems)
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Validation checks the Owner real proof evidence backfill package only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
$jsonPath = Join-Path $OutputRoot "owner-real-proof-evidence-backfill-package-validation.json"
$mdPath = Join-Path $OutputRoot "owner-real-proof-evidence-backfill-package-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Owner Real Proof Evidence Backfill Package Validation", "", "- validationState: ``$state``", "- failedBlockerCount: ``$($failed.Count)``", "- backfillItemCount: ``$($items.Count)``", "", "## Boundary", "", $validation.boundary)
Write-Host "OwnerRealProofEvidenceBackfillPackageValidationState=$state FailedBlockers=$($failed.Count) BackfillItems=$($items.Count)"
if ($Strict.IsPresent -and $failed.Count -gt 0) { throw "Owner real proof evidence backfill package validation failed." }
