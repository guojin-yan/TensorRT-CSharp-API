[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-publish-evidence-import-runbook.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerPublishEvidenceImportRunbook.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$sections = @(Get-PropertyOrDefault -Object $record -Name "sections" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12
$requiredIds = @("package-identity", "publish-transcripts", "clean-consumer", "host-runtime", "non-substitutes", "rollback-close")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-publish-evidence-import-runbook") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "contract-count" ([int](Get-PropertyOrDefault -Object $record -Name "contractRequiredFieldCount" -DefaultValue 0) -ge 178) "blocker" "Runbook must use the Owner public publish 178-field contract.")) | Out-Null
$items.Add((New-OwnerValidationItem "no-invented-fields" ([int](Get-PropertyOrDefault -Object $record -Name "sectionFieldMissingFromContractCount" -DefaultValue -1) -eq 0) "blocker" "Runbook section fields must exist in contract.")) | Out-Null
$items.Add((New-OwnerValidationItem "section-count" ($sections.Count -eq $requiredIds.Count) "blocker" "Every evidence import section must be present.")) | Out-Null
foreach ($id in $requiredIds) {
  $items.Add((New-OwnerValidationItem "section-$id" (@($sections | Where-Object { [string]$_.id -eq $id -and [bool]$_.ownerActionRequired -and -not [bool]$_.performsPublish -and -not [bool]$_.isProof -and [int]$_.fieldCount -ge 5 }).Count -eq 1) "blocker" "Missing or unsafe runbook section: $id")) | Out-Null
}
$items.Add((New-OwnerValidationItem "validator-links" ($text.Contains("Test-OwnerPublicPublishExecutionResultCandidate.ps1") -and $text.Contains("Test-PostPublishVerificationRecord.ps1") -and $text.Contains("Test-FinalOwnerRollbackReview.ps1") -and $text.Contains("Test-FinalOwnerCloseDecision.ps1")) "blocker" "Runbook must link publish, PostPublish, rollback, and close validators.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Runbook must not publish or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "final-owner-publish-evidence-import-runbook-validation-ready-non-proof" } else { "blocked-final-owner-publish-evidence-import-runbook-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "final-owner-publish-evidence-import-runbook-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  sectionCount = [int]$sections.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner evidence import runbook validation only; not proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-publish-evidence-import-runbook-validation.json"
$mdPath = Join-Path $OutputRoot "final-owner-publish-evidence-import-runbook-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Final Owner Publish Evidence Import Runbook Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- sectionCount: ``$($validation.sectionCount)``",
  "",
  $validation.boundary
)
Write-Host "FinalOwnerPublishEvidenceImportRunbookValidationState=$state FailedBlockers=$failedBlockerCount Sections=$($validation.sectionCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Final Owner publish evidence import runbook validation failed." }
