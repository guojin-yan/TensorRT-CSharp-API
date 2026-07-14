[CmdletBinding()]
param(
  [string]$ImportPath = "artifacts\final-release\final-owner-close-decision-import.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($ImportPath)) { $ImportPath = Join-Path $RepositoryRoot $ImportPath }
if (-not (Test-Path -LiteralPath $ImportPath -PathType Leaf)) { & (Join-Path $RepositoryRoot "eng\Import-FinalOwnerCloseDecision.ps1") -RepositoryRoot $RepositoryRoot }
$record = Get-Content -LiteralPath $ImportPath -Raw -Encoding utf8 | ConvertFrom-Json
$boundary = [string](Get-PropertyOrDefault $record "boundary" "")
$validationItems = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault $record "recordKind" "") -eq "final-owner-close-decision-import") "blocker" "recordKind must match."
  New-OwnerValidationItem "default-blocked" (([string](Get-PropertyOrDefault $record "importState" "")).Contains("blocked") -or [bool](Get-PropertyOrDefault $record "finalCloseDecisionReady" $false)) "blocker" "Default final close decision import must remain blocked."
  New-OwnerValidationItem "non-runtime-proof" (-not [bool](Get-PropertyOrDefault $record "isRuntimeExecutionProof" $true) -and -not [bool](Get-PropertyOrDefault $record "isPostPublishProof" $true)) "blocker" "Final close decision cannot claim runtime/post-publish proof."
  New-OwnerValidationItem "no-close-authority" (-not [bool](Get-PropertyOrDefault $record "canCloseReleaseIssue" $true) -and -not [bool](Get-PropertyOrDefault $record "isReleaseCloseProof" $true) -and [bool](Get-PropertyOrDefault $record "issueCloseExecutionForbidden" $false)) "blocker" "Import admission must never execute or authorize release issue close."
  New-OwnerValidationItem "required-field-contract" (@((Get-PropertyOrDefault $record "requiredFields" @())).Count -ge 20) "blocker" "Final close decision must expose the full Owner input contract."
  New-OwnerValidationItem "boundary" ($boundary.Contains("not runtime proof") -and $boundary.Contains("not post-publish proof") -and $boundary.Contains("not release close proof") -and $boundary.Contains("does not close the release issue") -and $boundary.Contains("not package push")) "blocker" "Boundary must preserve non-proof and manual-close status."
)
$failed = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failed.Count -eq 0) { "final-owner-close-decision-validation-ready-non-proof" } else { "blocked-final-owner-close-decision-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "final-owner-close-decision-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failed.Count
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
  boundary = "Validation checks final close decision import only. It is not runtime proof, not post-publish proof, not publish approval by itself, not release close approval while blocked, and not package push."
}
$jsonPath = Join-Path $OutputRoot "final-owner-close-decision-validation.json"
$mdPath = Join-Path $OutputRoot "final-owner-close-decision-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 8)
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Final Owner Close Decision Validation", "", "- validationState: ``$state``", "- failedBlockerCount: ``$($failed.Count)``", "", "## Boundary", "", $validation.boundary)
Write-Host "FinalOwnerCloseDecisionValidationState=$state FailedBlockers=$($failed.Count)"
if ($Strict.IsPresent -and $failed.Count -gt 0) { throw "Final Owner close decision validation failed." }
