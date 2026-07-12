[CmdletBinding()]
param(
  [string]$ImportPath = "artifacts\final-release\final-owner-rollback-review-import.json",
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
if (-not (Test-Path -LiteralPath $ImportPath -PathType Leaf)) { & (Join-Path $RepositoryRoot "eng\Import-FinalOwnerRollbackReview.ps1") -RepositoryRoot $RepositoryRoot }
$record = Get-Content -LiteralPath $ImportPath -Raw -Encoding utf8 | ConvertFrom-Json
$boundary = [string](Get-PropertyOrDefault $record "boundary" "")
function Test-StringArrayContainsAll {
  param(
    [AllowNull()][object]$Array,
    [string[]]$Expected
  )

  $values = @($Array | ForEach-Object { [string]$_ })
  foreach ($item in $Expected) {
    if (-not ($values | Where-Object { $_.Equals($item, [StringComparison]::OrdinalIgnoreCase) })) {
      return $false
    }
  }

  return $true
}

$requiredForbiddenActions = @("delete", "delist", "withdraw", "deprecate", "dotnet nuget delete", "nuget delete")
$validationItems = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault $record "recordKind" "") -eq "final-owner-rollback-review-import") "blocker" "recordKind must match."
  New-OwnerValidationItem "default-blocked" (([string](Get-PropertyOrDefault $record "importState" "")).Contains("blocked") -or [bool](Get-PropertyOrDefault $record "rollbackReviewReady" $false)) "blocker" "Default rollback review import must remain blocked."
  New-OwnerValidationItem "non-proof" (-not [bool](Get-PropertyOrDefault $record "canCloseReleaseIssue" $true) -and -not [bool](Get-PropertyOrDefault $record "isRuntimeExecutionProof" $true)) "blocker" "Rollback review cannot claim runtime proof or close release."
  New-OwnerValidationItem "boundary" ($boundary.Contains("not runtime proof") -and $boundary.Contains("not post-publish proof") -and $boundary.Contains("not package push")) "blocker" "Boundary must preserve non-proof status."
  New-OwnerValidationItem "rollback-no-execution" ([bool](Get-PropertyOrDefault $record "rollbackOrWithdrawExecutionForbidden" $false) -and [bool](Get-PropertyOrDefault $record "deleteDelistWithdrawDeprecateForbidden" $false)) "blocker" "Rollback/delete/delist/withdraw/deprecate execution must be explicitly forbidden."
  New-OwnerValidationItem "rollback-not-close-proof" (-not [bool](Get-PropertyOrDefault $record "rollbackPlanIsReleaseCloseProof" $true) -and -not [bool](Get-PropertyOrDefault $record "isReleaseCloseProof" $true)) "blocker" "Rollback plan cannot be release close proof."
  New-OwnerValidationItem "forbidden-release-actions" (Test-StringArrayContainsAll (Get-PropertyOrDefault $record "forbiddenReleaseActions" @()) $requiredForbiddenActions) "blocker" "Forbidden actions must include delete/delist/withdraw/deprecate command surfaces."
  New-OwnerValidationItem "boundary-forbids-release-actions" ($boundary.Contains("does not execute rollback", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("delete", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("delist", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("withdraw", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("deprecate", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must state rollback and package removal/deprecation commands are not executed."
)
$failed = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failed.Count -eq 0) { "final-owner-rollback-review-validation-ready-non-proof" } else { "blocked-final-owner-rollback-review-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "final-owner-rollback-review-validation"
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
  rollbackPlanRequired = [bool](Get-PropertyOrDefault $record "rollbackPlanRequired" $false)
  rollbackOrWithdrawExecutionForbidden = [bool](Get-PropertyOrDefault $record "rollbackOrWithdrawExecutionForbidden" $false)
  deleteDelistWithdrawDeprecateForbidden = [bool](Get-PropertyOrDefault $record "deleteDelistWithdrawDeprecateForbidden" $false)
  rollbackPlanIsReleaseCloseProof = [bool](Get-PropertyOrDefault $record "rollbackPlanIsReleaseCloseProof" $true)
  forbiddenReleaseActions = @((Get-PropertyOrDefault $record "forbiddenReleaseActions" @()))
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Validation checks rollback review import only. It does not execute rollback, delete, delist, withdraw, or deprecate actions. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, not release close proof, and not package push."
}
$jsonPath = Join-Path $OutputRoot "final-owner-rollback-review-validation.json"
$mdPath = Join-Path $OutputRoot "final-owner-rollback-review-validation.md"
$validation | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Final Owner Rollback Review Validation", "", "- validationState: ``$state``", "- failedBlockerCount: ``$($failed.Count)``", "", "## Boundary", "", $validation.boundary)
Write-Host "FinalOwnerRollbackReviewValidationState=$state FailedBlockers=$($failed.Count)"
if ($Strict.IsPresent -and $failed.Count -gt 0) { throw "Final Owner rollback review validation failed." }
