[CmdletBinding()]
param(
  [string]$ImportPath = "artifacts\final-release\github-ci-evidence-from-owner-input.json",
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
if (-not (Test-Path -LiteralPath $ImportPath -PathType Leaf)) { & (Join-Path $RepositoryRoot "eng\Import-GitHubCiEvidenceFromOwnerInput.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null }
$record = Get-Content -LiteralPath $ImportPath -Raw -Encoding utf8 | ConvertFrom-Json
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$items = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "github-ci-evidence-from-owner-input") "blocker" "recordKind must match."
  New-OwnerValidationItem "blocked-or-accepted-state" ([string](Get-PropertyOrDefault -Object $record -Name "importState" -DefaultValue "") -in @("blocked-github-ci-evidence-owner-input-required", "github-ci-evidence-owner-input-accepted-non-proof")) "blocker" "Import state must be explicit."
  New-OwnerValidationItem "non-proof-flags" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) "blocker" "CI import must not publish, use tokens, close, or claim proof."
  New-OwnerValidationItem "boundary" ($boundary.Contains("queued", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must reject queued/local substitutes."
)
if (-not [bool](Get-PropertyOrDefault -Object $record -Name "ciEvidenceAccepted" -DefaultValue $false)) {
  $items += New-OwnerValidationItem "blocked-has-findings" ([int](Get-PropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue 0) -gt 0) "blocker" "Blocked CI input must include action-required findings."
}
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "github-ci-evidence-from-owner-input-validation-ready-non-proof" } else { "invalid-github-ci-evidence-from-owner-input" }
$validation = [pscustomobject]@{
  recordKind = "github-ci-evidence-from-owner-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  ciEvidenceAccepted = [bool](Get-PropertyOrDefault -Object $record -Name "ciEvidenceAccepted" -DefaultValue $false)
  failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  validationItems = @($items)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "GitHub CI evidence validation is Owner evidence admission only; CI success is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "github-ci-evidence-from-owner-input-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "github-ci-evidence-from-owner-input-validation.md") -InputObject @("# GitHub CI Evidence From Owner Input Validation", "", "- validationState: ``$state``", "- ciEvidenceAccepted: ``$($validation.ciEvidenceAccepted)``", "- failedBlockerCount: ``$($failedBlockers.Count)``", "", $validation.boundary)
Write-Host "GitHubCiEvidenceFromOwnerInputValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "GitHub CI evidence Owner input validation failed." }
