[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-article-blocked-claim-owner-review-list.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PublicArticleBlockedClaimOwnerReviewList.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$reviewItems = @(Get-PropertyOrDefault -Object $record -Name "reviewItems" -DefaultValue @())
$actionGroups = @(Get-PropertyOrDefault -Object $record -Name "actionGroups" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12

$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-article-blocked-claim-owner-review-list") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "review-items" ($reviewItems.Count -eq [int](Get-PropertyOrDefault -Object $record -Name "blockedClaimCount" -DefaultValue -1) -and $reviewItems.Count -ge 1) "blocker" "Review list must include blocked claims from the scan.")) | Out-Null
$items.Add((New-OwnerValidationItem "action-classification" ($text.Contains("remove") -and $text.Contains("downgrade-to-roadmap") -and $text.Contains("wait-for-real-proof") -and $text.Contains("bind-to-evidence-field")) "blocker" "Review list must include all allowed action categories.")) | Out-Null
$items.Add((New-OwnerValidationItem "proof-gate-binding" ($text.Contains("post-publish-owner-input") -and $text.Contains("release-close-strict-closure") -and $text.Contains("forbidden-substitute-scan")) "blocker" "Review items must bind currently observed blocked claims to real proof fields/gates.")) | Out-Null
$items.Add((New-OwnerValidationItem "action-groups" ($actionGroups.Count -ge 1 -and [int](Get-PropertyOrDefault -Object $record -Name "actionGroupCount" -DefaultValue 0) -eq $actionGroups.Count) "blocker" "Action groups must be summarized.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Review list must not publish packages/articles or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "public-article-blocked-claim-owner-review-list-validation-ready-non-proof" } else { "blocked-public-article-blocked-claim-owner-review-list-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "public-article-blocked-claim-owner-review-list-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  blockedClaimCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedClaimCount" -DefaultValue 0)
  actionGroupCount = [int](Get-PropertyOrDefault -Object $record -Name "actionGroupCount" -DefaultValue 0)
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Public article blocked claim owner review list validation only; not article publication, not package publication, not proof promotion, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "public-article-blocked-claim-owner-review-list-validation.json"
$mdPath = Join-Path $OutputRoot "public-article-blocked-claim-owner-review-list-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Public Article Blocked Claim Owner Review List Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- blockedClaimCount: ``$($validation.blockedClaimCount)``",
  "- actionGroupCount: ``$($validation.actionGroupCount)``",
  "",
  $validation.boundary
)
Write-Host "PublicArticleBlockedClaimOwnerReviewListValidationState=$state FailedBlockers=$failedBlockerCount BlockedClaims=$($validation.blockedClaimCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Public article blocked claim owner review list validation failed." }
