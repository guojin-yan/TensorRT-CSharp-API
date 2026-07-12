[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-article-source-patch-apply-readiness-pack.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PublicArticleSourcePatchApplyReadinessPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = @(Get-PropertyOrDefault -Object $record -Name "readinessItems" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12
$validationItems = New-Object System.Collections.Generic.List[object]

$validationItems.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-article-source-patch-apply-readiness-pack") "blocker" "recordKind must match.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "readiness-count" ($items.Count -ge 64 -and [int](Get-PropertyOrDefault -Object $record -Name "readinessItemCount" -DefaultValue 0) -eq $items.Count) "blocker" "Readiness pack must include the patch proposals.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "artifact-only" (-not [bool](Get-PropertyOrDefault -Object $record -Name "writesSourceArticles" -DefaultValue $true) -and $text.Contains("artifact-only-no-source-overwrite") -and $text.Contains("appliesPatchNow")) "blocker" "Readiness pack must be artifact-only and must not overwrite article sources.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "mapping-fields" ($text.Contains("lineExists") -and $text.Contains("matchedTextStillPresent") -and $text.Contains("nearbyMatchFound") -and $text.Contains("canApplyAfterOwnerApproval")) "blocker" "Readiness items must include mapping and owner-approval fields.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "mapping-or-review" (($items | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "readinessState" -DefaultValue "") -notin @("source-line-mapped-ready-after-owner-approval","owner-review-required-source-line-not-mapped") }).Count -eq 0) "blocker" "Each item must be mapped or explicitly require Owner review.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "non-proof-language" ($text.Contains("证据导入前") -and $text.Contains("不能声明已发布") -and $text.Contains("proof complete")) "blocker" "Readiness output must retain non-proof language.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Readiness pack must not publish packages/articles or close release.")) | Out-Null

$failed = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "public-article-source-patch-apply-readiness-pack-validation-ready-non-proof" } else { "blocked-public-article-source-patch-apply-readiness-pack-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "public-article-source-patch-apply-readiness-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$validationItems.Count
  readinessItemCount = [int](Get-PropertyOrDefault -Object $record -Name "readinessItemCount" -DefaultValue 0)
  mappedItemCount = [int](Get-PropertyOrDefault -Object $record -Name "mappedItemCount" -DefaultValue 0)
  ownerReviewRequiredItemCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerReviewRequiredItemCount" -DefaultValue 0)
  validationItems = @($validationItems.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Public article source patch apply readiness validation only; not article publication, not package publication, not proof promotion, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "public-article-source-patch-apply-readiness-pack-validation.json"
$mdPath = Join-Path $OutputRoot "public-article-source-patch-apply-readiness-pack-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Public Article Source Patch Apply Readiness Pack Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- readinessItemCount: ``$($validation.readinessItemCount)``",
  "- mappedItemCount: ``$($validation.mappedItemCount)``",
  "- ownerReviewRequiredItemCount: ``$($validation.ownerReviewRequiredItemCount)``",
  "",
  $validation.boundary
)
Write-Host "PublicArticleSourcePatchApplyReadinessPackValidationState=$state FailedBlockers=$failedBlockerCount Items=$($validation.readinessItemCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Public article source patch apply readiness pack validation failed." }
