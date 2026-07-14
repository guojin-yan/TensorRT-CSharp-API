[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-input-skeleton.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$resolvedInputPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerExecutionInputSkeleton.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @()))
$ids = @($lanes | ForEach-Object { [string]$_.id })
$requiredIds = @("public-package-url-hash", "external-clean-consumer-post-publish", "article-publication", "yolovision-real-model", "final-owner-rollback-review", "final-owner-close-decision", "github-ci-evidence", "release-evidence-bundle-hash-review", "classification-audit-hash-review")
$missingIds = @($requiredIds | Where-Object { $ids -notcontains $_ })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$items = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-execution-input-skeleton") "blocker" "recordKind must match."
  New-OwnerValidationItem "lane-coverage" ($missingIds.Count -eq 0 -and $lanes.Count -ge 9) "blocker" "Skeleton must cover all final Owner proof lanes."
  New-OwnerValidationItem "required-fields" ([int](Get-PropertyOrDefault -Object $record -Name "requiredFieldCount" -DefaultValue 0) -ge 40) "blocker" "Skeleton must expose enough concrete Owner fields."
  New-OwnerValidationItem "template-artifact" (-not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $record -Name "templateArtifact" -DefaultValue ""))) "blocker" "Template artifact path is required."
  New-OwnerValidationItem "non-proof-flags" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) "blocker" "Skeleton must not publish, use tokens, close, or claim proof."
  New-OwnerValidationItem "boundary" ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must be explicit."
)
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "final-owner-execution-input-skeleton-validation-ready-non-proof" } else { "invalid-final-owner-execution-input-skeleton" }
$validation = [pscustomobject]@{
  recordKind = "final-owner-execution-input-skeleton-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  laneCount = $lanes.Count
  requiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "requiredFieldCount" -DefaultValue 0)
  missingRequiredLaneCount = $missingIds.Count
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
  boundary = "Final Owner execution input skeleton validation is schema/readiness only; it is not runtime proof, not post-publish proof, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "final-owner-execution-input-skeleton-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "final-owner-execution-input-skeleton-validation.md") -InputObject @("# Final Owner Execution Input Skeleton Validation", "", "- validationState: ``$state``", "- laneCount: ``$($validation.laneCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "FinalOwnerExecutionInputSkeletonValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Final Owner execution input skeleton validation failed." }
