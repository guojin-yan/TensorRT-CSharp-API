[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

function New-BackfillItem {
  param([int]$Order, [string]$Group, [string]$Id, [string]$TargetJson, [string]$TargetField, [string]$SourceHint, [string]$Validator)
  [pscustomobject]@{
    order = $Order
    group = $Group
    id = $Id
    targetJson = $TargetJson
    targetField = $TargetField
    sourceHint = $SourceHint
    strictValidator = $Validator
    ownerActionRequired = $true
    passed = $false
  }
}

$executionPackage = Read-JsonOrNull $RepositoryRoot "artifacts\final-release\final-owner-real-proof-execution-package.json"
$gapMatrix = Read-JsonOrNull $RepositoryRoot "artifacts\final-release\final-owner-real-proof-gap-matrix.json"
$gaps = @(Convert-ToArray (Get-PropertyOrDefault -Object $gapMatrix -Name "gaps" -DefaultValue @()))

$items = New-Object System.Collections.Generic.List[object]
$order = 0
foreach ($gap in $gaps) {
  $order++
  $category = [string](Get-PropertyOrDefault -Object $gap -Name "category" -DefaultValue "owner-real-proof")
  $gapType = [string](Get-PropertyOrDefault -Object $gap -Name "gapType" -DefaultValue "missing-field")
  $field = [string](Get-PropertyOrDefault -Object $gap -Name "fieldOrArtifact" -DefaultValue "ownerField")
  $sourceArtifact = [string](Get-PropertyOrDefault -Object $gap -Name "sourceArtifact" -DefaultValue "")
  $targetJson = if ($category -eq "post-publish-clean-consumer-proof") {
    "post-publish-clean-consumer-proof-result.owner.json"
  } elseif ($category -eq "rollback-review") {
    "final-owner-rollback-review.owner.json"
  } elseif ($category -eq "final-close-decision") {
    "final-owner-close-decision.owner.json"
  } else {
    "external-clean-consumer-execution-result.owner.json"
  }
  $validator = if ($category -eq "post-publish-clean-consumer-proof") {
    "eng/Import-PostPublishCleanConsumerProofResult.ps1 + eng/Test-PostPublishCleanConsumerProofResult.ps1 -Strict -FailOnNotProof"
  } elseif ($category -eq "rollback-review") {
    "eng/Import-FinalOwnerRollbackReview.ps1 + eng/Test-FinalOwnerRollbackReview.ps1 -Strict"
  } elseif ($category -eq "final-close-decision") {
    "eng/Import-FinalOwnerCloseDecision.ps1 + eng/Test-FinalOwnerCloseDecision.ps1 -Strict"
  } else {
    "eng/Import-ExternalCleanConsumerExecutionResult.ps1 + eng/Test-ExternalCleanConsumerExecutionResult.ps1 -Strict -FailOnNotProof"
  }

  $items.Add((New-BackfillItem $order $gapType "$category-$field" $targetJson $field $sourceArtifact $validator)) | Out-Null
}

if ($items.Count -eq 0) {
  $order++
  $items.Add((New-BackfillItem $order "missing-owner-confirmation" "owner-real-proof-no-gap-matrix-items" "owner-real-proof-staging-workspace" "ownerEvidence" "final-owner-real-proof-gap-matrix.json" "Run gap matrix export first.")) | Out-Null
}

$groupCounts = @($items.ToArray() | Group-Object group | ForEach-Object { [pscustomobject]@{ group = $_.Name; count = $_.Count } })

$record = [pscustomobject]@{
  recordKind = "owner-real-proof-evidence-backfill-package"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  packageState = "blocked-owner-real-proof-evidence-backfill-required"
  sourceExecutionPackageState = [string](Get-PropertyOrDefault -Object $executionPackage -Name "packageState" -DefaultValue "missing-final-owner-real-proof-execution-package")
  sourceGapMatrixState = [string](Get-PropertyOrDefault -Object $gapMatrix -Name "matrixState" -DefaultValue "missing-final-owner-real-proof-gap-matrix")
  backfillItemCount = $items.Count
  groupCounts = @($groupCounts)
  backfillItems = @($items.ToArray())
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
  boundary = "Owner real proof evidence backfill package is a field-level owner worklist only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-real-proof-evidence-backfill-package.json"
$mdPath = Join-Path $OutputRoot "owner-real-proof-evidence-backfill-package.md"
$record | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in ($items.ToArray() | Select-Object -First 160)) {
  "| $($item.order) | ``$($item.group)`` | ``$(ConvertTo-MarkdownCell $item.targetJson)`` | ``$(ConvertTo-MarkdownCell $item.targetField)`` | $(ConvertTo-MarkdownCell $item.strictValidator) |"
}
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Real Proof Evidence Backfill Package",
  "",
  "- packageState: ``$($record.packageState)``",
  "- backfillItemCount: ``$($record.backfillItemCount)``",
  "- passed: ``False``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "| Order | Group | Target JSON | Target Field | Validator |",
  "|---:|---|---|---|---|",
  @($rows),
  "",
  "## Boundary",
  "",
  $record.boundary
)

Write-Host "Wrote $jsonPath"
Write-Host "Wrote $mdPath"
