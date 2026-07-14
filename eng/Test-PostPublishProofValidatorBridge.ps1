[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-proof-validator-bridge.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PostPublishProofValidatorBridge.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$lanes = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @()))
$items = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "post-publish-proof-validator-bridge") "blocker" "recordKind must match."
  New-OwnerValidationItem "lane-coverage" ([int](Get-PropertyOrDefault -Object $record -Name "laneCount" -DefaultValue 0) -ge 4 -and $lanes.Count -ge 4) "blocker" "Bridge must cover public package, external CleanConsumer, article publication, and YoloVision lanes."
  New-OwnerValidationItem "public-hash-boundary" ([bool](Get-PropertyOrDefault -Object $record -Name "publicPackageHashCannotSubstitutePostPublishProof" -DefaultValue $false)) "blocker" "Public package hash/download verification cannot substitute post-publish runtime proof."
  New-OwnerValidationItem "shape-boundary" ([bool](Get-PropertyOrDefault -Object $record -Name "shapeValidCannotSubstitutePostPublishProof" -DefaultValue $false)) "blocker" "Staging shape-valid states cannot substitute post-publish proof."
  New-OwnerValidationItem "non-proof-flags" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) "blocker" "Bridge must not publish, use tokens, close, or claim proof."
  New-OwnerValidationItem "blocked-or-ready-state" ([string](Get-PropertyOrDefault -Object $record -Name "bridgeState" -DefaultValue "") -in @("blocked-post-publish-proof-validator-bridge-real-owner-proof-required", "post-publish-proof-validator-bridge-ready-for-owner-close-review-non-proof")) "blocker" "Bridge state must be explicit."
  New-OwnerValidationItem "boundary" ($boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("cannot substitute post-publish CleanConsumer runtime proof", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must reject public hash/staging substitutions."
)
if (-not [bool](Get-PropertyOrDefault -Object $record -Name "allPostPublishInputsAccepted" -DefaultValue $false)) {
  $items += New-OwnerValidationItem "blocked-reasons" ([int](Get-PropertyOrDefault -Object $record -Name "blockedReasonCount" -DefaultValue 0) -gt 0) "blocker" "Blocked bridge must include blocked reasons."
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "post-publish-proof-validator-bridge-validation-ready-non-proof" } else { "invalid-post-publish-proof-validator-bridge" }
$validation = [pscustomobject]@{
  recordKind = "post-publish-proof-validator-bridge-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  bridgeState = [string](Get-PropertyOrDefault -Object $record -Name "bridgeState" -DefaultValue "")
  laneCount = [int](Get-PropertyOrDefault -Object $record -Name "laneCount" -DefaultValue 0)
  inputShapeReadyLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "inputShapeReadyLaneCount" -DefaultValue 0)
  proofReadyLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "proofReadyLaneCount" -DefaultValue 0)
  blockedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0)
  publicPackageHashCannotSubstitutePostPublishProof = [bool](Get-PropertyOrDefault -Object $record -Name "publicPackageHashCannotSubstitutePostPublishProof" -DefaultValue $false)
  shapeValidCannotSubstitutePostPublishProof = [bool](Get-PropertyOrDefault -Object $record -Name "shapeValidCannotSubstitutePostPublishProof" -DefaultValue $false)
  allPostPublishInputsAccepted = [bool](Get-PropertyOrDefault -Object $record -Name "allPostPublishInputsAccepted" -DefaultValue $false)
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
  boundary = "Post-publish proof validator bridge validation is aggregation only; public package hashes and staging shape-valid states are not post-publish proof, not release close approval, and not package push."
}

Write-Utf8File -LiteralPath (Join-Path $OutputRoot "post-publish-proof-validator-bridge-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "post-publish-proof-validator-bridge-validation.md") -InputObject @("# Post-Publish Proof Validator Bridge Validation", "", "- validationState: ``$state``", "- lanes: ``$($validation.inputShapeReadyLaneCount)/$($validation.laneCount)`` input-shape-ready", "- proofReadyLaneCount: ``$($validation.proofReadyLaneCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "PostPublishProofValidatorBridgeValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Post-publish proof validator bridge validation failed." }
