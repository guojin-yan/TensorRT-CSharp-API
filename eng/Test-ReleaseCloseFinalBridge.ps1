[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-final-bridge.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-ReleaseCloseFinalBridge.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$gates = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "closeGates" -DefaultValue @()))
$rejected = @((Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "rejectedNonProofStates" -DefaultValue @())) | ForEach-Object { [string]$_ })
$requiredRejected = @("template", "dashboard", "dry-run", "local-feed", "ProjectReference", "direct-nupkg", "queued-workflow", "staging-shape-valid-only", "public-package-hash-only", "validation-ready-without-owner-proof")
$missingRejected = @($requiredRejected | Where-Object { $rejected -notcontains $_ })
$items = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-close-final-bridge") "blocker" "recordKind must match."
  New-OwnerValidationItem "gate-coverage" ([int](Get-PropertyOrDefault -Object $record -Name "gateCount" -DefaultValue 0) -ge 6 -and $gates.Count -ge 6) "blocker" "Bridge must inspect post-publish, rollback, final close, staging, release bundle hash, and classification audit gates."
  New-OwnerValidationItem "non-proof-flags" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) "blocker" "Bridge must not publish, use tokens, close issue, or claim proof."
  New-OwnerValidationItem "blocked-or-ready-state" ([string](Get-PropertyOrDefault -Object $record -Name "bridgeState" -DefaultValue "") -in @("blocked-release-close-final-bridge-real-owner-proof-required", "release-close-final-bridge-ready-for-owner-manual-close-review-non-proof")) "blocker" "Bridge state must be explicit."
  New-OwnerValidationItem "rejected-non-proof-states" ($missingRejected.Count -eq 0) "blocker" "Bridge must reject template/dashboard/dry-run/local-feed/ProjectReference/direct-nupkg/queued/staging/hash-only substitutions."
  New-OwnerValidationItem "hash-fields" (([string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $record -Name "releaseEvidenceBundleSha256" -DefaultValue "")) -or (Test-Sha256Text (Get-PropertyOrDefault -Object $record -Name "releaseEvidenceBundleSha256" -DefaultValue ""))) -and ([string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $record -Name "classificationAuditSha256" -DefaultValue "")) -or (Test-Sha256Text (Get-PropertyOrDefault -Object $record -Name "classificationAuditSha256" -DefaultValue "")))) "blocker" "Hash fields must be empty or SHA256."
  New-OwnerValidationItem "boundary" ($boundary.Contains("does not publish", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("does not close the release issue", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must reject publish/close/proof substitution."
)
if (-not [bool](Get-PropertyOrDefault -Object $record -Name "allCloseInputsReady" -DefaultValue $false)) {
  $items += New-OwnerValidationItem "blocked-reasons" ([int](Get-PropertyOrDefault -Object $record -Name "blockedReasonCount" -DefaultValue 0) -gt 0 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedGateCount" -DefaultValue 0) -gt 0) "blocker" "Blocked bridge must include blocked gate reasons."
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "release-close-final-bridge-validation-ready-non-proof" } else { "invalid-release-close-final-bridge" }
$validation = [pscustomobject]@{
  recordKind = "release-close-final-bridge-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  bridgeState = [string](Get-PropertyOrDefault -Object $record -Name "bridgeState" -DefaultValue "")
  gateCount = [int](Get-PropertyOrDefault -Object $record -Name "gateCount" -DefaultValue 0)
  readyGateCount = [int](Get-PropertyOrDefault -Object $record -Name "readyGateCount" -DefaultValue 0)
  blockedGateCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedGateCount" -DefaultValue 0)
  allCloseInputsReady = [bool](Get-PropertyOrDefault -Object $record -Name "allCloseInputsReady" -DefaultValue $false)
  acceptedRealInputCount = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "acceptedRealInputIds" -DefaultValue @())).Count
  rejectedNonProofStateCount = $rejected.Count
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
  boundary = "Release close final bridge validation is aggregation only; it does not publish, does not use tokens, does not close the release issue, is not release close approval, and not package push."
}

Write-Utf8File -LiteralPath (Join-Path $OutputRoot "release-close-final-bridge-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "release-close-final-bridge-validation.md") -InputObject @("# Release Close Final Bridge Validation", "", "- validationState: ``$state``", "- readyGateCount: ``$($validation.readyGateCount)/$($validation.gateCount)``", "- blockedGateCount: ``$($validation.blockedGateCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "ReleaseCloseFinalBridgeValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Release close final bridge validation failed." }
