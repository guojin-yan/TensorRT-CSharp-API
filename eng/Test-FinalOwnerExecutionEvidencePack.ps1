[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-evidence-pack.json",
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
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) { & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerExecutionEvidencePack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null }
$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$gates = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "gates" -DefaultValue @()))
$gateIds = @($gates | ForEach-Object { [string]$_.id })
$requiredGateIds = @("final-owner-execution-input-skeleton", "github-ci-evidence-from-owner-input", "release-evidence-bundle-hash-review", "classification-audit-hash-review", "post-publish-proof-validator-bridge", "release-close-final-bridge", "real-owner-proof-convergence-dashboard")
$missingGateIds = @($requiredGateIds | Where-Object { $gateIds -notcontains $_ })
$rejected = @((Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "rejectedNonProofStates" -DefaultValue @())) | ForEach-Object { [string]$_ })
$requiredRejected = @("template", "dashboard", "dry-run", "local-feed", "ProjectReference", "direct-nupkg", "queued-workflow", "hash-only", "validation-ready-without-owner-proof")
$missingRejected = @($requiredRejected | Where-Object { $rejected -notcontains $_ })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$items = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-execution-evidence-pack") "blocker" "recordKind must match."
  New-OwnerValidationItem "gate-coverage" ($missingGateIds.Count -eq 0 -and $gates.Count -ge 7) "blocker" "Pack must cover skeleton, CI, hash reviews, bridges, and unified Owner proof convergence."
  New-OwnerValidationItem "rejected-non-proof-states" ($missingRejected.Count -eq 0) "blocker" "Pack must reject common non-proof substitutes."
  New-OwnerValidationItem "blocked-or-ready-state" ([string](Get-PropertyOrDefault -Object $record -Name "packState" -DefaultValue "") -in @("blocked-final-owner-execution-evidence-pack-real-owner-proof-required", "final-owner-execution-evidence-pack-ready-for-owner-release-review-non-proof")) "blocker" "Pack state must be explicit."
  New-OwnerValidationItem "non-proof-flags" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) "blocker" "Pack must not publish, use tokens, close, or claim proof."
  New-OwnerValidationItem "boundary" ($boundary.Contains("does not publish", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("does not use tokens", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must reject publish/token/close/proof substitution."
)
if (-not [bool](Get-PropertyOrDefault -Object $record -Name "allOwnerExecutionInputsReady" -DefaultValue $false)) {
  $items += New-OwnerValidationItem "blocked-reasons" ([int](Get-PropertyOrDefault -Object $record -Name "blockedReasonCount" -DefaultValue 0) -gt 0 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedGateCount" -DefaultValue 0) -gt 0) "blocker" "Blocked pack must include blocked reasons."
}
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "final-owner-execution-evidence-pack-validation-ready-non-proof" } else { "invalid-final-owner-execution-evidence-pack" }
$validation = [pscustomobject]@{
  recordKind = "final-owner-execution-evidence-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  packState = [string](Get-PropertyOrDefault -Object $record -Name "packState" -DefaultValue "")
  gateCount = $gates.Count
  readyGateCount = [int](Get-PropertyOrDefault -Object $record -Name "readyGateCount" -DefaultValue 0)
  blockedGateCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedGateCount" -DefaultValue 0)
  allOwnerExecutionInputsReady = [bool](Get-PropertyOrDefault -Object $record -Name "allOwnerExecutionInputsReady" -DefaultValue $false)
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
  boundary = "Final Owner execution evidence pack validation is aggregation only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "final-owner-execution-evidence-pack-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "final-owner-execution-evidence-pack-validation.md") -InputObject @("# Final Owner Execution Evidence Pack Validation", "", "- validationState: ``$state``", "- readyGateCount: ``$($validation.readyGateCount)/$($validation.gateCount)``", "- blockedGateCount: ``$($validation.blockedGateCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "FinalOwnerExecutionEvidencePackValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Final Owner execution evidence pack validation failed." }
