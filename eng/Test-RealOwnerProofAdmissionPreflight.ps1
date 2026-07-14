[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-owner-proof-admission-preflight.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$resolvedInput = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInput -PathType Leaf)) { & (Join-Path $RepositoryRoot "eng\Export-RealOwnerProofAdmissionPreflight.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null }
$record = Get-Content -LiteralPath $resolvedInput -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @()))
$ids = @($lanes | ForEach-Object { [string]$_.id })
$requiredIds = @("public-package-url-hash", "external-clean-consumer-post-publish", "article-publication", "yolovision-real-model", "final-rollback-review", "final-close-decision", "github-ci-evidence", "release-evidence-bundle-hash-review", "classification-audit-hash-review")
$missingIds = @($requiredIds | Where-Object { $ids -notcontains $_ })
$rejected = @((Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "rejectedNonProofSubstitutes" -DefaultValue @())) | ForEach-Object { [string]$_ })
$requiredRejected = @("template", "dashboard", "dry-run", "local-feed", "ProjectReference", "direct-nupkg", "queued-workflow", "hash-only", "sample-build-only", "mock-output")
$missingRejected = @($requiredRejected | Where-Object { $rejected -notcontains $_ })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$items = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "real-owner-proof-admission-preflight") "blocker" "recordKind must match."
  New-OwnerValidationItem "lane-coverage" ($lanes.Count -eq 9 -and $missingIds.Count -eq 0) "blocker" "Preflight must cover all nine Owner proof lanes."
  New-OwnerValidationItem "structural-readiness" ([int](Get-PropertyOrDefault -Object $record -Name "structuralReadyLaneCount" -DefaultValue 0) -eq 9 -and @($lanes | Where-Object { -not [bool]$_.structuralReady }).Count -eq 0) "blocker" "All existing lane validators must be structurally valid."
  New-OwnerValidationItem "accepted-count-consistency" ([int](Get-PropertyOrDefault -Object $record -Name "acceptedLaneCount" -DefaultValue -1) -eq @($lanes | Where-Object { [bool]$_.accepted }).Count) "blocker" "Accepted lane count must match lane observations."
  New-OwnerValidationItem "blocked-state-has-actions" ([bool](Get-PropertyOrDefault -Object $record -Name "allRealOwnerProofInputsAccepted" -DefaultValue $false) -or ([int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0) -gt 0 -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerActionRequired" -DefaultValue $false))) "blocker" "Blocked preflight must preserve Owner action."
  New-OwnerValidationItem "rejected-substitutes" ($missingRejected.Count -eq 0) "blocker" "Preflight must reject common non-proof substitutes."
  New-OwnerValidationItem "non-proof-flags" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) "blocker" "Preflight aggregate must remain non-proof."
  New-OwnerValidationItem "boundary" ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must preserve proof and publish limits."
)
$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failed.Count -eq 0) { "real-owner-proof-admission-preflight-validation-ready-non-proof" } else { "invalid-real-owner-proof-admission-preflight" }
$validation = [pscustomobject]@{
  recordKind = "real-owner-proof-admission-preflight-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  preflightState = [string](Get-PropertyOrDefault -Object $record -Name "preflightState" -DefaultValue "")
  laneCount = $lanes.Count
  structuralReadyLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "structuralReadyLaneCount" -DefaultValue 0)
  acceptedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "acceptedLaneCount" -DefaultValue 0)
  blockedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0)
  allRealOwnerProofInputsAccepted = [bool](Get-PropertyOrDefault -Object $record -Name "allRealOwnerProofInputsAccepted" -DefaultValue $false)
  failedBlockerCount = $failed.Count
  validationItems = @($items)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Real Owner proof admission preflight validation checks aggregate structure only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "real-owner-proof-admission-preflight-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "real-owner-proof-admission-preflight-validation.md") -InputObject @("# Real Owner Proof Admission Preflight Validation", "", "- validationState: ``$state``", "- laneCount: ``$($validation.laneCount)``", "- acceptedLaneCount: ``$($validation.acceptedLaneCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "RealOwnerProofAdmissionPreflightValidationState=$state FailedBlockers=$($failed.Count)"
if ($Strict.IsPresent -and $failed.Count -gt 0) { throw "Real Owner proof admission preflight validation failed." }
