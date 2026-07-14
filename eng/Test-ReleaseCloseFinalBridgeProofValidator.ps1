[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-final-bridge-proof-validator.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$resolvedInputPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-ReleaseCloseFinalBridgeProofValidator.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = @(Test-OwnerPostPublishProofValidatorArtifact -Record $record -RecordKind "release-close-final-bridge-proof-validator" -ExpectedBlockedState "blocked-release-close-final-bridge-real-owner-proof-required" -RequiredBoundaryText "not release close approval")
$items += (New-OwnerValidationItem "dependency-results" ([int](Get-PropertyOrDefault -Object $record -Name "dependencyRequiredCount" -DefaultValue 0) -ge 4) "blocker" "Release close bridge must inspect the four upstream post-publish proof validators.")
$items += (New-OwnerValidationItem "can-close-false" (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Release close bridge validator must not close the issue by itself.")
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "release-close-final-bridge-proof-validator-ready-non-proof" } else { "invalid-release-close-final-bridge-proof-validator" }
$validation = [pscustomobject]@{
  recordKind = "release-close-final-bridge-proof-validator-validation"; generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O"); validationState = $state; validatorState = [string]$record.validatorState; ownerEvidenceAccepted = [bool]$record.ownerEvidenceAccepted; requiredFieldCount = [int]$record.requiredFieldCount; readyFieldCount = [int]$record.readyFieldCount; blockedFieldCount = [int]$record.blockedFieldCount; blockedReasonCount = [int]$record.blockedReasonCount; dependencyRequiredCount = [int]$record.dependencyRequiredCount; dependencyAcceptedCount = [int]$record.dependencyAcceptedCount; failedBlockerCount = $failedBlockers.Count; validationItems = @($items); performsPublish = $false; usesPublishToken = $false; canPublishPublicly = $false; canCloseReleaseIssue = $false; canPromoteRuntimeProof = $false; isRuntimeExecutionProof = $false; isPostPublishProof = $false; isReleaseCloseProof = $false; boundary = "Release close final bridge proof validator validation is strict Owner input admission only; not release close approval, not post-publish proof, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "release-close-final-bridge-proof-validator-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "release-close-final-bridge-proof-validator-validation.md") -InputObject @("# Release Close Final Bridge Proof Validator Validation", "", "- validationState: ``$state``", "- ownerEvidenceAccepted: ``$($validation.ownerEvidenceAccepted)``", "- dependencies: ``$($validation.dependencyAcceptedCount)/$($validation.dependencyRequiredCount)``", "- blockedReasonCount: ``$($validation.blockedReasonCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "ReleaseCloseFinalBridgeProofValidatorValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Release close final bridge proof validator validation failed." }
