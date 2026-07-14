[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\external-clean-consumer-post-publish-proof-validator.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-ExternalCleanConsumerPostPublishProofValidator.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = @(Test-OwnerPostPublishProofValidatorArtifact -Record $record -RecordKind "external-clean-consumer-post-publish-proof-validator" -ExpectedBlockedState "blocked-external-clean-consumer-real-owner-proof-required" -RequiredBoundaryText "not post-publish proof")
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "external-clean-consumer-post-publish-proof-validator-ready-non-proof" } else { "invalid-external-clean-consumer-post-publish-proof-validator" }
$validation = [pscustomobject]@{
  recordKind = "external-clean-consumer-post-publish-proof-validator-validation"; generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O"); validationState = $state; validatorState = [string]$record.validatorState; ownerEvidenceAccepted = [bool]$record.ownerEvidenceAccepted; requiredFieldCount = [int]$record.requiredFieldCount; readyFieldCount = [int]$record.readyFieldCount; blockedFieldCount = [int]$record.blockedFieldCount; blockedReasonCount = [int]$record.blockedReasonCount; failedBlockerCount = $failedBlockers.Count; validationItems = @($items); performsPublish = $false; usesPublishToken = $false; canPublishPublicly = $false; canCloseReleaseIssue = $false; canPromoteRuntimeProof = $false; isRuntimeExecutionProof = $false; isPostPublishProof = $false; isReleaseCloseProof = $false; boundary = "External clean consumer post-publish proof validator validation is strict Owner input admission only; not runtime proof, not post-publish proof, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "external-clean-consumer-post-publish-proof-validator-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "external-clean-consumer-post-publish-proof-validator-validation.md") -InputObject @("# External Clean Consumer Post-Publish Proof Validator Validation", "", "- validationState: ``$state``", "- ownerEvidenceAccepted: ``$($validation.ownerEvidenceAccepted)``", "- blockedReasonCount: ``$($validation.blockedReasonCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "ExternalCleanConsumerPostPublishProofValidatorValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "External clean consumer post-publish proof validator validation failed." }
