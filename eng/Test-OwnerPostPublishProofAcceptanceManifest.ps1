[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-post-publish-proof-acceptance-manifest.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-OwnerPostPublishProofAcceptanceManifest.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = @(Test-OwnerPostPublishProofAcceptanceManifestArtifact -Record $record)
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "owner-post-publish-proof-acceptance-manifest-ready-non-proof" } else { "invalid-owner-post-publish-proof-acceptance-manifest" }
$validation = [pscustomobject]@{
  recordKind = "owner-post-publish-proof-acceptance-manifest-validation"; generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O"); validationState = $state; manifestState = [string]$record.manifestState; validatorCount = [int]$record.validatorCount; acceptedValidatorCount = [int]$record.acceptedValidatorCount; blockedValidatorCount = [int]$record.blockedValidatorCount; allValidatorsAccepted = [bool]$record.allValidatorsAccepted; readyForManualReleaseCloseReview = [bool]$record.readyForManualReleaseCloseReview; releaseCloseReady = $false; closeIssueCommandReady = $false; failedBlockerCount = $failedBlockers.Count; validationItems = @($items); ownerActionRequired = -not [bool]$record.allValidatorsAccepted; performsPublish = $false; usesPublishToken = $false; canPublishPublicly = $false; canCloseReleaseIssue = $false; canPromoteRuntimeProof = $false; isRuntimeExecutionProof = $false; isPostPublishProof = $false; isReleaseCloseProof = $false; boundary = "Owner post-publish proof acceptance manifest validation is aggregation only; not post-publish proof, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "owner-post-publish-proof-acceptance-manifest-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "owner-post-publish-proof-acceptance-manifest-validation.md") -InputObject @("# Owner Post-Publish Proof Acceptance Manifest Validation", "", "- validationState: ``$state``", "- validators: ``$($validation.acceptedValidatorCount)/$($validation.validatorCount)``", "- releaseCloseReady: ``False``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "OwnerPostPublishProofAcceptanceManifestValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Owner post-publish proof acceptance manifest validation failed." }
