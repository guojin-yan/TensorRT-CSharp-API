[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-package-url-hash-proof-validator.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PublicPackageUrlHashProofValidator.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = @(Test-OwnerPostPublishProofValidatorArtifact -Record $record -RecordKind "public-package-url-hash-proof-validator" -ExpectedBlockedState "blocked-public-package-url-hash-real-owner-proof-required" -RequiredBoundaryText "not post-publish proof")
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "public-package-url-hash-proof-validator-ready-non-proof" } else { "invalid-public-package-url-hash-proof-validator" }
$validation = [pscustomobject]@{
  recordKind = "public-package-url-hash-proof-validator-validation"; generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O"); validationState = $state; validatorState = [string]$record.validatorState; ownerEvidenceAccepted = [bool]$record.ownerEvidenceAccepted; requiredFieldCount = [int]$record.requiredFieldCount; readyFieldCount = [int]$record.readyFieldCount; blockedFieldCount = [int]$record.blockedFieldCount; blockedReasonCount = [int]$record.blockedReasonCount; failedBlockerCount = $failedBlockers.Count; validationItems = @($items); performsPublish = $false; usesPublishToken = $false; canPublishPublicly = $false; canCloseReleaseIssue = $false; canPromoteRuntimeProof = $false; isRuntimeExecutionProof = $false; isPostPublishProof = $false; isReleaseCloseProof = $false; boundary = "Public package URL/hash proof validator validation is strict Owner input admission only; not post-publish proof, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "public-package-url-hash-proof-validator-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "public-package-url-hash-proof-validator-validation.md") -InputObject @("# Public Package URL/Hash Proof Validator Validation", "", "- validationState: ``$state``", "- ownerEvidenceAccepted: ``$($validation.ownerEvidenceAccepted)``", "- blockedReasonCount: ``$($validation.blockedReasonCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "PublicPackageUrlHashProofValidatorValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Public package URL/hash proof validator validation failed." }
