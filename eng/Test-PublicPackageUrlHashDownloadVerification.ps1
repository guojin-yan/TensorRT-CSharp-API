[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-package-url-hash-download-verification.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PublicPackageUrlHashDownloadVerification.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$realOwnerInputPresent = [bool]$record.realOwnerInputPresent
$downloadAllowed = [bool]$record.downloadAllowed
$downloadAttemptedCount = [int]$record.downloadAttemptedCount
$blockedReasonCount = [int]$record.blockedReasonCount
$blockedWithoutOwnerInputPassed = $realOwnerInputPresent -or ((-not $downloadAllowed) -and ($downloadAttemptedCount -eq 0) -and ($blockedReasonCount -gt 0))
$boundaryText = [string]$record.boundary
$boundaryPassed = $boundaryText.Contains("does not publish", [StringComparison]::OrdinalIgnoreCase) -and $boundaryText.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase)
$items.Add((New-OwnerValidationItem "record-kind" ([string]$record.recordKind -eq "public-package-url-hash-download-verification") "blocker" "Download verification recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "package-checks" ([int]$record.packageCheckCount -ge 2) "blocker" "Download verification must track NuGet managed and GitHub runtime package checks.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-without-owner-input" $blockedWithoutOwnerInputPassed "blocker" "Download verification must not download without real Owner input and accepted public package validator.")) | Out-Null
$items.Add((New-OwnerValidationItem "non-proof-flags" (Assert-OwnerPostPublishFalseFlags -Record $record) "blocker" "Download verification must not publish, close, or promote proof by itself.")) | Out-Null
$items.Add((New-OwnerValidationItem "boundary" $boundaryPassed "blocker" "Download verification boundary must explicitly reject proof substitution.")) | Out-Null
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "public-package-url-hash-download-verification-ready-non-proof" } else { "invalid-public-package-url-hash-download-verification" }
$validation = [pscustomobject]@{
  recordKind = "public-package-url-hash-download-verification-validation"; generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O"); validationState = $state; verificationState = [string]$record.verificationState; realOwnerInputPresent = [bool]$record.realOwnerInputPresent; downloadAllowed = [bool]$record.downloadAllowed; downloadVerificationReady = [bool]$record.downloadVerificationReady; packageCheckCount = [int]$record.packageCheckCount; downloadAttemptedCount = [int]$record.downloadAttemptedCount; hashMatchedCount = [int]$record.hashMatchedCount; blockedReasonCount = [int]$record.blockedReasonCount; failedBlockerCount = $failedBlockers.Count; validationItems = @($items.ToArray()); performsPublish = $false; usesPublishToken = $false; canPublishPublicly = $false; canCloseReleaseIssue = $false; canPromoteRuntimeProof = $false; isRuntimeExecutionProof = $false; isPostPublishProof = $false; isReleaseCloseProof = $false; boundary = "Public package URL/hash download verification validation is non-proof; not post-publish proof, not publish approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "public-package-url-hash-download-verification-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "public-package-url-hash-download-verification-validation.md") -InputObject @("# Public Package URL/Hash Download Verification Validation", "", "- validationState: ``$state``", "- downloadAllowed: ``$($validation.downloadAllowed)``", "- downloadAttemptedCount: ``$($validation.downloadAttemptedCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "PublicPackageUrlHashDownloadVerificationValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Public package URL/hash download verification validation failed." }
