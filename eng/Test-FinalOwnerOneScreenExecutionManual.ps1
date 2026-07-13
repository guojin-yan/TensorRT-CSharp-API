[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-one-screen-execution-manual.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerOneScreenExecutionManual.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$steps = @(Get-PropertyOrDefault -Object $record -Name "steps" -DefaultValue @())
$releaseCloseRealInputChain = @(Convert-ToArray -Value (Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChain" -DefaultValue @()))
$forbidden = @(Get-PropertyOrDefault -Object $record -Name "forbiddenNonProofSubstitutes" -DefaultValue @())
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$requiredSteps = @("preflight-freeze", "owner-authorization", "public-publish-command", "package-page-and-download", "post-publish-clean-consumer", "rollback-review", "close-decision", "strict-final-verification")
$requiredSubstitutes = @("local feed", "ProjectReference", "direct .nupkg", "dashboard", "dry-run", "manual approval", "queued GitHub Actions run", "missing self-hosted runner", "sidecar-only", "TensorRtExec report", "public package download proof alone", "post-publish validation-ready without proofCandidateReady", "release evidence bundle hash only", "strict close validator output without real proof")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-one-screen-execution-manual") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-non-proof" (([string](Get-PropertyOrDefault -Object $record -Name "manualState" -DefaultValue "")).Contains("blocked") -and -not [bool](Get-PropertyOrDefault -Object $record -Name "manualIsProof" -DefaultValue $true)) "blocker" "Manual must remain blocked/non-proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsGitHubPackagesPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsNuGetPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Manual must never execute publish or close issue.")) | Out-Null
$items.Add((New-OwnerValidationItem "step-count" ($steps.Count -eq $requiredSteps.Count) "blocker" "Manual must include the full Owner execution order.")) | Out-Null
foreach ($id in $requiredSteps) {
  $items.Add((New-OwnerValidationItem "step-$id" (@($steps | Where-Object { [string]$_.id -eq $id -and [bool]$_.notExecutedByAutomation -and -not [bool]$_.performsPublish -and -not [bool]$_.canCloseReleaseIssue -and -not [string]::IsNullOrWhiteSpace([string]$_.validatorCommand) }).Count -eq 1) "blocker" "Missing or unsafe manual step: $id")) | Out-Null
}
foreach ($substitute in $requiredSubstitutes) {
  $items.Add((New-OwnerValidationItem "substitute-$substitute" (@($forbidden | Where-Object { [string]$_ -eq $substitute }).Count -eq 1) "blocker" "Missing forbidden substitute: $substitute")) | Out-Null
}
$boundaryOk = $boundary.IndexOf("may show dotnet nuget push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("never executes publish", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("GitHub Packages", [StringComparison]::OrdinalIgnoreCase) -ge 0
$items.Add((New-OwnerValidationItem "boundary" $boundaryOk "blocker" "Boundary must allow only Owner-only placeholders and forbid automation publish.")) | Out-Null

$releaseCloseRealInputChainCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainCount" -DefaultValue 0)
$releaseCloseRealInputChainRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainRequiredFieldCount" -DefaultValue 0)
$releaseCloseRealInputChainRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainRejectedSubstituteCount" -DefaultValue 0)
$releaseCloseRealInputChainSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainSourceReadinessSignalCount" -DefaultValue 0)
$releaseCloseRealInputChainBlockedRealInputCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainBlockedRealInputCount" -DefaultValue 0)
$publicPackageDownloadProofRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofRequiredFieldCount" -DefaultValue 0)
$publicPackageDownloadProofRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofRejectedSubstituteCount" -DefaultValue 0)
$publicPackageDownloadProofSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofSourceReadinessSignalCount" -DefaultValue 0)
$publicPackageDownloadProofCandidateReady = [bool](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofCandidateReady" -DefaultValue $true)
$postPublishCleanConsumerProofRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofRequiredFieldCount" -DefaultValue 0)
$postPublishCleanConsumerProofRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofRejectedSubstituteCount" -DefaultValue 0)
$postPublishCleanConsumerProofBlockedRealInputCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofBlockedRealInputCount" -DefaultValue 0)
$postPublishCleanConsumerProofSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofSourceReadinessSignalCount" -DefaultValue 0)
$postPublishCleanConsumerProofCandidateReady = [bool](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofCandidateReady" -DefaultValue $true)
$postPublishCleanConsumerProofSourceProofLinkageReady = [bool](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofSourceProofLinkageReady" -DefaultValue $true)
$releaseEvidenceBundleSha256 = [string](Get-PropertyOrDefault -Object $record -Name "releaseEvidenceBundleSha256" -DefaultValue "")
$finalCloseStrictValidatorOutputState = [string](Get-PropertyOrDefault -Object $record -Name "finalCloseStrictValidatorOutputState" -DefaultValue "")

$items.Add((New-OwnerValidationItem "release-close-real-input-chain-count" ($releaseCloseRealInputChainCount -eq 8 -and $releaseCloseRealInputChain.Count -eq 8) "blocker" "Manual must inherit the eight-step release-close real input chain from the one-screen pack.")) | Out-Null
$items.Add((New-OwnerValidationItem "release-close-real-input-required-fields" ($releaseCloseRealInputChainRequiredFieldCount -ge 100) "blocker" "Release-close real input chain must expose at least 100 required fields.")) | Out-Null
$items.Add((New-OwnerValidationItem "release-close-real-input-rejected-substitutes" ($releaseCloseRealInputChainRejectedSubstituteCount -ge 30) "blocker" "Release-close real input chain must expose rejected substitute count.")) | Out-Null
$items.Add((New-OwnerValidationItem "release-close-real-input-source-signals" ($releaseCloseRealInputChainSourceReadinessSignalCount -eq 18) "blocker" "Release-close real input chain source readiness signal count must match one-screen pack.")) | Out-Null
$items.Add((New-OwnerValidationItem "release-close-real-input-blocked-inputs" ($releaseCloseRealInputChainBlockedRealInputCount -gt 0) "blocker" "Release-close real input chain must remain blocked on missing real inputs.")) | Out-Null
$items.Add((New-OwnerValidationItem "public-download-contract" ($publicPackageDownloadProofRequiredFieldCount -ge 30 -and $publicPackageDownloadProofRejectedSubstituteCount -eq 11 -and $publicPackageDownloadProofSourceReadinessSignalCount -eq 7 -and -not $publicPackageDownloadProofCandidateReady) "blocker" "Manual must carry public package download proof counts while keeping it blocked/non-proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "post-publish-contract" ($postPublishCleanConsumerProofRequiredFieldCount -ge 50 -and $postPublishCleanConsumerProofRejectedSubstituteCount -eq 11 -and $postPublishCleanConsumerProofBlockedRealInputCount -gt 0 -and $postPublishCleanConsumerProofSourceReadinessSignalCount -eq 11 -and -not $postPublishCleanConsumerProofCandidateReady -and -not $postPublishCleanConsumerProofSourceProofLinkageReady) "blocker" "Manual must carry post-publish CleanConsumer proof counts while keeping proof/linkage blocked.")) | Out-Null
$items.Add((New-OwnerValidationItem "release-evidence-bundle-sha" (Test-Sha256Text -Value $releaseEvidenceBundleSha256) "blocker" "Manual must carry a 64-hex release evidence bundle SHA from the one-screen pack.")) | Out-Null
$items.Add((New-OwnerValidationItem "strict-close-output-state" ($finalCloseStrictValidatorOutputState -eq "blocked-final-close-gate-owner-proof-required") "blocker" "Manual must carry the strict close validator output state and keep close blocked.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "final-owner-one-screen-execution-manual-validation-ready-non-proof" } else { "blocked-final-owner-one-screen-execution-manual-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "final-owner-one-screen-execution-manual-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  stepCount = [int]$steps.Count
  releaseCloseRealInputChainCount = $releaseCloseRealInputChainCount
  releaseCloseRealInputChainRequiredFieldCount = $releaseCloseRealInputChainRequiredFieldCount
  releaseCloseRealInputChainRejectedSubstituteCount = $releaseCloseRealInputChainRejectedSubstituteCount
  releaseCloseRealInputChainSourceReadinessSignalCount = $releaseCloseRealInputChainSourceReadinessSignalCount
  releaseCloseRealInputChainBlockedRealInputCount = $releaseCloseRealInputChainBlockedRealInputCount
  publicPackageDownloadProofRequiredFieldCount = $publicPackageDownloadProofRequiredFieldCount
  publicPackageDownloadProofRejectedSubstituteCount = $publicPackageDownloadProofRejectedSubstituteCount
  publicPackageDownloadProofSourceReadinessSignalCount = $publicPackageDownloadProofSourceReadinessSignalCount
  publicPackageDownloadProofCandidateReady = $publicPackageDownloadProofCandidateReady
  postPublishCleanConsumerProofRequiredFieldCount = $postPublishCleanConsumerProofRequiredFieldCount
  postPublishCleanConsumerProofRejectedSubstituteCount = $postPublishCleanConsumerProofRejectedSubstituteCount
  postPublishCleanConsumerProofBlockedRealInputCount = $postPublishCleanConsumerProofBlockedRealInputCount
  postPublishCleanConsumerProofSourceReadinessSignalCount = $postPublishCleanConsumerProofSourceReadinessSignalCount
  postPublishCleanConsumerProofCandidateReady = $postPublishCleanConsumerProofCandidateReady
  postPublishCleanConsumerProofSourceProofLinkageReady = $postPublishCleanConsumerProofSourceProofLinkageReady
  releaseEvidenceBundleSha256 = $releaseEvidenceBundleSha256
  finalCloseStrictValidatorOutputState = $finalCloseStrictValidatorOutputState
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  manualIsProof = $false
  performsPublish = $false
  performsGitHubPackagesPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseCloseProof = $false
  boundary = "Manual validation only; not proof, not publish approval, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "final-owner-one-screen-execution-manual-validation.json"
$mdPath = Join-Path $OutputRoot "final-owner-one-screen-execution-manual-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Final Owner One-Screen Execution Manual Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- stepCount: ``$($validation.stepCount)``",
  "- releaseCloseRealInputChainCount: ``$($validation.releaseCloseRealInputChainCount)``",
  "- releaseCloseRealInputChainRequiredFieldCount: ``$($validation.releaseCloseRealInputChainRequiredFieldCount)``",
  "- releaseCloseRealInputChainRejectedSubstituteCount: ``$($validation.releaseCloseRealInputChainRejectedSubstituteCount)``",
  "- releaseCloseRealInputChainSourceReadinessSignalCount: ``$($validation.releaseCloseRealInputChainSourceReadinessSignalCount)``",
  "- releaseCloseRealInputChainBlockedRealInputCount: ``$($validation.releaseCloseRealInputChainBlockedRealInputCount)``",
  "- publicPackageDownloadProofRequiredFieldCount: ``$($validation.publicPackageDownloadProofRequiredFieldCount)``",
  "- publicPackageDownloadProofCandidateReady: ``$($validation.publicPackageDownloadProofCandidateReady)``",
  "- postPublishCleanConsumerProofRequiredFieldCount: ``$($validation.postPublishCleanConsumerProofRequiredFieldCount)``",
  "- postPublishCleanConsumerProofCandidateReady: ``$($validation.postPublishCleanConsumerProofCandidateReady)``",
  "- postPublishCleanConsumerProofSourceProofLinkageReady: ``$($validation.postPublishCleanConsumerProofSourceProofLinkageReady)``",
  "- releaseEvidenceBundleSha256: ``$($validation.releaseEvidenceBundleSha256)``",
  "- finalCloseStrictValidatorOutputState: ``$($validation.finalCloseStrictValidatorOutputState)``",
  "",
  $validation.boundary
)
Write-Host "FinalOwnerOneScreenExecutionManualValidationState=$state FailedBlockers=$failedBlockerCount Steps=$($validation.stepCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Final Owner one-screen execution manual validation failed." }
