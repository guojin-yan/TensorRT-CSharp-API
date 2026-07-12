[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

function Get-TextOrEmpty {
  param([AllowNull()][object]$Record, [string]$Name)
  return [string](Get-PropertyOrDefault -Object $Record -Name $Name -DefaultValue "")
}

function New-CrossCheck {
  param([string]$Id, [bool]$Passed, [string]$CurrentValue, [string]$ExpectedValue, [string]$OwnerAction)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    currentValue = $CurrentValue
    expectedValue = $ExpectedValue
    ownerAction = $OwnerAction
    ownerActionRequired = -not $Passed
    isProof = $false
  }
}

$ownerTemplate = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-input.template.json"
$postPublishInput = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-verification-owner-input.template.json"
$postPublishRecord = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-verification-record.json"
$postPublishValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-verification-validation.json"
$intakePack = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-real-publish-evidence-intake-dry-run-pack-validation.json"

$postPackagePageUrl = Get-TextOrEmpty -Record $postPublishInput -Name "packagePageUrl"
$recordPackagePageUrl = Get-TextOrEmpty -Record $postPublishRecord -Name "packagePageUrl"
$postManagedUrl = Get-TextOrEmpty -Record (Get-PropertyOrDefault -Object $postPublishInput -Name "packageIdentity" -DefaultValue $null) -Name "managedPackageUrl"
$postRuntimeUrl = Get-TextOrEmpty -Record (Get-PropertyOrDefault -Object $postPublishInput -Name "packageIdentity" -DefaultValue $null) -Name "runtimePackageUrl"
$validationManagedUrl = Get-TextOrEmpty -Record $postPublishValidation -Name "managedPackageUrl"
$validationRuntimeUrl = Get-TextOrEmpty -Record $postPublishValidation -Name "runtimePackageUrl"
$postManagedSha = Get-TextOrEmpty -Record $postPublishInput -Name "downloadedManagedPackageSha256"
$postRuntimeSha = Get-TextOrEmpty -Record $postPublishInput -Name "downloadedRuntimePackageSha256"
$validationManagedSha = Get-TextOrEmpty -Record $postPublishValidation -Name "managedNupkgSha256"
$validationRuntimeSha = Get-TextOrEmpty -Record $postPublishValidation -Name "runtimeNupkgSha256"
$cleanRoot = Get-TextOrEmpty -Record $postPublishRecord -Name "cleanConsumerRoot"
$cleanRootOutside = [bool](Get-PropertyOrDefault -Object $postPublishRecord -Name "cleanConsumerRootOutsideRepository" -DefaultValue $false)
$noProjectReference = [bool](Get-PropertyOrDefault -Object $postPublishRecord -Name "noProjectReference" -DefaultValue $false)
$noLocalPackageSource = [bool](Get-PropertyOrDefault -Object $postPublishRecord -Name "noLocalPackageSource" -DefaultValue $false)
$noLocalNupkgPackageReference = [bool](Get-PropertyOrDefault -Object $postPublishRecord -Name "noLocalNupkgPackageReference" -DefaultValue $false)

$crossChecks = @(
  New-CrossCheck -Id "package-page-url-owner-input-to-record" -Passed ((-not (Test-OwnerPlaceholder $postPackagePageUrl)) -and $postPackagePageUrl -eq $recordPackagePageUrl -and $postPackagePageUrl -match '^https?://') -CurrentValue $recordPackagePageUrl -ExpectedValue $postPackagePageUrl -OwnerAction "Owner must provide the same public package page URL in PostPublish input and projected record."
  New-CrossCheck -Id "managed-download-url-owner-input-to-validation" -Passed ((-not (Test-OwnerPlaceholder $postManagedUrl)) -and $postManagedUrl -eq $validationManagedUrl -and $postManagedUrl -match '^https?://') -CurrentValue $validationManagedUrl -ExpectedValue $postManagedUrl -OwnerAction "Owner must provide managed public download URL from NuGet/GitHub Packages."
  New-CrossCheck -Id "runtime-download-url-owner-input-to-validation" -Passed ((-not (Test-OwnerPlaceholder $postRuntimeUrl)) -and $postRuntimeUrl -eq $validationRuntimeUrl -and $postRuntimeUrl -match '^https?://') -CurrentValue $validationRuntimeUrl -ExpectedValue $postRuntimeUrl -OwnerAction "Owner must provide runtime public download URL from NuGet/GitHub Packages."
  New-CrossCheck -Id "managed-sha256-owner-input-to-validation" -Passed ((Test-Sha256Text $postManagedSha) -and $postManagedSha -eq $validationManagedSha) -CurrentValue $validationManagedSha -ExpectedValue $postManagedSha -OwnerAction "Owner must align downloaded managed nupkg SHA256 with validation record."
  New-CrossCheck -Id "runtime-sha256-owner-input-to-validation" -Passed ((Test-Sha256Text $postRuntimeSha) -and $postRuntimeSha -eq $validationRuntimeSha) -CurrentValue $validationRuntimeSha -ExpectedValue $postRuntimeSha -OwnerAction "Owner must align downloaded runtime nupkg SHA256 with validation record."
  New-CrossCheck -Id "clean-consumer-root-outside-repository" -Passed ((-not (Test-OwnerPlaceholder $cleanRoot)) -and $cleanRootOutside) -CurrentValue "cleanConsumerRoot=$cleanRoot; outside=$cleanRootOutside" -ExpectedValue "clean consumer root outside repository" -OwnerAction "Owner must run clean consumer outside the source repository."
  New-CrossCheck -Id "no-project-reference-local-feed-direct-nupkg" -Passed ($noProjectReference -and $noLocalPackageSource -and $noLocalNupkgPackageReference) -CurrentValue "noProjectReference=$noProjectReference; noLocalPackageSource=$noLocalPackageSource; noLocalNupkgPackageReference=$noLocalNupkgPackageReference" -ExpectedValue "all true" -OwnerAction "Owner must remove ProjectReference, local feed, and direct nupkg references."
)

$failed = @($crossChecks | Where-Object { -not [bool]$_.passed })
$state = "blocked-post-publish-strict-cross-check-owner-evidence-required"
$record = [pscustomobject]@{
  recordKind = "post-publish-strict-cross-check-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  crossCheckState = $state
  crossCheckCount = $crossChecks.Count
  failedCrossCheckCount = $failed.Count
  crossChecks = @($crossChecks)
  intakeDryRunValidationState = [string](Get-PropertyOrDefault -Object $intakePack -Name "validationState" -DefaultValue "missing-owner-real-publish-evidence-intake-dry-run-pack-validation")
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "PostPublish strict cross-check pack only. It compares Owner public publish/PostPublish field shape and blocks placeholders, local feed, ProjectReference, direct nupkg, dashboards, dry-runs, queued workflows, missing runners, sidecars, and TensorRtExec reports; it is not proof and never publishes."
}

$jsonPath = Join-Path $OutputRoot "post-publish-strict-cross-check-pack.json"
$mdPath = Join-Path $OutputRoot "post-publish-strict-cross-check-pack.md"
$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# PostPublish Strict Cross-Check Pack") | Out-Null
$md.Add("") | Out-Null
$md.Add("- crossCheckState: ``$state``") | Out-Null
$md.Add("- failedCrossCheckCount: ``$($failed.Count)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Cross Check | Passed | Current | Expected | Owner Action |") | Out-Null
$md.Add("| --- | --- | --- | --- | --- |") | Out-Null
foreach ($check in $crossChecks) {
  $md.Add("| $($check.id) | $($check.passed) | $(ConvertTo-MarkdownCell $check.currentValue) | $(ConvertTo-MarkdownCell $check.expectedValue) | $(ConvertTo-MarkdownCell $check.ownerAction) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "PostPublishStrictCrossCheckPackState=$state CrossChecks=$($crossChecks.Count) Failed=$($failed.Count)"
