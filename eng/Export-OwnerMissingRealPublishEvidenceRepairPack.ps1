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

function New-RepairGroup {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string]$EvidenceStage,
    [object[]]$MissingItems,
    [string]$OwnerAction,
    [string]$ValidatorCommand
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    evidenceStage = $EvidenceStage
    missingItemCount = @($MissingItems).Count
    missingItems = @($MissingItems)
    ownerAction = $OwnerAction
    validatorCommand = $ValidatorCommand
    ownerActionRequired = $true
    blocked = $true
    performsPublish = $false
    canCloseReleaseIssue = $false
    isProof = $false
  }
}

function New-FieldItem {
  param([string]$Name, [string]$Group, [string]$Reason)
  [pscustomobject]@{ name = $Name; group = $Group; reason = $Reason; ownerActionRequired = $true }
}

$contract = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-input-contract.json"
if ($null -eq $contract) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerPublicPublishExecutionResultInputContract.ps1") -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputRoot
  $contract = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-input-contract.json"
}
$requiredFields = @(Get-PropertyOrDefault -Object $contract -Name "requiredFields" -DefaultValue @())
$ownerPublishMissing = @($requiredFields | ForEach-Object {
  New-FieldItem ([string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "")) ([string](Get-PropertyOrDefault -Object $_ -Name "group" -DefaultValue "owner-public-publish")) "Owner public publish result is still template/input-required; real field value must be supplied after Owner execution."
} | Where-Object { -not [string]::IsNullOrWhiteSpace($_.name) })

$postPublishMissingNames = @(
  "packagePageUrl",
  "downloadedManagedPackagePath",
  "downloadedManagedPackageSha256",
  "downloadedRuntimePackagePath",
  "downloadedRuntimePackageSha256",
  "runtimeNativeAssetResolutionReportPath",
  "runtimeNativeAssetResolutionReportSha256",
  "cleanConsumerRoot",
  "noProjectReference",
  "noLocalPackageSource",
  "noLocalNupkgPackageReference",
  "nativeAssetsCopied",
  "dependencyProbePassed",
  "runtimeSmokePassed",
  "runtimeSmokeExitCode"
)
$postPublishMissing = @($postPublishMissingNames | ForEach-Object { New-FieldItem $_ "post-publish-owner-input" "PostPublish owner input is still placeholder/false/null and requires repository-external clean consumer evidence." })

$crossCheck = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-strict-cross-check-pack.json"
$failedCrossChecks = @(Get-PropertyOrDefault -Object $crossCheck -Name "crossChecks" -DefaultValue @() | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) })
$failedCrossCheckItems = @($failedCrossChecks | ForEach-Object { New-FieldItem ([string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "unknown-cross-check")) "post-publish-cross-check" ([string](Get-PropertyOrDefault -Object $_ -Name "ownerAction" -DefaultValue "Owner must repair failed PostPublish cross-check.")) })

$closure = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\release-close-strict-evidence-closure.json"
$blockedLanes = @(Get-PropertyOrDefault -Object $closure -Name "lanes" -DefaultValue @() | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "laneState" -DefaultValue "") -like "blocked*" -or -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) })
if ($blockedLanes.Count -eq 0) {
  $blockedLanes = @(
    [pscustomobject]@{ id = "owner-public-publish-result"; ownerAction = "Owner must supply real public publish result." },
    [pscustomobject]@{ id = "post-publish-record"; ownerAction = "Owner must supply PostPublish record." },
    [pscustomobject]@{ id = "rollback-review"; ownerAction = "Owner must supply rollback review." },
    [pscustomobject]@{ id = "close-decision"; ownerAction = "Owner must supply close decision." },
    [pscustomobject]@{ id = "final-public-publish-acceptance-gate"; ownerAction = "Owner must satisfy final acceptance gate." },
    [pscustomobject]@{ id = "forbidden-substitute-scan"; ownerAction = "Owner must supply forbidden substitute scan." }
  )
}
$closureItems = @($blockedLanes | ForEach-Object { New-FieldItem ([string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "unknown-close-lane")) "release-close" ([string](Get-PropertyOrDefault -Object $_ -Name "ownerAction" -DefaultValue "Owner must repair release close lane.")) })

$groups = @(
  New-RepairGroup 1 "owner-public-publish-contract" "Owner public publish contract fields" "publish-result" $ownerPublishMissing "Fill the 178-field Owner public publish result from real public package execution evidence." "Export-OwnerPublicPublishExecutionResultInputTemplate.ps1; Test-OwnerPublicPublishExecutionResultPreflight.ps1 -Strict"
  New-RepairGroup 2 "post-publish-owner-input" "PostPublish owner input placeholders" "post-publish-clean-consumer" $postPublishMissing "Replace PostPublish placeholders with public package URL/hash and repository-external clean consumer evidence." "Export-PostPublishVerificationOwnerInputTemplate.ps1; Test-PostPublishVerificationOwnerInput.ps1 -Strict"
  New-RepairGroup 3 "post-publish-strict-cross-checks" "Failed PostPublish strict cross-checks" "post-publish-cross-check" $failedCrossCheckItems "Align Owner input, projected record, hashes, URLs, and no-substitute flags." "Export-PostPublishStrictCrossCheckPack.ps1; Test-PostPublishStrictCrossCheckPack.ps1 -Strict"
  New-RepairGroup 4 "release-close-blocked-lanes" "Release close blocked lanes" "release-close" $closureItems "Provide real publish/PostPublish/rollback/close evidence until strict closure accepts it." "Export-ReleaseCloseStrictEvidenceClosure.ps1; Test-ReleaseCloseStrictEvidenceClosure.ps1 -Strict"
)

$record = [pscustomobject]@{
  recordKind = "owner-missing-real-publish-evidence-repair-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  repairState = "blocked-owner-missing-real-publish-evidence-repair-required"
  contractRequiredFieldCount = [int](Get-PropertyOrDefault -Object $contract -Name "requiredFieldCount" -DefaultValue $ownerPublishMissing.Count)
  ownerPublishMissingFieldCount = $ownerPublishMissing.Count
  postPublishMissingFieldCount = $postPublishMissing.Count
  failedPostPublishCrossCheckCount = $failedCrossCheckItems.Count
  releaseCloseBlockedLaneCount = $closureItems.Count
  repairGroupCount = $groups.Count
  repairGroups = @($groups)
  ownerActionRequired = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner missing real publish evidence repair pack only. It enumerates missing Owner evidence and never publishes packages, validates public availability, promotes proof, or closes the release."
}

$jsonPath = Join-Path $OutputRoot "owner-missing-real-publish-evidence-repair-pack.json"
$mdPath = Join-Path $OutputRoot "owner-missing-real-publish-evidence-repair-pack.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Owner Missing Real Publish Evidence Repair Pack") | Out-Null
$md.Add("") | Out-Null
$md.Add("- repairState: ``$($record.repairState)``") | Out-Null
$md.Add("- contractRequiredFieldCount: ``$($record.contractRequiredFieldCount)``") | Out-Null
$md.Add("- ownerPublishMissingFieldCount: ``$($record.ownerPublishMissingFieldCount)``") | Out-Null
$md.Add("- postPublishMissingFieldCount: ``$($record.postPublishMissingFieldCount)``") | Out-Null
$md.Add("- failedPostPublishCrossCheckCount: ``$($record.failedPostPublishCrossCheckCount)``") | Out-Null
$md.Add("- releaseCloseBlockedLaneCount: ``$($record.releaseCloseBlockedLaneCount)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Group | Missing Items | Stage | Owner Action |") | Out-Null
$md.Add("| --- | --- | --- | --- |") | Out-Null
foreach ($group in $groups) {
  $md.Add("| ``$($group.id)`` | $($group.missingItemCount) | ``$($group.evidenceStage)`` | $(ConvertTo-MarkdownCell $group.ownerAction) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "OwnerMissingRealPublishEvidenceRepairPackState=$($record.repairState) Groups=$($record.repairGroupCount) OwnerFields=$($record.ownerPublishMissingFieldCount) PostPublishFields=$($record.postPublishMissingFieldCount) CrossChecks=$($record.failedPostPublishCrossCheckCount)"
