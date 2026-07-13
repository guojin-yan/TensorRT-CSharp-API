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

function New-RunbookSection {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string[]]$FieldNames,
    [string]$EvidenceSource,
    [string]$ValidatorCommand
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    fieldNames = @($FieldNames)
    fieldCount = @($FieldNames).Count
    evidenceSource = $EvidenceSource
    validatorCommand = $ValidatorCommand
    ownerActionRequired = $true
    blocked = $true
    performsPublish = $false
    canCloseReleaseIssue = $false
    isProof = $false
    boundary = "Owner evidence import runbook section only. It maps fields to evidence sources and never substitutes real public publish or PostPublish proof."
  }
}

$contract = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-input-contract.json"
if ($null -eq $contract) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerPublicPublishExecutionResultInputContract.ps1") -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputRoot
  $contract = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-input-contract.json"
}
$requiredFields = @(Get-PropertyOrDefault -Object $contract -Name "requiredFields" -DefaultValue @())
$fieldNames = @($requiredFields | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "") } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
$fieldSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($name in $fieldNames) { [void]$fieldSet.Add($name) }

$sections = @(
  New-RunbookSection 1 "package-identity" "Public package identity and hashes" @("publicPackageId", "publicPackageVersion", "publicPackageSource", "publicPackageUrl", "publicPackageDownloadedPath", "publicPackageSha256", "managedPackageId", "runtimePackageId", "managedPackageUrl", "runtimePackageUrl", "managedPackageSha256", "runtimePackageSha256") "Public package source page/download after Owner publish." "Test-OwnerPublicPublishExecutionResultPreflight.ps1 -Strict"
  New-RunbookSection 2 "publish-transcripts" "Publish command and transcript evidence" @("nugetPushCommand", "nugetPushSource", "nugetPushTranscriptPath", "nugetPushTranscriptSha256", "nugetPushExitCode", "publishCommandPlanPath", "publishCommandPlanSha256", "managedPublishCommandSha256", "runtimePublishCommandSha256") "Owner terminal transcript from manually authorized public publish." "Import-OwnerPublicPublishExecutionResultCandidate.ps1; Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict"
  New-RunbookSection 3 "clean-consumer" "Clean external consumer restore/build/smoke evidence" @("cleanConsumerRestoreRoot", "cleanConsumerRestoreCommand", "cleanConsumerRestoreNoLocalFeedEvidence", "cleanConsumerBuildCommand", "cleanConsumerRuntimeSmokeCommand", "cleanConsumerRuntimeSmokeExitCode", "cleanConsumerRuntimeSmokeReportPath", "cleanConsumerRuntimeSmokeReportSha256") "External consumer workspace outside source repository." "Export-PostPublishVerificationRecordFromOwnerInput.ps1; Test-PostPublishVerificationRecord.ps1 -Strict"
  New-RunbookSection 4 "host-runtime" "Host runtime identity and environment transcript" @("hostMachineName", "hostOs", "hostGpuName", "hostGpuDriverVersion", "hostCudaVersion", "hostTensorRtVersion", "hostCudnnVersion", "hostEnvironmentTranscriptPath", "hostEnvironmentTranscriptSha256") "Owner runtime host and environment capture." "Test-OwnerPublicPublishExecutionResultPreflight.ps1 -Strict"
  New-RunbookSection 5 "non-substitutes" "Forbidden substitute confirmations and scan" @("noLocalFeedConfirmation", "noProjectReferenceConfirmation", "noDirectNupkgConfirmation", "noDryRunOnlyConfirmation", "noManualApprovalOnlyConfirmation", "noQueuedWorkflowConfirmation", "noMissingSelfHostedRunnerConfirmation", "noSidecarOnlyConfirmation", "noTensorRtExecReportOnlyConfirmation", "forbiddenSubstituteScanPath", "forbiddenSubstituteScanSha256") "Forbidden substitute scan plus Owner confirmations." "Export-PublicPublishForbiddenSubstituteScan.ps1; Test-PublicPublishForbiddenSubstituteScan.ps1 -Strict"
  New-RunbookSection 6 "rollback-close" "Rollback review and final close linkage" @("rollbackPlanPath", "rollbackPlanSha256", "finalPublicPackageUrlApproval", "finalPublicPackageHashApproval", "finalPublicPackageApprovalTimestampUtc", "strictValidatorCommand", "strictValidatorOutputPath", "strictValidatorOutputSha256") "Rollback review, strict validator transcript, and final Owner close decision." "Import-FinalOwnerRollbackReview.ps1; Test-FinalOwnerRollbackReview.ps1 -Strict; Import-FinalOwnerCloseDecision.ps1; Test-FinalOwnerCloseDecision.ps1 -Strict"
)

$missing = @($sections | ForEach-Object { $_.fieldNames } | Where-Object { -not $fieldSet.Contains([string]$_) } | Sort-Object -Unique)
$record = [pscustomobject]@{
  recordKind = "final-owner-publish-evidence-import-runbook"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  runbookState = "blocked-final-owner-publish-evidence-import-owner-action-required"
  contractRequiredFieldCount = [int](Get-PropertyOrDefault -Object $contract -Name "requiredFieldCount" -DefaultValue $fieldNames.Count)
  sectionCount = $sections.Count
  sections = @($sections)
  sectionFieldMissingFromContract = @($missing)
  sectionFieldMissingFromContractCount = @($missing).Count
  ownerActionRequired = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner publish evidence import runbook only. It maps the 178-field Owner public publish contract into execution sections but does not execute publish, import real proof, validate public availability, or close the release."
}

$jsonPath = Join-Path $OutputRoot "final-owner-publish-evidence-import-runbook.json"
$mdPath = Join-Path $OutputRoot "final-owner-publish-evidence-import-runbook.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Final Owner Publish Evidence Import Runbook") | Out-Null
$md.Add("") | Out-Null
$md.Add("- runbookState: ``$($record.runbookState)``") | Out-Null
$md.Add("- contractRequiredFieldCount: ``$($record.contractRequiredFieldCount)``") | Out-Null
$md.Add("- sectionCount: ``$($record.sectionCount)``") | Out-Null
$md.Add("- sectionFieldMissingFromContractCount: ``$($record.sectionFieldMissingFromContractCount)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| # | Section | Fields | Source | Validator |") | Out-Null
$md.Add("| --- | --- | --- | --- | --- |") | Out-Null
foreach ($section in $sections) {
  $md.Add("| $($section.order) | $($section.id) | $($section.fieldCount) | $(ConvertTo-MarkdownCell $section.evidenceSource) | ``$(ConvertTo-MarkdownCell $section.validatorCommand)`` |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "FinalOwnerPublishEvidenceImportRunbookState=$($record.runbookState) Sections=$($record.sectionCount) MissingFields=$($record.sectionFieldMissingFromContractCount)"
