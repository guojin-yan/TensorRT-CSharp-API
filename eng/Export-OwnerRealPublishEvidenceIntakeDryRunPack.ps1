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
. (Join-Path $PSScriptRoot "OwnerPublicPublishExecutionResultCommon.ps1")

function New-IntakeGroup {
  param(
    [string]$Id,
    [string]$Title,
    [string[]]$RequiredFields,
    [string]$OwnerAction,
    [string]$ValidatorCommand
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    requiredFieldCount = @($RequiredFields).Count
    requiredFields = @($RequiredFields)
    ownerAction = $OwnerAction
    validatorCommand = $ValidatorCommand
    ownerActionRequired = $true
    ready = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isProof = $false
    boundary = "Dry-run intake group only. It lists required Owner evidence fields and never executes publish or promotes proof."
  }
}

$contract = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-input-contract.json"
if ($null -eq $contract) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerPublicPublishExecutionResultInputContract.ps1") -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputRoot
  $contract = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-input-contract.json"
}
$template = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-input.template.json"
$preflight = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-preflight.json"
$candidateValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-candidate-validation.json"
$postPublishOwnerInput = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-verification-owner-input.template.json"
$postPublishOwnerInputValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-verification-owner-input-validation.json"
$postPublishValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-verification-validation.json"
$strictClosure = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\release-close-strict-evidence-closure-validation.json"
$readonlyAudit = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-readonly-publish-audit-pack-validation.json"
$ownerManual = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-owner-one-screen-execution-manual-validation.json"

$requiredFields = @(Get-OwnerPropertyOrDefault -Object $contract -Name "requiredFields" -DefaultValue @())
$fieldNames = @($requiredFields | ForEach-Object { [string](Get-OwnerPropertyOrDefault -Object $_ -Name "name" -DefaultValue "") } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

$groups = @(
  New-IntakeGroup -Id "managed-runtime-public-package" -Title "Managed/runtime public package identity and hashes" -RequiredFields @("managedPackageId", "managedPackageUrl", "managedPackageSha256", "runtimePackageId", "runtimePackageUrl", "runtimePackageSha256", "packageManagedPackageVersion", "packageRuntimePackageVersion", "packageSourceChannel") -OwnerAction "Owner records public package page/download URL, version, source channel, and SHA256 for managed and runtime packages." -ValidatorCommand "Export-OwnerPublicPublishExecutionResultInputTemplate.ps1; Test-OwnerPublicPublishExecutionResultPreflight.ps1 -Strict"
  New-IntakeGroup -Id "publish-command-and-transcripts" -Title "Publish command plan and transcript hashes" -RequiredFields @("publishCommandPlanPath", "publishCommandPlanSha256", "managedPublishCommandSha256", "runtimePublishCommandSha256", "nugetPushTranscriptPath", "nugetPushTranscriptSha256", "nugetPushExitCode") -OwnerAction "Owner records command plan hash and publish stdout/stderr/merged transcript hashes after manual execution." -ValidatorCommand "Import-OwnerPublicPublishExecutionResultCandidate.ps1; Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict"
  New-IntakeGroup -Id "owner-authorization" -Title "Owner authorization and scope" -RequiredFields @("ownerAuthorizationId", "ownerAuthorizationScope", "ownerAuthorizationTimestampUtc", "ownerApprovalTranscriptPath", "ownerApprovalTranscriptSha256") -OwnerAction "Owner records explicit authorization id, scope, timestamp, and supporting transcript hash." -ValidatorCommand "Test-OwnerPublicPublishExecutionResultPreflight.ps1 -Strict"
  New-IntakeGroup -Id "source-runner-boundary" -Title "Source runner queue/infrastructure boundary" -RequiredFields @("sourceRunnerQueueStatus", "sourceRunnerInfrastructureStatus", "sourceRunnerOwnerAction", "noQueuedWorkflowConfirmation", "noMissingSelfHostedRunnerConfirmation") -OwnerAction "Owner confirms queued workflow and missing runner states are owner-infra-action only and not proof." -ValidatorCommand "Test-OwnerPublicPublishExecutionResultPreflight.ps1 -Strict"
  New-IntakeGroup -Id "forbidden-substitute-scan" -Title "Forbidden substitute scan linkage" -RequiredFields @("noLocalFeedConfirmation", "noProjectReferenceConfirmation", "noDirectNupkgConfirmation", "noCandidateDashboardRunbookSubstitutionConfirmation", "noBlockedDashboardSubstitutionConfirmation", "noDryRunOnlyConfirmation", "noManualApprovalOnlyConfirmation", "noSidecarOnlyConfirmation", "noTensorRtExecReportOnlyConfirmation", "forbiddenSubstituteScanPath", "forbiddenSubstituteScanSha256") -OwnerAction "Owner links the forbidden substitute scan and confirms no local feed, ProjectReference, direct nupkg, dashboard/runbook, dry-run, sidecar, or TensorRtExec report substituted proof." -ValidatorCommand "Export-PublicPublishForbiddenSubstituteScan.ps1; Test-PublicPublishForbiddenSubstituteScan.ps1 -Strict"
  New-IntakeGroup -Id "post-publish-clean-consumer" -Title "PostPublish clean consumer evidence" -RequiredFields @("publicPackageUrl", "publicPackageDownloadedPath", "publicPackageSha256", "managedPackageUrl", "managedPackageSha256", "runtimePackageUrl", "runtimePackageSha256", "cleanConsumerRestoreRoot", "cleanConsumerRestoreNoLocalFeedEvidence", "cleanConsumerRuntimeSmokeReportPath", "cleanConsumerRuntimeSmokeReportSha256") -OwnerAction "Owner records public package URL/download/hash fields in the Owner publish contract, then performs clean external consumer restore/build/smoke and records report hashes." -ValidatorCommand "Export-PostPublishVerificationRecordFromOwnerInput.ps1; Test-PostPublishVerificationRecord.ps1 -Strict"
)

$requiredFieldNameSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($name in $fieldNames) { [void]$requiredFieldNameSet.Add($name) }
$declaredButMissingInContract = @($groups | ForEach-Object { $_.requiredFields } | Where-Object { -not $requiredFieldNameSet.Contains([string]$_) } | Sort-Object -Unique)
$contractRequiredFieldCount = [int](Get-OwnerPropertyOrDefault -Object $contract -Name "requiredFieldCount" -DefaultValue 0)
$preflightState = [string](Get-PropertyOrDefault -Object $preflight -Name "preflightState" -DefaultValue "missing-owner-public-publish-execution-result-preflight")
$candidateState = [string](Get-PropertyOrDefault -Object $candidateValidation -Name "validationState" -DefaultValue "missing-owner-public-publish-execution-result-candidate-validation")
$postPublishOwnerInputState = [string](Get-PropertyOrDefault -Object $postPublishOwnerInputValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-owner-input-validation")
$postPublishValidationState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$strictClosureState = [string](Get-PropertyOrDefault -Object $strictClosure -Name "validationState" -DefaultValue "missing-release-close-strict-evidence-closure-validation")
$readonlyAuditState = [string](Get-PropertyOrDefault -Object $readonlyAudit -Name "validationState" -DefaultValue "missing-final-readonly-publish-audit-pack-validation")
$ownerManualState = [string](Get-PropertyOrDefault -Object $ownerManual -Name "validationState" -DefaultValue "missing-final-owner-one-screen-execution-manual-validation")

$fakeReadyCases = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "dashboard",
  "dry-run",
  "manual approval",
  "queued GitHub Actions run",
  "missing self-hosted runner",
  "sidecar-only",
  "TensorRtExec report"
) | ForEach-Object {
  [pscustomobject]@{
    substitute = $_
    fakeReadyBlocked = $true
    ownerActionRequired = $true
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isProof = $false
    reason = "$_ is a forbidden non-proof substitute and cannot satisfy Owner real publish evidence intake."
  }
}

$state = "blocked-owner-real-publish-evidence-intake-dry-run-owner-action-required"
$record = [pscustomobject]@{
  recordKind = "owner-real-publish-evidence-intake-dry-run-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  intakeState = $state
  contractRequiredFieldCount = $contractRequiredFieldCount
  intakeGroupCount = $groups.Count
  intakeGroups = @($groups)
  declaredButMissingInContract = @($declaredButMissingInContract)
  declaredButMissingInContractCount = $declaredButMissingInContract.Count
  sourceStates = [pscustomobject]@{
    ownerPublishPreflightState = $preflightState
    ownerPublishCandidateValidationState = $candidateState
    postPublishOwnerInputValidationState = $postPublishOwnerInputState
    postPublishValidationState = $postPublishValidationState
    strictClosureValidationState = $strictClosureState
    readonlyAuditValidationState = $readonlyAuditState
    ownerManualValidationState = $ownerManualState
  }
  fakeReadyCases = @($fakeReadyCases)
  fakeReadyCaseCount = @($fakeReadyCases).Count
  blockedFakeReadyCaseCount = @($fakeReadyCases | Where-Object { [bool]$_.fakeReadyBlocked }).Count
  ownerActionRequired = $true
  passed = $false
  dryRunOnly = $true
  performsPublish = $false
  performsGitHubPackagesPublish = $false
  performsNuGetPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner real publish evidence intake dry-run pack only. It lists required fields, sequencing, and fake-ready blockers; it never runs dotnet nuget push, never publishes GitHub Packages, never triggers workflows, never accepts local feed/ProjectReference/direct nupkg/dashboard/dry-run/queued workflow/missing runner/sidecar-only/TensorRtExec report as proof, and cannot close the release."
}

$jsonPath = Join-Path $OutputRoot "owner-real-publish-evidence-intake-dry-run-pack.json"
$mdPath = Join-Path $OutputRoot "owner-real-publish-evidence-intake-dry-run-pack.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Owner Real Publish Evidence Intake Dry-Run Pack") | Out-Null
$md.Add("") | Out-Null
$md.Add("- intakeState: ``$state``") | Out-Null
$md.Add("- contractRequiredFieldCount: ``$contractRequiredFieldCount``") | Out-Null
$md.Add("- intakeGroupCount: ``$($groups.Count)``") | Out-Null
$md.Add("- dryRunOnly: ``True``") | Out-Null
$md.Add("- performsPublish: ``False``") | Out-Null
$md.Add("") | Out-Null
$md.Add("## Intake Groups") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Group | Required Fields | Owner Action | Validator |") | Out-Null
$md.Add("| --- | --- | --- | --- |") | Out-Null
foreach ($group in $groups) {
  $md.Add("| $($group.id) | $($group.requiredFieldCount) | $(ConvertTo-MarkdownCell $group.ownerAction) | ``$(ConvertTo-MarkdownCell $group.validatorCommand)`` |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Fake-Ready Blockers") | Out-Null
$md.Add("") | Out-Null
foreach ($case in $fakeReadyCases) { $md.Add("- ``$($case.substitute)``: blocked, non-proof") | Out-Null }
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "OwnerRealPublishEvidenceIntakeDryRunPackState=$state Groups=$($groups.Count) FakeReadyBlocked=$($record.blockedFakeReadyCaseCount)"
