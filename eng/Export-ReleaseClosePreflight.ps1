[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot,
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Normalize-PostPublishRequiredEvidence {
  param([AllowNull()][object]$Evidence)

  $items = @($Evidence | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
  foreach ($requiredField in @("noLocalPackageSource", "noLocalNupkgPackageReference")) {
    if ($items -notcontains $requiredField) {
      $items += $requiredField
    }
  }

  return @($items)
}

function New-PreflightItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$CurrentStatus,
    [string]$OwnerAction,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    currentStatus = $CurrentStatus
    ownerAction = $OwnerAction
    boundary = $Boundary
  }
}

$externalProofExists = Test-Path -LiteralPath (Join-Path $RepositoryRoot "artifacts\final-release\external-runtime-proof-record.json") -PathType Leaf
$postPublishProofExists = Test-Path -LiteralPath (Join-Path $RepositoryRoot "artifacts\final-release\post-publish-verification-record.json") -PathType Leaf
$externalValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$ownerPlan = Read-JsonOrNull "artifacts\final-release\owner-authorized-publish-command-plan.json"
$ownerPlanValidation = Read-JsonOrNull "artifacts\final-release\owner-authorized-publish-command-plan-validation.json"
$releaseOwnerProofInputRecordValidation = Read-JsonOrNull "artifacts\final-release\release-owner-proof-input-record-validation.json"
$releaseIssueCloseRecordValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"
$fullAcceptance = Read-JsonOrNull "artifacts\final-release\release-candidate-full-acceptance-summary.json"
$cleanConsumerScan = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-project-scan.json"
$sampleRunEvidenceValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$sampleAssetManifestAudit = Read-JsonOrNull "artifacts\user-acceptance\sample-asset-manifest-audit.json"
$linuxValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"

$externalCanPromote = [bool](Get-PropertyOrDefault -Object $externalValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$postPublishProof = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
$postPublishCanClose = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$ownerPlanState = [string](Get-PropertyOrDefault -Object $ownerPlanValidation -Name "validationState" -DefaultValue "missing-owner-authorized-publish-command-plan-validation")
$ownerFailedCount = [int](Get-PropertyOrDefault -Object $ownerPlanValidation -Name "failedValidationItemCount" -DefaultValue -1)
$ownerPlaceholderOnly = [bool](Get-PropertyOrDefault -Object $ownerPlanValidation -Name "publishCommandsPlaceholderOnly" -DefaultValue $false)
$ownerAuthorizationProofGateState = [string](Get-PropertyOrDefault -Object $ownerPlanValidation -Name "ownerAuthorizationProofGateState" -DefaultValue "missing-owner-authorization-proof-gate")
$ownerAuthorizationRequiredFieldCount = [int](Get-PropertyOrDefault -Object $ownerPlanValidation -Name "ownerAuthorizationRequiredFieldCount" -DefaultValue 0)
$ownerAuthorizationMissingOwnerInputCount = [int](Get-PropertyOrDefault -Object $ownerPlanValidation -Name "ownerAuthorizationProofGateMissingOwnerInputCount" -DefaultValue -1)
$manualMaterializationPrerequisiteCount = [int](Get-PropertyOrDefault -Object $ownerPlanValidation -Name "manualMaterializationPrerequisiteCount" -DefaultValue 0)
$postPublishRequiredEvidenceCount = [int](Get-PropertyOrDefault -Object $ownerPlanValidation -Name "postPublishRequiredEvidenceCount" -DefaultValue 0)
$releaseOwnerProofInputValidationState = [string](Get-PropertyOrDefault -Object $releaseOwnerProofInputRecordValidation -Name "validationState" -DefaultValue "missing-release-owner-proof-input-record-validation")
$releaseOwnerProofInputClassification = [string](Get-PropertyOrDefault -Object $releaseOwnerProofInputRecordValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$releaseOwnerProofInputCanPromote = [bool](Get-PropertyOrDefault -Object $releaseOwnerProofInputRecordValidation -Name "canPromoteOwnerProofInput" -DefaultValue $false)
$releaseOwnerProofInputFailedValidationItemCount = [int](Get-PropertyOrDefault -Object $releaseOwnerProofInputRecordValidation -Name "failedValidationItemCount" -DefaultValue -1)
$releaseIssueCloseRecordValidationState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseRecordValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")
$releaseIssueCloseRecordProofClassification = [string](Get-PropertyOrDefault -Object $releaseIssueCloseRecordValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$releaseIssueCloseRecordCanPromote = [bool](Get-PropertyOrDefault -Object $releaseIssueCloseRecordValidation -Name "canPromoteReleaseIssueCloseRecord" -DefaultValue $false)
$releaseIssueCloseRecordCanClose = [bool](Get-PropertyOrDefault -Object $releaseIssueCloseRecordValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$releaseIssueCloseRecordFailedValidationItemCount = [int](Get-PropertyOrDefault -Object $releaseIssueCloseRecordValidation -Name "failedValidationItemCount" -DefaultValue -1)
$ownerAuthorizationRequiredFields = @(Get-PropertyOrDefault -Object $ownerPlan -Name "ownerAuthorizationRequiredFields" -DefaultValue @())
$manualMaterializationPrerequisites = @(Get-PropertyOrDefault -Object $ownerPlan -Name "manualMaterializationPrerequisites" -DefaultValue @())
$postPublishRequiredEvidence = @(Get-PropertyOrDefault -Object $ownerPlan -Name "postPublishRequiredEvidence" -DefaultValue @())
$postPublishRequiredEvidence = Normalize-PostPublishRequiredEvidence -Evidence $postPublishRequiredEvidence
$postPublishRequiredEvidenceCount = $postPublishRequiredEvidence.Count
$staleFindingCount = [int](Get-PropertyOrDefault -Object $staleAudit -Name "findingCount" -DefaultValue -1)
$acceptanceState = [string](Get-PropertyOrDefault -Object $fullAcceptance -Name "acceptanceState" -DefaultValue "missing-release-candidate-full-acceptance-summary")
$acceptanceCanClose = [bool](Get-PropertyOrDefault -Object $fullAcceptance -Name "canCloseReleaseIssue" -DefaultValue $false)
$acceptanceCanPublish = [bool](Get-PropertyOrDefault -Object $fullAcceptance -Name "canPublishPublicly" -DefaultValue $false)
$cleanConsumerScanPassed = [bool](Get-PropertyOrDefault -Object $cleanConsumerScan -Name "scanPassed" -DefaultValue $false)
$cleanConsumerScanState = [string](Get-PropertyOrDefault -Object $cleanConsumerScan -Name "scanState" -DefaultValue "missing-post-publish-clean-consumer-project-scan")
$sampleRunState = [string](Get-PropertyOrDefault -Object $sampleRunEvidenceValidation -Name "validationState" -DefaultValue "missing-sample-run-evidence-validation")
$sampleCanPromoteRealModel = [bool](Get-PropertyOrDefault -Object $sampleRunEvidenceValidation -Name "canPromoteRealModelRuntime" -DefaultValue $false)
$sampleManifestErrorCount = [int](Get-PropertyOrDefault -Object $sampleAssetManifestAudit -Name "errorCount" -DefaultValue -1)
$linuxState = [string](Get-PropertyOrDefault -Object $linuxValidation -Name "validationState" -DefaultValue "missing-linux-runner-evidence-validation")
$linuxProof = [bool](Get-PropertyOrDefault -Object $linuxValidation -Name "isRealLinuxRunnerProof" -DefaultValue $false)

$items = @(
  New-PreflightItem -Id "external-runtime-proof-record" -Passed ($externalProofExists -and $externalCanPromote) -CurrentStatus "exists=$externalProofExists; canPromoteRuntimeProof=$externalCanPromote" -OwnerAction "Provide a real external-runtime-proof-record.json and pass Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof." -Boundary "Template, draft, runbook, collection package, helper, build-only, parse-only, sidecar-only, dependency-probe-only, and blocked-by-cuda-driver are not runtime proof."
  New-PreflightItem -Id "release-owner-proof-input-record" -Passed $releaseOwnerProofInputCanPromote -CurrentStatus "validationState=$releaseOwnerProofInputValidationState; proofClassification=$releaseOwnerProofInputClassification; canPromoteOwnerProofInput=$releaseOwnerProofInputCanPromote; failedValidationItemCount=$releaseOwnerProofInputFailedValidationItemCount" -OwnerAction "Fill release-owner-proof-input-record.json with owner authorization, selected channel, package URL/hash, clean consumer, runtime log/hash, stdout/stderr summary, and host metadata, then pass Test-ReleaseOwnerProofInputRecord.ps1 -RequireExistingLogs -FailOnNotProof." -Boundary "Template-only, readiness snapshot, schema-only, managed-readiness, local feed, ProjectReference, direct .nupkg, and missing-log-hash records cannot satisfy owner proof input."
  New-PreflightItem -Id "owner-authorized-command-plan" -Passed ([string]::Equals($ownerPlanState, "ready-for-owner-manual-materialization", [System.StringComparison]::Ordinal) -and $ownerFailedCount -eq 0 -and $ownerPlaceholderOnly) -CurrentStatus "validationState=$ownerPlanState; failedValidationItemCount=$ownerFailedCount; publishCommandsPlaceholderOnly=$ownerPlaceholderOnly; ownerAuthorizationProofGateState=$ownerAuthorizationProofGateState; requiredFieldCount=$ownerAuthorizationRequiredFieldCount; missingOwnerInputCount=$ownerAuthorizationMissingOwnerInputCount; manualMaterializationPrerequisiteCount=$manualMaterializationPrerequisiteCount" -OwnerAction "Obtain explicit owner authorization, fill every owner authorization field, validate the owner command plan, and keep command materialization manual." -Boundary "Generated command plans, placeholders, templates, examples, and missing owner approval fields cannot authorize publication or release issue closure."
  New-PreflightItem -Id "linux-runner-proof" -Passed $linuxProof -CurrentStatus "linuxRuntimePackageKey=$LinuxRuntimePackageKey; validationState=$linuxState; isRealLinuxRunnerProof=$linuxProof" -OwnerAction "Run Test-LinuxRunnerEvidenceRecord.ps1 on a real Linux x64 compatible host and copy the validated record back under artifacts/linux-dry-run/$LinuxRuntimePackageKey." -Boundary "Windows handoff, template-only records, and blocked-by-cuda-driver are not Linux runner proof."
  New-PreflightItem -Id "real-model-runtime-proof" -Passed $sampleCanPromoteRealModel -CurrentStatus "sampleRunEvidence=$sampleRunState; canPromoteRealModelRuntime=$sampleCanPromoteRealModel; sampleManifestErrorCount=$sampleManifestErrorCount" -OwnerAction "Run Classification/YoloVision with real owner-provided assets and pass Test-SampleAssetManifest.ps1 plus Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog." -Boundary "Classification/YoloVision evidence can promote only real-model-runtime and never package-consumer-runtime."
  New-PreflightItem -Id "post-publish-clean-consumer-scan" -Passed $cleanConsumerScanPassed -CurrentStatus "scanState=$cleanConsumerScanState; scanPassed=$cleanConsumerScanPassed" -OwnerAction "Run Test-PostPublishCleanConsumerProject.ps1 against the clean consumer .csproj outside the repo." -Boundary "A scan is helper evidence only and cannot close the release issue by itself."
  New-PreflightItem -Id "post-publish-verification-record" -Passed ($postPublishProofExists -and $postPublishProof -and $postPublishCanClose) -CurrentStatus "exists=$postPublishProofExists; isPostPublishVerificationProof=$postPublishProof; canCloseReleaseIssue=$postPublishCanClose" -OwnerAction "Provide a real post-publish-verification-record.json and pass Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof." -Boundary "Local feed, ProjectReference, draft, or dependency-probe-only records cannot close the release issue."
  New-PreflightItem -Id "stale-release-claims" -Passed ($staleFindingCount -eq 0) -CurrentStatus "findingCount=$staleFindingCount" -OwnerAction "Keep release-facing docs/artifacts free from premature publish or close claims." -Boundary "Stale claims can block release acceptance even when scripts pass."
  New-PreflightItem -Id "full-acceptance-close-readiness" -Passed ($acceptanceCanPublish -and $acceptanceCanClose) -CurrentStatus "acceptanceState=$acceptanceState; canPublishPublicly=$acceptanceCanPublish; canCloseReleaseIssue=$acceptanceCanClose" -OwnerAction "Refresh full acceptance after real owner/external/post-publish proof is present." -Boundary "ready-for-owner-proof-collection is not public publish or close readiness."
  New-PreflightItem -Id "release-issue-close-record" -Passed ($releaseIssueCloseRecordCanPromote -and $releaseIssueCloseRecordCanClose) -CurrentStatus "validationState=$releaseIssueCloseRecordValidationState; proofClassification=$releaseIssueCloseRecordProofClassification; canPromoteReleaseIssueCloseRecord=$releaseIssueCloseRecordCanPromote; canCloseReleaseIssue=$releaseIssueCloseRecordCanClose; failedValidationItemCount=$releaseIssueCloseRecordFailedValidationItemCount" -OwnerAction "After every real proof gate passes, fill release-issue-close-record.json and pass Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady before owner closes the release issue." -Boundary "Template-only close records, readiness snapshots, schema-only records, preflight-only records, missing evidence bundle hash, and missing owner final close decision cannot close the release issue."
)

$failedItems = @($items | Where-Object { -not $_.passed })
$canCloseReleaseIssue = $failedItems.Count -eq 0
$preflightState = if ($canCloseReleaseIssue) { "ready-for-release-close-owner-review" } else { "blocked-real-proof-required" }

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$jsonPath = Join-Path $OutputRoot "release-close-preflight.json"
$markdownPath = Join-Path $OutputRoot "release-close-preflight.md"

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-close-preflight"
  preflightState = $preflightState
  canCloseReleaseIssue = $canCloseReleaseIssue
  canPublishPublicly = $acceptanceCanPublish
  performsPublish = $false
  failedItemCount = $failedItems.Count
  externalProofExists = $externalProofExists
  postPublishProofExists = $postPublishProofExists
  cleanConsumerScanState = $cleanConsumerScanState
  staleFindingCount = $staleFindingCount
  ownerPlanValidationState = $ownerPlanState
  ownerAuthorizationProofGateState = $ownerAuthorizationProofGateState
  ownerAuthorizationRequiredFieldCount = $ownerAuthorizationRequiredFieldCount
  ownerAuthorizationMissingOwnerInputCount = $ownerAuthorizationMissingOwnerInputCount
  releaseOwnerProofInputValidationState = $releaseOwnerProofInputValidationState
  releaseOwnerProofInputProofClassification = $releaseOwnerProofInputClassification
  releaseOwnerProofInputCanPromote = $releaseOwnerProofInputCanPromote
  releaseOwnerProofInputFailedValidationItemCount = $releaseOwnerProofInputFailedValidationItemCount
  releaseIssueCloseRecordValidationState = $releaseIssueCloseRecordValidationState
  releaseIssueCloseRecordProofClassification = $releaseIssueCloseRecordProofClassification
  releaseIssueCloseRecordCanPromote = $releaseIssueCloseRecordCanPromote
  releaseIssueCloseRecordCanCloseReleaseIssue = $releaseIssueCloseRecordCanClose
  releaseIssueCloseRecordFailedValidationItemCount = $releaseIssueCloseRecordFailedValidationItemCount
  manualMaterializationPrerequisiteCount = $manualMaterializationPrerequisiteCount
  postPublishRequiredEvidenceCount = $postPublishRequiredEvidenceCount
  ownerAuthorizationRequiredFields = $ownerAuthorizationRequiredFields
  manualMaterializationPrerequisites = $manualMaterializationPrerequisites
  postPublishRequiredEvidence = $postPublishRequiredEvidence
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  linuxRunnerValidationState = $linuxState
  isRealLinuxRunnerProof = $linuxProof
  sampleRunEvidenceValidationState = $sampleRunState
  canPromoteRealModelRuntime = $sampleCanPromoteRealModel
  sampleAssetManifestErrorCount = $sampleManifestErrorCount
  fullAcceptanceState = $acceptanceState
  preflightItems = $items
  ownerActionSummary = @(
    "Collect real external runtime proof on a compatible host.",
    "Fill and validate release-owner-proof-input-record.json with selected channel, package hashes, clean consumer logs, runtime smoke evidence, and host metadata.",
    "Obtain explicit owner authorization before manual publish execution, including owner identity, approval timestamp, target channel, command plan hash, proof bundle hash, rollback plan, credential handling, and NVIDIA redistribution approval.",
    "Collect Linux runner proof on a real Linux x64 compatible host.",
    "Collect Classification/YoloVision real-model-runtime proof with real model assets.",
    "After real publication, scan the clean consumer project and generate a post-publish verification record.",
    "Run post-publish validation with -RequireExistingLog -FailOnNotProof.",
    "Refresh release acceptance and close readiness only after every item passes.",
    "Fill and validate release-issue-close-record.json only after real proof, owner decision, stale audit, preflight, and evidence bundle hash all pass."
  )
  nonSubstituteProofKinds = @(
    "template",
    "draft",
    "runbook",
    "collection package",
    "local inventory",
    "local feed",
    "ProjectReference",
    "helper",
    "build-only",
    "parse-only",
    "sidecar-only",
    "input package",
    "dependency-probe-only",
    "Skipped=True",
    "blocked-by-cuda-driver",
    "bridge-only package consumer log",
    "bridge-only wrapper surface",
    "WrapperSurfaceEvidenceKind=compile-surface-proof",
    "IsRuntimeExecutionProof=False",
    "mismatched log SHA256",
    "Parser/ParserRefitter diagnostic snapshots",
    "copied managed diagnostic snapshot",
    "managed-readiness",
    "managed-readiness-only",
    "callback-allocator-readiness-snapshot",
    "CallbackAllocatorReadinessSnapshot",
    "TensorRtCallbackAllocatorReadinessSnapshot",
    "precheck-only",
    "dry-run-only",
    "schema-only",
    "template-only release issue close record",
    "missing owner final close decision",
    "mismatched evidence bundle SHA256",
    "Windows handoff for Linux proof"
  )
}

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Close Preflight")
$lines.Add("")
$lines.Add("- preflight state: ``$preflightState``")
$lines.Add("- can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("- can publish publicly: ``$acceptanceCanPublish``")
$lines.Add("- performs publish: ``False``")
$lines.Add("- failed item count: ``$($failedItems.Count)``")
$lines.Add("- owner authorization proof gate state: ``$ownerAuthorizationProofGateState``")
$lines.Add("- owner authorization required field count: ``$ownerAuthorizationRequiredFieldCount``")
$lines.Add("- owner authorization missing owner input count: ``$ownerAuthorizationMissingOwnerInputCount``")
$lines.Add("- release owner proof input validation state: ``$releaseOwnerProofInputValidationState``")
$lines.Add("- release owner proof input can promote: ``$releaseOwnerProofInputCanPromote``")
$lines.Add("- release issue close record validation state: ``$releaseIssueCloseRecordValidationState``")
$lines.Add("- release issue close record can promote: ``$releaseIssueCloseRecordCanPromote``")
$lines.Add("- manual materialization prerequisite count: ``$manualMaterializationPrerequisiteCount``")
$lines.Add("- post-publish required evidence count: ``$postPublishRequiredEvidenceCount``")
$lines.Add("")
$lines.Add("## Preflight Items")
$lines.Add("")
$lines.Add("| ID | Passed | Current status | Owner action | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($item in $items) {
  $lines.Add("| ``$($item.id)`` | ``$($item.passed)`` | $($item.currentStatus.Replace("|", "\|")) | $($item.ownerAction.Replace("|", "\|")) | $($item.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Owner Action Summary")
$lines.Add("")
foreach ($item in $record.ownerActionSummary) { $lines.Add("- $item") }
$lines.Add("")
$lines.Add("## Owner Authorization Required Fields")
$lines.Add("")
foreach ($field in $ownerAuthorizationRequiredFields) { $lines.Add("- ``$field``") }
$lines.Add("")
$lines.Add("## Manual Materialization Prerequisites")
$lines.Add("")
$lines.Add("| ID | Blocks materialization | Current state | Required evidence |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $manualMaterializationPrerequisites) {
  $lines.Add("| ``$($item.id)`` | ``$($item.blocksMaterialization)`` | $(([string]$item.currentState).Replace("|", "\|")) | $(([string]$item.requiredEvidence).Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Post-Publish Required Evidence")
$lines.Add("")
foreach ($field in $postPublishRequiredEvidence) { $lines.Add("- ``$field``") }
$lines.Add("")
$lines.Add("## Non-Substitute Proof Kinds")
$lines.Add("")
foreach ($item in $record.nonSubstituteProofKinds) { $lines.Add("- ``$item``") }
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close preflight written to $jsonPath"
Write-Host "PreflightState=$preflightState FailedItemCount=$($failedItems.Count) CanCloseReleaseIssue=$canCloseReleaseIssue PerformsPublish=False"
