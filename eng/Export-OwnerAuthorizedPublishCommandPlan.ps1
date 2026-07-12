[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
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

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function New-CommandPlanItem {
  param(
    [string]$Id,
    [string]$Channel,
    [string]$Command,
    [string]$RequiredAuthorization,
    [string]$RequiredEvidence,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    channel = $Channel
    command = $Command
    placeholderOnly = $true
    commandMaterializationState = "placeholder-only"
    materializedExecutableCommand = ""
    authorized = $false
    executable = $false
    performsPublish = $false
    ownerExecutionOnly = $true
    modelExecutionForbidden = $true
    copyOnlyAfterOwnerAuthorization = $true
    ownerAction = "Release owner must manually replace placeholders and execute outside this automation after explicit authorization."
    requiredAuthorization = $RequiredAuthorization
    requiredEvidence = $RequiredEvidence
    boundary = $Boundary
  }
}

$freezeSummary = Read-JsonOrNull "artifacts\release\release-candidate-freeze-summary.json"
$freezeChecklist = Read-JsonOrNull "artifacts\release\release-candidate-freeze-checklist.json"
$freezeValidation = Read-JsonOrNull "artifacts\release\release-candidate-freeze-validation.json"
$publishChecklist = Read-JsonOrNull "artifacts\final-release\release-publish-execution-checklist.json"
$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"
$externalRuntimeProofBackfillPlan = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-backfill-plan.json"
$postPublishVerificationBackfillPlan = Read-JsonOrNull "artifacts\final-release\post-publish-verification-backfill-plan.json"
$externalRuntimeProofCollectionPackage = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-collection-package.json"
$postPublishVerificationCollectionPackage = Read-JsonOrNull "artifacts\final-release\post-publish-verification-collection-package.json"
$ownerDecision = Read-JsonOrNull "artifacts\final-release\release-owner-decision-record.json"
$ownerApproval = Read-JsonOrNull "artifacts\final-release\release-owner-approval-input-validation.json"
$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"

$freezeState = [string](Get-PropertyOrDefault -Object $freezeSummary -Name "freezeState" -DefaultValue "missing-release-candidate-freeze-summary")
$freezeCanPublish = [bool](Get-PropertyOrDefault -Object $freezeSummary -Name "canPublish" -DefaultValue $false)
$freezeCanPromote = [bool](Get-PropertyOrDefault -Object $freezeSummary -Name "canPromote" -DefaultValue $false)
$freezeCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $freezeSummary -Name "canCloseReleaseIssue" -DefaultValue $false)
$freezePerformsPublish = [bool](Get-PropertyOrDefault -Object $freezeSummary -Name "performsPublish" -DefaultValue $true)
$freezeRequiresHumanOwner = [bool](Get-PropertyOrDefault -Object $freezeSummary -Name "requiresHumanOwner" -DefaultValue $true)
$realExternalRuntimeProofReady = [bool](Get-PropertyOrDefault -Object $freezeSummary -Name "realExternalRuntimeProofReady" -DefaultValue $false)
$realPostPublishVerificationReady = [bool](Get-PropertyOrDefault -Object $freezeSummary -Name "realPostPublishVerificationReady" -DefaultValue $false)
$runtimeProofStatus = [string](Get-PropertyOrDefault -Object $freezeSummary -Name "runtimeProofStatus" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidence -Name "runtimeProofStatus" -DefaultValue "missing-runtime-proof-status")))
$closeReadinessConsistent = [bool](Get-PropertyOrDefault -Object $freezeSummary -Name "closeReadinessConsistent" -DefaultValue $false)

$freezeValidationState = [string](Get-PropertyOrDefault -Object $freezeValidation -Name "validationState" -DefaultValue "missing-release-candidate-freeze-validation")
$freezeValidationFailedCount = [int](Get-PropertyOrDefault -Object $freezeValidation -Name "failedValidationItemCount" -DefaultValue -1)
$publishChecklistState = [string](Get-PropertyOrDefault -Object $publishChecklist -Name "executionState" -DefaultValue "missing-release-publish-execution-checklist")
$publishChecklistCanExecute = [bool](Get-PropertyOrDefault -Object $publishChecklist -Name "canExecutePublicPublish" -DefaultValue $false)
$publishChecklistPerformsPublish = [bool](Get-PropertyOrDefault -Object $publishChecklist -Name "performsPublish" -DefaultValue $true)
$finalPackageReviewState = [string](Get-PropertyOrDefault -Object $finalPackageReview -Name "bundleState" -DefaultValue "missing-final-package-review-bundle")
$finalPackageReviewPackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "packageCount" -DefaultValue 0)
$finalPackageReviewNativeAssetCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "nativeAssetCount" -DefaultValue 0)
$finalPackageReviewCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $finalPackageReview -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$externalRuntimeProofBackfillPlanState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofBackfillPlan -Name "planState" -DefaultValue "missing-external-runtime-proof-backfill-plan")
$externalRuntimeProofBackfillStepCount = @((Get-PropertyOrDefault -Object $externalRuntimeProofBackfillPlan -Name "backfillSteps" -DefaultValue @())).Count
$externalRuntimeProofBackfillCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofBackfillPlan -Name "canPromoteRuntimeProof" -DefaultValue $false)
$externalRuntimeProofCollectionPackageState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "packageState" -DefaultValue "missing-external-runtime-proof-collection-package")
$externalRuntimeProofCollectionPackageStepCount = @((Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "collectionSteps" -DefaultValue @())).Count
$externalRuntimeProofCollectionPackageCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "canPromoteRuntimeProof" -DefaultValue $false)
$externalRuntimeProofCollectionPackageCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "canCloseReleaseIssue" -DefaultValue $false)
$externalRuntimeProofCollectionPackageRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "isRuntimeExecutionEvidence" -DefaultValue $false)
$postPublishVerificationBackfillPlanState = [string](Get-PropertyOrDefault -Object $postPublishVerificationBackfillPlan -Name "planState" -DefaultValue "missing-post-publish-verification-backfill-plan")
$postPublishVerificationBackfillStepCount = @((Get-PropertyOrDefault -Object $postPublishVerificationBackfillPlan -Name "backfillSteps" -DefaultValue @())).Count
$postPublishVerificationBackfillCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $postPublishVerificationBackfillPlan -Name "canCloseReleaseIssue" -DefaultValue $false)
$postPublishVerificationCollectionPackageState = [string](Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "packageState" -DefaultValue "missing-post-publish-verification-collection-package")
$postPublishVerificationCollectionPackageStepCount = @((Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "collectionSteps" -DefaultValue @())).Count
$postPublishVerificationCollectionPackageProof = [bool](Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "isPostPublishVerificationProof" -DefaultValue $false)
$postPublishVerificationCollectionPackageCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "canCloseReleaseIssue" -DefaultValue $false)
$ownerDecisionState = [string](Get-PropertyOrDefault -Object $ownerDecision -Name "recordState" -DefaultValue "missing-release-owner-decision-record")
$ownerDecisionCanPublish = [bool](Get-PropertyOrDefault -Object $ownerDecision -Name "canPublishPublicly" -DefaultValue $false)
$ownerApprovalStatus = [string](Get-PropertyOrDefault -Object $ownerApproval -Name "overallStatus" -DefaultValue "missing-release-owner-approval-input-validation")
$ownerApprovalCanPublish = [bool](Get-PropertyOrDefault -Object $ownerApproval -Name "canPublishPublicly" -DefaultValue $false)
$externalRuntimeProofCanPromote = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$postPublishCanClose = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$staleFindingCount = [int](Get-PropertyOrDefault -Object $staleAudit -Name "findingCount" -DefaultValue -1)

$blockingReasons = New-Object System.Collections.Generic.List[string]
if (-not $freezeCanPublish) { $blockingReasons.Add("freeze canPublish is false") }
if (-not $ownerApprovalCanPublish) { $blockingReasons.Add("owner approval input is not publish-ready") }
if (-not $ownerDecisionCanPublish) { $blockingReasons.Add("owner decision record is not publish-ready") }
if (-not $publishChecklistCanExecute) { $blockingReasons.Add("release publish execution checklist is not executable") }
if (-not $realExternalRuntimeProofReady) { $blockingReasons.Add("missing real compatible-host external runtime proof") }
if ([string]::Equals($runtimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase)) { $blockingReasons.Add("blocked-by-cuda-driver is not smoke passed") }
if ($freezeValidationFailedCount -ne 0) { $blockingReasons.Add("freeze validation is not clean") }
if ($staleFindingCount -ne 0) { $blockingReasons.Add("stale release claims audit is not clean") }

$canMaterializeExecutableCommands = $false
$planState = if ($blockingReasons.Count -eq 0 -and $freezeCanPublish -and $ownerApprovalCanPublish -and $ownerDecisionCanPublish -and $publishChecklistCanExecute -and $realExternalRuntimeProofReady) {
  "ready-for-owner-manual-materialization"
}
else {
  "blocked-owner-authorization-required"
}

$publishCommands = @(
  New-CommandPlanItem -Id "nuget-org-managed-package" -Channel "nuget.org" -Command "dotnet nuget push <managed-package>.nupkg --api-key <NUGET_API_KEY> --source https://api.nuget.org/v3/index.json" -RequiredAuthorization "release owner explicit authorization plus NuGet API key handling" -RequiredEvidence "freeze publish readiness validator-promoted, owner decision validator-promoted for the selected channel, real external runtime proof ready, package SHA256 verified" -Boundary "Placeholder only; this script never pushes to nuget.org."
  New-CommandPlanItem -Id "nuget-org-runtime-package" -Channel "nuget.org" -Command "dotnet nuget push <runtime-package>.nupkg --api-key <NUGET_API_KEY> --source https://api.nuget.org/v3/index.json" -RequiredAuthorization "release owner explicit authorization plus NVIDIA redistribution review" -RequiredEvidence "runtime package key $RuntimePackageKey, native asset hashes, redistribution approval, compatible-host proof" -Boundary "Placeholder only; this script never publishes NVIDIA runtime components."
  New-CommandPlanItem -Id "github-packages-managed-package" -Channel "GitHub Packages" -Command "dotnet nuget push <managed-package>.nupkg --api-key <GITHUB_TOKEN> --source <github-packages-source>" -RequiredAuthorization "release owner explicit authorization plus scoped GitHub token" -RequiredEvidence "GitHub Packages source URI and package SHA256 reviewed" -Boundary "Placeholder only; this script never uploads GitHub Packages."
  New-CommandPlanItem -Id "github-packages-runtime-package" -Channel "GitHub Packages" -Command "dotnet nuget push <runtime-package>.nupkg --api-key <GITHUB_TOKEN> --source <github-packages-source>" -RequiredAuthorization "release owner explicit authorization plus scoped GitHub token and redistribution review" -RequiredEvidence "runtime package key $RuntimePackageKey, package SHA256, and channel source URI reviewed" -Boundary "Placeholder only; this script never uploads GitHub Packages."
  New-CommandPlanItem -Id "github-release-assets" -Channel "GitHub Release" -Command "gh release upload <tag> <package>.nupkg <package>.sha256" -RequiredAuthorization "release owner explicit authorization plus GitHub release tag confirmation" -RequiredEvidence "release tag, package identities, nupkg SHA256, and release notes reviewed" -Boundary "Placeholder only; this script never uploads GitHub Release assets."
)

$postPublishCommandPlan = @(
  [pscustomobject]@{ id = "create-clean-consumer"; command = "New-Item -ItemType Directory <clean-consumer-root>"; requiredEvidence = "cleanConsumerRoot outside the source repository"; boundary = "Source repository build is not post-publish proof." }
  [pscustomobject]@{ id = "restore-from-channel"; command = "dotnet restore <clean-consumer.csproj> --source <published-channel-source>"; requiredEvidence = "channelSourceUri, managed/runtime package source, restoreLogSha256"; boundary = "Local feed or ProjectReference cannot close release issue." }
  [pscustomobject]@{ id = "build-clean-consumer"; command = "dotnet build <clean-consumer.csproj> -c Release --no-restore"; requiredEvidence = "build log, native asset listing, nativeAssetListingSha256"; boundary = "Build success alone is not runtime smoke proof." }
  [pscustomobject]@{ id = "run-dependency-probe"; command = "dotnet run --project <clean-consumer.csproj> -- --dependency-probe --runtime-package-key $RuntimePackageKey"; requiredEvidence = "DependencyProbe BridgeInitialized and dependencyProbeLogSha256"; boundary = "DependencyProbe is diagnostics only." }
  [pscustomobject]@{ id = "run-runtime-smoke"; command = "dotnet run --project <clean-consumer.csproj> -- --runtime-package-key $RuntimePackageKey --smoke"; requiredEvidence = "runtimeSmokePassed=true, runtimeSmokeExitCode=0, smokeStatus=passed, smokeLogSha256, stdoutSummary, stderrSummary/no-stderr-emitted"; boundary = "blocked-by-cuda-driver is not smoke passed." }
  [pscustomobject]@{ id = "validate-post-publish-record"; command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof"; requiredEvidence = "isPostPublishVerificationProof must be true, release issue close readiness must be true, and all referenced log SHA256 values must match in the validator output"; boundary = "Only a real post-publish proof can close the release issue." }
)

$ownerAuthorizationRequiredFields = @(
  "ownerName",
  "ownerDecisionId",
  "approvalTimestampUtc",
  "targetChannel",
  "selectedRuntimePackageKey",
  "approvedManagedPackageId",
  "approvedRuntimePackageId",
  "approvedManagedPackageVersion",
  "approvedRuntimePackageVersion",
  "approvedCommandPlanSha256",
  "approvedProofBundleSha256",
  "rollbackPlan",
  "credentialHandlingAcknowledged",
  "nvidiaRedistributionApproval"
)

$manualMaterializationPrerequisites = @(
  [pscustomobject]@{ id = "owner-authorization"; requiredEvidence = "release-owner-approval-input-validation.json with validator-promoted owner identity and publication approval"; currentState = "ownerApprovalStatus=$ownerApprovalStatus; ownerApprovalCanPublishPublicly=$ownerApprovalCanPublish"; blocksMaterialization = -not $ownerApprovalCanPublish }
  [pscustomobject]@{ id = "owner-decision"; requiredEvidence = "release-owner-decision-record.json with validator-promoted selected-channel approval"; currentState = "ownerDecisionState=$ownerDecisionState; ownerDecisionCanPublishPublicly=$ownerDecisionCanPublish"; blocksMaterialization = -not $ownerDecisionCanPublish }
  [pscustomobject]@{ id = "freeze-summary"; requiredEvidence = "release-candidate-freeze-summary.json with canPublish=true and clean freeze validation"; currentState = "freezeState=$freezeState; canPublish=$freezeCanPublish; freezeValidationFailedCount=$freezeValidationFailedCount"; blocksMaterialization = (-not $freezeCanPublish) -or ($freezeValidationFailedCount -ne 0) }
  [pscustomobject]@{ id = "package-consumer-runtime-proof"; requiredEvidence = "external-runtime-proof-validation.json with proofClassification=package-consumer-runtime and canPromoteRuntimeProof=true"; currentState = "realExternalRuntimeProofReady=$realExternalRuntimeProofReady; externalRuntimeProofCanPromoteRuntimeProof=$externalRuntimeProofCanPromote"; blocksMaterialization = (-not $realExternalRuntimeProofReady) -or (-not $externalRuntimeProofCanPromote) }
  [pscustomobject]@{ id = "publish-checklist"; requiredEvidence = "release-publish-execution-checklist.json with canExecutePublicPublish=true"; currentState = "publishChecklistState=$publishChecklistState; canExecutePublicPublish=$publishChecklistCanExecute; performsPublish=$publishChecklistPerformsPublish"; blocksMaterialization = (-not $publishChecklistCanExecute) -or $publishChecklistPerformsPublish }
  [pscustomobject]@{ id = "stale-release-claims"; requiredEvidence = "stale-release-claims-audit.json with findingCount=0"; currentState = "staleFindingCount=$staleFindingCount"; blocksMaterialization = $staleFindingCount -ne 0 }
)

$postPublishRequiredEvidence = @(
  "selectedChannel",
  "channelSourceUri",
  "publishedPackageUrl",
  "managedPackageUrl",
  "runtimePackageUrl",
  "managedNupkgSha256",
  "runtimeNupkgSha256",
  "cleanConsumerRootOutsideRepository",
  "consumerProjectPath",
  "noProjectReference",
  "noLocalPackageSource",
  "noLocalNupkgPackageReference",
  "restoreLogPath",
  "nativeAssetListingSha256",
  "dependencyProbeLogPath",
  "dependencyProbeLogSha256",
  "runtimeSmokeLogPath",
  "runtimeSmokeLogSha256",
  "runtimeSmokePassed",
  "runtimeSmokeExitCode",
  "stdoutSummary",
  "stderrSummary",
  "hostMetadata"
)

$ownerAuthorizationProofGate = [pscustomobject]@{
  gateState = "blocked-owner-authorization-required"
  requiredFieldCount = $ownerAuthorizationRequiredFields.Count
  requiredFields = $ownerAuthorizationRequiredFields
  requiredArtifacts = @(
    "artifacts/final-release/release-owner-approval-input-record.json",
    "artifacts/final-release/release-owner-approval-input-validation.json",
    "artifacts/final-release/release-owner-decision-record.json",
    "artifacts/final-release/owner-authorized-publish-command-plan-validation.json"
  )
  requiredValidators = @(
    "Test-ReleaseOwnerApprovalInput.ps1",
    "Test-OwnerAuthorizedPublishCommandPlan.ps1"
  )
  missingOwnerInputCount = $ownerAuthorizationRequiredFields.Count
  canMaterializeExecutableCommands = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  boundary = "Owner authorization requires a real owner-filled approval input plus decision record; templates, examples, plans, and generated command placeholders cannot authorize publication."
}

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "owner-authorized-publish-command-plan"
  runtimePackageKey = $RuntimePackageKey
  planState = $planState
  performsPublish = $false
  requiresHumanOwner = $true
  requiresExplicitOwnerAuthorization = $true
  canMaterializeExecutableCommands = $canMaterializeExecutableCommands
  canPublish = $freezeCanPublish
  canPromote = $freezeCanPromote
  canCloseReleaseIssue = $freezeCanCloseReleaseIssue
  realExternalRuntimeProofReady = $realExternalRuntimeProofReady
  realPostPublishVerificationReady = $realPostPublishVerificationReady
  runtimeProofStatus = $runtimeProofStatus
  closeReadinessConsistent = $closeReadinessConsistent
  ownerApprovalStatus = $ownerApprovalStatus
  ownerApprovalCanPublishPublicly = $ownerApprovalCanPublish
  ownerDecisionState = $ownerDecisionState
  ownerDecisionCanPublishPublicly = $ownerDecisionCanPublish
  publishChecklistState = $publishChecklistState
  publishChecklistCanExecutePublicPublish = $publishChecklistCanExecute
  publishChecklistPerformsPublish = $publishChecklistPerformsPublish
  finalPackageReviewState = $finalPackageReviewState
  finalPackageReviewPackageCount = $finalPackageReviewPackageCount
  finalPackageReviewNativeAssetCount = $finalPackageReviewNativeAssetCount
  finalPackageReviewCanUseAsPublicPackageProof = $finalPackageReviewCanUseAsPublicPackageProof
  externalRuntimeProofBackfillPlanState = $externalRuntimeProofBackfillPlanState
  externalRuntimeProofBackfillStepCount = $externalRuntimeProofBackfillStepCount
  externalRuntimeProofBackfillCanPromoteRuntimeProof = $externalRuntimeProofBackfillCanPromoteRuntimeProof
  externalRuntimeProofCollectionPackageState = $externalRuntimeProofCollectionPackageState
  externalRuntimeProofCollectionPackageStepCount = $externalRuntimeProofCollectionPackageStepCount
  externalRuntimeProofCollectionPackageCanPromoteRuntimeProof = $externalRuntimeProofCollectionPackageCanPromoteRuntimeProof
  externalRuntimeProofCollectionPackageCanCloseReleaseIssue = $externalRuntimeProofCollectionPackageCanCloseReleaseIssue
  externalRuntimeProofCollectionPackageRuntimeExecutionEvidence = $externalRuntimeProofCollectionPackageRuntimeExecutionEvidence
  postPublishVerificationBackfillPlanState = $postPublishVerificationBackfillPlanState
  postPublishVerificationBackfillStepCount = $postPublishVerificationBackfillStepCount
  postPublishVerificationBackfillCanCloseReleaseIssue = $postPublishVerificationBackfillCanCloseReleaseIssue
  postPublishVerificationCollectionPackageState = $postPublishVerificationCollectionPackageState
  postPublishVerificationCollectionPackageStepCount = $postPublishVerificationCollectionPackageStepCount
  postPublishVerificationCollectionPackageProof = $postPublishVerificationCollectionPackageProof
  postPublishVerificationCollectionPackageCanCloseReleaseIssue = $postPublishVerificationCollectionPackageCanCloseReleaseIssue
  releaseCandidateFreezeState = $freezeState
  releaseCandidateFreezeValidationState = $freezeValidationState
  releaseCandidateFreezeFailedValidationItemCount = $freezeValidationFailedCount
  externalRuntimeProofCanPromoteRuntimeProof = $externalRuntimeProofCanPromote
  postPublishCanCloseReleaseIssue = $postPublishCanClose
  staleReleaseClaimsFindingCount = $staleFindingCount
  blockingReasons = @($blockingReasons)
  ownerAuthorizationRequiredFields = $ownerAuthorizationRequiredFields
  ownerAuthorizationProofGate = $ownerAuthorizationProofGate
  manualMaterializationPrerequisites = $manualMaterializationPrerequisites
  postPublishRequiredEvidence = $postPublishRequiredEvidence
  publishCommands = $publishCommands
  postPublishCommandPlan = $postPublishCommandPlan
  sourceEvidence = @(
    "artifacts/release/release-candidate-freeze-summary.json",
    "artifacts/release/release-candidate-freeze-checklist.json",
    "artifacts/release/release-candidate-freeze-validation.json",
    "artifacts/final-release/release-publish-execution-checklist.json",
    "artifacts/final-release/final-package-review-bundle.json",
    "artifacts/final-release/external-runtime-proof-backfill-plan.json",
    "artifacts/final-release/external-runtime-proof-collection-package.json",
    "artifacts/final-release/post-publish-verification-backfill-plan.json",
    "artifacts/final-release/post-publish-verification-collection-package.json",
    "artifacts/final-release/release-owner-approval-input-validation.json",
    "artifacts/final-release/release-owner-decision-record.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/stale-release-claims-audit.json"
  )
  safetyNotes = @(
    "This command plan is not publish execution.",
    "All publish commands are placeholders with authorized=false and executable=false.",
    "All publish commands keep placeholderOnly=true and materializedExecutableCommand empty.",
    "Large-model assistants and automation may generate this review artifact but must not execute publish, upload, delete, delist, or withdraw commands.",
    "The script does not run dotnet nuget push, GitHub Packages upload, GitHub Release upload, delete, delist, or withdraw.",
    "The final package review bundle is local package inventory, not public package proof.",
    "Backfill plans and collection packages are guidance only; they are not runtime proof, post-publish proof, publish approval, or release close approval.",
    "blocked-by-cuda-driver is not smoke passed.",
    "Template, draft, runbook, collection bundle, collection package, and dependency-probe-only records are not proof.",
    "Post-publish verification can close the release issue only after real channel publication and clean consumer smoke."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "owner-authorized-publish-command-plan.json"
$markdownPath = Join-Path $outputRoot "owner-authorized-publish-command-plan.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner Authorized Publish Command Plan")
$lines.Add("")
$lines.Add("- plan state: ``$planState``")
$lines.Add("- performs publish: ``False``")
$lines.Add("- requires explicit owner authorization: ``True``")
$lines.Add("- can materialize executable commands: ``False``")
$lines.Add("- model/automation execution forbidden: ``True``")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- release candidate freeze state: ``$freezeState``")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- external runtime proof backfill plan state: ``$externalRuntimeProofBackfillPlanState``")
$lines.Add("- external runtime proof collection package state: ``$externalRuntimeProofCollectionPackageState``")
$lines.Add("- post-publish verification backfill plan state: ``$postPublishVerificationBackfillPlanState``")
$lines.Add("- post-publish verification collection package state: ``$postPublishVerificationCollectionPackageState``")
$lines.Add("")
$lines.Add("This artifact is a safe owner review bundle. It never executes publish, upload, delete, delist, or withdraw actions.")
$lines.Add("")
$lines.Add("## Owner Authorization Proof Gate")
$lines.Add("")
$lines.Add("- gate state: ``$($ownerAuthorizationProofGate.gateState)``")
$lines.Add("- required field count: ``$($ownerAuthorizationProofGate.requiredFieldCount)``")
$lines.Add("- missing owner input count: ``$($ownerAuthorizationProofGate.missingOwnerInputCount)``")
$lines.Add("- can materialize executable commands: ``False``")
$lines.Add("- can publish publicly: ``False``")
$lines.Add("- can close release issue: ``False``")
$lines.Add("")
$lines.Add("Required owner fields:")
$lines.Add("")
foreach ($field in $ownerAuthorizationRequiredFields) {
  $lines.Add("- ``$field``")
}
$lines.Add("")
$lines.Add("## Manual Materialization Prerequisites")
$lines.Add("")
$lines.Add("| ID | Blocks materialization | Current state | Required evidence |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $manualMaterializationPrerequisites) {
  $lines.Add("| ``$($item.id)`` | ``$($item.blocksMaterialization)`` | $($item.currentState.Replace("|", "\|")) | $($item.requiredEvidence.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Post-Publish Required Evidence")
$lines.Add("")
foreach ($field in $postPublishRequiredEvidence) {
  $lines.Add("- ``$field``")
}
$lines.Add("")
$lines.Add("## Blocking Reasons")
$lines.Add("")
if ($blockingReasons.Count -eq 0) {
  $lines.Add("- none")
}
else {
  foreach ($reason in $blockingReasons) {
    $lines.Add("- $reason")
  }
}
$lines.Add("")
$lines.Add("## Publish Command Placeholders")
$lines.Add("")
$lines.Add("| ID | Channel | Placeholder only | Authorized | Executable | Materialized executable | Command | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- | --- |")
foreach ($command in $publishCommands) {
  $lines.Add("| ``$($command.id)`` | ``$($command.channel)`` | ``$($command.placeholderOnly)`` | ``$($command.authorized)`` | ``$($command.executable)`` | ``$($command.materializedExecutableCommand)`` | ``$($command.command)`` | $($command.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Post-Publish Command Plan")
$lines.Add("")
$lines.Add("| ID | Command | Required evidence | Boundary |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($command in $postPublishCommandPlan) {
  $lines.Add("| ``$($command.id)`` | ``$($command.command)`` | $($command.requiredEvidence.Replace("|", "\|")) | $($command.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Backfill Plans")
$lines.Add("")
$lines.Add("| Plan | State | Step count | Promotes or closes | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
$lines.Add("| ``external-runtime-proof-backfill-plan`` | ``$externalRuntimeProofBackfillPlanState`` | $externalRuntimeProofBackfillStepCount | ``$externalRuntimeProofBackfillCanPromoteRuntimeProof`` | Guidance only; not runtime proof or publish approval. |")
$lines.Add("| ``post-publish-verification-backfill-plan`` | ``$postPublishVerificationBackfillPlanState`` | $postPublishVerificationBackfillStepCount | ``$postPublishVerificationBackfillCanCloseReleaseIssue`` | Guidance only; not post-publish proof or release close approval. |")
$lines.Add("")
$lines.Add("## Collection Packages")
$lines.Add("")
$lines.Add("| Package | State | Step count | Promotes or closes | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
$lines.Add("| ``external-runtime-proof-collection-package`` | ``$externalRuntimeProofCollectionPackageState`` | $externalRuntimeProofCollectionPackageStepCount | ``$externalRuntimeProofCollectionPackageCanPromoteRuntimeProof`` / ``$externalRuntimeProofCollectionPackageCanCloseReleaseIssue`` | Copyable guidance only; not runtime proof or publish approval. |")
$lines.Add("| ``post-publish-verification-collection-package`` | ``$postPublishVerificationCollectionPackageState`` | $postPublishVerificationCollectionPackageStepCount | ``$postPublishVerificationCollectionPackageProof`` / ``$postPublishVerificationCollectionPackageCanCloseReleaseIssue`` | Copyable guidance only; not post-publish proof or release close approval. |")
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $record.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner authorized publish command plan written to $jsonPath"
Write-Host "Owner authorized publish command plan written to $markdownPath"
Write-Host "PlanState=$planState CanMaterializeExecutableCommands=$canMaterializeExecutableCommands PerformsPublish=False BlockingReasonCount=$($blockingReasons.Count)"
