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

function New-BackfillStep {
  param(
    [string]$Id,
    [int]$Order,
    [string]$Title,
    [string]$Command,
    [string]$RequiredEvidence,
    [string]$OutputArtifact,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    order = $Order
    title = $Title
    command = $Command
    requiredEvidence = $RequiredEvidence
    outputArtifact = $OutputArtifact
    boundary = $Boundary
    performsPublish = $false
    canCloseReleaseIssue = $false
  }
}

$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$ownerPlan = Read-JsonOrNull "artifacts\final-release\owner-authorized-publish-command-plan.json"
$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$ownerApproval = Read-JsonOrNull "artifacts\final-release\release-owner-approval-input-validation.json"
$ownerDecision = Read-JsonOrNull "artifacts\final-release\release-owner-decision-record.json"
$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"

$validationState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$proofClassification = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "postPublishProofClassification" -DefaultValue "missing-post-publish-proof-classification")
$proofClassificationPromotable = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "postPublishProofClassificationPromotable" -DefaultValue $false)
$isPostPublishVerificationProof = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$failedProofItemCount = [int](Get-PropertyOrDefault -Object $postPublishValidation -Name "failedProofItemCount" -DefaultValue -1)
$managedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "managedNupkgSha256Ready" -DefaultValue $false)
$runtimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "runtimeNupkgSha256Ready" -DefaultValue $false)
$consumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "consumerProjectIdentityReady" -DefaultValue $false)
$smokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)
$hostReady = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "hostReady" -DefaultValue $false)
$commandsReady = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "commandsReady" -DefaultValue $false)
$stdoutStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "stdoutStderrSummaryReady" -DefaultValue $false)
$allLogSha256Matches = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "allLogSha256Matches" -DefaultValue $false)
$runtimeSmokePassed = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "runtimeSmokePassed" -DefaultValue $false)
$runtimeSmokeExitCodeIsZero = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "runtimeSmokeExitCodeIsZero" -DefaultValue $false)
$noProjectReference = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "noProjectReference" -DefaultValue $false)
$ownerPlanState = [string](Get-PropertyOrDefault -Object $ownerPlan -Name "planState" -DefaultValue "missing-owner-authorized-publish-command-plan")
$ownerApprovalStatus = [string](Get-PropertyOrDefault -Object $ownerApproval -Name "overallStatus" -DefaultValue "missing-owner-approval-input-validation")
$ownerApprovalCanPublish = [bool](Get-PropertyOrDefault -Object $ownerApproval -Name "canPublishPublicly" -DefaultValue $false)
$ownerDecisionCanPublish = [bool](Get-PropertyOrDefault -Object $ownerDecision -Name "canPublishPublicly" -DefaultValue $false)
$releaseEvidenceCanClose = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "canCloseReleaseIssue" -DefaultValue $false)
$finalPackageReviewState = [string](Get-PropertyOrDefault -Object $finalPackageReview -Name "bundleState" -DefaultValue "missing-final-package-review-bundle")
$finalPackageReviewPackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "packageCount" -DefaultValue 0)
$finalPackageReviewCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $finalPackageReview -Name "canUseAsPublicPackageProof" -DefaultValue $false)

$recordPath = "artifacts/final-release/post-publish-verification-record.json"
$templatePath = "artifacts/final-release/post-publish-verification-record-template.json"
$validationPath = "artifacts/final-release/post-publish-verification-validation.json"

$steps = @(
  New-BackfillStep -Id "confirm-owner-authorized-publish" -Order 1 -Title "Confirm owner-authorized package publication" -Command "Review release-owner-approval-input-validation.json, release-owner-decision-record.json, and owner-authorized-publish-command-plan.json before any manual publish command is run." -RequiredEvidence "owner approval validator-promoted, owner decision validator-promoted, selected channel, rollback plan, API key handling, NVIDIA redistribution approval" -OutputArtifact "artifacts/final-release/release-owner-approval-input-validation.json" -Boundary "This plan never runs dotnet nuget push or gh release upload."
  New-BackfillStep -Id "copy-template-to-record" -Order 2 -Title "Copy post-publish template to real record" -Command "Copy-Item -LiteralPath $templatePath -Destination $recordPath" -RequiredEvidence "recordKind will be changed to post-publish-verification-record and templateOnly=false only after real publication" -OutputArtifact $recordPath -Boundary "A copied template is not proof."
  New-BackfillStep -Id "capture-published-package-identity" -Order 3 -Title "Capture published package identity and hashes" -Command "Download managed/runtime packages from <selected-channel>; Get-FileHash -Algorithm SHA256 for each downloaded nupkg" -RequiredEvidence "selectedChannel, channelSourceUri, managed/runtime package id, version, URL, downloaded nupkg SHA256" -OutputArtifact $recordPath -Boundary "Local nupkg path and final-package-review-bundle are not public channel proof."
  New-BackfillStep -Id "create-clean-consumer" -Order 4 -Title "Create clean consumer outside repository" -Command "New-Item -ItemType Directory <clean-consumer-root>; dotnet new console; add package references from the selected channel only" -RequiredEvidence "cleanConsumerRoot outside source repository, consumerProjectName, consumerProjectPath ending in .csproj, no ProjectReference" -OutputArtifact $recordPath -Boundary "Source repository restore or ProjectReference invalidates post-publish proof."
  New-BackfillStep -Id "restore-build-and-capture-logs" -Order 5 -Title "Restore and build from the published channel" -Command "dotnet restore <clean-consumer.csproj> --source <channelSourceUri>; dotnet build <clean-consumer.csproj> -c Release --no-restore; Get-FileHash for restore log and native asset listing" -RequiredEvidence "restoreCommand, buildCommand, managed/runtime package source, restoreLogSha256, nativeAssetListingSha256, nativeAssetsCopied=true" -OutputArtifact $recordPath -Boundary "Build success alone is not runtime smoke proof."
  New-BackfillStep -Id "run-dependency-probe-and-smoke" -Order 6 -Title "Run dependency probe and runtime smoke on compatible host" -Command "dotnet run --project <clean-consumer.csproj> -- --dependency-probe --runtime-package-key $RuntimePackageKey; dotnet run --project <clean-consumer.csproj> -- --runtime-package-key $RuntimePackageKey --smoke" -RequiredEvidence "host CUDA/TensorRT/cuDNN metadata, dependencyProbePassed=true, runtimeSmokePassed=true, runtimeSmokeExitCode=0, smokeStatus=passed, smokeLogSha256" -OutputArtifact $recordPath -Boundary "DependencyProbe is diagnostics only; blocked-by-cuda-driver is not smoke passed."
  New-BackfillStep -Id "review-stdout-stderr" -Order 7 -Title "Review stdout and stderr summaries" -Command "Fill stdoutSummary and stderrSummary from reviewed restore/build/probe/smoke logs; use no-stderr-emitted only when stderr was actually empty." -RequiredEvidence "non-empty stdoutSummary and stderrSummary" -OutputArtifact $recordPath -Boundary "Log paths without reviewed summaries cannot close the release issue."
  New-BackfillStep -Id "validate-post-publish-record" -Order 8 -Title "Validate real post-publish verification record" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath $recordPath -RequireExistingLog -FailOnNotProof" -RequiredEvidence "validator-promoted post-publish proof state; validator-promoted close readiness; failedProofItemCount=0; all log SHA256 values match" -OutputArtifact $validationPath -Boundary "Only validator-promoted real post-publish proof can close the release issue."
  New-BackfillStep -Id "refresh-release-close-readiness" -Order 9 -Title "Refresh release close readiness" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFreezeSummary.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateFreezeSummary.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerAuthorizedPublishCommandPlan.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerAuthorizedPublishCommandPlan.ps1" -RequiredEvidence "release evidence and freeze summary reflect validator-promoted close readiness only after real post-publish proof" -OutputArtifact "artifacts/final-release/release-evidence-bundle.json" -Boundary "Refreshing records is not a package push and cannot fabricate proof."
)

$blockingReasons = New-Object System.Collections.Generic.List[string]
if (-not $proofClassificationPromotable) { $blockingReasons.Add("post-publish proof classification is not promotable") }
if (-not $isPostPublishVerificationProof) { $blockingReasons.Add("post-publish verification proof is not real") }
if (-not $canCloseReleaseIssue) { $blockingReasons.Add("release issue cannot close yet") }
if ($failedProofItemCount -ne 0) { $blockingReasons.Add("post-publish validator has failed proof items") }
if (-not ($managedNupkgSha256Ready -and $runtimeNupkgSha256Ready)) { $blockingReasons.Add("published managed/runtime nupkg SHA256 evidence is incomplete") }
if (-not $consumerProjectIdentityReady) { $blockingReasons.Add("clean consumer identity is incomplete") }
if (-not $smokeCommandRuntimeKeyReady) { $blockingReasons.Add("smoke command does not include the target runtime package key") }
if (-not $hostReady) { $blockingReasons.Add("compatible host metadata is incomplete") }
if (-not $commandsReady) { $blockingReasons.Add("restore/build/smoke command capture is incomplete") }
if (-not $stdoutStderrSummaryReady) { $blockingReasons.Add("stdout/stderr summaries are incomplete") }
if (-not $allLogSha256Matches) { $blockingReasons.Add("restore/native/dependency/smoke log SHA256 evidence is missing or mismatched") }
if (-not ($runtimeSmokePassed -and $runtimeSmokeExitCodeIsZero)) { $blockingReasons.Add("runtime smoke has not passed with exitCode=0") }
if (-not $noProjectReference) { $blockingReasons.Add("clean consumer no ProjectReference confirmation is missing") }
if (-not ($ownerApprovalCanPublish -and $ownerDecisionCanPublish)) { $blockingReasons.Add("owner approval and owner decision are not publish-ready") }

$planState = if ($blockingReasons.Count -eq 0) { "ready-after-real-post-publish-proof" } else { "blocked-real-post-publish-proof-required" }

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "post-publish-verification-backfill-plan"
  runtimePackageKey = $RuntimePackageKey
  planState = $planState
  performsPublish = $false
  approvesPublicRelease = $false
  isPostPublishVerificationProof = $false
  canCloseReleaseIssue = $false
  validationState = $validationState
  postPublishProofClassification = $proofClassification
  postPublishProofClassificationPromotable = $proofClassificationPromotable
  currentValidationIsPostPublishVerificationProof = $isPostPublishVerificationProof
  currentValidationCanCloseReleaseIssue = $canCloseReleaseIssue
  failedProofItemCount = $failedProofItemCount
  ownerPlanState = $ownerPlanState
  ownerApprovalStatus = $ownerApprovalStatus
  ownerApprovalCanPublishPublicly = $ownerApprovalCanPublish
  ownerDecisionCanPublishPublicly = $ownerDecisionCanPublish
  releaseEvidenceCanCloseReleaseIssue = $releaseEvidenceCanClose
  finalPackageReviewState = $finalPackageReviewState
  finalPackageReviewPackageCount = $finalPackageReviewPackageCount
  finalPackageReviewCanUseAsPublicPackageProof = $finalPackageReviewCanUseAsPublicPackageProof
  managedNupkgSha256Ready = $managedNupkgSha256Ready
  runtimeNupkgSha256Ready = $runtimeNupkgSha256Ready
  consumerProjectIdentityReady = $consumerProjectIdentityReady
  smokeCommandRuntimeKeyReady = $smokeCommandRuntimeKeyReady
  hostReady = $hostReady
  commandsReady = $commandsReady
  stdoutStderrSummaryReady = $stdoutStderrSummaryReady
  allLogSha256Matches = $allLogSha256Matches
  runtimeSmokePassed = $runtimeSmokePassed
  runtimeSmokeExitCodeIsZero = $runtimeSmokeExitCodeIsZero
  noProjectReference = $noProjectReference
  blockingReasons = @($blockingReasons)
  backfillSteps = @($steps)
  expectedArtifacts = [pscustomobject]@{
    templatePath = $templatePath
    realRecordPath = $recordPath
    validationPath = $validationPath
  }
  sourceEvidence = @(
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/owner-authorized-publish-command-plan.json",
    "artifacts/final-release/release-owner-approval-input-validation.json",
    "artifacts/final-release/release-owner-decision-record.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/final-package-review-bundle.json"
  )
  safetyNotes = @(
    "This backfill plan does not publish packages.",
    "This backfill plan is not post-publish proof.",
    "final-package-review-bundle is local package inventory, not public channel proof.",
    "Local feed, ProjectReference, templates, drafts, runbooks, collection bundles, and dependency-probe-only records cannot close the release issue.",
    "blocked-by-cuda-driver is not smoke passed.",
    "Only post-publish-verification-record.json validated with -RequireExistingLog -FailOnNotProof can close the release issue."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "post-publish-verification-backfill-plan.json"
$markdownPath = Join-Path $outputRoot "post-publish-verification-backfill-plan.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Post-Publish Verification Backfill Plan")
$lines.Add("")
$lines.Add("- plan state: ``$planState``")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- proof classification: ``$proofClassification``")
$lines.Add("- current validation can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("- current validation is post-publish proof: ``$isPostPublishVerificationProof``")
$lines.Add("- failed proof item count: ``$failedProofItemCount``")
$lines.Add("- blocking reasons: $($blockingReasons.Count)")
$lines.Add("")
$lines.Add("## Backfill Steps")
$lines.Add("")
$lines.Add("| Order | ID | Command | Required evidence | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($step in $steps) {
  $lines.Add("| $($step.order) | ``$($step.id)`` | $($step.command.Replace("|", "\|")) | $($step.requiredEvidence.Replace("|", "\|")) | $($step.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Blocking Reasons")
$lines.Add("")
foreach ($reason in $blockingReasons) {
  $lines.Add("- $reason")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $record.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish verification backfill plan written to $jsonPath"
Write-Host "Post-publish verification backfill plan written to $markdownPath"
