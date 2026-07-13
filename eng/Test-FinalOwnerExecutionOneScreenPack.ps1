[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-one-screen-pack.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final owner execution one-screen pack not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @())
$gaps = Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "ownerInputGapTable" -DefaultValue @())
$finalPublicProofPath = Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "finalPublicProofPath" -DefaultValue @())
$ownerExecutionSequence = Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "ownerExecutionSequence" -DefaultValue @())
$releaseCloseRealInputChain = Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChain" -DefaultValue @())
$sourceArtifacts = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @()) | ForEach-Object { [string]$_ })
$nonSubstitutes = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "nonSubstituteProofKinds" -DefaultValue @()) | ForEach-Object { [string]$_ })
$finalCloseProofAdmissionLaneIds = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "finalCloseProofAdmissionLaneIds" -DefaultValue @()) | ForEach-Object { [string]$_ })
$finalCloseProofAdmissionRequiredFields = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "finalCloseProofAdmissionRequiredFields" -DefaultValue @()) | ForEach-Object { [string]$_ })
$finalCloseRejectedNonProofStates = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "finalCloseRejectedNonProofStates" -DefaultValue @()) | ForEach-Object { [string]$_ })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$allText = $record | ConvertTo-Json -Depth 22

$requiredLaneIds = @(
  "clean-external-package-consumer",
  "post-publish-owner-verification",
  "owner-public-publish-result-input",
  "post-publish-proof-record-contract",
  "final-release-close-owner-approval",
  "release-evidence-and-public-docs-freeze"
)
$laneIds = @($lanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })

$requiredGapIds = @(
  "clean-consumer-root",
  "package-source-url",
  "managed-package-identity",
  "runtime-package-identity",
  "native-asset-listing",
  "restore-build-run-logs",
  "log-sha256",
  "runtime-exit-and-smoke-status",
  "runtime-execution-timestamps",
  "host-metadata",
  "owner-review",
  "post-publish-downloaded-package-hash",
  "dual-package-nuget-route-owner-proof",
  "dual-package-github-runtime-route-owner-proof",
  "rollback-review",
  "final-close-decision",
  "strict-validator-chain"
)
$gapIds = @($gaps | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })

$requiredFinalPublicProofStepIds = @(
  "github-actions-run-evidence",
  "owner-public-publish-result",
  "public-package-download-proof",
  "public-package-download-owner-execution-pack",
  "post-publish-clean-consumer-proof-result",
  "post-publish-user-verification-pack",
  "final-public-release-closure-bridge",
  "release-issue-close-owner-decision-input",
  "dual-package-final-close-lanes"
)
$finalPublicProofStepIds = @($finalPublicProofPath | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })

$requiredOwnerExecutionStepIds = @(
  "remote-ci-github-actions-run-proof",
  "public-publish-owner-result",
  "public-package-download-proof",
  "clean-external-consumer-smoke-proof",
  "post-publish-verification-proof",
  "rollback-review",
  "final-close-decision"
)
$ownerExecutionStepIds = @($ownerExecutionSequence | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$invalidOwnerExecutionSteps = @($ownerExecutionSequence | Where-Object {
  -not [bool](Get-PropertyOrDefault -Object $_ -Name "blocked" -DefaultValue $false) -or
    [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "strictValidator" -DefaultValue "")) -or
    [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "requiredEvidence" -DefaultValue "")) -or
    [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "ownerAction" -DefaultValue ""))
})
$requiredReleaseCloseRealInputChainIds = @(
  "github-actions-run-evidence",
  "owner-public-publish-result",
  "public-package-download-proof",
  "post-publish-clean-consumer-proof-result",
  "rollback-review",
  "final-close-decision",
  "release-evidence-bundle-sha",
  "strict-close-validator-output"
)
$releaseCloseRealInputChainIds = @($releaseCloseRealInputChain | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$invalidReleaseCloseRealInputSteps = @($releaseCloseRealInputChain | Where-Object {
  -not [bool](Get-PropertyOrDefault -Object $_ -Name "blocked" -DefaultValue $false) -or
    -not [bool](Get-PropertyOrDefault -Object $_ -Name "ownerActionRequired" -DefaultValue $false) -or
    [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -or
    [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -or
    [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) -or
    [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "strictValidator" -DefaultValue "")) -or
    [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "blockedReason" -DefaultValue ""))
})

$requiredAdmissionLaneIds = @(
  "github-actions-run-proof",
  "owner-public-publish-result",
  "public-package-download-proof",
  "post-publish-clean-consumer-proof",
  "release-issue-close-record-strict-validation"
)
$requiredAdmissionFields = @(
  "publicPackageSourceUrl",
  "publicPackageDownloadUrl",
  "managedNupkgSha256",
  "runtimeNupkgSha256",
  "externalCleanConsumerProjectIdentity",
  "smokeCommandRuntimePackageKey",
  "hostCudaVersion",
  "hostTensorRtVersion",
  "hostCudnnVersion",
  "stdoutSha256",
  "stderrSha256",
  "mergedTranscriptSha256",
  "githubRunId",
  "githubHeadSha",
  "githubLogSha256",
  "githubArtifactSha256",
  "ownerReviewer",
  "ownerAuthorizationLink",
  "rollbackReview",
  "finalCloseDecision"
)
$requiredRejectedStates = @(
  "template-only",
  "candidate-only",
  "draft-rich-but-not-proof",
  "draft-blocked-by-cuda-driver",
  "not-requested",
  "validation-ready-without-proof-candidate",
  "dashboard-only",
  "runbook-only",
  "local-feed-only",
  "project-reference-only"
)

$requiredSources = @(
  "artifacts/final-release/clean-external-package-consumer-owner-runbook.json",
  "artifacts/final-release/post-publish-owner-verification-runbook.json",
  "artifacts/final-release/owner-public-publish-execution-result-input-contract.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json",
  "artifacts/final-release/final-release-close-owner-approval-contract.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/release-evidence-classification-audit.json",
  "artifacts/final-release/public-proof-claim-boundary-audit.json",
  "artifacts/final-release/clean-consumer-external-proof-closure-pack.json",
  "artifacts/final-release/github-actions-run-evidence-import-validation.json",
  "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json",
  "artifacts/final-release/public-package-download-proof-candidate-validation.json",
  "artifacts/final-release/public-package-download-proof-owner-execution-pack-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json",
  "artifacts/final-release/post-publish-user-verification-pack-validation.json",
  "artifacts/final-release/final-public-release-closure-bridge-validation.json",
  "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
  "artifacts/final-release/final-owner-rollback-review-validation.json",
  "artifacts/final-release/final-owner-close-decision-validation.json",
  "artifacts/final-release/release-issue-final-close-decision-validation.json",
  "artifacts/final-release/dual-package-publish-preflight-matrix.json",
  "artifacts/final-release/dual-package-publish-preflight-matrix-validation.json",
  "artifacts/final-release/final-close-gate-convergence.json",
  "artifacts/final-release/final-close-gate-convergence-validation.json"
)

$requiredMarkers = @(
  "final owner execution one-screen pack",
  "owner one-screen execution guidance",
  "owner input gap table",
  "final public proof path",
  "release candidate public proof final audit",
  "dual package final close lanes",
  "dual package publish preflight matrix",
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "pre-publish smoke reused as post-publish proof",
  "package-managed-dry-run",
  "dashboard-only",
  "artifact-only",
  "queued workflow",
  "missing runner",
  "sidecar-only",
  "local test"
)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-execution-one-screen-pack") -Severity "blocker" -Detail "recordKind must be final-owner-execution-one-screen-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "packState" -DefaultValue "") -eq "blocked-final-owner-execution-one-screen-real-owner-input-required") -Severity "blocker" -Detail "Pack must remain blocked until real Owner inputs are supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-lanes" -Passed (@($requiredLaneIds | Where-Object { $laneIds -notcontains $_ }).Count -eq 0 -and $lanes.Count -ge 6) -Severity "blocker" -Detail "Pack must include all final Owner execution lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-owner-gap-table" -Passed (@($requiredGapIds | Where-Object { $gapIds -notcontains $_ }).Count -eq 0 -and $gaps.Count -ge 17) -Severity "blocker" -Detail "Pack must include a real Owner input gap table covering logs, hashes, timestamps, host metadata, post-publish evidence, dual-package route proof, rollback, and final close.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-final-public-proof-path" -Passed (@($requiredFinalPublicProofStepIds | Where-Object { $finalPublicProofStepIds -notcontains $_ }).Count -eq 0 -and $finalPublicProofPath.Count -ge 9) -Severity "blocker" -Detail "Pack must include the final public proof path: GitHub Actions, Owner publish result, public download, owner download execution pack, post-publish clean consumer, post-publish user verification pack, final bridge, release issue close decision, and dual-package final close lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-execution-sequence" -Passed (@($requiredOwnerExecutionStepIds | Where-Object { $ownerExecutionStepIds -notcontains $_ }).Count -eq 0 -and $ownerExecutionSequence.Count -eq 7 -and $invalidOwnerExecutionSteps.Count -eq 0) -Severity "blocker" -Detail "Pack must expose the seven-step Owner execution sequence: remote CI, public publish result, public download proof, clean external consumer proof, post-publish proof, rollback review, and final close decision.")) | Out-Null
$items.Add((New-ValidationItem -Id "release-close-real-input-chain" -Passed (@($requiredReleaseCloseRealInputChainIds | Where-Object { $releaseCloseRealInputChainIds -notcontains $_ }).Count -eq 0 -and $releaseCloseRealInputChain.Count -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedReleaseCloseRealInputChainCount" -DefaultValue 0) -eq 8 -and $invalidReleaseCloseRealInputSteps.Count -eq 0) -Severity "blocker" -Detail "Pack must expose the ordered release-close real input chain: GitHub Actions, Owner publish, public download, post-publish CleanConsumer, rollback, final decision, evidence bundle SHA, and strict close validator output.")) | Out-Null
$items.Add((New-ValidationItem -Id "final-close-admission-contract-surface" -Passed (@($requiredAdmissionLaneIds | Where-Object { $finalCloseProofAdmissionLaneIds -notcontains $_ }).Count -eq 0 -and @($requiredAdmissionFields | Where-Object { $finalCloseProofAdmissionRequiredFields -notcontains $_ }).Count -eq 0 -and @($requiredRejectedStates | Where-Object { $finalCloseRejectedNonProofStates -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Pack must surface FinalClose admission lanes, required proof fields, and rejected non-proof states from final-close-gate-convergence.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-download-contract-counts" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofRequiredFieldCount" -DefaultValue 0) -ge 30 -and [int](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofRejectedSubstituteCount" -DefaultValue 0) -eq 11 -and [int](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofSourceReadinessSignalCount" -DefaultValue 0) -eq 7 -and -not [bool](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofCandidateReady" -DefaultValue $true)) -Severity "blocker" -Detail "Pack must surface public package download proof required fields, rejected substitutes, source readiness signals, and blocked proofCandidateReady state.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-contract-counts" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofRequiredFieldCount" -DefaultValue 0) -ge 50 -and [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofRejectedSubstituteCount" -DefaultValue 0) -eq 11 -and [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofSourceReadinessSignalCount" -DefaultValue 0) -eq 11 -and [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofBlockedRealInputCount" -DefaultValue 0) -gt 0 -and -not [bool](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofCandidateReady" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofSourceProofLinkageReady" -DefaultValue $true)) -Severity "blocker" -Detail "Pack must surface post-publish CleanConsumer required fields, rejected substitutes, blocked real inputs, source readiness signals, and blocked proof/source linkage state.")) | Out-Null
$items.Add((New-ValidationItem -Id "release-evidence-bundle-sha-and-strict-close-output" -Passed (([string](Get-PropertyOrDefault -Object $record -Name "releaseEvidenceBundleSha256" -DefaultValue "")) -match "^[0-9a-f]{64}$" -and [string](Get-PropertyOrDefault -Object $record -Name "finalCloseStrictValidatorOutputState" -DefaultValue "") -eq "blocked-final-close-gate-owner-proof-required" -and [int](Get-PropertyOrDefault -Object $record -Name "finalCloseStrictValidatorFailedBlockerCount" -DefaultValue 999) -eq 0) -Severity "blocker" -Detail "Pack must carry the release evidence bundle SHA and the strict close validator output state without treating them as close approval.")) | Out-Null
$items.Add((New-ValidationItem -Id "all-lanes-blocked-non-proof" -Passed (@($lanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "blocked" -DefaultValue $false) -or [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) }).Count -eq 0) -Severity "blocker" -Detail "Every lane must remain blocked, non-publish, non-promoting, and non-closing.")) | Out-Null
$items.Add((New-ValidationItem -Id "final-public-proof-path-blocked-non-proof" -Passed (@($finalPublicProofPath | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "blocked" -DefaultValue $false) -or [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "isRuntimeExecutionProof" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "isPostPublishProof" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "isReleaseCloseProof" -DefaultValue $true) }).Count -eq 0) -Severity "blocker" -Detail "Every final public proof path step must remain blocked and non-proof until real owner evidence is accepted.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-proof-cannot-substitute-post-publish" -Passed ($allText.Contains("public package download proof alone", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("post-publish validation-ready without proofCandidateReady", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-PostPublishCleanConsumerProofResult.ps1", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Pack must explicitly reject public package download proof as a substitute for post-publish CleanConsumer proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifacts" -Passed (@($requiredSources | Where-Object { $sourceArtifacts -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Pack must source upstream owner runbooks/contracts, release evidence, public docs freeze, and closure pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-validator-coverage" -Passed ($allText.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-PostPublishCleanConsumerProofRecordDraft.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-ReleaseEvidenceClassificationAudit.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-FinalReleaseCloseApprovalRealInputFromOwnerResult.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-GitHubActionsRunEvidenceImport.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-OwnerPublicPublishExecutionResultCandidate.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-PublicPackageDownloadProofCandidate.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-PublicPackageDownloadProofOwnerExecutionPack.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-PostPublishUserVerificationPack.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-FinalPublicReleaseClosureBridge.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-ReleaseIssueCloseOwnerDecisionInput.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-DualPackagePublishPreflightMatrix.ps1", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Pack must expose strict validators for runtime proof, post-publish proof, release evidence classification, final close approval, dual-package route proof, and the final public proof path.")) | Out-Null
$items.Add((New-ValidationItem -Id "dual-package-final-close-surface" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "dualPackageRouteCount" -DefaultValue 0) -eq 2 -and [int](Get-PropertyOrDefault -Object $record -Name "dualPackageFinalCloseBlockedLaneCount" -DefaultValue 0) -eq 2 -and -not [bool](Get-PropertyOrDefault -Object $record -Name "dualPackageAcceptsSubstituteProof" -DefaultValue $true) -and $allText.Contains("Test-DualPackagePublishPreflightMatrix.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("dual-package-final-close-lanes", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Pack must surface both dual-package final close lanes as blocked, non-substitute Owner proof requirements.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-non-substitute-markers" -Passed (@($requiredMarkers | Where-Object { $nonSubstitutes -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Pack must propagate final owner one-screen, owner guidance, gap table, and forbidden substitute markers.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsRuntimeExecution" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Pack must remain non-proof, non-publish, non-runtime, and non-close.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must state proof, publish, close, and package push exclusions.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "blocked-final-owner-execution-one-screen-real-owner-input-required" } else { "invalid-final-owner-execution-one-screen-pack" }

$validation = [pscustomobject]@{
  recordKind = "final-owner-execution-one-screen-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  inputPath = $resolvedInputPath
  laneCount = $lanes.Count
  ownerInputGapCount = $gaps.Count
  finalPublicProofPathCount = $finalPublicProofPath.Count
  ownerExecutionSequenceCount = $ownerExecutionSequence.Count
  releaseCloseRealInputChainCount = $releaseCloseRealInputChain.Count
  releaseCloseRealInputChainRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainRequiredFieldCount" -DefaultValue 0)
  releaseCloseRealInputChainRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainRejectedSubstituteCount" -DefaultValue 0)
  releaseCloseRealInputChainSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainSourceReadinessSignalCount" -DefaultValue 0)
  releaseCloseRealInputChainBlockedRealInputCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainBlockedRealInputCount" -DefaultValue 0)
  publicPackageDownloadProofRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofRequiredFieldCount" -DefaultValue 0)
  publicPackageDownloadProofRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofRejectedSubstituteCount" -DefaultValue 0)
  publicPackageDownloadProofSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofSourceReadinessSignalCount" -DefaultValue 0)
  publicPackageDownloadProofCandidateReady = [bool](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofCandidateReady" -DefaultValue $false)
  postPublishCleanConsumerProofRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofRequiredFieldCount" -DefaultValue 0)
  postPublishCleanConsumerProofRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofRejectedSubstituteCount" -DefaultValue 0)
  postPublishCleanConsumerProofBlockedRealInputCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofBlockedRealInputCount" -DefaultValue 0)
  postPublishCleanConsumerProofSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofSourceReadinessSignalCount" -DefaultValue 0)
  postPublishCleanConsumerProofCandidateReady = [bool](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofCandidateReady" -DefaultValue $false)
  postPublishCleanConsumerProofSourceProofLinkageReady = [bool](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofSourceProofLinkageReady" -DefaultValue $false)
  releaseEvidenceBundleSha256 = [string](Get-PropertyOrDefault -Object $record -Name "releaseEvidenceBundleSha256" -DefaultValue "")
  finalCloseStrictValidatorOutputState = [string](Get-PropertyOrDefault -Object $record -Name "finalCloseStrictValidatorOutputState" -DefaultValue "")
  finalCloseProofAdmissionLaneCount = $finalCloseProofAdmissionLaneIds.Count
  finalCloseProofAdmissionRequiredFieldCount = $finalCloseProofAdmissionRequiredFields.Count
  finalCloseRejectedNonProofStateCount = $finalCloseRejectedNonProofStates.Count
  dualPackageRouteCount = [int](Get-PropertyOrDefault -Object $record -Name "dualPackageRouteCount" -DefaultValue 0)
  dualPackageFinalCloseBlockedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "dualPackageFinalCloseBlockedLaneCount" -DefaultValue 0)
  dualPackageAcceptsSubstituteProof = [bool](Get-PropertyOrDefault -Object $record -Name "dualPackageAcceptsSubstituteProof" -DefaultValue $true)
  failedBlockerCount = $failedBlockers.Count
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = $validationItems
  boundary = "Validation confirms the FinalOwner one-screen pack shape and non-proof boundaries only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-one-screen-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-one-screen-pack-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validationItems) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Final Owner Execution One-Screen Pack Validation

| Field | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| laneCount | ``$($validation.laneCount)`` |
| ownerInputGapCount | ``$($validation.ownerInputGapCount)`` |
| finalPublicProofPathCount | ``$($validation.finalPublicProofPathCount)`` |
| ownerExecutionSequenceCount | ``$($validation.ownerExecutionSequenceCount)`` |
| releaseCloseRealInputChainCount | ``$($validation.releaseCloseRealInputChainCount)`` |
| releaseCloseRealInputChainRequiredFieldCount | ``$($validation.releaseCloseRealInputChainRequiredFieldCount)`` |
| releaseCloseRealInputChainRejectedSubstituteCount | ``$($validation.releaseCloseRealInputChainRejectedSubstituteCount)`` |
| releaseCloseRealInputChainSourceReadinessSignalCount | ``$($validation.releaseCloseRealInputChainSourceReadinessSignalCount)`` |
| releaseCloseRealInputChainBlockedRealInputCount | ``$($validation.releaseCloseRealInputChainBlockedRealInputCount)`` |
| publicPackageDownloadProofRequiredFieldCount | ``$($validation.publicPackageDownloadProofRequiredFieldCount)`` |
| publicPackageDownloadProofRejectedSubstituteCount | ``$($validation.publicPackageDownloadProofRejectedSubstituteCount)`` |
| publicPackageDownloadProofSourceReadinessSignalCount | ``$($validation.publicPackageDownloadProofSourceReadinessSignalCount)`` |
| publicPackageDownloadProofCandidateReady | ``$($validation.publicPackageDownloadProofCandidateReady)`` |
| postPublishCleanConsumerProofRequiredFieldCount | ``$($validation.postPublishCleanConsumerProofRequiredFieldCount)`` |
| postPublishCleanConsumerProofRejectedSubstituteCount | ``$($validation.postPublishCleanConsumerProofRejectedSubstituteCount)`` |
| postPublishCleanConsumerProofBlockedRealInputCount | ``$($validation.postPublishCleanConsumerProofBlockedRealInputCount)`` |
| postPublishCleanConsumerProofSourceReadinessSignalCount | ``$($validation.postPublishCleanConsumerProofSourceReadinessSignalCount)`` |
| postPublishCleanConsumerProofCandidateReady | ``$($validation.postPublishCleanConsumerProofCandidateReady)`` |
| postPublishCleanConsumerProofSourceProofLinkageReady | ``$($validation.postPublishCleanConsumerProofSourceProofLinkageReady)`` |
| releaseEvidenceBundleSha256 | ``$($validation.releaseEvidenceBundleSha256)`` |
| finalCloseStrictValidatorOutputState | ``$($validation.finalCloseStrictValidatorOutputState)`` |
| finalCloseProofAdmissionLaneCount | ``$($validation.finalCloseProofAdmissionLaneCount)`` |
| finalCloseProofAdmissionRequiredFieldCount | ``$($validation.finalCloseProofAdmissionRequiredFieldCount)`` |
| finalCloseRejectedNonProofStateCount | ``$($validation.finalCloseRejectedNonProofStateCount)`` |
| dualPackageRouteCount | ``$($validation.dualPackageRouteCount)`` |
| dualPackageFinalCloseBlockedLaneCount | ``$($validation.dualPackageFinalCloseBlockedLaneCount)`` |
| dualPackageAcceptsSubstituteProof | ``$($validation.dualPackageAcceptsSubstituteProof)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final owner execution one-screen pack validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) CanPublish=False CanClose=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final owner execution one-screen pack validation failed with $($failedBlockers.Count) blocker(s)."
}
