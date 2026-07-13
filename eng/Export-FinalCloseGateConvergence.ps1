[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function ConvertTo-StringArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value | ForEach-Object { [string]$_ }) }
  return @([string]$Value)
}

function Get-LaneById {
  param([AllowNull()][object[]]$Lanes, [string]$Id)
  return @($Lanes | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq $Id } | Select-Object -First 1)[0]
}

function New-GateLane {
  param([string]$Id, [AllowNull()][object]$Record, [string]$StateProperty, [string]$DefaultState, [string]$NextOwnerAction)
  $state = [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue $DefaultState)
  $blockedReasons = ConvertTo-StringArray (Get-PropertyOrDefault -Object $Record -Name "blockedReasons" -DefaultValue @())
  $blockedReason = [string](Get-PropertyOrDefault -Object $Record -Name "blockedReason" -DefaultValue "")
  if (-not [string]::IsNullOrWhiteSpace($blockedReason) -and $blockedReasons -notcontains $blockedReason) {
    $blockedReasons = @($blockedReasons + $blockedReason)
  }
  $sourceOwnerResultRows = ConvertTo-StringArray (Get-PropertyOrDefault -Object $Record -Name "sourceOwnerResultRow" -DefaultValue @())
  $resultArtifactPaths = ConvertTo-StringArray (Get-PropertyOrDefault -Object $Record -Name "resultArtifactPaths" -DefaultValue @())
  $hashProof = Get-PropertyOrDefault -Object $Record -Name "hashProof" -DefaultValue $null
  $validatorCommand = [string](Get-PropertyOrDefault -Object $Record -Name "validatorCommand" -DefaultValue "")
  [pscustomobject]@{
    id = $Id
    state = $state
    ready = $false
    strictValidatorRequired = $true
    strictValidatorInputOnly = $true
    bridgeInputOnly = [bool](Get-PropertyOrDefault -Object $Record -Name "bridgeInputOnly" -DefaultValue $true)
    validatorCommand = $validatorCommand
    sourceArtifactStateProperty = $StateProperty
    blockedReason = if ($blockedReasons.Count -gt 0) { [string]$blockedReasons[0] } else { "strict-validator-real-proof-required" }
    blockedReasons = @($blockedReasons)
    sourceOwnerResultRows = @($sourceOwnerResultRows)
    resultArtifactPaths = @($resultArtifactPaths)
    hashProof = $hashProof
    nextOwnerAction = $NextOwnerAction
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
    boundary = "Final close gate convergence lane only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

function New-RemoteProofGateLane {
  param(
    [string]$Id,
    [AllowNull()][object]$RemoteLane,
    [string]$NextOwnerAction,
    [string]$RequiredEvidence,
    [bool]$RequireProofReady = $false,
    [string]$ProofReadyProperty = ""
  )

  $state = [string](Get-PropertyOrDefault -Object $RemoteLane -Name "state" -DefaultValue "missing-remote-ci-and-public-publish-proof-backfill-gate-lane")
  $ready = [bool](Get-PropertyOrDefault -Object $RemoteLane -Name "ready" -DefaultValue $false)
  $stateReady = [bool](Get-PropertyOrDefault -Object $RemoteLane -Name "stateReady" -DefaultValue $false)
  $remoteRequireProofReady = [bool](Get-PropertyOrDefault -Object $RemoteLane -Name "requireProofReady" -DefaultValue $RequireProofReady)
  $remoteProofReadyProperty = [string](Get-PropertyOrDefault -Object $RemoteLane -Name "proofReadyProperty" -DefaultValue $ProofReadyProperty)
  $proofReady = [bool](Get-PropertyOrDefault -Object $RemoteLane -Name "proofReady" -DefaultValue $false)
  [pscustomobject]@{
    id = $Id
    state = $state
    ready = $ready
    strictValidatorRequired = $true
    strictValidatorInputOnly = $true
    bridgeInputOnly = $true
    validatorCommand = "eng\Test-RemoteCiAndPublicPublishProofBackfillGate.ps1 -Strict"
    sourceArtifactStateProperty = "remoteProofGateLane"
    blockedReason = if ($ready) { "" } else { "remote-ci-public-publish-and-post-publish-real-proof-required" }
    blockedReasons = if ($ready) { @() } else { @("remote-ci-public-publish-and-post-publish-real-proof-required") }
    sourceOwnerResultRows = @()
    resultArtifactPaths = @("artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.json", "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.json")
    hashProof = $null
    nextOwnerAction = $NextOwnerAction
    requiredEvidence = $RequiredEvidence
    remoteProofGateLaneId = $Id
    remoteGateStateReady = $stateReady
    requireProofReady = $remoteRequireProofReady
    proofReadyProperty = $remoteProofReadyProperty
    proofReady = $proofReady
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
    boundary = "Final close gate remote proof dependency lane only; not runtime proof, not post-publish proof, not GitHub Actions proof, not publish approval, not release close approval, and not package push."
  }
}

function New-DualPackageGateLane {
  param([AllowNull()][object]$Route)

  $routeId = [string](Get-PropertyOrDefault -Object $Route -Name "id" -DefaultValue "missing-dual-package-route")
  $blockedReasons = ConvertTo-StringArray (Get-PropertyOrDefault -Object $Route -Name "blockedReasons" -DefaultValue @("dual-package-owner-action-required"))
  [pscustomobject]@{
    id = "dual-package-$routeId"
    state = "blocked-dual-package-publish-route-owner-proof-required"
    ready = $false
    strictValidatorRequired = $true
    strictValidatorInputOnly = $true
    bridgeInputOnly = $true
    validatorCommand = "eng\Test-DualPackagePublishPreflightMatrix.ps1 -Strict"
    sourceArtifactStateProperty = "dualPackagePublishPreflightRoute"
    blockedReason = if ($blockedReasons.Count -gt 0) { [string]$blockedReasons[0] } else { "dual-package-owner-action-required" }
    blockedReasons = @($blockedReasons)
    sourceOwnerResultRows = @()
    resultArtifactPaths = @("artifacts/final-release/dual-package-publish-preflight-matrix.json", "artifacts/final-release/dual-package-publish-preflight-matrix.md", "artifacts/final-release/dual-package-publish-preflight-matrix-validation.json")
    hashProof = $null
    nextOwnerAction = [string](Get-PropertyOrDefault -Object $Route -Name "nextOwnerAction" -DefaultValue "owner-authorize-package-route-and-import-real-proof")
    requiredEvidence = "Owner authorization, external clean consumer proof, public/package-source download proof, and post-publish clean consumer proof for this package route."
    dualPackageRouteId = $routeId
    distributionChannel = [string](Get-PropertyOrDefault -Object $Route -Name "distributionChannel" -DefaultValue "")
    packageId = [string](Get-PropertyOrDefault -Object $Route -Name "packageId" -DefaultValue "")
    externalProofRequired = [bool](Get-PropertyOrDefault -Object $Route -Name "externalProofRequired" -DefaultValue $true)
    externalProofMissingReason = [string](Get-PropertyOrDefault -Object $Route -Name "externalProofMissingReason" -DefaultValue "external-proof-missing")
    postPublishProofRequired = [bool](Get-PropertyOrDefault -Object $Route -Name "postPublishProofRequired" -DefaultValue $true)
    postPublishProofMissingReason = [string](Get-PropertyOrDefault -Object $Route -Name "postPublishProofMissingReason" -DefaultValue "post-publish-proof-missing")
    ownerActionRequired = [bool](Get-PropertyOrDefault -Object $Route -Name "ownerActionRequired" -DefaultValue $true)
    acceptsSubstituteProof = [bool](Get-PropertyOrDefault -Object $Route -Name "acceptsSubstituteProof" -DefaultValue $true)
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
    boundary = "Dual package final-close lane only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$publicPublishRealResult = Read-JsonOrNull "artifacts\final-release\public-publish-real-result-owner-input-contract-validation.json"
$cleanConsumerProof = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-record-contract-validation.json"
$postPublishVerification = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$strictOwnerDecisionImport = Read-JsonOrNull "artifacts\final-release\release-issue-close-strict-owner-decision-import-validation.json"
$releaseIssueCloseRecord = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$strictCloseReady = Read-JsonOrNull "artifacts\final-release\strict-close-ready-convergence-dashboard-validation.json"
$realProofImportBridge = Read-JsonOrNull "artifacts\final-release\release-close-real-proof-import-bridge-validation.json"
$ownerResultImport = Read-JsonOrNull "artifacts\final-release\owner-external-proof-execution-result-import-validation.json"
$candidateFromOwnerResult = Read-JsonOrNull "artifacts\final-release\real-proof-record-candidate-from-owner-result-import-validation.json"
$realExternalImportValidator = Read-JsonOrNull "artifacts\final-release\real-external-proof-record-import-validator-validation.json"
$remoteProofBackfillGate = Read-JsonOrNull "artifacts\final-release\remote-ci-and-public-publish-proof-backfill-gate.json"
$remoteProofBackfillGateValidation = Read-JsonOrNull "artifacts\final-release\remote-ci-and-public-publish-proof-backfill-gate-validation.json"
$dualPackagePublishPreflightMatrix = Read-JsonOrNull "artifacts\final-release\dual-package-publish-preflight-matrix.json"
$remoteProofBackfillGateState = [string](Get-PropertyOrDefault -Object $remoteProofBackfillGateValidation -Name "validationState" -DefaultValue "missing-remote-ci-and-public-publish-proof-backfill-gate-validation")
$remoteProofLanes = @((Get-PropertyOrDefault -Object $remoteProofBackfillGate -Name "lanes" -DefaultValue @()))
$dualPackageRoutes = @((Get-PropertyOrDefault -Object $dualPackagePublishPreflightMatrix -Name "routes" -DefaultValue @()))
$dualPackageGateLanes = @($dualPackageRoutes | ForEach-Object { New-DualPackageGateLane -Route $_ })

$gateLanes = @(
  New-RemoteProofGateLane -Id "github-actions-run-proof" -RemoteLane (Get-LaneById -Lanes $remoteProofLanes -Id "github-actions-run-proof") -RequiredEvidence "Real GitHub Actions workflow run URL, run id, head SHA, conclusion, log hash, and artifact hash." -NextOwnerAction "Import a real successful GitHub Actions run proof for the pushed commit; queued workflow and local tests remain substitutes."
  New-RemoteProofGateLane -Id "owner-public-publish-result" -RemoteLane (Get-LaneById -Lanes $remoteProofLanes -Id "owner-public-publish-result") -RequiredEvidence "Owner public publish result with public package URLs, package identity, hashes, transcript hashes, reviewer, and authorization linkage." -NextOwnerAction "Import actual public NuGet/GitHub publish result from Owner execution."
  New-RemoteProofGateLane -Id "public-package-download-proof" -RemoteLane (Get-LaneById -Lanes $remoteProofLanes -Id "public-package-download-proof") -RequiredEvidence "Public package download source/URL, package identity, timestamp, and SHA256 from a non-local source." -NextOwnerAction "Download the public package from the public source and import hash proof."
  New-RemoteProofGateLane -Id "post-publish-clean-consumer-proof" -RemoteLane (Get-LaneById -Lanes $remoteProofLanes -Id "post-publish-clean-consumer-proof") -RequiredEvidence "Repository-external clean consumer restore/build/smoke proof with proofCandidateReady=true after public publication." -RequireProofReady $true -ProofReadyProperty "proofCandidateReady" -NextOwnerAction "Provide repository-external clean consumer proof from public packages; validation-ready without proofCandidateReady remains blocked."
) + $dualPackageGateLanes + @(
  New-GateLane -Id "public-publish-real-result" -Record $publicPublishRealResult -StateProperty "validationState" -DefaultState "missing-public-publish-real-result-owner-input-contract-validation" -NextOwnerAction "Provide real public package source, URL, SHA256, timestamp, transcript, reviewer, and rollback review."
  New-GateLane -Id "post-publish-clean-consumer-proof-record-contract" -Record $cleanConsumerProof -StateProperty "validationState" -DefaultState "missing-post-publish-clean-consumer-proof-record-contract-validation" -NextOwnerAction "Provide repository-external clean consumer restore/build/smoke evidence and no-substitute scan."
  New-GateLane -Id "post-publish-verification" -Record $postPublishVerification -StateProperty "validationState" -DefaultState "missing-post-publish-verification-validation" -NextOwnerAction "Run or import post-publish verification after real public package publish."
  New-GateLane -Id "owner-external-proof-result-import" -Record $ownerResultImport -StateProperty "validationState" -DefaultState "missing-owner-external-proof-execution-result-import-validation" -NextOwnerAction "Import owner-provided passed execution results with stdout, stderr, merged transcript, artifact paths, and SHA256 values."
  New-GateLane -Id "real-external-proof-record-import-validator" -Record $realExternalImportValidator -StateProperty "validationState" -DefaultState "missing-real-external-proof-record-import-validator-validation" -NextOwnerAction "Run strict validator before any owner result candidate can feed release close."
  New-GateLane -Id "owner-result-candidate-bridge" -Record $candidateFromOwnerResult -StateProperty "validationState" -DefaultState "missing-real-proof-record-candidate-from-owner-result-import-validation" -NextOwnerAction "Keep imported owner candidates as strict-validator input only until real proof validator accepts them."
  New-GateLane -Id "release-close-real-proof-import-bridge" -Record $realProofImportBridge -StateProperty "validationState" -DefaultState "missing-release-close-real-proof-import-bridge-validation" -NextOwnerAction "Bridge only strict validator accepted real proof records into final close readiness."
  New-GateLane -Id "strict-owner-decision-import" -Record $strictOwnerDecisionImport -StateProperty "validationState" -DefaultState "missing-release-issue-close-strict-owner-decision-import-validation" -NextOwnerAction "Import strict close owner decision after real proof contracts are filled."
  New-GateLane -Id "strict-close-ready-dashboard" -Record $strictCloseReady -StateProperty "validationState" -DefaultState "missing-strict-close-ready-convergence-dashboard-validation" -NextOwnerAction "Re-run strict close ready dashboard after all proof contracts pass."
  New-GateLane -Id "release-issue-close-record-strict-validation" -Record $releaseIssueCloseRecord -StateProperty "validationState" -DefaultState "missing-release-issue-close-record-validation" -NextOwnerAction "Run Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady as final owner close gate."
)

$blockedLanes = @($gateLanes | Where-Object { -not [bool]$_.ready })
$forbiddenSubstituteMarkers = @(
  "candidate",
  "draft",
  "dashboard",
  "dry-run",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "template",
  "build-only",
  "blocked-by-cuda-driver"
)
$strictValidatorSourceArtifacts = @(
  "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
  "artifacts/final-release/real-external-proof-record-import-validator-validation.json",
  "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json",
  "artifacts/final-release/release-close-real-proof-import-bridge-validation.json",
  "artifacts/final-release/release-issue-close-record-validation.json",
  "artifacts/final-release/dual-package-publish-preflight-matrix-validation.json"
)

$finalCloseProofAdmissionRequiredFields = @(
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

$rejectedNonProofStates = @(
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

$acceptedProofAdmissionContract = @(
  [pscustomobject]@{
    laneId = "github-actions-run-proof"
    requiredAcceptedState = "accepted-github-actions-run-proof"
    strictValidator = "eng\Test-GitHubActionsRunEvidenceImport.ps1 -Strict"
    requiredEvidenceFields = @("githubRunId", "githubHeadSha", "githubLogSha256", "githubArtifactSha256", "ownerReviewer")
    acceptedOnlyAfterStrictValidator = $true
    rejectsNonProofStates = $rejectedNonProofStates
    canPromoteRuntimeProof = $false
  },
  [pscustomobject]@{
    laneId = "owner-public-publish-result"
    requiredAcceptedState = "accepted-owner-public-publish-result"
    strictValidator = "eng\Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict"
    requiredEvidenceFields = @("publicPackageSourceUrl", "publicPackageDownloadUrl", "managedNupkgSha256", "runtimeNupkgSha256", "ownerAuthorizationLink", "ownerReviewer", "mergedTranscriptSha256")
    acceptedOnlyAfterStrictValidator = $true
    rejectsNonProofStates = $rejectedNonProofStates
    canPromoteRuntimeProof = $false
  },
  [pscustomobject]@{
    laneId = "public-package-download-proof"
    requiredAcceptedState = "accepted-public-package-download-proof"
    strictValidator = "eng\Test-PublicPackageDownloadProofCandidate.ps1 -Strict"
    requiredEvidenceFields = @("publicPackageSourceUrl", "publicPackageDownloadUrl", "managedNupkgSha256", "runtimeNupkgSha256", "ownerReviewer")
    acceptedOnlyAfterStrictValidator = $true
    rejectsNonProofStates = $rejectedNonProofStates
    canPromoteRuntimeProof = $false
  },
  [pscustomobject]@{
    laneId = "post-publish-clean-consumer-proof"
    requiredAcceptedState = "accepted-post-publish-clean-consumer-proof"
    strictValidator = "eng\Test-PostPublishCleanConsumerProofResult.ps1 -Strict -RequireExistingFiles -RequireHashMatch -FailOnNotProof"
    requiredEvidenceFields = @("externalCleanConsumerProjectIdentity", "smokeCommandRuntimePackageKey", "hostCudaVersion", "hostTensorRtVersion", "hostCudnnVersion", "stdoutSha256", "stderrSha256", "mergedTranscriptSha256", "ownerReviewer")
    acceptedOnlyAfterStrictValidator = $true
    rejectsNonProofStates = $rejectedNonProofStates
    canPromoteRuntimeProof = $false
  },
  [pscustomobject]@{
    laneId = "release-issue-close-record-strict-validation"
    requiredAcceptedState = "accepted-release-issue-close-record"
    strictValidator = "eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
    requiredEvidenceFields = @("rollbackReview", "finalCloseDecision", "ownerReviewer", "ownerAuthorizationLink", "mergedTranscriptSha256")
    acceptedOnlyAfterStrictValidator = $true
    rejectsNonProofStates = $rejectedNonProofStates
    canPromoteRuntimeProof = $false
  }
)

$record = [pscustomobject]@{
  recordKind = "final-close-gate-convergence"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  convergenceState = "blocked-final-close-gate-owner-proof-required"
  laneCount = $gateLanes.Count
  blockedLaneCount = $blockedLanes.Count
  readyLaneCount = 0
  gateLanes = @($gateLanes)
  remoteCiAndPublicPublishProofBackfillGateState = $remoteProofBackfillGateState
  remoteProofRequiredLaneIds = @("github-actions-run-proof", "owner-public-publish-result", "public-package-download-proof", "post-publish-clean-consumer-proof")
  dualPackagePublishPreflightState = [string](Get-PropertyOrDefault -Object $dualPackagePublishPreflightMatrix -Name "recordKind" -DefaultValue "missing-dual-package-publish-preflight-matrix")
  dualPackageRouteCount = $dualPackageRoutes.Count
  dualPackageRouteOwnerActions = @($dualPackageRoutes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "nextOwnerAction" -DefaultValue "") })
  dualPackageExternalProofMissingReasons = @($dualPackageRoutes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "externalProofMissingReason" -DefaultValue "") })
  dualPackagePostPublishProofMissingReasons = @($dualPackageRoutes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "postPublishProofMissingReason" -DefaultValue "") })
  dualPackageAcceptsSubstituteProof = [bool](Get-PropertyOrDefault -Object $dualPackagePublishPreflightMatrix -Name "acceptsSubstituteProof" -DefaultValue $true)
  sourceArtifacts = @(
    "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.json",
    "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.md",
    "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.json",
    "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.md",
    "artifacts/final-release/public-publish-real-result-owner-input-contract-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
    "artifacts/final-release/real-external-proof-record-import-validator-validation.json",
    "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json",
    "artifacts/final-release/release-close-real-proof-import-bridge-validation.json",
    "artifacts/final-release/dual-package-publish-preflight-matrix.json",
    "artifacts/final-release/dual-package-publish-preflight-matrix.md",
    "artifacts/final-release/release-issue-close-strict-owner-decision-import-validation.json",
    "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json"
  )
  strictValidatorSourceArtifacts = $strictValidatorSourceArtifacts
  forbiddenSubstituteMarkers = $forbiddenSubstituteMarkers
  finalCloseProofAdmissionRequiredFields = $finalCloseProofAdmissionRequiredFields
  rejectedNonProofStates = $rejectedNonProofStates
  acceptedProofAdmissionContract = $acceptedProofAdmissionContract
  acceptedProofAdmissionContractLaneIds = @($acceptedProofAdmissionContract | ForEach-Object { $_.laneId })
  acceptedProofAdmissionContractCount = $acceptedProofAdmissionContract.Count
  finalCloseAcceptedProofSources = @(
    "strict-validator-accepted-real-external-proof-record",
    "strict-release-issue-close-record-validation"
  )
  rejectedCloseSubstitutes = @(
    "candidate",
    "draft",
    "dashboard",
    "dry-run",
    "local feed",
    "ProjectReference",
    "direct .nupkg",
    "template",
    "build-only",
    "blocked-by-cuda-driver"
  )
  summary = [pscustomobject]@{
    strictValidatorRequired = $true
    candidateInputOnly = $true
    bridgeInputOnly = $true
    dashboardsCloseNothing = $true
    ownerRealProofRequired = $true
    dualPackageOwnerActionRequired = $true
  }
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  proofAdmissionRule = "Final close may only proceed after every required lane has strict-validator accepted real evidence with required hashes, public package URLs, owner authorization, rollback review, and final close decision. Template, draft, candidate, dashboard, runbook, local feed, ProjectReference, direct nupkg, build-only, and driver-blocked records are rejected."
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  boundary = "This convergence view summarizes final close blockers only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-close-gate-convergence.json"
$markdownPath = Join-Path $OutputRoot "final-close-gate-convergence.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Final Close Gate Convergence",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| convergenceState | ``$($record.convergenceState)`` |",
  "| laneCount | ``$($record.laneCount)`` |",
  "| blockedLaneCount | ``$($record.blockedLaneCount)`` |",
  "| acceptedProofAdmissionContractCount | ``$($record.acceptedProofAdmissionContractCount)`` |",
  "| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |",
  "",
  "## Proof Admission Contract",
  "",
  "| Lane | Accepted State | Strict Validator | Required Fields |",
  "| --- | --- | --- | --- |"
)

foreach ($admission in $acceptedProofAdmissionContract) {
  $markdown += "| $($admission.laneId) | ``$($admission.requiredAcceptedState)`` | ``$($admission.strictValidator)`` | $($admission.requiredEvidenceFields -join ', ') |"
}

$markdown += @(
  "",
  "## Gate Lanes",
  "",
  "| Lane | State | Next Owner Action |",
  "| --- | --- | --- |"
)

foreach ($lane in $gateLanes) {
  $markdown += "| $($lane.id) | ``$($lane.state)`` | $($lane.nextOwnerAction) |"
}

$markdown += @("", "## Boundary", "", $record.boundary)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final close gate convergence written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ConvergenceState=$($record.convergenceState) Lanes=$($record.laneCount) Blocked=$($record.blockedLaneCount)"
