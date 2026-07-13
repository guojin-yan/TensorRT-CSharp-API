[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-ReadyUrl {
  param([AllowNull()][object]$Value, [string]$Prefix = "https://")
  $text = ([string]$Value).Trim()
  return -not [string]::IsNullOrWhiteSpace($text) -and $text.StartsWith($Prefix, [StringComparison]::OrdinalIgnoreCase)
}

function Get-StringArrayProperty {
  param([AllowNull()][object]$Object, [string]$Name)
  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue @()
  return @($value | ForEach-Object { [string]$_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function New-ClosureConsistencyCheck {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function New-ClosureLane {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Artifact,
    [AllowNull()][object]$Record,
    [string]$StateProperty,
    [string]$RequiredState,
    [string]$OwnerAction,
    [string]$Boundary,
    [string[]]$RequiredBeforeClose,
    [bool]$RequireProofReady = $false,
    [string]$ProofReadyProperty = "proofCandidateReady",
    [string]$ValidatorPath = "",
    [string]$RequiredEvidence = ""
  )

  $state = [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue "missing-$Id")
  $proofReady = [bool](Get-PropertyOrDefault -Object $Record -Name $ProofReadyProperty -DefaultValue $false)
  $stateReady = $state -eq $RequiredState
  $ready = if ($RequireProofReady) { $stateReady -and $proofReady } else { $stateReady }
  $exists = $null -ne $Record
  $evidence = if ([string]::IsNullOrWhiteSpace($RequiredEvidence)) { $RequiredBeforeClose -join "; " } else { $RequiredEvidence }
  return [pscustomobject]@{
    laneId = $Id
    title = $Title
    artifact = $Artifact
    artifactExists = $exists
    state = $state
    requiredState = $RequiredState
    stateReady = $stateReady
    requireProofReady = $RequireProofReady
    proofReadyProperty = $ProofReadyProperty
    proofReady = $proofReady
    ready = $ready
    ownerAction = $OwnerAction
    requiredEvidence = $evidence
    validatorPath = $ValidatorPath
    requiredBeforeClose = @($RequiredBeforeClose)
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPackageConsumerRuntimeProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isPackageConsumerRuntimeProof" -DefaultValue $false)
    isPostPublishProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isPostPublishProof" -DefaultValue $false)
    isReleaseCloseProof = $false
    boundary = $Boundary
  }
}

function Get-MatrixLaneById {
  param([AllowNull()][object]$Matrix, [string]$Id)
  foreach ($lane in @((Get-PropertyOrDefault -Object $Matrix -Name "lanes" -DefaultValue @()))) {
    $laneId = [string](Get-PropertyOrDefault -Object $lane -Name "id" -DefaultValue "")
    if ($laneId.Equals($Id, [StringComparison]::OrdinalIgnoreCase)) {
      return $lane
    }
  }

  return $null
}

$ownerAuthorization = Read-JsonOrNull "artifacts\final-release\owner-publish-authorization-input-validation.json"
$ownerPublishExecutionResult = Read-JsonOrNull "artifacts\final-release\owner-publish-execution-result-input-validation.json"
$githubActionsRunEvidence = Read-JsonOrNull "artifacts\final-release\github-actions-run-evidence-import-validation.json"
$ownerPublicPublishResult = Read-JsonOrNull "artifacts\final-release\owner-public-publish-execution-result-candidate-validation.json"
$publicDownload = Read-JsonOrNull "artifacts\final-release\public-package-download-proof-candidate-validation.json"
$publicPackageDownloadOwnerExecutionPack = Read-JsonOrNull "artifacts\final-release\public-package-download-proof-owner-execution-pack-validation.json"
$cleanConsumerSmoke = Read-JsonOrNull "artifacts\final-release\clean-external-consumer-smoke-input-validation.json"
$postPublishProof = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-validation.json"
$postPublishUserVerificationPack = Read-JsonOrNull "artifacts\final-release\post-publish-user-verification-pack-validation.json"
$releaseCloseDecision = Read-JsonOrNull "artifacts\final-release\release-issue-close-owner-decision-input-validation.json"
$strictCloseDashboard = Read-JsonOrNull "artifacts\final-release\strict-close-ready-convergence-dashboard-validation.json"
$preReleaseReadinessMatrix = Read-JsonOrNull "artifacts\final-release\pre-release-package-proof-readiness-matrix.json"

$preReleaseMatrixState = [string](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "matrixState" -DefaultValue "missing-pre-release-package-proof-readiness-matrix")
$preReleaseMatrixReady = $preReleaseMatrixState.Equals("pre-release-package-proof-ready", [StringComparison]::OrdinalIgnoreCase)
$preReleaseReadyLaneCount = [int](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "readyLaneCount" -DefaultValue 0)
$preReleaseBlockedLaneCount = [int](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "blockedLaneCount" -DefaultValue 0)
$preReleaseCurrentHead = [string](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "currentHead" -DefaultValue "")
$preReleaseSourceQualityRunId = [string](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "sourceQualityRunId" -DefaultValue "")
$preReleaseMatrixLanes = @((Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "lanes" -DefaultValue @()))
$preReleaseRequiredLaneIds = @(
  "source-quality-ci",
  "current-head-package-dry-run",
  "owner-dispatch-pack",
  "public-package-download",
  "clean-external-package-consumer-runtime",
  "post-publish-clean-consumer-proof"
)
$preReleaseLaneIds = @($preReleaseMatrixLanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$preReleaseMissingLaneIds = @($preReleaseRequiredLaneIds | Where-Object { $preReleaseLaneIds -notcontains $_ })
$preReleaseLaneMetadataFindings = New-Object System.Collections.Generic.List[string]
$preReleasePrematurePromoteFindings = New-Object System.Collections.Generic.List[string]
foreach ($lane in $preReleaseMatrixLanes) {
  $laneId = [string](Get-PropertyOrDefault -Object $lane -Name "id" -DefaultValue "")
  $requiredEvidence = [string](Get-PropertyOrDefault -Object $lane -Name "requiredEvidence" -DefaultValue "")
  $validatorPath = [string](Get-PropertyOrDefault -Object $lane -Name "validatorPath" -DefaultValue "")
  $ready = [bool](Get-PropertyOrDefault -Object $lane -Name "ready" -DefaultValue $false)
  $canPromotePublic = [bool](Get-PropertyOrDefault -Object $lane -Name "canPromotePublicProof" -DefaultValue $false)
  $canPromoteRuntime = [bool](Get-PropertyOrDefault -Object $lane -Name "canPromoteRuntimeProof" -DefaultValue $false)
  $canPromotePostPublish = [bool](Get-PropertyOrDefault -Object $lane -Name "canPromotePostPublishProof" -DefaultValue $false)
  if ([string]::IsNullOrWhiteSpace($requiredEvidence) -or [string]::IsNullOrWhiteSpace($validatorPath)) {
    $preReleaseLaneMetadataFindings.Add($laneId) | Out-Null
  }

  if (-not $ready -and ($canPromotePublic -or $canPromoteRuntime -or $canPromotePostPublish)) {
    $preReleasePrematurePromoteFindings.Add("$laneId:promote-flag-on-blocked-lane") | Out-Null
  }

  if (@("source-quality-ci", "current-head-package-dry-run", "owner-dispatch-pack") -contains $laneId -and ($canPromotePublic -or $canPromoteRuntime -or $canPromotePostPublish)) {
    $preReleasePrematurePromoteFindings.Add("$laneId:non-proof-lane-promote-flag") | Out-Null
  }
}
$preReleaseLaneMetadataReady = $null -ne $preReleaseReadinessMatrix -and $preReleaseMissingLaneIds.Count -eq 0 -and $preReleaseLaneMetadataFindings.Count -eq 0
$preReleasePromoteFlagsSafe = $preReleasePrematurePromoteFindings.Count -eq 0

$lanes = @(
  New-ClosureLane `
    -Id "pre-release-package-proof-readiness" `
    -Title "Pre-release package proof readiness matrix" `
    -Artifact "artifacts/final-release/pre-release-package-proof-readiness-matrix.json" `
    -Record $preReleaseReadinessMatrix `
    -StateProperty "matrixState" `
    -RequiredState "pre-release-package-proof-ready" `
    -OwnerAction "Unblock current-head package dry-run, public package download, clean external consumer runtime, and post-publish proof lanes according to each matrix lane's validatorPath and requiredEvidence." `
    -RequiredBeforeClose @("matrix lane validatorPath", "matrix lane requiredEvidence", "current-head package dry-run evidence", "public package download proof", "clean external consumer runtime proof", "post-publish proof") `
    -Boundary "Pre-release readiness matrix is an upstream classifier only; source-quality, dispatch packs, preflights, templates, local artifacts, and blocked lanes cannot promote public/runtime/post-publish proof." `
    -ValidatorPath "eng\Export-PreReleasePackageProofReadinessMatrix.ps1" `
    -RequiredEvidence "All pre-release proof lanes expose validatorPath/requiredEvidence and are ready without premature promote flags."
  New-ClosureLane `
    -Id "github-actions-run-proof" `
    -Title "GitHub Actions run evidence import" `
    -Artifact "artifacts/final-release/github-actions-run-evidence-import-validation.json" `
    -Record $githubActionsRunEvidence `
    -StateProperty "validationState" `
    -RequiredState "github-actions-run-evidence-ready" `
    -OwnerAction "Import a real successful GitHub Actions run for the exact release commit, including run URL, run id, head SHA, workflow log hash, and artifact manifest hash." `
    -RequiredBeforeClose @("real GitHub Actions run URL", "run id", "head SHA", "workflow log SHA256", "artifact manifest SHA256") `
    -Boundary "GitHub Actions run evidence import validation is read-only; it does not trigger workflows, publish packages, prove runtime smoke, or close release issues." `
    -RequireProofReady $true `
    -ProofReadyProperty "githubActionsRunEvidenceReady"
  New-ClosureLane `
    -Id "owner-public-publish-result" `
    -Title "Owner public publish execution result candidate" `
    -Artifact "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json" `
    -Record $ownerPublicPublishResult `
    -StateProperty "validationState" `
    -RequiredState "owner-public-publish-execution-result-candidate-ready" `
    -OwnerAction "Import Owner-supplied public publish result with public package URL/version/SHA, managed/runtime package URLs, GitHub release asset, reviewer, authorization, and GitHub Actions linkage." `
    -RequiredBeforeClose @("owner reviewer", "public package URL", "public package version", "public package SHA256", "managed/runtime package URLs", "GitHub release asset SHA256") `
    -Boundary "Owner public publish result candidate validation is strict-validator input only; it does not execute publish, use tokens, prove clean consumer runtime, or close the release issue." `
    -RequireProofReady $true `
    -ProofReadyProperty "proofCandidateReady"
  New-ClosureLane `
    -Id "owner-publish-authorization" `
    -Title "Owner publish authorization input" `
    -Artifact "artifacts/final-release/owner-publish-authorization-input-validation.json" `
    -Record $ownerAuthorization `
    -StateProperty "validationState" `
    -RequiredState "owner-publish-authorization-input-ready-for-owner-run" `
    -OwnerAction "Owner reviews publish commands, hashes, release notes, rollback plan, and explicitly approves only an owner-run publish." `
    -RequiredBeforeClose @("owner identity", "package hashes reviewed", "publish command reviewed", "post-publish proof still required") `
    -Boundary "Authorization validation never publishes, stores tokens, proves public download, proves runtime smoke, or closes the release issue."
  New-ClosureLane `
    -Id "owner-publish-execution-result" `
    -Title "Owner publish execution result input" `
    -Artifact "artifacts/final-release/owner-publish-execution-result-input-validation.json" `
    -Record $ownerPublishExecutionResult `
    -StateProperty "validationState" `
    -RequiredState "owner-publish-execution-result-input-ready" `
    -OwnerAction "Owner imports redacted publish transcript, public package URLs, downloaded package hashes, release notes, rollback plan, and no-token confirmations after the owner-run publish." `
    -RequiredBeforeClose @("owner-run publish result", "redacted transcript hashes", "public package URLs", "downloaded package hashes", "rollback review") `
    -Boundary "Owner publish execution result validation does not publish, use tokens, prove clean consumer runtime smoke, prove post-publish verification, or close the release issue."
  New-ClosureLane `
    -Id "public-package-download-proof" `
    -Title "Public package download proof candidate" `
    -Artifact "artifacts/final-release/public-package-download-proof-candidate-validation.json" `
    -Record $publicDownload `
    -StateProperty "validationState" `
    -RequiredState "public-package-download-proof-candidate-ready" `
    -OwnerAction "Validate the imported public package download candidate after downloading managed/runtime packages from public package sources and backfilling URLs, paths, and SHA256 values." `
    -RequiredBeforeClose @("public managed package URL", "public runtime package URL", "downloaded managed SHA256", "downloaded runtime SHA256") `
    -Boundary "Public download proof is not a local feed, direct nupkg, dry-run artifact, or GitHub Actions artifact substitute."
  New-ClosureLane `
    -Id "public-package-download-owner-execution-pack" `
    -Title "Public package download proof owner execution pack" `
    -Artifact "artifacts/final-release/public-package-download-proof-owner-execution-pack-validation.json" `
    -Record $publicPackageDownloadOwnerExecutionPack `
    -StateProperty "validationState" `
    -RequiredState "public-package-download-proof-owner-execution-ready-for-owner-review" `
    -OwnerAction "Owner uses the manual execution pack to download public managed/runtime packages, capture SHA256/size fields, fill the public download proof input, import the candidate, and refresh evidence." `
    -RequiredBeforeClose @("owner download commands reviewed", "public URLs captured", "downloaded package hashes captured", "candidate validator run", "post-publish verification pack refreshed") `
    -Boundary "The Owner execution pack is manual guidance only; it is not public package download proof by itself, not runtime proof, not post-publish proof, not publish approval, not package push, and cannot close the release issue." `
    -ValidatorPath "eng\Test-PublicPackageDownloadProofOwnerExecutionPack.ps1 -Strict" `
    -RequiredEvidence "A completed owner-filled public-package-download-proof-input plus public-package-download-proof-candidate-validation.json, not this guidance pack alone."
  New-ClosureLane `
    -Id "clean-external-consumer-smoke" `
    -Title "Clean external consumer smoke input" `
    -Artifact "artifacts/final-release/clean-external-consumer-smoke-input-validation.json" `
    -Record $cleanConsumerSmoke `
    -StateProperty "validationState" `
    -RequiredState "clean-external-consumer-smoke-input-ready" `
    -OwnerAction "Run a repository-external clean consumer using package references and capture stdout/stderr/runtime probe hashes." `
    -RequiredBeforeClose @("external consumer root", "no ProjectReference", "runtime-package-key smoke", "native assets copied", "exit code zero") `
    -Boundary "A sample, ProjectReference, local RestoreSources, direct nupkg, build-only run, or dependency-probe-only run cannot replace smoke proof."
  New-ClosureLane `
    -Id "post-publish-proof" `
    -Title "Post-publish clean consumer proof result" `
    -Artifact "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json" `
    -Record $postPublishProof `
    -StateProperty "validationState" `
    -RequiredState "post-publish-clean-consumer-proof-result-validation-ready" `
    -OwnerAction "After public publish, import and validate repository-external clean consumer restore/build/run logs, public package hashes, host metadata, and proofCandidateReady=true." `
    -RequiredBeforeClose @("HTTPS public package metadata", "downloaded public package hashes", "external consumer smoke logs", "host/runtime metadata") `
    -Boundary "Validation-ready alone cannot close this lane; proofCandidateReady must be true, and the bridge itself is not runtime proof, not post-publish proof, not publish approval, and not release close approval." `
    -RequireProofReady $true `
    -ProofReadyProperty "proofCandidateReady"
  New-ClosureLane `
    -Id "post-publish-user-verification-pack" `
    -Title "Post-publish user verification pack" `
    -Artifact "artifacts/final-release/post-publish-user-verification-pack-validation.json" `
    -Record $postPublishUserVerificationPack `
    -StateProperty "validationState" `
    -RequiredState "post-publish-user-verification-ready-for-owner-close-review" `
    -OwnerAction "Refresh and review the user verification pack after public download, clean external consumer, post-publish proof, final audit, and strict close lanes are all ready." `
    -RequiredBeforeClose @("public download lane ready", "clean external consumer lane ready", "post-publish proof lane ready", "final post-publish audit ready", "strict close dashboard ready") `
    -Boundary "The post-publish user verification pack is owner action aggregation only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close release issue." `
    -ValidatorPath "eng\Test-PostPublishUserVerificationPack.ps1 -Strict" `
    -RequiredEvidence "All post-publish user verification lanes ready with real public download and clean external consumer evidence."
  New-ClosureLane `
    -Id "release-issue-close-owner-decision" `
    -Title "Release issue close owner decision input" `
    -Artifact "artifacts/final-release/release-issue-close-owner-decision-input-validation.json" `
    -Record $releaseCloseDecision `
    -StateProperty "validationState" `
    -RequiredState "release-issue-close-owner-decision-input-ready" `
    -OwnerAction "Owner signs final close decision only after all proof lanes and rollback review are ready." `
    -RequiredBeforeClose @("release issue URL", "owner final close decision", "rollback plan", "approved proof hashes") `
    -Boundary "Owner close decision validation is separate from publishing and cannot close the issue by itself."
  New-ClosureLane `
    -Id "strict-close-ready-convergence-dashboard" `
    -Title "Strict close ready convergence dashboard" `
    -Artifact "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json" `
    -Record $strictCloseDashboard `
    -StateProperty "validationState" `
    -RequiredState "strict-close-ready-convergence-dashboard-ready" `
    -OwnerAction "Refresh the strict close dashboard after every real owner input and proof validator passes." `
    -RequiredBeforeClose @("all strict close lanes ready", "classification audit clean", "no proof substitutes") `
    -Boundary "The dashboard summarizes readiness only; it is not package push, publish approval, runtime proof, or issue closure."
)

$githubActionsReady = [bool](Get-PropertyOrDefault -Object $githubActionsRunEvidence -Name "githubActionsRunEvidenceReady" -DefaultValue $false)
$githubActionsRunId = [string](Get-PropertyOrDefault -Object $githubActionsRunEvidence -Name "runId" -DefaultValue "")
$githubActionsRunUrl = [string](Get-PropertyOrDefault -Object $githubActionsRunEvidence -Name "runUrl" -DefaultValue "")
$githubActionsHeadSha = [string](Get-PropertyOrDefault -Object $githubActionsRunEvidence -Name "headSha" -DefaultValue "")
$githubActionsWorkflowRunLogSha256 = [string](Get-PropertyOrDefault -Object $githubActionsRunEvidence -Name "workflowRunLogSha256" -DefaultValue "")
$githubActionsArtifactManifestSha256 = [string](Get-PropertyOrDefault -Object $githubActionsRunEvidence -Name "artifactManifestSha256" -DefaultValue "")

$ownerPublicPublishReady = [bool](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "proofCandidateReady" -DefaultValue $false)
$ownerPublicPublishSourceGitHubActionsReady = [bool](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "sourceGitHubActionsRunEvidenceReady" -DefaultValue $false)
$ownerPublicPackageUrl = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "publicPackageUrl" -DefaultValue "")
$ownerPublicPackageVersion = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "publicPackageVersion" -DefaultValue "")
$ownerPublicPackageSha256 = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "publicPackageSha256" -DefaultValue "")
$ownerManagedPackageUrl = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "managedPackageUrl" -DefaultValue "")
$ownerRuntimePackageUrl = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "runtimePackageUrl" -DefaultValue "")
$ownerGitHubReleaseUrl = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "githubReleaseUrl" -DefaultValue "")
$ownerGitHubReleaseAssetUrl = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "githubReleaseAssetUrl" -DefaultValue "")
$ownerGitHubReleaseAssetSha256 = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "githubReleaseAssetSha256" -DefaultValue "")
$ownerReviewer = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "ownerReviewer" -DefaultValue "")
$ownerReviewTimestampUtc = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "ownerReviewTimestampUtc" -DefaultValue "")

$publicDownloadReady = [bool](Get-PropertyOrDefault -Object $publicDownload -Name "proofCandidateReady" -DefaultValue $false)
$publicDownloadSourceGitHubActionsReady = [bool](Get-PropertyOrDefault -Object $publicDownload -Name "sourceGitHubActionsRunEvidenceReady" -DefaultValue $false)
$publicDownloadSourceOwnerReady = [bool](Get-PropertyOrDefault -Object $publicDownload -Name "sourceOwnerPublicPublishResultReady" -DefaultValue $false)
$publicDownloadManagedPackageUrl = [string](Get-PropertyOrDefault -Object $publicDownload -Name "managedPackagePageUrl" -DefaultValue "")
$publicDownloadManagedPackageDownloadUrl = [string](Get-PropertyOrDefault -Object $publicDownload -Name "managedPackageDownloadUrl" -DefaultValue "")
$publicDownloadRuntimePackageUrl = [string](Get-PropertyOrDefault -Object $publicDownload -Name "runtimePackagePageUrl" -DefaultValue "")
$publicDownloadRuntimePackageDownloadUrl = [string](Get-PropertyOrDefault -Object $publicDownload -Name "runtimePackageDownloadUrl" -DefaultValue "")
$publicDownloadSourceOwnerPackageUrl = [string](Get-PropertyOrDefault -Object $publicDownload -Name "sourceOwnerPublicPackageUrl" -DefaultValue "")
$publicDownloadSourceOwnerPackageVersion = [string](Get-PropertyOrDefault -Object $publicDownload -Name "sourceOwnerPublicPackageVersion" -DefaultValue "")
$publicDownloadSourceOwnerPackageSha256 = [string](Get-PropertyOrDefault -Object $publicDownload -Name "sourceOwnerPublicPackageSha256" -DefaultValue "")
$publicDownloadGitHubReleaseUrl = [string](Get-PropertyOrDefault -Object $publicDownload -Name "githubReleaseUrl" -DefaultValue "")
$publicDownloadGitHubReleaseAssetUrl = [string](Get-PropertyOrDefault -Object $publicDownload -Name "githubReleaseAssetUrl" -DefaultValue "")
$publicDownloadGitHubReleaseAssetSha256 = [string](Get-PropertyOrDefault -Object $publicDownload -Name "githubReleaseAssetSha256" -DefaultValue "")
$publicPackageDownloadOwnerExecutionState = [string](Get-PropertyOrDefault -Object $publicPackageDownloadOwnerExecutionPack -Name "validationState" -DefaultValue "missing-public-package-download-proof-owner-execution-pack-validation")
$publicPackageDownloadOwnerExecutionBlocked = $publicPackageDownloadOwnerExecutionState -eq "blocked-public-package-download-proof-owner-execution-required"
$publicPackageDownloadOwnerExecutionSafe =
  $null -eq $publicPackageDownloadOwnerExecutionPack -or (
  -not [bool](Get-PropertyOrDefault -Object $publicPackageDownloadOwnerExecutionPack -Name "performsPublish" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $publicPackageDownloadOwnerExecutionPack -Name "usesPublishToken" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $publicPackageDownloadOwnerExecutionPack -Name "canPublishPublicly" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $publicPackageDownloadOwnerExecutionPack -Name "canCloseReleaseIssue" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $publicPackageDownloadOwnerExecutionPack -Name "isPostPublishProof" -DefaultValue $true))
$postPublishProofReady = [bool](Get-PropertyOrDefault -Object $postPublishProof -Name "proofCandidateReady" -DefaultValue $false)
$postPublishSourceProofLinkageReady = [bool](Get-PropertyOrDefault -Object $postPublishProof -Name "sourceProofLinkageReady" -DefaultValue $false)
$postPublishSourceGitHubActionsReady = [bool](Get-PropertyOrDefault -Object $postPublishProof -Name "sourceGitHubActionsRunEvidenceReady" -DefaultValue $false)
$postPublishSourceOwnerReady = [bool](Get-PropertyOrDefault -Object $postPublishProof -Name "sourceOwnerPublicPublishResultReady" -DefaultValue $false)
$postPublishSourcePublicDownloadReady = [bool](Get-PropertyOrDefault -Object $postPublishProof -Name "sourcePublicPackageDownloadProofReady" -DefaultValue $false)
$postPublishPublicPackageUrl = [string](Get-PropertyOrDefault -Object $postPublishProof -Name "publicPackageUrl" -DefaultValue "")
$postPublishPublicPackageVersion = [string](Get-PropertyOrDefault -Object $postPublishProof -Name "managedPackageVersion" -DefaultValue "")
$postPublishPublicPackageSha256 = [string](Get-PropertyOrDefault -Object $postPublishProof -Name "downloadedManagedPackageSha256" -DefaultValue "")
$postPublishSourceOwnerPackageUrl = [string](Get-PropertyOrDefault -Object $postPublishProof -Name "sourceOwnerPublicPackageUrl" -DefaultValue "")
$postPublishSourceOwnerPackageVersion = [string](Get-PropertyOrDefault -Object $postPublishProof -Name "sourceOwnerPublicPackageVersion" -DefaultValue "")
$postPublishSourceOwnerPackageSha256 = [string](Get-PropertyOrDefault -Object $postPublishProof -Name "sourceOwnerPublicPackageSha256" -DefaultValue "")
$postPublishUserVerificationState = [string](Get-PropertyOrDefault -Object $postPublishUserVerificationPack -Name "validationState" -DefaultValue "missing-post-publish-user-verification-pack-validation")
$postPublishUserVerificationBlocked = $postPublishUserVerificationState -eq "blocked-post-publish-user-verification-required"
$postPublishUserVerificationSafe =
  $null -eq $postPublishUserVerificationPack -or (
  -not [bool](Get-PropertyOrDefault -Object $postPublishUserVerificationPack -Name "performsPublish" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $postPublishUserVerificationPack -Name "usesPublishToken" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $postPublishUserVerificationPack -Name "canPublishPublicly" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $postPublishUserVerificationPack -Name "canCloseReleaseIssue" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $postPublishUserVerificationPack -Name "isPostPublishProof" -DefaultValue $true))
$publicDownloadForbiddenFindings = Get-StringArrayProperty -Object $publicDownload -Name "forbiddenSubstituteFindings"
$ownerForbiddenFindings = Get-StringArrayProperty -Object $ownerPublicPublishResult -Name "forbiddenSubstituteFindings"
$githubActionsForbiddenFindings = Get-StringArrayProperty -Object $githubActionsRunEvidence -Name "forbiddenSubstituteFindings"
$allForbiddenFindings = @($publicDownloadForbiddenFindings + $ownerForbiddenFindings + $githubActionsForbiddenFindings) | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Select-Object -Unique
$forbiddenFindingCount = @($allForbiddenFindings).Count

$crossLaneConsistencyChecks = @(
  New-ClosureConsistencyCheck -Id "github-actions-run-evidence-ready" -Passed $githubActionsReady -Severity "action-required" -Detail "A real GitHub Actions run evidence validation must be ready before release close review."
  New-ClosureConsistencyCheck -Id "pre-release-readiness-matrix-present" -Passed ($null -ne $preReleaseReadinessMatrix) -Severity "action-required" -Detail "Pre-release readiness matrix must be generated before final public release closure review."
  New-ClosureConsistencyCheck -Id "pre-release-readiness-lanes-present" -Passed ($preReleaseMissingLaneIds.Count -eq 0) -Severity "blocker" -Detail $(if ($preReleaseMissingLaneIds.Count -eq 0) { "All required pre-release readiness lanes are present." } else { "Missing pre-release lane(s): $($preReleaseMissingLaneIds -join ', ')" })
  New-ClosureConsistencyCheck -Id "pre-release-readiness-lane-metadata-present" -Passed $preReleaseLaneMetadataReady -Severity "blocker" -Detail $(if ($preReleaseLaneMetadataReady) { "Every pre-release lane exposes requiredEvidence and validatorPath." } else { "Lane metadata missing for: $($preReleaseLaneMetadataFindings -join ', ')" })
  New-ClosureConsistencyCheck -Id "pre-release-readiness-no-premature-promote-flags" -Passed $preReleasePromoteFlagsSafe -Severity "blocker" -Detail $(if ($preReleasePromoteFlagsSafe) { "No blocked or non-proof pre-release lane exposes public/runtime/post-publish promote flags." } else { "Premature promote flag(s): $($preReleasePrematurePromoteFindings -join ', ')" })
  New-ClosureConsistencyCheck -Id "pre-release-readiness-ready-for-close" -Passed $preReleaseMatrixReady -Severity "action-required" -Detail "Pre-release package proof readiness matrix must be ready before owner close review."
  New-ClosureConsistencyCheck -Id "github-actions-run-url-present" -Passed (Test-ReadyUrl -Value $githubActionsRunUrl -Prefix "https://github.com/") -Severity "action-required" -Detail "GitHub Actions run URL must be a real github.com actions URL."
  New-ClosureConsistencyCheck -Id "github-actions-head-sha-format" -Passed ($githubActionsHeadSha -match "^[0-9a-fA-F]{40}$") -Severity "action-required" -Detail "GitHub Actions head SHA must identify the reviewed commit."
  New-ClosureConsistencyCheck -Id "github-actions-log-and-artifact-hashes" -Passed ((Test-Sha256Format -Value $githubActionsWorkflowRunLogSha256) -and (Test-Sha256Format -Value $githubActionsArtifactManifestSha256)) -Severity "action-required" -Detail "Workflow log SHA256 and artifact manifest SHA256 must be present."
  New-ClosureConsistencyCheck -Id "owner-public-publish-result-ready" -Passed $ownerPublicPublishReady -Severity "action-required" -Detail "Owner public publish result candidate must be ready."
  New-ClosureConsistencyCheck -Id "owner-public-publish-links-github-actions" -Passed ($ownerPublicPublishSourceGitHubActionsReady -and $githubActionsReady) -Severity "action-required" -Detail "Owner public publish result must link to ready GitHub Actions run evidence."
  New-ClosureConsistencyCheck -Id "public-download-proof-ready" -Passed $publicDownloadReady -Severity "action-required" -Detail "Public package download proof candidate must be ready."
  New-ClosureConsistencyCheck -Id "public-download-links-source-proofs" -Passed ($publicDownloadSourceGitHubActionsReady -and $publicDownloadSourceOwnerReady) -Severity "action-required" -Detail "Public download proof must link to ready GitHub Actions and Owner public publish result."
  New-ClosureConsistencyCheck -Id "public-package-download-owner-execution-pack-present" -Passed ($null -ne $publicPackageDownloadOwnerExecutionPack) -Severity "action-required" -Detail "Owner public package download execution pack validation must be present before final closure review."
  New-ClosureConsistencyCheck -Id "public-package-download-owner-execution-pack-blocked" -Passed $publicPackageDownloadOwnerExecutionBlocked -Severity "action-required" -Detail "Owner execution pack must remain blocked until real owner public package download evidence is supplied; it is guidance only and not proof."
  New-ClosureConsistencyCheck -Id "public-package-download-owner-execution-pack-safe" -Passed $publicPackageDownloadOwnerExecutionSafe -Severity "blocker" -Detail "Owner execution pack must keep publish/token/proof/close flags false."
  New-ClosureConsistencyCheck -Id "owner-and-public-download-package-url-match" -Passed (-not [string]::IsNullOrWhiteSpace($ownerPublicPackageUrl) -and $ownerPublicPackageUrl.Equals($publicDownloadSourceOwnerPackageUrl, [StringComparison]::OrdinalIgnoreCase) -and $ownerPublicPackageUrl.Equals($publicDownloadManagedPackageUrl, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "Owner public package URL must match the public download candidate managed package page URL."
  New-ClosureConsistencyCheck -Id "owner-and-public-download-version-match" -Passed (-not [string]::IsNullOrWhiteSpace($ownerPublicPackageVersion) -and $ownerPublicPackageVersion.Equals($publicDownloadSourceOwnerPackageVersion, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "Owner public package version must match the public download candidate source version."
  New-ClosureConsistencyCheck -Id "owner-and-public-download-sha-match" -Passed ((Test-Sha256Format -Value $ownerPublicPackageSha256) -and $ownerPublicPackageSha256.Equals($publicDownloadSourceOwnerPackageSha256, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "Owner public package SHA256 must match the public download candidate source SHA256."
  New-ClosureConsistencyCheck -Id "runtime-package-url-public" -Passed ((Test-ReadyUrl -Value $ownerRuntimePackageUrl -Prefix "https://www.nuget.org/packages/") -and (Test-ReadyUrl -Value $publicDownloadRuntimePackageUrl -Prefix "https://www.nuget.org/packages/") -and (Test-ReadyUrl -Value $publicDownloadRuntimePackageDownloadUrl)) -Severity "action-required" -Detail "Runtime package page and download URLs must be public HTTPS URLs."
  New-ClosureConsistencyCheck -Id "github-release-asset-consistent" -Passed ((Test-ReadyUrl -Value $ownerGitHubReleaseAssetUrl -Prefix "https://github.com/") -and $ownerGitHubReleaseAssetUrl.Equals($publicDownloadGitHubReleaseAssetUrl, [StringComparison]::OrdinalIgnoreCase) -and (Test-Sha256Format -Value $ownerGitHubReleaseAssetSha256) -and $ownerGitHubReleaseAssetSha256.Equals($publicDownloadGitHubReleaseAssetSha256, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "GitHub release asset URL and SHA256 must match between Owner publish result and public download proof."
  New-ClosureConsistencyCheck -Id "owner-reviewer-and-timestamp-present" -Passed (-not [string]::IsNullOrWhiteSpace($ownerReviewer) -and -not [string]::IsNullOrWhiteSpace($ownerReviewTimestampUtc)) -Severity "action-required" -Detail "Owner reviewer and review timestamp must be present."
  New-ClosureConsistencyCheck -Id "post-publish-proof-candidate-ready" -Passed $postPublishProofReady -Severity "action-required" -Detail "Post-publish clean consumer proof result must have proofCandidateReady=true."
  New-ClosureConsistencyCheck -Id "post-publish-links-source-proofs" -Passed ($postPublishSourceProofLinkageReady -and $postPublishSourceGitHubActionsReady -and $postPublishSourceOwnerReady -and $postPublishSourcePublicDownloadReady) -Severity "action-required" -Detail "Post-publish proof must link to ready GitHub Actions, Owner public publish, and public package download proofs."
  New-ClosureConsistencyCheck -Id "post-publish-user-verification-pack-present" -Passed ($null -ne $postPublishUserVerificationPack) -Severity "action-required" -Detail "Post-publish user verification pack validation must be present before final closure review."
  New-ClosureConsistencyCheck -Id "post-publish-user-verification-pack-blocked" -Passed $postPublishUserVerificationBlocked -Severity "action-required" -Detail "Post-publish user verification pack must remain blocked until real public download, clean consumer, post-publish proof, and final audit lanes are ready."
  New-ClosureConsistencyCheck -Id "post-publish-user-verification-pack-safe" -Passed $postPublishUserVerificationSafe -Severity "blocker" -Detail "Post-publish user verification pack must keep publish/token/proof/close flags false."
  New-ClosureConsistencyCheck -Id "post-publish-owner-package-url-match" -Passed (-not [string]::IsNullOrWhiteSpace($postPublishPublicPackageUrl) -and $postPublishPublicPackageUrl.Equals($ownerPublicPackageUrl, [StringComparison]::OrdinalIgnoreCase) -and $postPublishPublicPackageUrl.Equals($postPublishSourceOwnerPackageUrl, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "Post-publish public package URL must match Owner public publish URL."
  New-ClosureConsistencyCheck -Id "post-publish-owner-package-version-match" -Passed (-not [string]::IsNullOrWhiteSpace($postPublishPublicPackageVersion) -and $postPublishPublicPackageVersion.Equals($ownerPublicPackageVersion, [StringComparison]::OrdinalIgnoreCase) -and $postPublishPublicPackageVersion.Equals($postPublishSourceOwnerPackageVersion, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "Post-publish package version must match Owner public publish version."
  New-ClosureConsistencyCheck -Id "post-publish-owner-package-sha-match" -Passed ((Test-Sha256Format -Value $postPublishPublicPackageSha256) -and $postPublishPublicPackageSha256.Equals($ownerPublicPackageSha256, [StringComparison]::OrdinalIgnoreCase) -and $postPublishPublicPackageSha256.Equals($postPublishSourceOwnerPackageSha256, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "Post-publish downloaded managed package SHA256 must match Owner/public download source SHA256."
  New-ClosureConsistencyCheck -Id "forbidden-substitutes-absent" -Passed ($forbiddenFindingCount -eq 0) -Severity "blocker" -Detail $(if ($forbiddenFindingCount -eq 0) { "No forbidden substitute findings were propagated from GitHub Actions, Owner publish, or public download proof lanes." } else { "Forbidden substitute findings were propagated: $($allForbiddenFindings -join ', ')" })
)

$blockedLanes = @($lanes | Where-Object { -not $_.ready })
$readyLanes = @($lanes | Where-Object { $_.ready })
$failedConsistencyBlockers = @($crossLaneConsistencyChecks | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedConsistencyActionRequired = @($crossLaneConsistencyChecks | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$allSafe = $true
foreach ($lane in $lanes) {
  $allSafe = $allSafe -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "performsPublish" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "usesPublishToken" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canPublishPublicly" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canCloseReleaseIssue" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "isReleaseCloseProof" -DefaultValue $true)
}

$bridgeState = if ($failedConsistencyBlockers.Count -gt 0) {
  "invalid-final-public-release-closure-bridge"
}
elseif ($blockedLanes.Count -eq 0 -and $failedConsistencyActionRequired.Count -eq 0) {
  "final-public-release-closure-bridge-ready-for-owner-close-review"
}
else {
  "blocked-final-public-release-closure-real-owner-proof-required"
}

$record = [pscustomobject]@{
  recordKind = "final-public-release-closure-bridge"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  bridgeState = $bridgeState
  laneCount = $lanes.Count
  readyLaneCount = $readyLanes.Count
  blockedLaneCount = $blockedLanes.Count
  missingArtifactCount = @($lanes | Where-Object { -not $_.artifactExists }).Count
  failedConsistencyBlockerCount = $failedConsistencyBlockers.Count
  failedConsistencyActionRequiredCount = $failedConsistencyActionRequired.Count
  forbiddenSubstituteFindingCount = $forbiddenFindingCount
  notExecutedByAutomation = $true
  performsPublish = $false
  usesPublishToken = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  allLanesSideEffectSafe = $allSafe
  closureLanes = @($lanes)
  crossLaneConsistencyChecks = @($crossLaneConsistencyChecks)
  closureProofSourceSummary = [pscustomobject]@{
    preReleaseReadinessMatrixState = $preReleaseMatrixState
    preReleaseReadinessMatrixReady = $preReleaseMatrixReady
    preReleaseCurrentHead = $preReleaseCurrentHead
    preReleaseSourceQualityRunId = $preReleaseSourceQualityRunId
    preReleaseReadyLaneCount = $preReleaseReadyLaneCount
    preReleaseBlockedLaneCount = $preReleaseBlockedLaneCount
    preReleaseLaneMetadataReady = $preReleaseLaneMetadataReady
    preReleasePromoteFlagsSafe = $preReleasePromoteFlagsSafe
    preReleaseMissingLaneIds = @($preReleaseMissingLaneIds)
    preReleaseLaneMetadataFindings = @($preReleaseLaneMetadataFindings.ToArray())
    preReleasePrematurePromoteFindings = @($preReleasePrematurePromoteFindings.ToArray())
    githubActionsRunEvidenceReady = $githubActionsReady
    githubActionsRunId = $githubActionsRunId
    githubActionsRunUrl = $githubActionsRunUrl
    githubActionsHeadSha = $githubActionsHeadSha
    githubActionsWorkflowRunLogSha256 = $githubActionsWorkflowRunLogSha256
    githubActionsArtifactManifestSha256 = $githubActionsArtifactManifestSha256
    ownerPublicPublishResultReady = $ownerPublicPublishReady
    ownerPublicPublishSourceGitHubActionsReady = $ownerPublicPublishSourceGitHubActionsReady
    ownerPublicPackageUrl = $ownerPublicPackageUrl
    ownerPublicPackageVersion = $ownerPublicPackageVersion
    ownerPublicPackageSha256 = $ownerPublicPackageSha256
    ownerManagedPackageUrl = $ownerManagedPackageUrl
    ownerRuntimePackageUrl = $ownerRuntimePackageUrl
    ownerGitHubReleaseUrl = $ownerGitHubReleaseUrl
    ownerGitHubReleaseAssetUrl = $ownerGitHubReleaseAssetUrl
    ownerGitHubReleaseAssetSha256 = $ownerGitHubReleaseAssetSha256
    ownerReviewer = $ownerReviewer
    ownerReviewTimestampUtc = $ownerReviewTimestampUtc
    publicDownloadProofReady = $publicDownloadReady
    publicDownloadSourceGitHubActionsReady = $publicDownloadSourceGitHubActionsReady
    publicDownloadSourceOwnerPublicPublishResultReady = $publicDownloadSourceOwnerReady
    publicDownloadManagedPackageUrl = $publicDownloadManagedPackageUrl
    publicDownloadManagedPackageDownloadUrl = $publicDownloadManagedPackageDownloadUrl
    publicDownloadRuntimePackageUrl = $publicDownloadRuntimePackageUrl
    publicDownloadRuntimePackageDownloadUrl = $publicDownloadRuntimePackageDownloadUrl
    publicDownloadSourceOwnerPackageUrl = $publicDownloadSourceOwnerPackageUrl
    publicDownloadSourceOwnerPackageVersion = $publicDownloadSourceOwnerPackageVersion
    publicDownloadSourceOwnerPackageSha256 = $publicDownloadSourceOwnerPackageSha256
    publicDownloadGitHubReleaseUrl = $publicDownloadGitHubReleaseUrl
    publicDownloadGitHubReleaseAssetUrl = $publicDownloadGitHubReleaseAssetUrl
    publicDownloadGitHubReleaseAssetSha256 = $publicDownloadGitHubReleaseAssetSha256
    publicPackageDownloadOwnerExecutionPackState = $publicPackageDownloadOwnerExecutionState
    publicPackageDownloadOwnerExecutionPackBlocked = $publicPackageDownloadOwnerExecutionBlocked
    publicPackageDownloadOwnerExecutionPackSafe = $publicPackageDownloadOwnerExecutionSafe
    postPublishProofReady = $postPublishProofReady
    postPublishSourceProofLinkageReady = $postPublishSourceProofLinkageReady
    postPublishSourceGitHubActionsReady = $postPublishSourceGitHubActionsReady
    postPublishSourceOwnerReady = $postPublishSourceOwnerReady
    postPublishSourcePublicDownloadReady = $postPublishSourcePublicDownloadReady
    postPublishPublicPackageUrl = $postPublishPublicPackageUrl
    postPublishPublicPackageVersion = $postPublishPublicPackageVersion
    postPublishPublicPackageSha256 = $postPublishPublicPackageSha256
    postPublishSourceOwnerPackageUrl = $postPublishSourceOwnerPackageUrl
    postPublishSourceOwnerPackageVersion = $postPublishSourceOwnerPackageVersion
    postPublishSourceOwnerPackageSha256 = $postPublishSourceOwnerPackageSha256
    postPublishUserVerificationPackState = $postPublishUserVerificationState
    postPublishUserVerificationPackBlocked = $postPublishUserVerificationBlocked
    postPublishUserVerificationPackSafe = $postPublishUserVerificationSafe
  }
  sourceArtifacts = @($lanes | ForEach-Object { $_.artifact })
  nextOwnerActions = @($blockedLanes | ForEach-Object { [pscustomobject]@{ laneId = $_.laneId; state = $_.state; ownerAction = $_.ownerAction; requiredBeforeClose = $_.requiredBeforeClose } })
  safetyBoundary = "Final public release closure bridge only joins owner authorization, owner publish execution result, public package download proof, public package download owner guidance, clean external consumer smoke, post-publish proof, post-publish user verification, release issue close owner decision, and strict close dashboard. It does not publish packages, use tokens, claim runtime proof, claim post-publish proof, or close the release issue."
}

$jsonPath = Join-Path $OutputRoot "final-public-release-closure-bridge.json"
$markdownPath = Join-Path $OutputRoot "final-public-release-closure-bridge.md"
$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$laneRows = $record.closureLanes | ForEach-Object {
  "| ``$($_.laneId)`` | ``$($_.state)`` | ``$($_.ready)`` | $($_.ownerAction.Replace("|", "\|")) |"
}

$markdown = @"
# Final Public Release Closure Bridge

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| bridgeState | ``$($record.bridgeState)`` |
| laneCount | ``$($record.laneCount)`` |
| readyLaneCount | ``$($record.readyLaneCount)`` |
| blockedLaneCount | ``$($record.blockedLaneCount)`` |
| missingArtifactCount | ``$($record.missingArtifactCount)`` |
| failedConsistencyBlockerCount | ``$($record.failedConsistencyBlockerCount)`` |
| failedConsistencyActionRequiredCount | ``$($record.failedConsistencyActionRequiredCount)`` |
| forbiddenSubstituteFindingCount | ``$($record.forbiddenSubstituteFindingCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| usesPublishToken | ``$($record.usesPublishToken)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Closure Lanes

| Lane | State | Ready | Owner Action |
|---|---:|---:|---|
$($laneRows -join "`r`n")

## Cross-Lane Consistency

| Check | Passed | Severity | Detail |
|---|---:|---|---|
$(($record.crossLaneConsistencyChecks | ForEach-Object { "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |" }) -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final public release closure bridge written to $jsonPath"
Write-Host "BridgeState=$($record.bridgeState) Ready=$($record.readyLaneCount) Blocked=$($record.blockedLaneCount)"
