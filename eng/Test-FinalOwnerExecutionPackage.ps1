[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-package.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final owner execution package not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$packageState = [string](Get-PropertyOrDefault -Object $record -Name "packageState" -DefaultValue "")
$steps = @((Get-PropertyOrDefault -Object $record -Name "executionSteps" -DefaultValue @()))
$actionIds = Convert-ToStringArray ($steps | ForEach-Object { Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "" })
$actionRequiredIds = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "actionRequiredIds" -DefaultValue @())
$forbiddenSubstitutes = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())
$validatorCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "validatorCommands" -DefaultValue @())
$ownerCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "ownerCommandSequence" -DefaultValue @())
$expectedResultArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "expectedResultArtifacts" -DefaultValue @())
$sourceArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())
$sourceArtifactEvidence = @((Get-PropertyOrDefault -Object $record -Name "sourceArtifactEvidence" -DefaultValue @()))
$sourceArtifactEvidenceCount = [int](Get-PropertyOrDefault -Object $record -Name "sourceArtifactEvidenceCount" -DefaultValue 0)
$sourceArtifactEvidenceMissingCount = [int](Get-PropertyOrDefault -Object $record -Name "sourceArtifactEvidenceMissingCount" -DefaultValue -1)
$sourceArtifactEvidenceSha256Count = [int](Get-PropertyOrDefault -Object $record -Name "sourceArtifactEvidenceSha256Count" -DefaultValue 0)
$sourceArtifactEvidenceNonProofBoundaryCount = [int](Get-PropertyOrDefault -Object $record -Name "sourceArtifactEvidenceNonProofBoundaryCount" -DefaultValue 0)
$sourceArtifactEvidenceIds = Convert-ToStringArray ($sourceArtifactEvidence | ForEach-Object { Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "" })
$releaseCloseRealInputChain = @((Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChain" -DefaultValue @()))
$ownerReleaseCloseHardGates = @((Get-PropertyOrDefault -Object $record -Name "ownerReleaseCloseHardGates" -DefaultValue @()))
$releaseCloseRealInputChainRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainRequiredFieldCount" -DefaultValue 0)
$releaseCloseRealInputChainRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainRejectedSubstituteCount" -DefaultValue 0)
$releaseCloseRealInputChainSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainSourceReadinessSignalCount" -DefaultValue 0)
$releaseCloseRealInputChainBlockedRealInputCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainBlockedRealInputCount" -DefaultValue 0)
$hardGateIds = Convert-ToStringArray ($ownerReleaseCloseHardGates | ForEach-Object { Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "" })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "final-owner-execution-package") -Severity "blocker" -Detail "recordKind must be final-owner-execution-package.")) | Out-Null
$items.Add((New-ValidationItem -Id "package-state" -Passed ($packageState -eq "blocked-final-owner-execution-required") -Severity "blocker" -Detail "Package must remain blocked until owner provides real proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "does-not-publish-or-promote" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPackagePush" -DefaultValue $true))) -Severity "blocker" -Detail "Package must not publish, promote proof, push packages, or close release.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-runtime-smoke-field-alignment-projected" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke-field-alignment" -and [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke-field-alignment-valid" -and [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus" -DefaultValue "") -eq "Smoke=not-requested" -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount" -DefaultValue 0) -ge 30 -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount" -DefaultValue -1) -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount" -DefaultValue -1) -eq 0 -and $sourceArtifacts -contains "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json" -and $sourceArtifacts -contains "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json") -Severity "blocker" -Detail "Final owner execution package must project owner runtime smoke field alignment as zero-missing non-proof coverage.")) | Out-Null
$items.Add((New-ValidationItem -Id "eight-execution-steps" -Passed ($steps.Count -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "executionStepCount" -DefaultValue 0) -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedExecutionStepCount" -DefaultValue 0) -eq 8) -Severity "blocker" -Detail "Package must expose exactly eight blocked execution steps: two owner runbook preflight items plus six final owner actions.")) | Out-Null
$items.Add((New-ValidationItem -Id "release-close-real-input-chain-projected" -Passed ($releaseCloseRealInputChain.Count -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainCount" -DefaultValue 0) -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedReleaseCloseRealInputChainCount" -DefaultValue 0) -eq 8 -and $releaseCloseRealInputChainRequiredFieldCount -ge 100 -and $releaseCloseRealInputChainRejectedSubstituteCount -ge 30 -and $releaseCloseRealInputChainSourceReadinessSignalCount -ge 18 -and $releaseCloseRealInputChainBlockedRealInputCount -gt 0) -Severity "blocker" -Detail "Package must project the eight-step release-close real input chain and its field/substitute/source-linkage counts.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-release-close-hard-gates" -Passed ($ownerReleaseCloseHardGates.Count -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "ownerReleaseCloseHardGateCount" -DefaultValue 0) -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedOwnerReleaseCloseHardGateCount" -DefaultValue 0) -eq 8) -Severity "blocker" -Detail "Package must expose the eight blocked owner release-close hard gates.")) | Out-Null
$items.Add((New-ValidationItem -Id "hard-gate-readiness-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofCandidateReady" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "postPublishProofCandidateReady" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "postPublishProofSourceLinkageReady" -DefaultValue $true)) -and [bool](Get-PropertyOrDefault -Object $record -Name "publicDownloadCannotSubstitutePostPublishProof" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "postPublishValidationReadyCannotSubstituteProofCandidateReady" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "bundleHashCannotSubstituteFinalCloseDecision" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "strictCloseOutputCannotCloseIssue" -DefaultValue $false)) -Severity "blocker" -Detail "Public download, validation-ready, bundle hash, and strict validator output must remain non-substitutes for real post-publish/final-close proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "hard-gate-source-states" -Passed (-not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $record -Name "releaseEvidenceBundleSha256" -DefaultValue "")) -and [string](Get-PropertyOrDefault -Object $record -Name "finalCloseStrictValidatorOutputState" -DefaultValue "") -eq "blocked-final-close-gate-owner-proof-required" -and [string](Get-PropertyOrDefault -Object $record -Name "releaseIssueCloseOwnerDecisionValidationState" -DefaultValue "") -eq "blocked-release-issue-close-owner-decision-input-required" -and [int](Get-PropertyOrDefault -Object $record -Name "finalCloseProofAdmissionRequiredFieldCount" -DefaultValue 0) -ge 20 -and [int](Get-PropertyOrDefault -Object $record -Name "finalCloseRejectedNonProofStateCount" -DefaultValue 0) -ge 10) -Severity "blocker" -Detail "Package must carry bundle SHA, strict close output state, final close decision state, and final close admission contract counts without approving close.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifact-evidence-counts" -Passed ($sourceArtifactEvidence.Count -ge 14 -and $sourceArtifactEvidenceCount -eq $sourceArtifactEvidence.Count -and $sourceArtifactEvidenceMissingCount -eq 0 -and $sourceArtifactEvidenceSha256Count -eq $sourceArtifactEvidence.Count -and $sourceArtifactEvidenceNonProofBoundaryCount -eq $sourceArtifactEvidence.Count) -Severity "blocker" -Detail "Source artifact evidence must cover all key source artifacts with existing files, SHA256 hashes, and non-proof boundaries.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifact-evidence-shape" -Passed (@($sourceArtifactEvidence | Where-Object {
        [bool](Get-PropertyOrDefault -Object $_ -Name "exists" -DefaultValue $false) -and
        [System.Text.RegularExpressions.Regex]::IsMatch([string](Get-PropertyOrDefault -Object $_ -Name "sha256" -DefaultValue ""), "^[0-9a-f]{64}$") -and
        -not [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -and
        -not [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
        -not [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -and
        -not [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) -and
        -not [bool](Get-PropertyOrDefault -Object $_ -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
        -not [bool](Get-PropertyOrDefault -Object $_ -Name "isPostPublishProof" -DefaultValue $true) -and
        -not [bool](Get-PropertyOrDefault -Object $_ -Name "isReleaseCloseProof" -DefaultValue $true) -and
        ([string](Get-PropertyOrDefault -Object $_ -Name "boundary" -DefaultValue "")).Contains("cannot substitute real Owner proof", [StringComparison]::OrdinalIgnoreCase) -and
        ([string](Get-PropertyOrDefault -Object $_ -Name "boundary" -DefaultValue "")).Contains("post-publish CleanConsumer proof", [StringComparison]::OrdinalIgnoreCase) -and
        ([string](Get-PropertyOrDefault -Object $_ -Name "boundary" -DefaultValue "")).Contains("final close decision", [StringComparison]::OrdinalIgnoreCase)
      }).Count -eq $sourceArtifactEvidence.Count) -Severity "blocker" -Detail "Every source artifact evidence item must remain hash-only traceability and not proof, publish, or close authority.")) | Out-Null

foreach ($expectedGate in @(
  "github-actions-run-evidence",
  "owner-public-publish-result",
  "public-package-download-proof",
  "post-publish-clean-consumer-proof-result",
  "rollback-review",
  "final-close-decision",
  "release-evidence-bundle-sha",
  "strict-close-validator-output"
)) {
  $items.Add((New-ValidationItem -Id "hard-gate-$expectedGate-present" -Passed ($hardGateIds -contains $expectedGate) -Severity "blocker" -Detail "Hard gate $expectedGate must be present.")) | Out-Null
}

foreach ($expected in @(
  "00-clean-external-package-consumer-owner-runbook",
  "00-post-publish-owner-verification-runbook",
  "01-real-model-runtime-owner-evidence",
  "02-package-consumer-runtime-clean-external-proof",
  "03-post-publish-verification-public-channel",
  "04-final-owner-real-input-template-pack",
  "05-owner-external-result-import-real-files",
  "06-owner-result-candidate-bridge-strict-promotion"
)) {
  $items.Add((New-ValidationItem -Id "action-$expected-present" -Passed ($actionIds -contains $expected) -Severity "blocker" -Detail "Action $expected must be present.")) | Out-Null
}

foreach ($expected in @(
  "clean-external-package-consumer-owner-runbook-required",
  "post-publish-owner-verification-runbook-required",
  "real-model-runtime-owner-proof-required",
  "package-consumer-runtime-owner-proof-required",
  "post-publish-verification-owner-proof-required",
  "final-owner-real-input-template-pack-owner-input-required",
  "owner-external-proof-result-import-owner-proof-required",
  "owner-result-candidate-bridge-real-proof-required"
)) {
  $items.Add((New-ValidationItem -Id "action-required-$expected-present" -Passed ($actionRequiredIds -contains $expected) -Severity "blocker" -Detail "Action-required id $expected must be present.")) | Out-Null
}

foreach ($step in $steps) {
  $id = [string](Get-PropertyOrDefault -Object $step -Name "id" -DefaultValue "")
  $inputContract = Get-PropertyOrDefault -Object $step -Name "inputContract" -DefaultValue $null
  $stepOwnerCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $step -Name "ownerCommands" -DefaultValue @())
  $stepValidatorCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $step -Name "validatorCommands" -DefaultValue @())
  $stepExpectedArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $step -Name "expectedResultArtifacts" -DefaultValue @())
  $stepForbiddenSubstitutes = Convert-ToStringArray (Get-PropertyOrDefault -Object $step -Name "forbiddenSubstitutes" -DefaultValue @())
  $stepBoundary = [string](Get-PropertyOrDefault -Object $step -Name "boundary" -DefaultValue "")
  $promotionBoundary = [string](Get-PropertyOrDefault -Object $step -Name "promotionBoundary" -DefaultValue "")
  $sourceArtifact = [string](Get-PropertyOrDefault -Object $inputContract -Name "sourceArtifact" -DefaultValue "")

  $nonPromoting = (-not [bool](Get-PropertyOrDefault -Object $step -Name "performsPublish" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $step -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $step -Name "canPublishPublicly" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $step -Name "canCloseReleaseIssue" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $step -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $step -Name "isPostPublishProof" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $step -Name "isReleaseCloseProof" -DefaultValue $true))

  $items.Add((New-ValidationItem -Id "step-$id-command-contract" -Passed (@($stepOwnerCommands).Count -ge 2 -and @($stepValidatorCommands).Count -ge 1 -and @($stepExpectedArtifacts).Count -ge 2 -and @($stepForbiddenSubstitutes).Count -ge 5 -and -not [string]::IsNullOrWhiteSpace($sourceArtifact)) -Severity "blocker" -Detail "Each step must expose owner commands, validators, result artifacts, forbidden substitutes, and an input contract.")) | Out-Null
  $items.Add((New-ValidationItem -Id "step-$id-non-promoting" -Passed $nonPromoting -Severity "blocker" -Detail "Step $id must not publish, close, or promote proof.")) | Out-Null
  $items.Add((New-ValidationItem -Id "step-$id-boundary" -Passed ($stepBoundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $stepBoundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $stepBoundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase) -and -not [string]::IsNullOrWhiteSpace($promotionBoundary)) -Severity "blocker" -Detail "Step $id must preserve non-proof and promotion boundaries.")) | Out-Null
}

foreach ($gate in $ownerReleaseCloseHardGates) {
  $id = [string](Get-PropertyOrDefault -Object $gate -Name "id" -DefaultValue "")
  $boundary = [string](Get-PropertyOrDefault -Object $gate -Name "boundary" -DefaultValue "")
  $nonPromotingGate = [bool](Get-PropertyOrDefault -Object $gate -Name "blocked" -DefaultValue $false) -and
    [bool](Get-PropertyOrDefault -Object $gate -Name "ownerActionRequired" -DefaultValue $false) -and
    (-not [bool](Get-PropertyOrDefault -Object $gate -Name "performsPublish" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $gate -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $gate -Name "canPublishPublicly" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $gate -Name "canCloseReleaseIssue" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $gate -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $gate -Name "isPostPublishProof" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $gate -Name "isReleaseCloseProof" -DefaultValue $true))

  $items.Add((New-ValidationItem -Id "hard-gate-$id-non-promoting" -Passed ($nonPromotingGate -and -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $gate -Name "sourceArtifact" -DefaultValue "")) -and -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $gate -Name "strictValidator" -DefaultValue ""))) -Severity "blocker" -Detail "Hard gate $id must remain blocked, owner-action-required, non-promoting, and tied to a source artifact plus validator.")) | Out-Null
  $items.Add((New-ValidationItem -Id "hard-gate-$id-boundary" -Passed ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Hard gate $id must preserve proof, close, and publish exclusions.")) | Out-Null
}

foreach ($marker in @("local feed", "ProjectReference", "direct .nupkg", "build-only", "dry-run", "template", "candidate", "dashboard", "blocked-by-cuda-driver", "repository-external", "public package source URL", "stdoutPath", "stderrPath", "mergedTranscriptPath", "nonSubstituteConfirmations", "does not run dotnet nuget push", "public package download proof alone", "post-publish validation-ready without proofCandidateReady", "release evidence bundle hash only", "strict close validator output without real proof")) {
  $items.Add((New-ValidationItem -Id "forbidden-substitute-$($marker.Replace(' ', '-').Replace('.', 'dot'))-visible" -Passed (($forbiddenSubstitutes -join "`n").Contains($marker, [StringComparison]::OrdinalIgnoreCase) -or $raw.Contains($marker, [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Forbidden substitute marker '$marker' must remain visible.")) | Out-Null
}

foreach ($needle in @("Test-CleanExternalPackageConsumerOwnerRunbook.ps1", "Test-PostPublishOwnerVerificationRunbook.ps1", "Test-SampleRunEvidenceRecord.ps1", "Test-PackageConsumerRuntimeProofRecord.ps1", "Test-PostPublishVerificationRecord.ps1", "Test-OwnerExternalProofExecutionResultImport.ps1", "Test-RealProofRecordCandidateFromOwnerResultImport.ps1")) {
  $items.Add((New-ValidationItem -Id "validator-$($needle.Replace('.ps1',''))-present" -Passed (($validatorCommands -join "`n").Contains($needle, [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Validator $needle must be included.")) | Out-Null
}

foreach ($needle in @("clean-external-package-consumer-owner-runbook-validation.json", "post-publish-owner-verification-runbook-validation.json", "owner-external-proof-execution-result.input.json", "sample-run-evidence-record-validation.json", "package-consumer-runtime-proof-record-validation.json", "post-publish-verification-validation.json", "final-owner-real-input-template-pack-validation.json", "owner-external-proof-execution-result-import-validation.json", "real-proof-record-candidate-from-owner-result-import-validation.json")) {
  $items.Add((New-ValidationItem -Id "result-artifact-$($needle.Replace('.json',''))-present" -Passed (($expectedResultArtifacts -join "`n").Contains($needle, [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Expected result artifact $needle must be listed.")) | Out-Null
}

foreach ($needle in @(
  "artifacts/final-release/final-owner-execution-one-screen-pack.json",
  "artifacts/final-release/final-owner-execution-one-screen-pack-validation.json",
  "artifacts/final-release/final-owner-strict-close-execution-order-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json",
  "artifacts/final-release/public-package-download-proof-candidate-validation.json",
  "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json",
  "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
  "artifacts/final-release/final-close-gate-convergence-validation.json"
)) {
  $items.Add((New-ValidationItem -Id "source-artifact-$($needle.Replace('artifacts/final-release/','').Replace('.json',''))-present" -Passed ($sourceArtifacts -contains $needle) -Severity "blocker" -Detail "Source artifact $needle must be listed.")) | Out-Null
}

foreach ($expectedEvidenceId in @(
  "final-owner-proof-action-worklist",
  "final-owner-proof-action-worklist-validation",
  "final-owner-execution-one-screen-pack",
  "final-owner-execution-one-screen-pack-validation",
  "final-owner-strict-close-execution-order-validation",
  "owner-public-publish-execution-result-candidate-validation",
  "public-package-download-proof-candidate-validation",
  "post-publish-clean-consumer-proof-result-validation",
  "release-issue-close-owner-decision-input-validation",
  "final-close-gate-convergence-validation",
  "release-evidence-bundle",
  "release-publish-readiness-evidence-pack",
  "package-consumer-owner-runtime-smoke-field-alignment",
  "package-consumer-owner-runtime-smoke-field-alignment-validation"
)) {
  $items.Add((New-ValidationItem -Id "source-artifact-evidence-$expectedEvidenceId-present" -Passed ($sourceArtifactEvidenceIds -contains $expectedEvidenceId) -Severity "blocker" -Detail "Source artifact evidence $expectedEvidenceId must be present.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-final-owner-execution-package" } else { "blocked-final-owner-execution-required" }
$blockedExecutionStepCount = @($steps | Where-Object {
    [bool](Get-PropertyOrDefault -Object $_ -Name "blocked" -DefaultValue $false)
  }).Count
$validationItems = @($items.ToArray())

$validation = [pscustomobject]@{
  recordKind = "final-owner-execution-package-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  inputPath = $resolvedInputPath
  validationState = $validationState
  actionCount = $actionIds.Count
  executionStepCount = $steps.Count
  blockedExecutionStepCount = $blockedExecutionStepCount
  ownerCommandCount = $ownerCommands.Count
  validatorCommandCount = $validatorCommands.Count
  expectedResultArtifactCount = $expectedResultArtifacts.Count
  packageConsumerOwnerRuntimeSmokeFieldAlignmentState = [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentState" -DefaultValue "")
  packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState = [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState" -DefaultValue "")
  packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount" -DefaultValue -1)
  releaseCloseRealInputChainCount = $releaseCloseRealInputChain.Count
  releaseCloseRealInputChainRequiredFieldCount = $releaseCloseRealInputChainRequiredFieldCount
  releaseCloseRealInputChainRejectedSubstituteCount = $releaseCloseRealInputChainRejectedSubstituteCount
  releaseCloseRealInputChainSourceReadinessSignalCount = $releaseCloseRealInputChainSourceReadinessSignalCount
  releaseCloseRealInputChainBlockedRealInputCount = $releaseCloseRealInputChainBlockedRealInputCount
  ownerReleaseCloseHardGateCount = $ownerReleaseCloseHardGates.Count
  blockedOwnerReleaseCloseHardGateCount = @($ownerReleaseCloseHardGates | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "blocked" -DefaultValue $false) }).Count
  publicPackageDownloadProofCandidateReady = [bool](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofCandidateReady" -DefaultValue $true)
  postPublishProofCandidateReady = [bool](Get-PropertyOrDefault -Object $record -Name "postPublishProofCandidateReady" -DefaultValue $true)
  postPublishProofSourceLinkageReady = [bool](Get-PropertyOrDefault -Object $record -Name "postPublishProofSourceLinkageReady" -DefaultValue $true)
  releaseEvidenceBundleSha256 = [string](Get-PropertyOrDefault -Object $record -Name "releaseEvidenceBundleSha256" -DefaultValue "")
  releaseIssueCloseOwnerDecisionValidationState = [string](Get-PropertyOrDefault -Object $record -Name "releaseIssueCloseOwnerDecisionValidationState" -DefaultValue "")
  finalCloseStrictValidatorOutputState = [string](Get-PropertyOrDefault -Object $record -Name "finalCloseStrictValidatorOutputState" -DefaultValue "")
  sourceArtifactEvidenceCount = $sourceArtifactEvidenceCount
  sourceArtifactEvidenceMissingCount = $sourceArtifactEvidenceMissingCount
  sourceArtifactEvidenceSha256Count = $sourceArtifactEvidenceSha256Count
  sourceArtifactEvidenceNonProofBoundaryCount = $sourceArtifactEvidenceNonProofBoundaryCount
  publicDownloadCannotSubstitutePostPublishProof = [bool](Get-PropertyOrDefault -Object $record -Name "publicDownloadCannotSubstitutePostPublishProof" -DefaultValue $false)
  bundleHashCannotSubstituteFinalCloseDecision = [bool](Get-PropertyOrDefault -Object $record -Name "bundleHashCannotSubstituteFinalCloseDecision" -DefaultValue $false)
  strictCloseOutputCannotCloseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "strictCloseOutputCannotCloseIssue" -DefaultValue $false)
  failedBlockerCount = $failedBlockers.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = $validationItems
  boundary = "Final owner execution package validation checks handoff shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-package-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-package-validation.md"
$validation | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $items) {
  "| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $($item.detail.Replace('|', '\|')) |"
}

$markdown = @"
# Final Owner Execution Package Validation

Generated at: ``$($validation.generatedAtUtc)``

## Summary

- recordKind: ``$($validation.recordKind)``
- validationState: ``$($validation.validationState)``
- executionStepCount: ``$($validation.executionStepCount)``
- blockedExecutionStepCount: ``$($validation.blockedExecutionStepCount)``
- ownerCommandCount: ``$($validation.ownerCommandCount)``
- validatorCommandCount: ``$($validation.validatorCommandCount)``
- expectedResultArtifactCount: ``$($validation.expectedResultArtifactCount)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentState: ``$($validation.packageConsumerOwnerRuntimeSmokeFieldAlignmentState)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState: ``$($validation.packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount: ``$($validation.packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount)``
- releaseCloseRealInputChainCount: ``$($validation.releaseCloseRealInputChainCount)``
- releaseCloseRealInputChainRequiredFieldCount: ``$($validation.releaseCloseRealInputChainRequiredFieldCount)``
- releaseCloseRealInputChainRejectedSubstituteCount: ``$($validation.releaseCloseRealInputChainRejectedSubstituteCount)``
- releaseCloseRealInputChainSourceReadinessSignalCount: ``$($validation.releaseCloseRealInputChainSourceReadinessSignalCount)``
- releaseCloseRealInputChainBlockedRealInputCount: ``$($validation.releaseCloseRealInputChainBlockedRealInputCount)``
- ownerReleaseCloseHardGateCount: ``$($validation.ownerReleaseCloseHardGateCount)``
- blockedOwnerReleaseCloseHardGateCount: ``$($validation.blockedOwnerReleaseCloseHardGateCount)``
- publicPackageDownloadProofCandidateReady: ``$($validation.publicPackageDownloadProofCandidateReady)``
- postPublishProofCandidateReady: ``$($validation.postPublishProofCandidateReady)``
- postPublishProofSourceLinkageReady: ``$($validation.postPublishProofSourceLinkageReady)``
- releaseEvidenceBundleSha256: ``$($validation.releaseEvidenceBundleSha256)``
- releaseIssueCloseOwnerDecisionValidationState: ``$($validation.releaseIssueCloseOwnerDecisionValidationState)``
- finalCloseStrictValidatorOutputState: ``$($validation.finalCloseStrictValidatorOutputState)``
- sourceArtifactEvidenceCount: ``$($validation.sourceArtifactEvidenceCount)``
- sourceArtifactEvidenceMissingCount: ``$($validation.sourceArtifactEvidenceMissingCount)``
- sourceArtifactEvidenceSha256Count: ``$($validation.sourceArtifactEvidenceSha256Count)``
- sourceArtifactEvidenceNonProofBoundaryCount: ``$($validation.sourceArtifactEvidenceNonProofBoundaryCount)``
- publicDownloadCannotSubstitutePostPublishProof: ``$($validation.publicDownloadCannotSubstitutePostPublishProof)``
- bundleHashCannotSubstituteFinalCloseDecision: ``$($validation.bundleHashCannotSubstituteFinalCloseDecision)``
- strictCloseOutputCannotCloseIssue: ``$($validation.strictCloseOutputCannotCloseIssue)``
- failedBlockerCount: ``$($validation.failedBlockerCount)``
- performsPublish: ``False``
- canPromoteRuntimeProof: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Validation Items

| ID | Passed | Severity | Detail |
| --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Final owner execution package validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$validationState ExecutionSteps=$($validation.executionStepCount) FailedBlockers=$($validation.failedBlockerCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Final owner execution package validation failed with $($failedBlockers.Count) blocker(s)."
}
