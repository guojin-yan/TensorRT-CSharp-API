[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-strict-close-execution-order.json",
  [string]$RepositoryRoot,
  [string]$OutputRoot,
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

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

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

function ConvertTo-Array {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-StringArray {
  param([AllowNull()][object]$Value)
  return @(ConvertTo-Array $Value | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
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
  throw "Final Owner StrictClose execution order not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]

$steps = @((Get-PropertyOrDefault -Object $record -Name "executionSteps" -DefaultValue @()))
$sourceRecords = @((Get-PropertyOrDefault -Object $record -Name "sourceRecords" -DefaultValue @()))
$sourceArtifacts = ConvertTo-StringArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())
$stepIds = @($steps | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$sourceIds = @($sourceRecords | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-strict-close-execution-order") -Severity "blocker" -Detail "recordKind must be final-owner-strict-close-execution-order.")) | Out-Null
$items.Add((New-ValidationItem -Id "order-state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "orderState" -DefaultValue "") -eq "blocked-final-owner-strict-close-owner-execution-required") -Severity "blocker" -Detail "Order must remain blocked until real owner proof and strict close validators pass.")) | Out-Null
$items.Add((New-ValidationItem -Id "step-counts" -Passed ($steps.Count -eq 7 -and [int](Get-PropertyOrDefault -Object $record -Name "stepCount" -DefaultValue 0) -eq 7 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedStepCount" -DefaultValue 0) -eq 7) -Severity "blocker" -Detail "Execution order must expose seven blocked macro steps.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-counts" -Passed ($sourceRecords.Count -ge 11 -and [int](Get-PropertyOrDefault -Object $record -Name "sourceRecordCount" -DefaultValue 0) -ge 11) -Severity "blocker" -Detail "Execution order must summarize all core owner execution source records plus the final one-screen pack evidence chain.")) | Out-Null

$countExpectations = @{
  sourceActionCount = 7
  sourceExecutionStepCount = 7
  cleanExternalRunbookStepCount = 9
  postPublishRunbookStepCount = 6
  publicPublishExecutionLaneCount = 10
  publicPublishCrossCheckCount = 11
  finalOwnerCloseReadinessCheckCount = 12
  finalReleaseCloseBlockerCount = 19
  ownerInputContractSurfaceCount = 5
  ownerInputContractCanonicalFieldCount = 14
  ownerInputContractRunbookInputCount = 2
}

foreach ($entry in $countExpectations.GetEnumerator()) {
  $value = [int](Get-PropertyOrDefault -Object $record -Name $entry.Key -DefaultValue 0)
  $items.Add((New-ValidationItem -Id "count-$($entry.Key)" -Passed ($value -ge [int]$entry.Value) -Severity "blocker" -Detail "$($entry.Key) must be at least $($entry.Value).")) | Out-Null
}

foreach ($expected in @(
  "01-contract-convergence-preflight",
  "02-clean-external-consumer-prepublish-proof",
  "03-public-publish-manual-owner-command",
  "04-post-publish-clean-consumer-proof",
  "05-owner-result-import-and-strict-validator",
  "06-final-readiness-and-blocker-dashboard",
  "07-release-issue-close-owner-decision"
)) {
  $items.Add((New-ValidationItem -Id "step-$expected-present" -Passed ($stepIds -contains $expected) -Severity "blocker" -Detail "Step $expected must be present.")) | Out-Null
}

foreach ($expected in @(
  "final-owner-proof-action-worklist",
  "final-owner-execution-package",
  "clean-external-package-consumer-owner-runbook",
  "post-publish-owner-verification-runbook",
  "public-publish-final-owner-execution-pack",
  "public-publish-command-cross-check",
  "final-owner-close-readiness-checkpoint",
  "final-release-close-blocker-dashboard",
  "owner-input-contract-convergence",
  "final-owner-execution-one-screen-pack",
  "final-owner-execution-one-screen-pack-validation"
)) {
  $items.Add((New-ValidationItem -Id "source-$expected-present" -Passed ($sourceIds -contains $expected) -Severity "blocker" -Detail "Source summary $expected must be present.")) | Out-Null
}

$releaseCloseRealInputChain = @(ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChain" -DefaultValue @()))
$releaseCloseRealInputChainCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainCount" -DefaultValue 0)
$releaseCloseRealInputChainRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainRequiredFieldCount" -DefaultValue 0)
$releaseCloseRealInputChainRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainRejectedSubstituteCount" -DefaultValue 0)
$releaseCloseRealInputChainSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainSourceReadinessSignalCount" -DefaultValue 0)
$releaseCloseRealInputChainBlockedRealInputCount = [int](Get-PropertyOrDefault -Object $record -Name "releaseCloseRealInputChainBlockedRealInputCount" -DefaultValue 0)
$publicPackageDownloadProofRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofRequiredFieldCount" -DefaultValue 0)
$publicPackageDownloadProofRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofRejectedSubstituteCount" -DefaultValue 0)
$publicPackageDownloadProofSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofSourceReadinessSignalCount" -DefaultValue 0)
$publicPackageDownloadProofCandidateReady = [bool](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofCandidateReady" -DefaultValue $true)
$postPublishCleanConsumerProofRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofRequiredFieldCount" -DefaultValue 0)
$postPublishCleanConsumerProofRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofRejectedSubstituteCount" -DefaultValue 0)
$postPublishCleanConsumerProofBlockedRealInputCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofBlockedRealInputCount" -DefaultValue 0)
$postPublishCleanConsumerProofSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofSourceReadinessSignalCount" -DefaultValue 0)
$postPublishCleanConsumerProofCandidateReady = [bool](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofCandidateReady" -DefaultValue $true)
$postPublishCleanConsumerProofSourceProofLinkageReady = [bool](Get-PropertyOrDefault -Object $record -Name "postPublishCleanConsumerProofSourceProofLinkageReady" -DefaultValue $true)
$releaseEvidenceBundleSha256 = [string](Get-PropertyOrDefault -Object $record -Name "releaseEvidenceBundleSha256" -DefaultValue "")
$finalCloseStrictValidatorOutputState = [string](Get-PropertyOrDefault -Object $record -Name "finalCloseStrictValidatorOutputState" -DefaultValue "")

$items.Add((New-ValidationItem -Id "release-close-real-input-chain-count" -Passed ($releaseCloseRealInputChainCount -eq 8 -and $releaseCloseRealInputChain.Count -eq 8) -Severity "blocker" -Detail "Execution order must import the eight-step release-close real input chain from the final one-screen pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "release-close-real-input-chain-required-fields" -Passed ($releaseCloseRealInputChainRequiredFieldCount -ge 100) -Severity "blocker" -Detail "Release-close real input chain must expose the full required-field surface.")) | Out-Null
$items.Add((New-ValidationItem -Id "release-close-real-input-chain-rejected-substitutes" -Passed ($releaseCloseRealInputChainRejectedSubstituteCount -ge 30) -Severity "blocker" -Detail "Release-close real input chain must expose rejected substitute coverage.")) | Out-Null
$items.Add((New-ValidationItem -Id "release-close-real-input-chain-source-signals" -Passed ($releaseCloseRealInputChainSourceReadinessSignalCount -eq 18) -Severity "blocker" -Detail "Release-close real input chain source readiness signal count must stay aligned with the one-screen pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "release-close-real-input-chain-blocked-inputs" -Passed ($releaseCloseRealInputChainBlockedRealInputCount -gt 0) -Severity "blocker" -Detail "Release-close real input chain must remain blocked on real Owner inputs.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-download-proof-non-substitute-contract" -Passed ($publicPackageDownloadProofRequiredFieldCount -ge 30 -and $publicPackageDownloadProofRejectedSubstituteCount -eq 11 -and $publicPackageDownloadProofSourceReadinessSignalCount -eq 7 -and -not $publicPackageDownloadProofCandidateReady) -Severity "blocker" -Detail "Public package download proof counts must be visible but cannot substitute post-publish proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-proof-blocked-contract" -Passed ($postPublishCleanConsumerProofRequiredFieldCount -ge 50 -and $postPublishCleanConsumerProofRejectedSubstituteCount -eq 11 -and $postPublishCleanConsumerProofBlockedRealInputCount -gt 0 -and $postPublishCleanConsumerProofSourceReadinessSignalCount -eq 11 -and -not $postPublishCleanConsumerProofCandidateReady -and -not $postPublishCleanConsumerProofSourceProofLinkageReady) -Severity "blocker" -Detail "Post-publish CleanConsumer proof must remain blocked until real proof and source linkage are supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "release-evidence-bundle-sha-visible" -Passed ([System.Text.RegularExpressions.Regex]::IsMatch($releaseEvidenceBundleSha256, "^[0-9a-f]{64}$")) -Severity "blocker" -Detail "Execution order must carry the release evidence bundle SHA without treating it as close approval.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-close-validator-output-visible" -Passed ($finalCloseStrictValidatorOutputState -eq "blocked-final-close-gate-owner-proof-required") -Severity "blocker" -Detail "Execution order must carry the strict close validator output and keep the close gate blocked.")) | Out-Null

$topLevelSafe = [bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)
$items.Add((New-ValidationItem -Id "top-level-non-proof" -Passed $topLevelSafe -Severity "blocker" -Detail "Execution order must not publish, promote proof, or close release.")) | Out-Null

foreach ($step in $steps) {
  $id = [string](Get-PropertyOrDefault -Object $step -Name "id" -DefaultValue "")
  $inputArtifacts = ConvertTo-StringArray (Get-PropertyOrDefault -Object $step -Name "inputArtifacts" -DefaultValue @())
  $outputArtifacts = ConvertTo-StringArray (Get-PropertyOrDefault -Object $step -Name "outputArtifacts" -DefaultValue @())
  $validatorScripts = ConvertTo-StringArray (Get-PropertyOrDefault -Object $step -Name "validatorScripts" -DefaultValue @())
  $failureStopRule = [string](Get-PropertyOrDefault -Object $step -Name "failureStopRule" -DefaultValue "")
  $blockedUntil = [string](Get-PropertyOrDefault -Object $step -Name "blockedUntil" -DefaultValue "")
  $boundary = [string](Get-PropertyOrDefault -Object $step -Name "boundary" -DefaultValue "")
  $stepSafe = [bool](Get-PropertyOrDefault -Object $step -Name "notExecutedByAutomation" -DefaultValue $false) -and
    [bool](Get-PropertyOrDefault -Object $step -Name "ownerExecutionOnly" -DefaultValue $false) -and
    -not [bool](Get-PropertyOrDefault -Object $step -Name "performsPublish" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $step -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $step -Name "canPublishPublicly" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $step -Name "canCloseReleaseIssue" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $step -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $step -Name "isPostPublishProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $step -Name "isReleaseCloseProof" -DefaultValue $true)

  $items.Add((New-ValidationItem -Id "step-$id-contract" -Passed (@($inputArtifacts).Count -ge 2 -and @($outputArtifacts).Count -ge 2 -and @($validatorScripts).Count -ge 1 -and -not [string]::IsNullOrWhiteSpace($failureStopRule) -and -not [string]::IsNullOrWhiteSpace($blockedUntil)) -Severity "blocker" -Detail "Each step must expose input artifacts, output artifacts, validator scripts, blocked-until text, and failure stop rule.")) | Out-Null
  $items.Add((New-ValidationItem -Id "step-$id-non-proof" -Passed $stepSafe -Severity "blocker" -Detail "Step $id must remain owner-only and non-proof.")) | Out-Null
  $items.Add((New-ValidationItem -Id "step-$id-boundary" -Passed ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Step $id must state all non-proof boundaries.")) | Out-Null
}

$packageConsumerProofStep = $steps | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "02-clean-external-consumer-prepublish-proof" } | Select-Object -First 1
$packageConsumerValidators = ConvertTo-StringArray (Get-PropertyOrDefault -Object $packageConsumerProofStep -Name "validatorScripts" -DefaultValue @())
$items.Add((New-ValidationItem -Id "package-consumer-proof-strong-gate" -Passed (@($packageConsumerValidators | Where-Object { $_.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-Strict", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1) -Severity "blocker" -Detail "Clean external package consumer step must require Strict, existing logs, and FailOnNotProof.")) | Out-Null

$postPublishProofStep = $steps | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "04-post-publish-clean-consumer-proof" } | Select-Object -First 1
$postPublishValidators = ConvertTo-StringArray (Get-PropertyOrDefault -Object $postPublishProofStep -Name "validatorScripts" -DefaultValue @())
$items.Add((New-ValidationItem -Id "post-publish-proof-strong-gate" -Passed (@($postPublishValidators | Where-Object { $_.Contains("Test-PostPublishVerificationRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1) -Severity "blocker" -Detail "Post-publish clean consumer step must require existing logs and FailOnNotProof.")) | Out-Null

foreach ($artifact in @(
  "final-owner-proof-action-worklist.json",
  "final-owner-execution-package.json",
  "clean-external-package-consumer-owner-runbook.json",
  "post-publish-owner-verification-runbook.json",
  "public-publish-final-owner-execution-pack.json",
  "public-publish-command-cross-check.json",
  "final-owner-close-readiness-checkpoint.json",
  "final-release-close-blocker-dashboard.json",
  "owner-input-contract-convergence.json",
  "final-owner-execution-one-screen-pack.json",
  "final-owner-execution-one-screen-pack-validation.json",
  "release-evidence-bundle.json",
  "release-evidence-classification-audit.json"
)) {
  $items.Add((New-ValidationItem -Id "artifact-$($artifact.Replace('.', '-'))-listed" -Passed (($sourceArtifacts -join "`n").Contains($artifact, [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Source artifact $artifact must be listed.")) | Out-Null
}

foreach ($marker in @("template", "draft", "candidate", "dashboard", "dry-run", "build-only", "local feed", "ProjectReference", "direct nupkg", "direct .nupkg", "runbook as proof", "manual handoff as proof", "bundle-ready as proof", "blocked-by-cuda-driver", "public package download proof alone", "post-publish validation-ready without proofCandidateReady", "release evidence bundle hash only", "strict close validator output without real proof", "dotnet nuget push")) {
  $items.Add((New-ValidationItem -Id "forbidden-$($marker.Replace(' ', '-').Replace('.', 'dot'))-visible" -Passed ($raw.Contains($marker, [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Forbidden substitute or manual-publish marker '$marker' must remain visible.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-final-owner-strict-close-execution-order" } else { "blocked-final-owner-strict-close-owner-execution-required" }

$validation = [pscustomobject]@{
  recordKind = "final-owner-strict-close-execution-order-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  stepCount = $steps.Count
  blockedStepCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedStepCount" -DefaultValue 0)
  sourceRecordCount = $sourceRecords.Count
  sourceActionCount = [int](Get-PropertyOrDefault -Object $record -Name "sourceActionCount" -DefaultValue 0)
  sourceExecutionStepCount = [int](Get-PropertyOrDefault -Object $record -Name "sourceExecutionStepCount" -DefaultValue 0)
  cleanExternalRunbookStepCount = [int](Get-PropertyOrDefault -Object $record -Name "cleanExternalRunbookStepCount" -DefaultValue 0)
  postPublishRunbookStepCount = [int](Get-PropertyOrDefault -Object $record -Name "postPublishRunbookStepCount" -DefaultValue 0)
  publicPublishExecutionLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "publicPublishExecutionLaneCount" -DefaultValue 0)
  publicPublishCrossCheckCount = [int](Get-PropertyOrDefault -Object $record -Name "publicPublishCrossCheckCount" -DefaultValue 0)
  finalOwnerCloseReadinessCheckCount = [int](Get-PropertyOrDefault -Object $record -Name "finalOwnerCloseReadinessCheckCount" -DefaultValue 0)
  finalReleaseCloseBlockerCount = [int](Get-PropertyOrDefault -Object $record -Name "finalReleaseCloseBlockerCount" -DefaultValue 0)
  ownerInputContractSurfaceCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerInputContractSurfaceCount" -DefaultValue 0)
  ownerInputContractCanonicalFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerInputContractCanonicalFieldCount" -DefaultValue 0)
  ownerInputContractRunbookInputCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerInputContractRunbookInputCount" -DefaultValue 0)
  releaseCloseRealInputChainCount = $releaseCloseRealInputChainCount
  releaseCloseRealInputChainRequiredFieldCount = $releaseCloseRealInputChainRequiredFieldCount
  releaseCloseRealInputChainRejectedSubstituteCount = $releaseCloseRealInputChainRejectedSubstituteCount
  releaseCloseRealInputChainSourceReadinessSignalCount = $releaseCloseRealInputChainSourceReadinessSignalCount
  releaseCloseRealInputChainBlockedRealInputCount = $releaseCloseRealInputChainBlockedRealInputCount
  publicPackageDownloadProofRequiredFieldCount = $publicPackageDownloadProofRequiredFieldCount
  publicPackageDownloadProofRejectedSubstituteCount = $publicPackageDownloadProofRejectedSubstituteCount
  publicPackageDownloadProofSourceReadinessSignalCount = $publicPackageDownloadProofSourceReadinessSignalCount
  publicPackageDownloadProofCandidateReady = $publicPackageDownloadProofCandidateReady
  postPublishCleanConsumerProofRequiredFieldCount = $postPublishCleanConsumerProofRequiredFieldCount
  postPublishCleanConsumerProofRejectedSubstituteCount = $postPublishCleanConsumerProofRejectedSubstituteCount
  postPublishCleanConsumerProofBlockedRealInputCount = $postPublishCleanConsumerProofBlockedRealInputCount
  postPublishCleanConsumerProofSourceReadinessSignalCount = $postPublishCleanConsumerProofSourceReadinessSignalCount
  postPublishCleanConsumerProofCandidateReady = $postPublishCleanConsumerProofCandidateReady
  postPublishCleanConsumerProofSourceProofLinkageReady = $postPublishCleanConsumerProofSourceProofLinkageReady
  releaseEvidenceBundleSha256 = $releaseEvidenceBundleSha256
  finalCloseStrictValidatorOutputState = $finalCloseStrictValidatorOutputState
  failedBlockerCount = $failedBlockers.Count
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "Final Owner StrictClose execution order validation checks order shape and non-proof boundaries only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-strict-close-execution-order-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-strict-close-execution-order-validation.md"
$validation | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Final Owner StrictClose Execution Order Validation")
$lines.Add("")
$lines.Add("该 validator 只检查 Owner 执行顺序 artifact 的形状、计数、输入/输出/validator/failureStopRule 和 non-proof 边界，不执行发布、不生成 proof、不关闭 release issue。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| validationState | ``$($validation.validationState)`` |")
$lines.Add("| stepCount | ``$($validation.stepCount)`` |")
$lines.Add("| sourceRecordCount | ``$($validation.sourceRecordCount)`` |")
$lines.Add("| failedBlockerCount | ``$($validation.failedBlockerCount)`` |")
$lines.Add("| releaseCloseRealInputChainCount | ``$($validation.releaseCloseRealInputChainCount)`` |")
$lines.Add("| releaseCloseRealInputChainRequiredFieldCount | ``$($validation.releaseCloseRealInputChainRequiredFieldCount)`` |")
$lines.Add("| releaseCloseRealInputChainRejectedSubstituteCount | ``$($validation.releaseCloseRealInputChainRejectedSubstituteCount)`` |")
$lines.Add("| releaseCloseRealInputChainSourceReadinessSignalCount | ``$($validation.releaseCloseRealInputChainSourceReadinessSignalCount)`` |")
$lines.Add("| releaseCloseRealInputChainBlockedRealInputCount | ``$($validation.releaseCloseRealInputChainBlockedRealInputCount)`` |")
$lines.Add("| publicPackageDownloadProofRequiredFieldCount | ``$($validation.publicPackageDownloadProofRequiredFieldCount)`` |")
$lines.Add("| publicPackageDownloadProofCandidateReady | ``$($validation.publicPackageDownloadProofCandidateReady)`` |")
$lines.Add("| postPublishCleanConsumerProofRequiredFieldCount | ``$($validation.postPublishCleanConsumerProofRequiredFieldCount)`` |")
$lines.Add("| postPublishCleanConsumerProofCandidateReady | ``$($validation.postPublishCleanConsumerProofCandidateReady)`` |")
$lines.Add("| postPublishCleanConsumerProofSourceProofLinkageReady | ``$($validation.postPublishCleanConsumerProofSourceProofLinkageReady)`` |")
$lines.Add("| releaseEvidenceBundleSha256 | ``$($validation.releaseEvidenceBundleSha256)`` |")
$lines.Add("| finalCloseStrictValidatorOutputState | ``$($validation.finalCloseStrictValidatorOutputState)`` |")
$lines.Add("| performsPublish | ``$($validation.performsPublish)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Validation Items")
$lines.Add("")
$lines.Add("| ID | Passed | Severity | Detail |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $items) {
  $lines.Add("| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($validation.boundary)

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final Owner StrictClose execution order validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$($validation.validationState) Steps=$($validation.stepCount) Sources=$($validation.sourceRecordCount) FailedBlockers=$($validation.failedBlockerCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Final Owner StrictClose execution order validation failed with $($failedBlockers.Count) blocker(s)."
}
