[CmdletBinding()]
param(
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
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Test-BlockedState {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return $text.StartsWith("blocked", [StringComparison]::OrdinalIgnoreCase) -or
    $text.StartsWith("missing", [StringComparison]::OrdinalIgnoreCase) -or
    $text.StartsWith("incomplete", [StringComparison]::OrdinalIgnoreCase) -or
    $text.StartsWith("invalid", [StringComparison]::OrdinalIgnoreCase) -or
    $text.StartsWith("template", [StringComparison]::OrdinalIgnoreCase)
}

function New-ConvergenceInputRow {
  param([object]$Mapping)

  $id = [string](Get-PropertyOrDefault -Object $Mapping -Name "sourceTaskId" -DefaultValue ([string](Get-PropertyOrDefault -Object $Mapping -Name "id" -DefaultValue "unknown-input")))
  $state = [string](Get-PropertyOrDefault -Object $Mapping -Name "mappingState" -DefaultValue "missing-mapping-state")
  $missing = [bool](Get-PropertyOrDefault -Object $Mapping -Name "missingRealInput" -DefaultValue $true)

  [pscustomobject]@{
    id = "input-$id"
    category = "input"
    currentState = $state
    targetArtifact = [string](Get-PropertyOrDefault -Object $Mapping -Name "targetArtifact" -DefaultValue "")
    targetField = [string](Get-PropertyOrDefault -Object $Mapping -Name "targetField" -DefaultValue "")
    requiredEvidence = [string](Get-PropertyOrDefault -Object $Mapping -Name "requiredEvidence" -DefaultValue "Real owner input required.")
    recommendedNextCommand = [string](Get-PropertyOrDefault -Object $Mapping -Name "firstCommand" -DefaultValue "")
    validatorCommand = [string](Get-PropertyOrDefault -Object $Mapping -Name "validatorCommand" -DefaultValue "")
    blockingReason = if ($missing) { "Owner real input is still placeholder or missing." } else { "Input is present but still needs strict close validation." }
    isRealProof = $false
    isOwnerActionRequired = $true
    canCloseReleaseIssue = $false
  }
}

function New-ConvergenceValidationRow {
  param(
    [string]$Id,
    [AllowNull()][object]$Validation,
    [string]$TargetArtifact,
    [string]$RequiredEvidence,
    [string]$RecommendedNextCommand,
    [string]$ValidatorCommand
  )

  $state = [string](Get-PropertyOrDefault -Object $Validation -Name "validationState" -DefaultValue "missing-$Id")
  $failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $Validation -Name "failedActionRequiredCount" -DefaultValue 1)
  $failedBlockerCount = [int](Get-PropertyOrDefault -Object $Validation -Name "failedBlockerCount" -DefaultValue 0)
  $blocked = (Test-BlockedState -Value $state) -or $failedActionRequiredCount -gt 0 -or $failedBlockerCount -gt 0

  [pscustomobject]@{
    id = $Id
    category = "validator"
    currentState = $state
    targetArtifact = $TargetArtifact
    targetField = "validationState"
    requiredEvidence = $RequiredEvidence
    recommendedNextCommand = $RecommendedNextCommand
    validatorCommand = $ValidatorCommand
    blockingReason = if ($blocked) { "Validator remains blocked/action-required; real proof or real owner input is still missing." } else { "Validator shape is ready; still verify strict close chain before promotion." }
    isRealProof = $false
    isOwnerActionRequired = $blocked
    canCloseReleaseIssue = $false
  }
}

$realInputMap = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-real-input-map.json"
$realInputMapValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-real-input-map-validation.json"
$ownerProofPack = Read-JsonOrNull "artifacts\final-release\owner-proof-real-backfill-execution-pack.json"
$strictCandidate = Read-JsonOrNull "artifacts\final-release\release-close-strict-record-candidate.json"
$postPublishOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-owner-input-validation.json"
$packageConsumerRuntimeProofCandidateValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-candidate-validation.json"
$releaseIssueFinalCloseDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-final-close-decision-validation.json"
$releaseIssueCloseRecordCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-candidate-validation.json"
$releaseIssueCloseRecordOverlayCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-overlay-candidate-validation.json"
$ownerExternalExecutionResultBackfillKitValidation = Read-JsonOrNull "artifacts\final-release\owner-external-execution-result-backfill-kit-validation.json"
$releaseCloseStrictRecordCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-close-strict-record-candidate-validation.json"

if ($null -eq $realInputMap) {
  throw "Missing release issue close record real input map. Run Export-ReleaseIssueCloseRecordRealInputMap.ps1 first."
}

$realInputMappings = @(Get-PropertyOrDefault -Object $realInputMap -Name "realInputMappings" -DefaultValue @())
$nonSubstituteKinds = @(Get-PropertyOrDefault -Object $realInputMap -Name "nonSubstituteProofKinds" -DefaultValue (Get-PropertyOrDefault -Object $ownerProofPack -Name "nonSubstituteProofKinds" -DefaultValue @(
  "template",
  "draft",
  "candidate",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "schema-only",
  "preflight-only",
  "dependency-probe-only",
  "blocked-by-cuda-driver"
)))
$strictCloseValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
$strictValidators = @(Get-PropertyOrDefault -Object $realInputMap -Name "strictValidators" -DefaultValue @($strictCloseValidatorCommand))
if ($strictValidators -notcontains $strictCloseValidatorCommand) {
  $strictValidators += $strictCloseValidatorCommand
}

$convergenceRows = @()
$convergenceRows += @($realInputMappings | ForEach-Object { New-ConvergenceInputRow -Mapping $_ })
$convergenceRows += New-ConvergenceValidationRow -Id "post-publish-verification-owner-input-validation" -Validation $postPublishOwnerInputValidation -TargetArtifact "artifacts/final-release/post-publish-verification-owner-input-validation.json" -RequiredEvidence "Real public-channel package source, clean consumer identity, logs, hashes, and command capture." -RecommendedNextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationOwnerInput.ps1 -Strict" -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationOwnerInput.ps1 -Strict"
$convergenceRows += New-ConvergenceValidationRow -Id "package-consumer-runtime-proof-candidate-validation" -Validation $packageConsumerRuntimeProofCandidateValidation -TargetArtifact "artifacts/final-release/package-consumer-runtime-proof-candidate-validation.json" -RequiredEvidence "Clean external consumer runtime smoke with matching log SHA256 and no local package/project references." -RecommendedNextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict" -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict"
$convergenceRows += New-ConvergenceValidationRow -Id "release-issue-final-close-decision-validation" -Validation $releaseIssueFinalCloseDecisionValidation -TargetArtifact "artifacts/final-release/release-issue-final-close-decision-validation.json" -RequiredEvidence "Final owner close decision after rollback review, post-publish proof, and runtime smoke review." -RecommendedNextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueFinalCloseDecision.ps1 -Strict" -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueFinalCloseDecision.ps1 -Strict"
$convergenceRows += New-ConvergenceValidationRow -Id "release-issue-close-record-candidate-validation" -Validation $releaseIssueCloseRecordCandidateValidation -TargetArtifact "artifacts/final-release/release-issue-close-record-candidate-validation.json" -RequiredEvidence "Validator-passing release close candidate with real owner input and real post-publish proof." -RecommendedNextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordCandidate.ps1 -Strict" -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordCandidate.ps1 -Strict"
$convergenceRows += New-ConvergenceValidationRow -Id "release-issue-close-record-overlay-candidate-validation" -Validation $releaseIssueCloseRecordOverlayCandidateValidation -TargetArtifact "artifacts/final-release/release-issue-close-record-overlay-candidate-validation.json" -RequiredEvidence "Overlay candidate with real owner close inputs, paths, and hashes." -RecommendedNextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordOverlayCandidate.ps1 -Strict" -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordOverlayCandidate.ps1 -Strict"
$convergenceRows += New-ConvergenceValidationRow -Id "owner-external-execution-result-backfill-kit-validation" -Validation $ownerExternalExecutionResultBackfillKitValidation -TargetArtifact "artifacts/final-release/owner-external-execution-result-backfill-kit-validation.json" -RequiredEvidence "Owner external execution logs, hashes, result states, and reviewed command output." -RecommendedNextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalExecutionResultBackfillKit.ps1 -Strict" -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalExecutionResultBackfillKit.ps1 -Strict"
$convergenceRows += New-ConvergenceValidationRow -Id "release-close-strict-record-candidate-validation" -Validation $releaseCloseStrictRecordCandidateValidation -TargetArtifact "artifacts/final-release/release-close-strict-record-candidate-validation.json" -RequiredEvidence "Strict close candidate with zero missing owner input, zero missing real proof, and hash consistency." -RecommendedNextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseStrictRecordCandidate.ps1 -Strict" -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseStrictRecordCandidate.ps1 -Strict"

$inputRows = @($convergenceRows | Where-Object { $_.category -eq "input" })
$validatorRows = @($convergenceRows | Where-Object { $_.category -eq "validator" })
$missingRealInputCount = [int](Get-PropertyOrDefault -Object $realInputMapValidation -Name "missingRealInputCount" -DefaultValue (@($inputRows | Where-Object { $_.currentState -like "blocked*" }).Count))
$blockedValidatorCount = @($validatorRows | Where-Object { $_.isOwnerActionRequired }).Count
$blockedProofCount = [int](Get-PropertyOrDefault -Object $releaseCloseStrictRecordCandidateValidation -Name "missingRealProofCount" -DefaultValue $blockedValidatorCount)
if ($blockedProofCount -lt 1) {
  $blockedProofCount = $blockedValidatorCount
}

$recommendedOwnerSequence = @(
  "1. Backfill the real public channel package source, not local feed or direct nupkg.",
  "2. Run clean consumer runtime smoke outside the repository with the target runtime package key.",
  "3. Backfill smoke log path, SHA256, exit code, host metadata, and runtime key.",
  "4. Backfill release issue id and release issue url.",
  "5. Backfill rollback owner, rollback trigger, and rollback plan.",
  "6. Backfill final close decision only after real post-publish and runtime smoke proof pass.",
  "7. Refresh overlay candidate, strict candidate, owner input convergence matrix, and release evidence bundle.",
  "8. Run the strict close validator: Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady only after real proof exists."
)

$sourceArtifacts = @(
  "artifacts/final-release/release-issue-close-record-real-input-map.json",
  "artifacts/final-release/release-issue-close-record-real-input-map-validation.json",
  "artifacts/final-release/owner-proof-real-backfill-execution-pack.json",
  "artifacts/final-release/release-close-strict-record-candidate.json",
  "artifacts/final-release/post-publish-verification-owner-input-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-candidate-validation.json",
  "artifacts/final-release/release-issue-final-close-decision-validation.json",
  "artifacts/final-release/release-issue-close-record-candidate-validation.json",
  "artifacts/final-release/release-issue-close-record-overlay-candidate-validation.json",
  "artifacts/final-release/owner-external-execution-result-backfill-kit-validation.json",
  "artifacts/final-release/release-close-strict-record-candidate-validation.json"
)

$convergence = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "owner-proof-real-input-convergence"
  convergenceState = "blocked-owner-real-input-convergence-required"
  releaseIssueCloseRecordRealInputMapState = [string](Get-PropertyOrDefault -Object $realInputMap -Name "mapState" -DefaultValue "missing-release-issue-close-record-real-input-map")
  releaseIssueCloseRecordRealInputMapValidationState = [string](Get-PropertyOrDefault -Object $realInputMapValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-real-input-map-validation")
  inputMappingCount = $inputRows.Count
  missingRealInputCount = $missingRealInputCount
  blockedValidatorCount = $blockedValidatorCount
  blockedProofCount = $blockedProofCount
  convergenceRows = $convergenceRows
  recommendedOwnerSequence = $recommendedOwnerSequence
  strictValidators = $strictValidators
  nonSubstituteProofKinds = $nonSubstituteKinds
  sourceArtifacts = $sourceArtifacts
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Owner proof real input convergence is an owner-facing validation matrix only. It does not publish, upload, approve public release, close the release issue, or promote template/draft/candidate/hash-only/local-feed evidence to proof."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-proof-real-input-convergence.json"
$markdownPath = Join-Path $artifactRoot "owner-proof-real-input-convergence.md"
$convergence | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rowLines = $convergenceRows | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.category)`` | ``$($_.currentState)`` | ``$($_.targetArtifact)`` | ``$($_.targetField)`` | $($_.blockingReason.Replace("|", "\|")) |"
}
$sequenceLines = $recommendedOwnerSequence | ForEach-Object { "- $_" }
$nonSubstituteLines = $nonSubstituteKinds | ForEach-Object { "- ``$_``" }
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }

$markdown = @"
# Owner Proof Real Input Convergence

生成时间：$($convergence.generatedAtUtc)

该收敛矩阵把 release close 所需 Owner 输入、post-publish proof、package consumer runtime proof、final close decision 和 strict close validators 放到同一个 owner-facing 视图。它只显示下一步输入和验证顺序，不发布包、不批准公开发布、不关闭 release issue。

| 项目 | 当前值 |
|---|---|
| convergenceState | ``$($convergence.convergenceState)`` |
| releaseIssueCloseRecordRealInputMapState | ``$($convergence.releaseIssueCloseRecordRealInputMapState)`` |
| releaseIssueCloseRecordRealInputMapValidationState | ``$($convergence.releaseIssueCloseRecordRealInputMapValidationState)`` |
| inputMappingCount | ``$($convergence.inputMappingCount)`` |
| missingRealInputCount | ``$($convergence.missingRealInputCount)`` |
| blockedValidatorCount | ``$($convergence.blockedValidatorCount)`` |
| blockedProofCount | ``$($convergence.blockedProofCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Convergence Rows

| ID | Category | Current State | Target Artifact | Target Field | Blocking Reason |
|---|---|---|---|---|---|
$($rowLines -join "`r`n")

## Recommended Owner Sequence

$($sequenceLines -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")

## Boundary

$($convergence.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner proof real input convergence written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ConvergenceState=$($convergence.convergenceState) InputMappings=$($convergence.inputMappingCount) MissingRealInputs=$($convergence.missingRealInputCount) BlockedValidators=$($convergence.blockedValidatorCount) BlockedProofs=$($convergence.blockedProofCount) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"
