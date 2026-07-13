[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

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

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-BackfillLine {
  param(
    [string]$Id,
    [string]$Title,
    [string]$CurrentState,
    [string]$OwnerCommand,
    [string[]]$RequiredOwnerResults,
    [string[]]$RequiredLogs,
    [string[]]$RequiredHashes,
    [string[]]$RequiredResultArtifactPaths,
    [string[]]$BackfillJsonFieldPaths,
    [string[]]$ForbiddenSubstitutes,
    [string[]]$Validators,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    currentState = $CurrentState
    ownerCommand = $OwnerCommand
    requiredOwnerResults = $RequiredOwnerResults
    requiredLogs = $RequiredLogs
    requiredHashes = $RequiredHashes
    requiredResultArtifactPaths = $RequiredResultArtifactPaths
    backfillJsonFieldPaths = $BackfillJsonFieldPaths
    forbiddenSubstitutes = $ForbiddenSubstitutes
    validators = $Validators
    stdoutPathRequired = $true
    stderrPathRequired = $true
    mergedTranscriptPathRequired = $true
    passedTrueRequired = $true
    hashProofRequired = $true
    strictValidatorInputOnly = $true
    canPromoteDirectlyToProof = $false
    ownerActionStatus = "owner-external-execution-result-required"
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    boundary = $Boundary
  }
}

$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$packageConsumerOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input-validation.json"
$postPublishOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-owner-input-validation.json"
$releaseCloseOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-owner-input-validation.json"
$finalCloseDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-final-close-decision-validation.json"
$realExternalProofOverlayPackValidation = Read-JsonOrNull "artifacts\final-release\real-external-proof-overlay-pack-validation.json"
$releaseIssueCloseRecordOverlayCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-overlay-candidate-validation.json"
$releaseIssueCloseRecordOverlayCandidate = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-overlay-candidate.json"

$bundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$packageConsumerOwnerState = [string](Get-PropertyOrDefault -Object $packageConsumerOwnerInputValidation -Name "validationState" -DefaultValue "missing-package-consumer-runtime-proof-owner-input-validation")
$postPublishOwnerState = [string](Get-PropertyOrDefault -Object $postPublishOwnerInputValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-owner-input-validation")
$releaseCloseOwnerState = [string](Get-PropertyOrDefault -Object $releaseCloseOwnerInputValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-owner-input-validation")
$finalCloseDecisionState = [string](Get-PropertyOrDefault -Object $finalCloseDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-final-close-decision-validation")
$overlayPackValidationState = [string](Get-PropertyOrDefault -Object $realExternalProofOverlayPackValidation -Name "validationState" -DefaultValue "missing-real-external-proof-overlay-pack-validation")
$overlayCandidateValidationState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseRecordOverlayCandidateValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-overlay-candidate-validation")
$overlayCandidateState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseRecordOverlayCandidate -Name "candidateState" -DefaultValue "missing-release-issue-close-record-overlay-candidate")

$globalForbiddenSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "candidate",
  "draft",
  "dashboard",
  "dry-run",
  "dependency-probe-only",
  "blocked-by-cuda-driver",
  "build-only",
  "schema-only",
  "template-only"
)

$ownerResultInputSchemaPaths = @(
  '$.resultInputs[].resultInputId',
  '$.resultInputs[].executionInputId',
  '$.resultInputs[].candidateId',
  '$.resultInputs[].proofLane',
  '$.resultInputs[].runtimePackageKey',
  '$.resultInputs[].hostMetadata.os',
  '$.resultInputs[].hostMetadata.arch',
  '$.resultInputs[].hostMetadata.gpu',
  '$.resultInputs[].hostMetadata.driverVersion',
  '$.resultInputs[].hostMetadata.cudaVersion',
  '$.resultInputs[].hostMetadata.tensorRtVersion',
  '$.resultInputs[].hostMetadata.dotnetVersion',
  '$.resultInputs[].packageIdentity.packageId',
  '$.resultInputs[].packageIdentity.packageVersion',
  '$.resultInputs[].packageIdentity.nupkgPath',
  '$.resultInputs[].packageIdentity.nupkgSha256',
  '$.resultInputs[].packageIdentity.packageSource',
  '$.resultInputs[].executedCommandLine',
  '$.resultInputs[].workingDirectory',
  '$.resultInputs[].stdoutPath',
  '$.resultInputs[].stdoutSha256',
  '$.resultInputs[].stderrPath',
  '$.resultInputs[].stderrSha256',
  '$.resultInputs[].mergedTranscriptPath',
  '$.resultInputs[].mergedTranscriptSha256',
  '$.resultInputs[].validatorOutputPath',
  '$.resultInputs[].validatorOutputSha256',
  '$.resultInputs[].exitCode',
  '$.resultInputs[].passed',
  '$.resultInputs[].startedAtUtc',
  '$.resultInputs[].endedAtUtc',
  '$.resultInputs[].ownerReviewer',
  '$.resultInputs[].ownerReviewTimestampUtc',
  '$.resultInputs[].nonSubstituteConfirmations'
)

$ownerResultHashProofPaths = @(
  '$.resultInputs[].packageIdentity.nupkgSha256',
  '$.resultInputs[].stdoutSha256',
  '$.resultInputs[].stderrSha256',
  '$.resultInputs[].mergedTranscriptSha256',
  '$.resultInputs[].validatorOutputSha256'
)

$backfillLines = @(
  (New-BackfillLine -Id "package-consumer-runtime-proof" -Title "Package consumer runtime proof execution result" -CurrentState $packageConsumerOwnerState -OwnerCommand "Run a repository-external clean consumer using public packages only; then capture restore/build/native-asset/runtime smoke logs and SHA256 values. Copy the completed resultInputs[] object into artifacts/final-release/owner-external-proof-execution-result.input.json." -RequiredOwnerResults @("public package source", "clean external consumer root", "consumer project identity", "runtime package key smoke command", "compatible host metadata", "runtime smoke exit code 0") -RequiredLogs @("restore log", "build log", "native asset listing", "runtime smoke log") -RequiredHashes @("managed nupkg SHA256", "runtime nupkg SHA256", "restore log SHA256", "build log SHA256", "runtime smoke log SHA256") -RequiredResultArtifactPaths @("stdoutPath", "stderrPath", "mergedTranscriptPath", "restoreLogPath", "buildLogPath", "nativeAssetListingPath", "runtimeSmokeLogPath") -BackfillJsonFieldPaths $ownerResultInputSchemaPaths -ForbiddenSubstitutes $globalForbiddenSubstitutes -Validators @("Import-OwnerExternalProofExecutionResult.ps1", "Test-OwnerExternalProofExecutionResultImport.ps1 -Strict", "Export-RealProofRecordCandidateFromOwnerResultImport.ps1", "Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict", "Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof") -Boundary "Requires real clean external consumer runtime smoke against public packages; owner input and candidate metadata are strict-validator input only, not proof.")
  (New-BackfillLine -Id "post-publish-verification" -Title "Post-publish verification execution result" -CurrentState $postPublishOwnerState -OwnerCommand "After owner-approved public publish, restore and run a clean external consumer from the public channel; capture stdout, stderr, merged transcript, package URLs, logs, and SHA256 values. Backfill the matching resultInputs[] item only after public package source is used." -RequiredOwnerResults @("selected public channel", "published package URL", "managed/runtime package identity", "clean consumer root", "consumer project path", "host metadata", "runtime smoke command", "runtime smoke exit code 0") -RequiredLogs @("restore log", "native asset listing", "dependency probe log", "runtime smoke log") -RequiredHashes @("managed nupkg SHA256", "runtime nupkg SHA256", "restore log SHA256", "native asset listing SHA256", "dependency probe log SHA256", "runtime smoke log SHA256") -RequiredResultArtifactPaths @("stdoutPath", "stderrPath", "mergedTranscriptPath", "restoreLogPath", "nativeAssetListingPath", "dependencyProbeLogPath", "runtimeSmokeLogPath") -BackfillJsonFieldPaths $ownerResultInputSchemaPaths -ForbiddenSubstitutes $globalForbiddenSubstitutes -Validators @("Import-OwnerExternalProofExecutionResult.ps1", "Test-OwnerExternalProofExecutionResultImport.ps1 -Strict", "Export-RealExternalProofRecordImportValidator.ps1", "Test-RealExternalProofRecordImportValidator.ps1 -Strict", "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof") -Boundary "Requires a real post-publish public channel package and clean external consumer smoke; local feed and direct package files remain non-proof.")
  (New-BackfillLine -Id "release-issue-close-record-owner-input" -Title "Release issue close owner input result" -CurrentState $releaseCloseOwnerState -OwnerCommand "Review strict validator outputs, rollback plan, owner decision, release issue metadata, and log/hash inventory before filling the matching resultInputs[] item for release close owner input." -RequiredOwnerResults @("real post-publish proof validation", "rollback plan", "rollback owner", "rollback trigger", "owner final close decision", "release issue id", "release issue URL") -RequiredLogs @("post-publish validation log", "release close validator log") -RequiredHashes @("release evidence bundle SHA256", "release close preflight SHA256", "stale claims audit SHA256", "post-publish validation SHA256") -RequiredResultArtifactPaths @("postPublishValidationLogPath", "releaseCloseValidatorLogPath", "releaseEvidenceBundlePath", "strictValidatorOutputPath") -BackfillJsonFieldPaths $ownerResultInputSchemaPaths -ForbiddenSubstitutes $globalForbiddenSubstitutes -Validators @("Import-OwnerExternalProofExecutionResult.ps1", "Test-OwnerExternalProofExecutionResultImport.ps1 -Strict", "Export-ReleaseCloseRealProofImportBridge.ps1", "Test-ReleaseCloseRealProofImportBridge.ps1 -Strict", "Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady") -Boundary "Close owner input cannot close the release issue without real post-publish proof, rollback approval, and strict close validation.")
  (New-BackfillLine -Id "release-issue-final-close-decision" -Title "Release issue final close decision result" -CurrentState $finalCloseDecisionState -OwnerCommand "Record owner final close decision only after every strict real proof validator passes; include review log, release issue URL, rollback review, and evidence hashes in the matching resultInputs[] item." -RequiredOwnerResults @("owner name", "decision timestamp UTC", "release issue id", "release issue URL", "final close decision", "rollback review", "real post-publish proof confirmation", "runtime smoke confirmation", "log/hash review confirmation") -RequiredLogs @("final close decision review log", "strict close validator log") -RequiredHashes @("final evidence freeze SHA256", "release evidence bundle SHA256", "post-publish validation SHA256", "release close candidate validation SHA256") -RequiredResultArtifactPaths @("finalCloseDecisionReviewLogPath", "strictCloseValidatorLogPath", "finalEvidenceFreezePath", "releaseEvidenceBundlePath") -BackfillJsonFieldPaths $ownerResultInputSchemaPaths -ForbiddenSubstitutes $globalForbiddenSubstitutes -Validators @("Import-OwnerExternalProofExecutionResult.ps1", "Test-OwnerExternalProofExecutionResultImport.ps1 -Strict", "Export-ReleaseCloseRealProofImportBridge.ps1", "Test-ReleaseCloseRealProofImportBridge.ps1 -Strict", "Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady") -Boundary "Final close decision remains owner input only until all real proof records and strict close validation pass.")
)

$requiredOwnerExecutionResults = @($backfillLines | ForEach-Object { $_.requiredOwnerResults } | Select-Object -Unique)
$requiredLogs = @($backfillLines | ForEach-Object { $_.requiredLogs } | Select-Object -Unique)
$requiredHashes = @($backfillLines | ForEach-Object { $_.requiredHashes } | Select-Object -Unique)
$requiredResultArtifactPaths = @($backfillLines | ForEach-Object { $_.requiredResultArtifactPaths } | Select-Object -Unique)
$backfillJsonFieldPaths = @($backfillLines | ForEach-Object { $_.backfillJsonFieldPaths } | Select-Object -Unique)
$placeholderFields = @($requiredOwnerExecutionResults)
$missingExternalLogs = @($requiredLogs)
$missingSha256Values = @($requiredHashes)

$sourceArtifacts = @(
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/owner-external-proof-execution-result.input.template.json",
  "artifacts/final-release/owner-external-proof-execution-result-input-template-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
  "artifacts/final-release/post-publish-verification-owner-input-validation.json",
  "artifacts/final-release/release-issue-close-record-owner-input-validation.json",
  "artifacts/final-release/release-issue-final-close-decision-validation.json",
  "artifacts/final-release/real-external-proof-overlay-pack-validation.json",
  "artifacts/final-release/release-issue-close-record-overlay-candidate.json",
  "artifacts/final-release/release-issue-close-record-overlay-candidate-validation.json"
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "owner-external-execution-result-backfill-kit"
  kitState = "blocked-owner-external-execution-results-required"
  runtimePackageKey = $RuntimePackageKey
  releaseEvidenceBundleState = $bundleState
  packageConsumerRuntimeProofOwnerInputValidationState = $packageConsumerOwnerState
  postPublishVerificationOwnerInputValidationState = $postPublishOwnerState
  releaseIssueCloseRecordOwnerInputValidationState = $releaseCloseOwnerState
  releaseIssueFinalCloseDecisionValidationState = $finalCloseDecisionState
  realExternalProofOverlayPackValidationState = $overlayPackValidationState
  releaseIssueCloseRecordOverlayCandidateState = $overlayCandidateState
  releaseIssueCloseRecordOverlayCandidateValidationState = $overlayCandidateValidationState
  requiredOwnerExecutionResultCount = $requiredOwnerExecutionResults.Count
  placeholderFieldCount = $placeholderFields.Count
  missingExternalLogCount = $missingExternalLogs.Count
  missingSha256Count = $missingSha256Values.Count
  requiredResultArtifactPathCount = $requiredResultArtifactPaths.Count
  backfillJsonFieldPathCount = $backfillJsonFieldPaths.Count
  readyForStrictValidatorLaneCount = 0
  localFeedReferenceCount = 1
  projectReferenceCount = 1
  directNupkgReferenceCount = 1
  blockedByCudaDriverCount = 1
  backfillLineCount = $backfillLines.Count
  backfillLines = $backfillLines
  placeholderFields = $placeholderFields
  missingExternalLogs = $missingExternalLogs
  missingSha256Values = $missingSha256Values
  requiredResultArtifactPaths = $requiredResultArtifactPaths
  backfillJsonFieldPaths = $backfillJsonFieldPaths
  ownerResultInputSchemaSummary = [pscustomobject]@{
    inputPath = "artifacts/final-release/owner-external-proof-execution-result.input.json"
    templatePath = "artifacts/final-release/owner-external-proof-execution-result.input.template.json"
    templateValidationPath = "artifacts/final-release/owner-external-proof-execution-result-input-template-validation.json"
    rootArrayPath = '$.resultInputs[]'
    importScript = "eng/Import-OwnerExternalProofExecutionResult.ps1"
    importValidationScript = "eng/Test-OwnerExternalProofExecutionResultImport.ps1 -Strict"
    requiredPathCount = $ownerResultInputSchemaPaths.Count
    requiredPaths = $ownerResultInputSchemaPaths
    hashProofPaths = $ownerResultHashProofPaths
    requiredReadyConditions = @(
      "resultInputId matches a template item",
      "requiredResultFields all filled",
      "packageIdentity.nupkgPath file exists under allowed evidence roots",
      "stdoutPath, stderrPath, mergedTranscriptPath, and validatorOutputPath files exist",
      "all SHA256 values are 64 hex characters and match referenced files",
      "exitCode is 0",
      "passed is true",
      "ownerReviewer and ownerReviewTimestampUtc are real values",
      "nonSubstituteConfirmations contains at least 10 entries",
      "forbidden substitute markers are absent"
    )
  }
  forbiddenSubstitutes = $globalForbiddenSubstitutes
  strictValidators = @($backfillLines | ForEach-Object { $_.validators } | Select-Object -Unique)
  cleanExternalConsumerContract = [pscustomobject]@{
    mustBeOutsideRepository = $true
    mustUsePublicPackageSource = $true
    projectReferenceForbidden = $true
    localFeedForbiddenForPostPublishProof = $true
    directNupkgForbiddenForPostPublishProof = $true
    stdoutPathRequired = $true
    stderrPathRequired = $true
    mergedTranscriptPathRequired = $true
    hashProofRequired = $true
  }
  proofPromotionBoundary = [pscustomobject]@{
    ownerResultImportIsInputOnly = $true
    candidateFromOwnerResultIsInputOnly = $true
    strictValidatorRequired = $true
    releaseCloseBridgeInputOnly = $true
    articlesAndDashboardsAreNotProof = $true
  }
  sourceArtifacts = $sourceArtifacts
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Backfill kit is owner execution guidance only. It cannot publish packages, prove runtime execution, or close release issue."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-external-execution-result-backfill-kit.json"
$markdownPath = Join-Path $artifactRoot "owner-external-execution-result-backfill-kit.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$lineRows = $backfillLines | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.currentState)`` | $($_.requiredOwnerResults.Count) | $($_.requiredLogs.Count) | $($_.requiredHashes.Count) | $($_.requiredResultArtifactPaths.Count) | $($_.boundary.Replace("|", "\|")) |"
}
$placeholderLines = $placeholderFields | ForEach-Object { "- ``$_``" }
$logLines = $missingExternalLogs | ForEach-Object { "- ``$_``" }
$hashLines = $missingSha256Values | ForEach-Object { "- ``$_``" }
$artifactPathLines = $requiredResultArtifactPaths | ForEach-Object { "- ``$_``" }
$fieldPathLines = $backfillJsonFieldPaths | ForEach-Object { "- ``$_``" }
$schemaPathLines = $ownerResultInputSchemaPaths | ForEach-Object { "- ``$_``" }
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }

$markdown = @"
# Owner External Execution Result Backfill Kit

生成时间：$($record.generatedAtUtc)

该 kit 把真实 Owner 外部执行结果回填集中为一份可执行清单。它不执行发布、不上传包、不关闭 release issue，也不会把 template/candidate/guidance/hash 一致性晋级为 proof。

| 项目 | 当前值 |
|---|---|
| kitState | ``$($record.kitState)`` |
| runtimePackageKey | ``$RuntimePackageKey`` |
| releaseEvidenceBundleState | ``$bundleState`` |
| packageConsumerRuntimeProofOwnerInputValidationState | ``$packageConsumerOwnerState`` |
| postPublishVerificationOwnerInputValidationState | ``$postPublishOwnerState`` |
| releaseIssueCloseRecordOwnerInputValidationState | ``$releaseCloseOwnerState`` |
| releaseIssueFinalCloseDecisionValidationState | ``$finalCloseDecisionState`` |
| realExternalProofOverlayPackValidationState | ``$overlayPackValidationState`` |
| releaseIssueCloseRecordOverlayCandidateValidationState | ``$overlayCandidateValidationState`` |
| requiredOwnerExecutionResultCount | ``$($record.requiredOwnerExecutionResultCount)`` |
| placeholderFieldCount | ``$($record.placeholderFieldCount)`` |
| missingExternalLogCount | ``$($record.missingExternalLogCount)`` |
| missingSha256Count | ``$($record.missingSha256Count)`` |
| requiredResultArtifactPathCount | ``$($record.requiredResultArtifactPathCount)`` |
| backfillJsonFieldPathCount | ``$($record.backfillJsonFieldPathCount)`` |
| readyForStrictValidatorLaneCount | ``$($record.readyForStrictValidatorLaneCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Backfill Lines

| ID | Current State | Owner Results | Logs | Hashes | Artifact Paths | Boundary |
|---|---|---:|---:|---:|---:|---|
$($lineRows -join "`r`n")

## Placeholder Owner Results

$($placeholderLines -join "`r`n")

## Missing External Logs

$($logLines -join "`r`n")

## Missing SHA256 Values

$($hashLines -join "`r`n")

## Required Result Artifact Paths

$($artifactPathLines -join "`r`n")

## Backfill JSON Field Paths

$($fieldPathLines -join "`r`n")

## Owner Result Input Schema

输入文件：``artifacts/final-release/owner-external-proof-execution-result.input.json``

导入脚本：``eng/Import-OwnerExternalProofExecutionResult.ps1``

严格校验：``eng/Test-OwnerExternalProofExecutionResultImport.ps1 -Strict``

$($schemaPathLines -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner external execution result backfill kit written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "KitState=$($record.kitState) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"
