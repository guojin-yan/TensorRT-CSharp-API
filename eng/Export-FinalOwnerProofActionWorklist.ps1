[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
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

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-Action {
  param(
    [string]$Id,
    [string]$LaneId,
    [string]$Phase,
    [string]$State,
    [string]$ActionRequiredId,
    [string]$Detail,
    [string[]]$RequiredInputs,
    [string[]]$OwnerCommands,
    [string[]]$ValidatorCommands,
    [string[]]$ForbiddenSubstitutes,
    [string]$SourceArtifact,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    laneId = $LaneId
    phase = $Phase
    state = $State
    blocked = $true
    actionRequiredId = $ActionRequiredId
    detail = $Detail
    requiredInputs = @($RequiredInputs)
    requiredInputCount = @($RequiredInputs).Count
    ownerCommands = @($OwnerCommands)
    validatorCommands = @($ValidatorCommands)
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    sourceArtifact = $SourceArtifact
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = $Boundary
  }
}

function Find-ActionRequiredItem {
  param([object[]]$Items, [string]$Id)

  return @($Items | Where-Object {
    [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq $Id
  } | Select-Object -First 1)
}

$finalGate = Read-JsonOrNull "artifacts\final-release\final-publish-proof-gate-report.json"
$releaseDashboard = Read-JsonOrNull "artifacts\final-release\release-proof-dashboard.json"
$finalCloseDashboard = Read-JsonOrNull "artifacts\final-release\final-release-close-blocker-dashboard.json"
$ownerResultCandidateBridge = Read-JsonOrNull "artifacts\final-release\real-proof-record-candidate-from-owner-result-import-validation.json"
$ownerExternalResultImport = Read-JsonOrNull "artifacts\final-release\owner-external-proof-execution-result-import-validation.json"
$publicDocsGate = Read-JsonOrNull "artifacts\final-release\public-docs-package-metadata-gate.json"
$cleanExternalPackageConsumerOwnerRunbook = Read-JsonOrNull "artifacts\final-release\clean-external-package-consumer-owner-runbook.json"
$cleanExternalPackageConsumerOwnerRunbookValidation = Read-JsonOrNull "artifacts\final-release\clean-external-package-consumer-owner-runbook-validation.json"
$postPublishOwnerVerificationRunbook = Read-JsonOrNull "artifacts\final-release\post-publish-owner-verification-runbook.json"
$postPublishOwnerVerificationRunbookValidation = Read-JsonOrNull "artifacts\final-release\post-publish-owner-verification-runbook-validation.json"

$gateItems = @()
if ($null -ne $finalGate) {
  $gateItems = @(Get-PropertyOrDefault -Object $finalGate -Name "validationItems" -DefaultValue @())
}

$actionRequiredItems = @($gateItems | Where-Object {
  -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) -and
  [string](Get-PropertyOrDefault -Object $_ -Name "severity" -DefaultValue "") -eq "action-required"
})

$commonForbidden = @(
  "preflight",
  "dashboard",
  "draft",
  "template",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "build-only",
  "dry-run",
  "matrix",
  "report",
  "sidecar-only report",
  "screenshot",
  "skipped run",
  "blocked-by-cuda-driver",
  "dependency-probe-only"
)

$realModelId = "real-model-runtime-owner-proof-required"
$packageId = "package-consumer-runtime-owner-proof-required"
$postPublishId = "post-publish-verification-owner-proof-required"
$finalOwnerTemplatePackId = "final-owner-real-input-template-pack-owner-input-required"
$ownerResultImportId = "owner-external-proof-result-import-owner-proof-required"
$candidateBridgeId = "owner-result-candidate-bridge-real-proof-required"

$realModelItem = Find-ActionRequiredItem -Items $actionRequiredItems -Id $realModelId
$packageItem = Find-ActionRequiredItem -Items $actionRequiredItems -Id $packageId
$postPublishItem = Find-ActionRequiredItem -Items $actionRequiredItems -Id $postPublishId
$finalOwnerTemplatePackItem = Find-ActionRequiredItem -Items $actionRequiredItems -Id $finalOwnerTemplatePackId
$ownerResultImportItem = Find-ActionRequiredItem -Items $actionRequiredItems -Id $ownerResultImportId
$candidateBridgeItem = Find-ActionRequiredItem -Items $actionRequiredItems -Id $candidateBridgeId
$cleanExternalRunbookState = [string](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbook -Name "runbookState" -DefaultValue "missing-clean-external-package-consumer-owner-runbook")
$cleanExternalRunbookValidationState = [string](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbookValidation -Name "validationState" -DefaultValue "missing-clean-external-package-consumer-owner-runbook-validation")
$cleanExternalRunbookStepCount = [int](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbook -Name "stepCount" -DefaultValue 0)
$cleanExternalRunbookFailedBlockerCount = [int](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbookValidation -Name "failedBlockerCount" -DefaultValue -1)
$postPublishRunbookState = [string](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbook -Name "runbookState" -DefaultValue "missing-post-publish-owner-verification-runbook")
$postPublishRunbookValidationState = [string](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbookValidation -Name "validationState" -DefaultValue "missing-post-publish-owner-verification-runbook-validation")
$postPublishRunbookStepCount = [int](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbook -Name "stepCount" -DefaultValue 0)
$postPublishRunbookFailedBlockerCount = [int](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbookValidation -Name "failedBlockerCount" -DefaultValue -1)

$actions = @(
  New-Action `
    -Id "00-clean-external-package-consumer-owner-runbook" `
    -LaneId "package-consumer-runtime" `
    -Phase "owner-runbook-preflight" `
    -State $cleanExternalRunbookValidationState `
    -ActionRequiredId "clean-external-package-consumer-owner-runbook-required" `
    -Detail "Run the clean external package consumer owner runbook before importing package-consumer runtime proof. runbookState=$cleanExternalRunbookState; steps=$cleanExternalRunbookStepCount; failedBlockers=$cleanExternalRunbookFailedBlockerCount." `
    -RequiredInputs @(
      "repository-external clean consumer workspace",
      "public or owner-approved package source",
      "stdoutPath",
      "stderrPath",
      "mergedTranscriptPath",
      "package and log SHA256 values",
      "validatorOutputSha256",
      "ownerReviewer and ownerReviewTimestampUtc",
      "nonSubstituteConfirmations"
    ) `
    -OwnerCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CleanExternalPackageConsumerOwnerRunbook.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CleanExternalPackageConsumerOwnerRunbook.ps1 -Strict",
      "Follow artifacts/final-release/clean-external-package-consumer-owner-runbook.md in a repository-external project"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CleanExternalPackageConsumerOwnerRunbook.ps1 -Strict"
    ) `
    -ForbiddenSubstitutes (@($commonForbidden) + @("runbook as proof", "local package source", "direct package file")) `
    -SourceArtifact "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json" `
    -Boundary "Clean external package consumer runbook is owner-executable guidance only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  New-Action `
    -Id "00-post-publish-owner-verification-runbook" `
    -LaneId "post-publish-verification" `
    -Phase "owner-runbook-preflight" `
    -State $postPublishRunbookValidationState `
    -ActionRequiredId "post-publish-owner-verification-runbook-required" `
    -Detail "Run the post-publish owner verification runbook only after real public publish. runbookState=$postPublishRunbookState; steps=$postPublishRunbookStepCount; failedBlockers=$postPublishRunbookFailedBlockerCount." `
    -RequiredInputs @(
      "selected public or owner-approved package channel",
      "published package URLs",
      "public package source URL",
      "downloaded nupkg SHA256 values",
      "repository-external post-publish clean consumer",
      "stdoutPath",
      "stderrPath",
      "mergedTranscriptPath",
      "host metadata and owner review"
    ) `
    -OwnerCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishOwnerVerificationRunbook.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishOwnerVerificationRunbook.ps1 -Strict",
      "Follow artifacts/final-release/post-publish-owner-verification-runbook.md after real public publish"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishOwnerVerificationRunbook.ps1 -Strict"
    ) `
    -ForbiddenSubstitutes (@($commonForbidden) + @("pre-publish package-consumer proof", "runbook as proof", "owner authorization only")) `
    -SourceArtifact "artifacts/final-release/post-publish-owner-verification-runbook-validation.json" `
    -Boundary "Post-publish owner verification runbook is owner-executable guidance only; it does not run dotnet nuget push and is not post-publish proof, publish approval, release close approval, runtime proof, or package push."
  New-Action `
    -Id "01-real-model-runtime-owner-evidence" `
    -LaneId "real-model-runtime" `
    -Phase "real-model-runtime" `
    -State "blocked-owner-real-model-evidence-required" `
    -ActionRequiredId $realModelId `
    -Detail ([string](Get-PropertyOrDefault -Object $realModelItem -Name "detail" -DefaultValue "real-model-runtime still requires real model logs, hashes, host metadata, and owner review.")) `
    -RequiredInputs @(
      "real model source URL/path and license/provenance",
      "model SHA256 and input asset SHA256",
      "TensorRtExec build report path and SHA256",
      "YoloVision run log path and SHA256",
      "YoloVision output JSON path and SHA256",
      "host OS, GPU, CUDA driver/runtime, TensorRT runtime",
      "owner reviewer, reviewedAtUtc, accepted=true"
    ) `
    -OwnerCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-YoloVisionRealAssetOwnerProofInputTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-YoloVisionRealAssetOwnerProofInput.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
    ) `
    -ForbiddenSubstitutes (@($commonForbidden) + @("sample matrix", "parse-only", "asset candidate")) `
    -SourceArtifact "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json" `
    -Boundary "Real-model-runtime proof requires real assets and real execution logs; it cannot close package-consumer-runtime, post-publish, or release-close lanes."
  New-Action `
    -Id "02-package-consumer-runtime-clean-external-proof" `
    -LaneId "package-consumer-runtime" `
    -Phase "package-consumer-runtime" `
    -State "blocked-clean-external-consumer-proof-required" `
    -ActionRequiredId $packageId `
    -Detail ([string](Get-PropertyOrDefault -Object $packageItem -Name "detail" -DefaultValue "package-consumer-runtime still requires clean external consumer proof with public package source.")) `
    -RequiredInputs @(
      "public package source URI",
      "managed package id/version",
      "runtime package id/version",
      "clean consumer root outside repository",
      "no ProjectReference confirmation",
      "no local feed confirmation",
      "no direct .nupkg confirmation",
      "restore/build/runtime smoke logs",
      "runtime smoke exitCode=0 and passed=true",
      "stdout/stderr summary",
      "smoke log SHA256",
      "host OS, GPU, CUDA driver/runtime, TensorRT runtime",
      "owner reviewer and reviewedAtUtc"
    ) `
    -OwnerCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof"
    ) `
    -ForbiddenSubstitutes (@($commonForbidden) + @("sample-run-evidence", "real-model-runtime", "local package source")) `
    -SourceArtifact "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json" `
    -Boundary "Package-consumer-runtime proof must come from a clean external consumer and a real public package source; local feed, ProjectReference, direct .nupkg, and sample-run evidence are not proof."
  New-Action `
    -Id "03-post-publish-verification-public-channel" `
    -LaneId "post-publish-verification" `
    -Phase "post-publish-verification" `
    -State "blocked-real-public-publish-and-clean-install-required" `
    -ActionRequiredId $postPublishId `
    -Detail ([string](Get-PropertyOrDefault -Object $postPublishItem -Name "detail" -DefaultValue "post-publish verification requires public channel publish and clean install/run logs.")) `
    -RequiredInputs @(
      "public channel name and source URI",
      "published managed package URL",
      "published runtime package URL when applicable",
      "downloaded nupkg SHA256 values",
      "clean consumer root outside repository",
      "restore log path and SHA256",
      "dependency probe log path and SHA256",
      "runtime smoke log path and SHA256",
      "runtime smoke exitCode=0 and passed=true",
      "host metadata",
      "owner reviewer and reviewedAtUtc"
    ) `
    -OwnerCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationOwnerInputTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordFromOwnerInput.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof"
    ) `
    -ForbiddenSubstitutes (@($commonForbidden) + @("pre-publish package-consumer proof", "owner authorization only")) `
    -SourceArtifact "artifacts/final-release/post-publish-verification-owner-input.template.json" `
    -Boundary "Post-publish verification can only happen after real public channel publish; pre-publish consumer proof and owner authorization cannot replace it."
  New-Action `
    -Id "04-final-owner-real-input-template-pack" `
    -LaneId "final-owner-real-input-template-pack" `
    -Phase "owner-real-input-template-pack" `
    -State "blocked-final-owner-real-input-required" `
    -ActionRequiredId $finalOwnerTemplatePackId `
    -Detail ([string](Get-PropertyOrDefault -Object $finalOwnerTemplatePackItem -Name "detail" -DefaultValue "Final owner real input template pack still requires real owner-filled stdout/stderr/log/SHA256/exitCode/host/package evidence for all five lanes.")) `
    -RequiredInputs @(
      "owner-authorization owner input JSON",
      "package-consumer-runtime proof owner input JSON",
      "linux-runner proof owner input JSON",
      "real-model-runtime owner input JSON",
      "post-publish verification owner input JSON",
      "real stdout/stderr/log paths and SHA256 values",
      "exitCode and host identity for every executable lane",
      "owner reviewer and non-substitute confirmations"
    ) `
    -OwnerCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalOwnerRealInputTemplates.ps1",
      "Fill artifacts/final-release/owner-real-inputs/*owner-input.template.json with real Owner evidence and save as the corresponding owner input files",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalOwnerRealInputTemplates.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalPublishProofGate.ps1 -Strict"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalOwnerRealInputTemplates.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalPublishProofGate.ps1 -Strict"
    ) `
    -ForbiddenSubstitutes (@($commonForbidden) + @("template as proof", "template pack as proof", "placeholder owner input")) `
    -SourceArtifact "artifacts/final-release/final-owner-real-input-template-pack-validation.json" `
    -Boundary "Final Owner real input templates are landing contracts only; they do not publish, do not approve public release, do not close release issues, and do not promote runtime or post-publish proof."
  New-Action `
    -Id "05-owner-external-result-import-real-files" `
    -LaneId "owner-external-proof-result-import" `
    -Phase "owner-result-import" `
    -State ([string](Get-PropertyOrDefault -Object $ownerExternalResultImport -Name "validationState" -DefaultValue "blocked-owner-external-proof-execution-result-required")) `
    -ActionRequiredId $ownerResultImportId `
    -Detail ([string](Get-PropertyOrDefault -Object $ownerResultImportItem -Name "detail" -DefaultValue "owner external proof result import still requires real existing logs, matching SHA256, exitCode=0, non-substitute confirmations, and owner review for every lane.")) `
    -RequiredInputs @(
      "existing log files under allowed evidence roots",
      "valid SHA256 for every referenced evidence file",
      "matching log SHA256 values",
      "exitCode=0 where runtime execution proof is claimed",
      "explicit non-substitute confirmations",
      "owner review for every lane"
    ) `
    -OwnerCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerExternalProofExecutionBundle.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-OwnerExternalProofExecutionResult.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalProofExecutionResultImport.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealExternalProofRecordImportValidator.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealExternalProofRecordImportValidator.ps1 -Strict"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalProofExecutionResultImport.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealExternalProofRecordImportValidator.ps1 -Strict"
    ) `
    -ForbiddenSubstitutes (@($commonForbidden) + @("missing file", "mismatched hash", "placeholder hash")) `
    -SourceArtifact "artifacts/final-release/owner-external-proof-execution-result-import.json" `
    -Boundary "Owner result import is an admission layer; imported rows with missing files, invalid hashes, or substitute evidence cannot become proof."
  New-Action `
    -Id "06-owner-result-candidate-bridge-strict-promotion" `
    -LaneId "owner-result-candidate-bridge" `
    -Phase "candidate-bridge" `
    -State ([string](Get-PropertyOrDefault -Object $ownerResultCandidateBridge -Name "validationState" -DefaultValue "blocked-owner-result-import-candidate-required")) `
    -ActionRequiredId $candidateBridgeId `
    -Detail ([string](Get-PropertyOrDefault -Object $candidateBridgeItem -Name "detail" -DefaultValue "owner result candidates still require strict real proof validator and promotion guard before any runtime or post-publish proof claim.")) `
    -RequiredInputs @(
      "readyForPromotionGuard=true owner result contract",
      "strict validator ready candidate",
      "package-consumer or post-publish lane classification",
      "real proof record validator pass",
      "promotion guard keeps candidate non-proof until strict record passes"
    ) `
    -OwnerCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofRecordCandidateFromOwnerResultImport.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalPublishProofGate.ps1 -Strict"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalPublishProofGate.ps1 -Strict"
    ) `
    -ForbiddenSubstitutes (@($commonForbidden) + @("candidate as proof", "ready contract as proof", "dashboard as proof")) `
    -SourceArtifact "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json" `
    -Boundary "Owner-result candidate bridge is strict-validator input only; candidates are not runtime proof, post-publish proof, public publish approval, or release close proof."
)

$actionIds = @($actionRequiredItems | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$expectedActionIds = @($realModelId, $packageId, $postPublishId, $finalOwnerTemplatePackId, $ownerResultImportId, $candidateBridgeId)
$missingActionIds = @($expectedActionIds | Where-Object { $actionIds -notcontains $_ })

$record = [ordered]@{
  recordKind = "final-owner-proof-action-worklist"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  worklistState = "blocked-final-owner-proof-action-required"
  sourceArtifacts = @(
    "artifacts/final-release/final-publish-proof-gate-report.json",
    "artifacts/final-release/release-proof-dashboard.json",
    "artifacts/final-release/final-release-close-blocker-dashboard.json",
    "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
    "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json",
    "artifacts/final-release/final-owner-real-input-template-pack-validation.json",
    "artifacts/final-release/clean-external-package-consumer-owner-runbook.json",
    "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json",
    "artifacts/final-release/post-publish-owner-verification-runbook.json",
    "artifacts/final-release/post-publish-owner-verification-runbook-validation.json",
    "artifacts/final-release/public-docs-package-metadata-gate.json"
  )
  finalGateState = [string](Get-PropertyOrDefault -Object $finalGate -Name "validationState" -DefaultValue "missing-final-publish-proof-gate-report")
  finalGateFailedBlockerCount = [int](Get-PropertyOrDefault -Object $finalGate -Name "failedBlockerCount" -DefaultValue -1)
  finalGateActionRequiredCount = [int](Get-PropertyOrDefault -Object $finalGate -Name "failedActionRequiredCount" -DefaultValue -1)
  releaseDashboardState = [string](Get-PropertyOrDefault -Object $releaseDashboard -Name "dashboardState" -DefaultValue "missing-release-proof-dashboard")
  releaseDashboardOwnerProofActionRequiredLaneCount = [int](Get-PropertyOrDefault -Object $releaseDashboard -Name "ownerProofActionRequiredLaneCount" -DefaultValue 0)
  finalCloseDashboardState = [string](Get-PropertyOrDefault -Object $finalCloseDashboard -Name "dashboardState" -DefaultValue "missing-final-release-close-blocker-dashboard")
  publicDocsGateState = [string](Get-PropertyOrDefault -Object $publicDocsGate -Name "gateState" -DefaultValue "missing-public-docs-package-metadata-gate")
  publicDocsBlockedMatchCount = [int](Get-PropertyOrDefault -Object $publicDocsGate -Name "blockedMatchCount" -DefaultValue -1)
  missingActionRequiredIds = @($missingActionIds)
  missingActionRequiredIdCount = @($missingActionIds).Count
  actionCount = $actions.Count
  blockedActionCount = @($actions | Where-Object { $_.blocked }).Count
  ownerActionLanes = @("real-model-runtime", "package-consumer-runtime", "post-publish-verification", "final-owner-real-input-template-pack", "public-owner-confirmation")
  ownerResultCandidateBridgeState = [string](Get-PropertyOrDefault -Object $ownerResultCandidateBridge -Name "validationState" -DefaultValue "missing-real-proof-record-candidate-from-owner-result-import-validation")
  ownerResultCandidateBridgeCandidateCount = [int](Get-PropertyOrDefault -Object $ownerResultCandidateBridge -Name "candidateCount" -DefaultValue 0)
  cleanExternalPackageConsumerOwnerRunbookState = $cleanExternalRunbookState
  cleanExternalPackageConsumerOwnerRunbookValidationState = $cleanExternalRunbookValidationState
  cleanExternalPackageConsumerOwnerRunbookStepCount = $cleanExternalRunbookStepCount
  cleanExternalPackageConsumerOwnerRunbookFailedBlockerCount = $cleanExternalRunbookFailedBlockerCount
  postPublishOwnerVerificationRunbookState = $postPublishRunbookState
  postPublishOwnerVerificationRunbookValidationState = $postPublishRunbookValidationState
  postPublishOwnerVerificationRunbookStepCount = $postPublishRunbookStepCount
  postPublishOwnerVerificationRunbookFailedBlockerCount = $postPublishRunbookFailedBlockerCount
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  actions = @($actions)
  nextOwnerOrder = @(
    "00-clean-external-package-consumer-owner-runbook",
    "01-real-model-runtime-owner-evidence",
    "02-package-consumer-runtime-clean-external-proof",
    "00-post-publish-owner-verification-runbook",
    "03-post-publish-verification-public-channel",
    "04-final-owner-real-input-template-pack",
    "05-owner-external-result-import-real-files",
    "06-owner-result-candidate-bridge-strict-promotion"
  )
  boundary = "Final owner proof action worklist is a one-screen handoff artifact only. It does not publish, does not promote runtime proof, does not verify post-publish proof, and cannot close the release issue. Candidates, dashboards, preflight output, templates, local feed, ProjectReference, direct .nupkg, build-only, dry-run, matrix, report, skipped run, and blocked-by-driver evidence remain non-proof."
}

$jsonPath = Join-Path $OutputRoot "final-owner-proof-action-worklist.json"
$markdownPath = Join-Path $OutputRoot "final-owner-proof-action-worklist.md"
$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($action in $actions) {
  "| ``$(ConvertTo-MarkdownCell $action.id)`` | ``$(ConvertTo-MarkdownCell $action.laneId)`` | ``$(ConvertTo-MarkdownCell $action.state)`` | ``$($action.requiredInputCount)`` | $(ConvertTo-MarkdownCell $action.boundary) |"
}

$commandSections = foreach ($action in $actions) {
  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("### $($action.id)")
  $lines.Add("")
  $lines.Add("- actionRequiredId: ``$($action.actionRequiredId)``")
  $lines.Add("- sourceArtifact: ``$($action.sourceArtifact)``")
  $lines.Add("- detail: $(ConvertTo-MarkdownCell $action.detail)")
  $lines.Add("- owner commands:")
  foreach ($command in $action.ownerCommands) {
    $lines.Add("  - ``$command``")
  }
  $lines.Add("- forbidden substitutes: ``$((@($action.forbiddenSubstitutes) | Select-Object -Unique) -join ', ')``")
  $lines -join "`r`n"
}

$markdown = @"
# Final Owner Proof Action Worklist

Generated at: ``$($record.generatedAtUtc)``

## Summary

- recordKind: ``$($record.recordKind)``
- worklistState: ``$($record.worklistState)``
- finalGateState: ``$($record.finalGateState)``
- finalGateFailedBlockerCount: ``$($record.finalGateFailedBlockerCount)``
- finalGateActionRequiredCount: ``$($record.finalGateActionRequiredCount)``
- releaseDashboardOwnerProofActionRequiredLaneCount: ``$($record.releaseDashboardOwnerProofActionRequiredLaneCount)``
- publicDocsBlockedMatchCount: ``$($record.publicDocsBlockedMatchCount)``
- actionCount: ``$($record.actionCount)``
- blockedActionCount: ``$($record.blockedActionCount)``
- performsPublish: ``False``
- canPromoteRuntimeProof: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Actions

| Action | Lane | State | Required Inputs | Boundary |
| --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Owner Commands

$($commandSections -join "`r`n`r`n")

## Boundary

$($record.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Final owner proof action worklist written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "WorklistState=$($record.worklistState) Actions=$($record.actionCount) Blocked=$($record.blockedActionCount) MissingActionRequiredIds=$($record.missingActionRequiredIdCount)"
