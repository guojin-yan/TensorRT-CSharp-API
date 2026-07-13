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

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-OverlayLine {
  param(
    [string]$Id,
    [string]$Title,
    [string]$TemplatePath,
    [string]$ValidationPath,
    [string]$ValidationState,
    [string[]]$AutoFields,
    [string[]]$OwnerRequiredFields,
    [string[]]$RequiredLogs,
    [string[]]$RequiredHashes,
    [string[]]$StrictValidators,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    templatePath = $TemplatePath
    validationPath = $ValidationPath
    validationState = $ValidationState
    autoFields = $AutoFields
    ownerRequiredFields = $OwnerRequiredFields
    requiredLogs = $RequiredLogs
    requiredHashes = $RequiredHashes
    strictValidators = $StrictValidators
    ownerActionStatus = "owner-action-required"
    canPromoteProof = $false
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
$finalEvidenceFreeze = Read-JsonOrNull "artifacts\final-release\final-evidence-freeze.json"
$finalEvidenceFreezeValidation = Read-JsonOrNull "artifacts\final-release\final-evidence-freeze-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$releaseCloseCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-candidate-validation.json"

$bundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$packageConsumerOwnerState = [string](Get-PropertyOrDefault -Object $packageConsumerOwnerInputValidation -Name "validationState" -DefaultValue "missing-package-consumer-runtime-proof-owner-input-validation")
$postPublishOwnerState = [string](Get-PropertyOrDefault -Object $postPublishOwnerInputValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-owner-input-validation")
$releaseCloseOwnerState = [string](Get-PropertyOrDefault -Object $releaseCloseOwnerInputValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-owner-input-validation")
$finalCloseDecisionState = [string](Get-PropertyOrDefault -Object $finalCloseDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-final-close-decision-validation")
$finalEvidenceFreezeState = [string](Get-PropertyOrDefault -Object $finalEvidenceFreeze -Name "freezeState" -DefaultValue "missing-final-evidence-freeze")
$finalEvidenceFreezeValidationState = [string](Get-PropertyOrDefault -Object $finalEvidenceFreezeValidation -Name "validationState" -DefaultValue "missing-final-evidence-freeze-validation")
$postPublishValidationState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$releaseCloseCandidateValidationState = [string](Get-PropertyOrDefault -Object $releaseCloseCandidateValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-candidate-validation")

$overlayLines = @(
  (New-OverlayLine -Id "package-consumer-runtime-proof" -Title "Package consumer runtime proof owner input" -TemplatePath "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json" -ValidationPath "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json" -ValidationState $packageConsumerOwnerState -AutoFields @("runtimePackageKey", "managedPackageId", "runtimePackageId", "strictValidationCommand") -OwnerRequiredFields @("publicPackageSource", "cleanConsumerRoot", "consumerProjectPath", "managedNupkgSha256", "runtimeNupkgSha256", "restoreLogPath", "buildLogPath", "smokeLogPath", "smokeLogSha256", "runtimeSmokeExitCode") -RequiredLogs @("restore log", "build log", "runtime smoke log", "native asset listing") -RequiredHashes @("managed nupkg SHA256", "runtime nupkg SHA256", "restore log SHA256", "build log SHA256", "smoke log SHA256") -StrictValidators @("Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict", "Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof") -Boundary "Requires clean external consumer and public package source; local feed, ProjectReference, direct .nupkg, dependency-probe-only, and blocked-by-cuda-driver are not proof.")
  (New-OverlayLine -Id "post-publish-verification" -Title "Post-publish verification owner input" -TemplatePath "artifacts/final-release/post-publish-verification-owner-input.template.json" -ValidationPath "artifacts/final-release/post-publish-verification-owner-input-validation.json" -ValidationState $postPublishOwnerState -AutoFields @("runtimePackageKey", "expectedRuntimePackageKey", "managedPackageId", "runtimePackageId", "smokeCommand") -OwnerRequiredFields @("selectedChannel", "channelSourceUri", "publishedPackageUrl", "managedNupkgSha256", "runtimeNupkgSha256", "cleanConsumerRoot", "consumerProjectPath", "restoreLogPath", "runtimeSmokeLogPath", "runtimeSmokeExitCode") -RequiredLogs @("restore log", "native asset listing", "dependency probe log", "runtime smoke log") -RequiredHashes @("managed nupkg SHA256", "runtime nupkg SHA256", "restore log SHA256", "native asset listing SHA256", "runtime smoke log SHA256") -StrictValidators @("Test-PostPublishVerificationOwnerInput.ps1 -Strict", "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof") -Boundary "Requires real post-publish public channel package identity and clean external consumer smoke.")
  (New-OverlayLine -Id "release-close-owner-input" -Title "Release issue close record owner input" -TemplatePath "artifacts/final-release/release-issue-close-record-owner-input.template.json" -ValidationPath "artifacts/final-release/release-issue-close-record-owner-input-validation.json" -ValidationState $releaseCloseOwnerState -AutoFields @("releaseEvidenceBundlePath", "postPublishProofValidationPath", "strictCloseValidatorCommand") -OwnerRequiredFields @("postPublishProofValidationState", "rollbackPlan", "rollbackOwner", "rollbackTrigger", "ownerFinalCloseDecision", "releaseIssueId", "releaseIssueUrl") -RequiredLogs @("post-publish validation log", "release close validator log") -RequiredHashes @("release evidence bundle SHA256", "release close preflight SHA256", "stale claims audit SHA256", "post-publish validation SHA256") -StrictValidators @("Test-ReleaseIssueCloseRecordOwnerInput.ps1 -Strict", "Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady") -Boundary "Requires real post-publish proof validation, rollback approval, and final owner decision.")
  (New-OverlayLine -Id "release-issue-final-close-decision" -Title "Release issue final close decision" -TemplatePath "artifacts/final-release/release-issue-final-close-decision.template.json" -ValidationPath "artifacts/final-release/release-issue-final-close-decision-validation.json" -ValidationState $finalCloseDecisionState -AutoFields @("finalEvidenceFreezePath", "releaseEvidenceBundlePath", "postPublishVerificationValidationPath", "releaseIssueCloseRecordCandidateValidationPath") -OwnerRequiredFields @("ownerName", "ownerDecisionTimestampUtc", "releaseIssueId", "releaseIssueUrl", "ownerFinalCloseDecision", "rollbackPlanReviewed", "confirmsRealPostPublishProof", "confirmsRuntimeSmokePassed", "confirmsLogsAndSha256Reviewed") -RequiredLogs @("final close decision review log", "strict close validator log") -RequiredHashes @("final evidence freeze SHA256", "final evidence freeze validation SHA256", "release evidence bundle SHA256", "post-publish validation SHA256", "release close candidate validation SHA256") -StrictValidators @("Test-ReleaseIssueFinalCloseDecision.ps1 -Strict", "Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady") -Boundary "Final decision template is owner input only and cannot close release issue by itself.")
)

$missingOwnerInputFields = @($overlayLines | ForEach-Object { $_.ownerRequiredFields } | Select-Object -Unique)
$sourceArtifacts = @(
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/final-evidence-freeze.json",
  "artifacts/final-release/final-evidence-freeze-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
  "artifacts/final-release/post-publish-verification-owner-input.template.json",
  "artifacts/final-release/post-publish-verification-owner-input-validation.json",
  "artifacts/final-release/post-publish-verification-validation.json",
  "artifacts/final-release/release-issue-close-record-owner-input.template.json",
  "artifacts/final-release/release-issue-close-record-owner-input-validation.json",
  "artifacts/final-release/release-issue-close-record-candidate-validation.json",
  "artifacts/final-release/release-issue-final-close-decision.template.json",
  "artifacts/final-release/release-issue-final-close-decision-validation.json"
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "real-external-proof-overlay-pack"
  overlayState = "blocked-real-owner-input-required"
  runtimePackageKey = $RuntimePackageKey
  releaseEvidenceBundleState = $bundleState
  finalEvidenceFreezeState = $finalEvidenceFreezeState
  finalEvidenceFreezeValidationState = $finalEvidenceFreezeValidationState
  postPublishVerificationValidationState = $postPublishValidationState
  releaseIssueCloseCandidateValidationState = $releaseCloseCandidateValidationState
  releaseIssueFinalCloseDecisionValidationState = $finalCloseDecisionState
  overlayLineCount = $overlayLines.Count
  missingOwnerInputFieldCount = $missingOwnerInputFields.Count
  overlayLines = $overlayLines
  missingOwnerInputFields = $missingOwnerInputFields
  requiredGlobalRules = @(
    "clean consumer root must be outside repository",
    "public package source must not be local feed",
    "ProjectReference is forbidden",
    "direct .nupkg reference is forbidden",
    "runtime smoke exit code must be 0",
    "all referenced logs must exist and match SHA256",
    "blocked-by-cuda-driver is not smoke passed"
  )
  sourceArtifacts = $sourceArtifacts
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Overlay pack is owner input guidance only. It cannot publish packages, promote proof, or close release issue."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "real-external-proof-overlay-pack.json"
$markdownPath = Join-Path $artifactRoot "real-external-proof-overlay-pack.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lineRows = $overlayLines | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.validationState)`` | $($_.ownerRequiredFields.Count) | $($_.boundary.Replace("|", "\|")) |"
}
$fieldLines = $missingOwnerInputFields | ForEach-Object { "- ``$_``" }
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }
$ruleLines = $record.requiredGlobalRules | ForEach-Object { "- $_" }

$markdown = @"
# Real External Proof Overlay Pack

生成时间：$($record.generatedAtUtc)

该 overlay pack 把 package-consumer-runtime、post-publish verification、release close owner input 和 final close decision 的真实 Owner 回填字段集中到一处。它不执行发布、不上传包、不关闭 release issue，也不会把 template/candidate/guidance 晋级为 proof。

| 项目 | 当前值 |
|---|---|
| overlayState | ``$($record.overlayState)`` |
| runtimePackageKey | ``$RuntimePackageKey`` |
| releaseEvidenceBundleState | ``$bundleState`` |
| finalEvidenceFreezeValidationState | ``$finalEvidenceFreezeValidationState`` |
| postPublishVerificationValidationState | ``$postPublishValidationState`` |
| releaseIssueFinalCloseDecisionValidationState | ``$finalCloseDecisionState`` |
| overlayLineCount | ``$($record.overlayLineCount)`` |
| missingOwnerInputFieldCount | ``$($record.missingOwnerInputFieldCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Overlay Lines

| ID | Validation State | Owner Field Count | Boundary |
|---|---|---:|---|
$($lineRows -join "`r`n")

## Owner Required Fields

$($fieldLines -join "`r`n")

## Global Rules

$($ruleLines -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real external proof overlay pack written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "OverlayState=$($record.overlayState) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"
