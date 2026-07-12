[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
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

function New-FreezeArtifact {
  param(
    [string]$Id,
    [string]$RelativePath,
    [string]$State,
    [string]$Boundary
  )

  $path = Join-Path $RepositoryRoot $RelativePath
  $exists = Test-Path -LiteralPath $path -PathType Leaf
  $sha256 = if ($exists) {
    (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  else {
    ""
  }

  [pscustomobject]@{
    id = $Id
    path = $RelativePath
    exists = $exists
    sha256 = $sha256
    state = $State
    boundary = $Boundary
  }
}

$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$postPublishOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-owner-input-validation.json"
$postPublishVerificationValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$releaseIssueCloseOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-owner-input-validation.json"
$releaseIssueCloseCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-candidate-validation.json"
$ownerExecutionPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$ownerExecutionPackageValidation = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package-validation.json"
$releaseIssueFinalCloseDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-final-close-decision-validation.json"

$bundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$bundleCanPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canPublishPublicly" -DefaultValue $false)
$bundleCanClose = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canCloseReleaseIssue" -DefaultValue $false)
$bundlePerformsPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "performsPublish" -DefaultValue $false)
$postPublishOwnerState = [string](Get-PropertyOrDefault -Object $postPublishOwnerInputValidation -Name "validationState" -DefaultValue "missing-post-publish-owner-input-validation")
$postPublishValidationState = [string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$releaseCloseOwnerState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseOwnerInputValidation -Name "validationState" -DefaultValue "missing-release-close-owner-input-validation")
$releaseCloseCandidateState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseCandidateValidation -Name "validationState" -DefaultValue "missing-release-close-candidate-validation")
$ownerExecutionState = [string](Get-PropertyOrDefault -Object $ownerExecutionPackage -Name "packageState" -DefaultValue "missing-owner-release-execution-package")
$ownerExecutionValidationState = [string](Get-PropertyOrDefault -Object $ownerExecutionPackageValidation -Name "validationState" -DefaultValue "missing-owner-release-execution-package-validation")
$finalCloseDecisionState = [string](Get-PropertyOrDefault -Object $releaseIssueFinalCloseDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-final-close-decision-validation")

$freezeArtifacts = @(
  (New-FreezeArtifact -Id "release-evidence-bundle" -RelativePath "artifacts/final-release/release-evidence-bundle.json" -State $bundleState -Boundary "Evidence bundle aggregation is not owner approval or publish proof.")
  (New-FreezeArtifact -Id "post-publish-owner-input-validation" -RelativePath "artifacts/final-release/post-publish-verification-owner-input-validation.json" -State $postPublishOwnerState -Boundary "Owner input validation is not post-publish proof.")
  (New-FreezeArtifact -Id "post-publish-verification-validation" -RelativePath "artifacts/final-release/post-publish-verification-validation.json" -State $postPublishValidationState -Boundary "Post-publish validation must remain incomplete until real public-channel smoke proof exists.")
  (New-FreezeArtifact -Id "release-close-owner-input-validation" -RelativePath "artifacts/final-release/release-issue-close-record-owner-input-validation.json" -State $releaseCloseOwnerState -Boundary "Close owner input is not close approval.")
  (New-FreezeArtifact -Id "release-close-candidate-validation" -RelativePath "artifacts/final-release/release-issue-close-record-candidate-validation.json" -State $releaseCloseCandidateState -Boundary "Close candidate is an input surface only.")
  (New-FreezeArtifact -Id "owner-release-execution-package" -RelativePath "artifacts/final-release/owner-release-execution-package.json" -State $ownerExecutionState -Boundary "Owner execution package is guidance only.")
  (New-FreezeArtifact -Id "owner-release-execution-package-validation" -RelativePath "artifacts/final-release/owner-release-execution-package-validation.json" -State $ownerExecutionValidationState -Boundary "Execution package validation is guidance validation only.")
  (New-FreezeArtifact -Id "release-issue-final-close-decision-validation" -RelativePath "artifacts/final-release/release-issue-final-close-decision-validation.json" -State $finalCloseDecisionState -Boundary "Final close decision template validation cannot close the issue.")
)

$missingArtifacts = @($freezeArtifacts | Where-Object { -not $_.exists })
$missingHashes = @($freezeArtifacts | Where-Object { [string]::IsNullOrWhiteSpace($_.sha256) })
$freezeState = if ($missingArtifacts.Count -eq 0 -and $missingHashes.Count -eq 0 -and -not $bundleCanPublish -and -not $bundleCanClose -and -not $bundlePerformsPublish) {
  "blocked-evidence-frozen-owner-action-required"
}
else {
  "blocked-freeze-source-artifact-required"
}

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "final-evidence-freeze"
  freezeState = $freezeState
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  releaseEvidenceBundleState = $bundleState
  postPublishOwnerInputValidationState = $postPublishOwnerState
  postPublishVerificationValidationState = $postPublishValidationState
  releaseIssueCloseOwnerInputValidationState = $releaseCloseOwnerState
  releaseIssueCloseCandidateValidationState = $releaseCloseCandidateState
  ownerReleaseExecutionPackageState = $ownerExecutionState
  ownerReleaseExecutionPackageValidationState = $ownerExecutionValidationState
  releaseIssueFinalCloseDecisionValidationState = $finalCloseDecisionState
  sourceArtifactCount = $freezeArtifacts.Count
  missingSourceArtifactCount = $missingArtifacts.Count
  missingHashCount = $missingHashes.Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  bundleCanPublishPublicly = $bundleCanPublish
  bundleCanCloseReleaseIssue = $bundleCanClose
  bundlePerformsPublish = $bundlePerformsPublish
  freezeArtifacts = $freezeArtifacts
  remainingBlockers = @(
    "real public package source restore/build/run is still owner action",
    "managed/runtime nupkg SHA256 values still require real public-channel owner input",
    "clean external consumer smoke must pass on compatible CUDA/TensorRT host",
    "post-publish verification remains incomplete until strict proof validation passes",
    "release issue close candidate remains blocked until rollback approval and final owner close decision are real"
  )
  safetyBoundary = "Final evidence freeze records hashes and states only. It does not publish packages, approve publication, or close the release issue."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "final-evidence-freeze.json"
$markdownPath = Join-Path $artifactRoot "final-evidence-freeze.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$artifactRows = $freezeArtifacts | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.exists)`` | ``$($_.state)`` | ``$($_.sha256)`` | $($_.boundary.Replace("|", "\|")) |"
}
$blockerLines = $record.remainingBlockers | ForEach-Object { "- $_" }

$markdown = @"
# Final Evidence Freeze

生成时间：$($record.generatedAtUtc)

该 freeze 只冻结当前 release evidence 与 close proof 相关 artifact 的路径、状态和 SHA256。它不执行发布、不上传包、不关闭 release issue。

| 项目 | 当前值 |
|---|---|
| freezeState | ``$($record.freezeState)`` |
| releaseEvidenceBundleState | ``$($record.releaseEvidenceBundleState)`` |
| postPublishVerificationValidationState | ``$($record.postPublishVerificationValidationState)`` |
| releaseIssueCloseCandidateValidationState | ``$($record.releaseIssueCloseCandidateValidationState)`` |
| ownerReleaseExecutionPackageValidationState | ``$($record.ownerReleaseExecutionPackageValidationState)`` |
| releaseIssueFinalCloseDecisionValidationState | ``$($record.releaseIssueFinalCloseDecisionValidationState)`` |
| sourceArtifactCount | ``$($record.sourceArtifactCount)`` |
| missingSourceArtifactCount | ``$($record.missingSourceArtifactCount)`` |
| missingHashCount | ``$($record.missingHashCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Freeze Artifacts

| ID | Exists | State | SHA256 | Boundary |
|---|---:|---|---|---|
$($artifactRows -join "`r`n")

## Remaining Blockers

$($blockerLines -join "`r`n")

## Safety Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "Final evidence freeze written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "FreezeState=$freezeState"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
