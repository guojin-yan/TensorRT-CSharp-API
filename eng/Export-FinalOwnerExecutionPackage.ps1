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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-ResultArtifactForAction {
  param([string]$ActionId)

  switch ($ActionId) {
    "00-clean-external-package-consumer-owner-runbook" {
      return @(
        "artifacts/final-release/clean-external-package-consumer-owner-runbook.json",
        "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json",
        "artifacts/final-release/owner-external-proof-execution-result.input.json"
      )
    }
    "00-post-publish-owner-verification-runbook" {
      return @(
        "artifacts/final-release/post-publish-owner-verification-runbook.json",
        "artifacts/final-release/post-publish-owner-verification-runbook-validation.json",
        "artifacts/final-release/owner-external-proof-execution-result.input.json"
      )
    }
    "01-real-model-runtime-owner-evidence" {
      return @(
        "artifacts/user-acceptance/sample-run-evidence-record.json",
        "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
        "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json"
      )
    }
    "02-package-consumer-runtime-clean-external-proof" {
      return @(
        "artifacts/final-release/package-consumer-runtime-proof-record.json",
        "artifacts/final-release/package-consumer-runtime-proof-record-validation.json",
        "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json"
      )
    }
    "03-post-publish-verification-public-channel" {
      return @(
        "artifacts/final-release/post-publish-verification-record.json",
        "artifacts/final-release/post-publish-verification-validation.json",
        "artifacts/final-release/post-publish-verification-owner-input-validation.json"
      )
    }
    "04-final-owner-real-input-template-pack" {
      return @(
        "artifacts/final-release/final-owner-real-input-template-pack.json",
        "artifacts/final-release/final-owner-real-input-template-pack-validation.json",
        "artifacts/final-release/owner-real-inputs"
      )
    }
    "05-owner-external-result-import-real-files" {
      return @(
        "artifacts/final-release/owner-external-proof-execution-result-import.json",
        "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
        "artifacts/final-release/real-external-proof-record-import-validator.json"
      )
    }
    "06-owner-result-candidate-bridge-strict-promotion" {
      return @(
        "artifacts/final-release/real-proof-record-candidate-from-owner-result-import.json",
        "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json",
        "artifacts/final-release/final-publish-proof-gate-report.json"
      )
    }
    default {
      return @()
    }
  }
}

function Get-InputContractForAction {
  param([object]$Action)

  $sourceArtifact = [string](Get-PropertyOrDefault -Object $Action -Name "sourceArtifact" -DefaultValue "")
  $resultArtifacts = @(Get-ResultArtifactForAction -ActionId ([string](Get-PropertyOrDefault -Object $Action -Name "id" -DefaultValue "")))

  [pscustomobject]@{
    sourceArtifact = $sourceArtifact
    expectedResultArtifacts = @($resultArtifacts)
    requiredFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $Action -Name "requiredInputs" -DefaultValue @())
    ownerMustProvideRealFiles = $true
    requiresExistingLog = $true
    requiresSha256 = $true
    requiresExitCodeZeroWhenRuntimeProof = $true
    requiresOwnerReview = $true
    acceptsTemplateOnly = $false
    acceptsCandidateAsProof = $false
    acceptsDashboardAsProof = $false
  }
}

function New-ExecutionStepFromAction {
  param([object]$Action, [int]$Order)

  $id = [string](Get-PropertyOrDefault -Object $Action -Name "id" -DefaultValue "")
  $laneId = [string](Get-PropertyOrDefault -Object $Action -Name "laneId" -DefaultValue "")
  $ownerCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $Action -Name "ownerCommands" -DefaultValue @())
  $validatorCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $Action -Name "validatorCommands" -DefaultValue @())
  $forbiddenSubstitutes = Convert-ToStringArray (Get-PropertyOrDefault -Object $Action -Name "forbiddenSubstitutes" -DefaultValue @())
  $inputContract = Get-InputContractForAction -Action $Action

  [pscustomobject]@{
    order = $Order
    id = $id
    laneId = $laneId
    phase = [string](Get-PropertyOrDefault -Object $Action -Name "phase" -DefaultValue $laneId)
    state = [string](Get-PropertyOrDefault -Object $Action -Name "state" -DefaultValue "blocked-owner-action-required")
    blocked = $true
    actionRequiredId = [string](Get-PropertyOrDefault -Object $Action -Name "actionRequiredId" -DefaultValue "")
    detail = [string](Get-PropertyOrDefault -Object $Action -Name "detail" -DefaultValue "")
    inputContract = $inputContract
    requiredInputs = Convert-ToStringArray (Get-PropertyOrDefault -Object $Action -Name "requiredInputs" -DefaultValue @())
    requiredInputCount = [int](Get-PropertyOrDefault -Object $Action -Name "requiredInputCount" -DefaultValue 0)
    ownerCommands = @($ownerCommands)
    ownerCommandCount = @($ownerCommands).Count
    validatorCommands = @($validatorCommands)
    validatorCommandCount = @($validatorCommands).Count
    expectedResultArtifacts = @($inputContract.expectedResultArtifacts)
    expectedResultArtifactCount = @($inputContract.expectedResultArtifacts).Count
    forbiddenSubstitutes = @($forbiddenSubstitutes)
    forbiddenSubstituteCount = @($forbiddenSubstitutes).Count
    promotionBoundary = [string](Get-PropertyOrDefault -Object $Action -Name "boundary" -DefaultValue "")
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner execution step only. It prepares real owner inputs, result artifacts, and validators; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$worklist = Read-JsonOrNull "artifacts\final-release\final-owner-proof-action-worklist.json"
$worklistValidation = Read-JsonOrNull "artifacts\final-release\final-owner-proof-action-worklist-validation.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$readinessPack = Read-JsonOrNull "artifacts\final-release\release-publish-readiness-evidence-pack.json"
$ownerRuntimeSmokeFieldAlignment = Read-JsonOrNull "artifacts\final-release\package-consumer-owner-runtime-smoke-field-alignment.json"
$ownerRuntimeSmokeFieldAlignmentValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-owner-runtime-smoke-field-alignment-validation.json"
$ownerRuntimeSmokeFieldAlignmentState = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "alignmentState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment")
$ownerRuntimeSmokeFieldAlignmentValidationState = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignmentValidation -Name "validationState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment-validation")
$ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "runtimeSmokeStatus" -DefaultValue "Smoke=missing")
$ownerRuntimeSmokeFieldAlignmentFieldCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "fieldCount" -DefaultValue 0)
$ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "missingRequiredFieldCount" -DefaultValue -1)
$ownerRuntimeSmokeFieldAlignmentFailedBlockerCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignmentValidation -Name "failedBlockerCount" -DefaultValue -1)

$actions = @()
if ($null -ne $worklist) {
  $actions = @((Get-PropertyOrDefault -Object $worklist -Name "actions" -DefaultValue @()))
}

$executionSteps = @()
for ($i = 0; $i -lt $actions.Count; $i++) {
  $executionSteps += New-ExecutionStepFromAction -Action $actions[$i] -Order ($i + 1)
}

$allForbiddenSubstitutes = @($executionSteps | ForEach-Object { $_.forbiddenSubstitutes } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ } | Select-Object -Unique)
$allValidatorCommands = @($executionSteps | ForEach-Object { $_.validatorCommands } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ } | Select-Object -Unique)
$allOwnerCommands = @($executionSteps | ForEach-Object { $_.ownerCommands } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ } | Select-Object -Unique)
$allResultArtifacts = @($executionSteps | ForEach-Object { $_.expectedResultArtifacts } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ } | Select-Object -Unique)

$record = [ordered]@{
  recordKind = "final-owner-execution-package"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  packageState = "blocked-final-owner-execution-required"
  sourceWorklistState = [string](Get-PropertyOrDefault -Object $worklist -Name "worklistState" -DefaultValue "missing-final-owner-proof-action-worklist")
  sourceWorklistValidationState = [string](Get-PropertyOrDefault -Object $worklistValidation -Name "validationState" -DefaultValue "missing-final-owner-proof-action-worklist-validation")
  releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  releasePublishReadinessEvidencePackState = [string](Get-PropertyOrDefault -Object $readinessPack -Name "packState" -DefaultValue "missing-release-publish-readiness-evidence-pack")
  packageConsumerOwnerRuntimeSmokeFieldAlignmentState = $ownerRuntimeSmokeFieldAlignmentState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState = $ownerRuntimeSmokeFieldAlignmentValidationState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = $ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount = $ownerRuntimeSmokeFieldAlignmentFieldCount
  packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = $ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount = $ownerRuntimeSmokeFieldAlignmentFailedBlockerCount
  actionCount = $actions.Count
  executionStepCount = $executionSteps.Count
  blockedExecutionStepCount = @($executionSteps | Where-Object { $_.blocked }).Count
  actionRequiredIds = @($executionSteps | ForEach-Object { $_.actionRequiredId })
  executionSteps = @($executionSteps)
  ownerCommandSequence = @($allOwnerCommands)
  validatorCommands = @($allValidatorCommands)
  expectedResultArtifacts = @($allResultArtifacts)
  forbiddenSubstitutes = @($allForbiddenSubstitutes)
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isPackagePush = $false
  requiresOwnerManualPublish = $true
  requiresRealExternalProof = $true
  requiresPostPublishProof = $true
  requiresReleaseCloseOwnerDecision = $true
  sourceArtifacts = @(
    "artifacts/final-release/final-owner-proof-action-worklist.json",
    "artifacts/final-release/final-owner-proof-action-worklist-validation.json",
    "artifacts/final-release/clean-external-package-consumer-owner-runbook.json",
    "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json",
    "artifacts/final-release/post-publish-owner-verification-runbook.json",
    "artifacts/final-release/post-publish-owner-verification-runbook-validation.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-publish-readiness-evidence-pack.json",
    "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json",
    "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json"
  )
  boundary = "Final owner execution package maps two owner runbook preflight items and five final owner actions to commands, input contracts, expected result artifacts, and validators. It is owner handoff only; it does not publish, does not promote runtime proof, does not verify post-publish proof, cannot close the release issue, and is not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-package.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-package.md"
$record | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($step in $executionSteps) {
  "| ``$(ConvertTo-MarkdownCell $step.id)`` | ``$(ConvertTo-MarkdownCell $step.laneId)`` | ``$(ConvertTo-MarkdownCell $step.state)`` | ``$($step.requiredInputCount)`` | ``$($step.expectedResultArtifactCount)`` | $(ConvertTo-MarkdownCell $step.promotionBoundary) |"
}

$sections = foreach ($step in $executionSteps) {
  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("### $($step.id)")
  $lines.Add("")
  $lines.Add("- actionRequiredId: ``$($step.actionRequiredId)``")
  $lines.Add("- sourceArtifact: ``$($step.inputContract.sourceArtifact)``")
  $lines.Add("- expected result artifacts: ``$($step.expectedResultArtifacts -join ', ')``")
  $lines.Add("- owner commands:")
  foreach ($command in $step.ownerCommands) {
    $lines.Add("  - ``$command``")
  }
  $lines.Add("- validator commands:")
  foreach ($command in $step.validatorCommands) {
    $lines.Add("  - ``$command``")
  }
  $lines.Add("- forbidden substitutes: ``$($step.forbiddenSubstitutes -join ', ')``")
  $lines.Add("- boundary: $(ConvertTo-MarkdownCell $step.promotionBoundary)")
  $lines -join "`r`n"
}

$markdown = @"
# Final Owner Execution Package

Generated at: ``$($record.generatedAtUtc)``

## Summary

- recordKind: ``$($record.recordKind)``
- packageState: ``$($record.packageState)``
- sourceWorklistState: ``$($record.sourceWorklistState)``
- sourceWorklistValidationState: ``$($record.sourceWorklistValidationState)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentState: ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentState)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState: ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus: ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount: ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount: ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount)``
- packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount: ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount)``
- executionStepCount: ``$($record.executionStepCount)``
- blockedExecutionStepCount: ``$($record.blockedExecutionStepCount)``
- performsPublish: ``False``
- canPromoteRuntimeProof: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Execution Steps

| Action | Lane | State | Required Inputs | Result Artifacts | Boundary |
| --- | --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Owner Step Details

$($sections -join "`r`n`r`n")

## Boundary

$($record.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Final owner execution package written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackageState=$($record.packageState) ExecutionSteps=$($record.executionStepCount) Blocked=$($record.blockedExecutionStepCount)"
