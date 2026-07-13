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

function Test-IsMissingRealInput {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text.StartsWith("<", [StringComparison]::Ordinal)
}

function Get-ReleaseCloseField {
  param([string]$Id)

  switch ($Id) {
    "rollback-plan" { "rollbackPlan" }
    "rollback-owner" { "rollbackOwner" }
    "rollback-trigger" { "rollbackTrigger" }
    "owner-final-close-decision" { "ownerFinalCloseDecision" }
    "release-issue-id" { "releaseIssueId" }
    "release-issue-url" { "releaseIssueUrl" }
    "public-channel-package-source" { "publicChannelPackageSource" }
    "clean-consumer-runtime-smoke-log" { "cleanConsumerRuntimeSmokeLog" }
    default { $Id.Replace("-", "") }
  }
}

function Get-TargetArtifact {
  param([string]$Id, [string]$TaskTargetArtifact)

  switch ($Id) {
    "rollback-plan" { "artifacts/final-release/release-issue-close-record-owner-input.template.json" }
    "rollback-owner" { "artifacts/final-release/release-issue-close-record-owner-input.template.json" }
    "rollback-trigger" { "artifacts/final-release/release-issue-close-record-owner-input.template.json" }
    "owner-final-close-decision" { "artifacts/final-release/release-issue-close-record-owner-input.template.json" }
    "release-issue-id" { "artifacts/final-release/release-issue-close-record-owner-input.template.json" }
    "release-issue-url" { "artifacts/final-release/release-issue-close-record-owner-input.template.json" }
    "public-channel-package-source" { "artifacts/final-release/post-publish-verification-record.input-draft.json" }
    "clean-consumer-runtime-smoke-log" { "artifacts/final-release/package-consumer-runtime-proof-candidate.json" }
    default {
      if ([string]::IsNullOrWhiteSpace($TaskTargetArtifact)) {
        "artifacts/final-release/release-issue-close-record-owner-input.template.json"
      }
      else {
        $TaskTargetArtifact
      }
    }
  }
}

function Get-TargetField {
  param([string]$Id)

  switch ($Id) {
    "rollback-plan" { "rollbackPlan" }
    "rollback-owner" { "rollbackOwner" }
    "rollback-trigger" { "rollbackTrigger" }
    "owner-final-close-decision" { "ownerFinalCloseDecision" }
    "release-issue-id" { "releaseIssueId" }
    "release-issue-url" { "releaseIssueUrl" }
    "public-channel-package-source" { "packageSource" }
    "clean-consumer-runtime-smoke-log" { "smokeLogSha256" }
    default { $Id }
  }
}

function Get-RequiredEvidence {
  param([string]$Id)

  switch ($Id) {
    "rollback-plan" { "Owner-reviewed rollback plan, including trigger, owner, and recovery command/reference." }
    "rollback-owner" { "Named rollback owner accountable after release issue close." }
    "rollback-trigger" { "Concrete rollback trigger that can be evaluated after public release." }
    "owner-final-close-decision" { "Explicit owner approval to close only after real post-publish and runtime proof pass." }
    "release-issue-id" { "Real release issue identifier from the public tracker or release governance system." }
    "release-issue-url" { "Real release issue URL matching the release issue id." }
    "public-channel-package-source" { "Public package source/channel URI, not local feed, direct nupkg, or ProjectReference." }
    "clean-consumer-runtime-smoke-log" { "Clean consumer runtime smoke log from outside the repository with matching SHA256 and runtime package key." }
    default { "Real owner-filled release close input." }
  }
}

function Get-FirstCommand {
  param([string]$Id, [string]$TaskFirstCommand)

  switch ($Id) {
    "rollback-plan" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOwnerInput.ps1" }
    "rollback-owner" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOwnerInput.ps1" }
    "rollback-trigger" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOwnerInput.ps1" }
    "owner-final-close-decision" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueFinalCloseDecisionTemplate.ps1" }
    "release-issue-id" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOwnerInput.ps1" }
    "release-issue-url" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOwnerInput.ps1" }
    "public-channel-package-source" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecord.ps1" }
    "clean-consumer-runtime-smoke-log" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofCandidate.ps1" }
    default {
      if ([string]::IsNullOrWhiteSpace($TaskFirstCommand)) {
        "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOwnerInput.ps1"
      }
      else {
        $TaskFirstCommand
      }
    }
  }
}

function New-RealInputMapping {
  param(
    [object]$Task,
    [string]$StrictCloseValidatorCommand,
    [string[]]$NonSubstituteKinds
  )

  $id = [string](Get-PropertyOrDefault -Object $Task -Name "id" -DefaultValue "unknown-owner-input")
  $currentValue = [string](Get-PropertyOrDefault -Object $Task -Name "currentValue" -DefaultValue "")
  $taskTargetArtifact = [string](Get-PropertyOrDefault -Object $Task -Name "targetArtifact" -DefaultValue "")
  $taskFirstCommand = [string](Get-PropertyOrDefault -Object $Task -Name "firstCommand" -DefaultValue "")
  $missing = Test-IsMissingRealInput -Value $currentValue

  [pscustomobject]@{
    id = "map-$id"
    sourceTaskId = $id
    currentValue = $currentValue
    targetArtifact = Get-TargetArtifact -Id $id -TaskTargetArtifact $taskTargetArtifact
    targetField = Get-TargetField -Id $id
    releaseCloseField = Get-ReleaseCloseField -Id $id
    requiredEvidence = Get-RequiredEvidence -Id $id
    firstCommand = Get-FirstCommand -Id $id -TaskFirstCommand $taskFirstCommand
    validatorCommand = [string](Get-PropertyOrDefault -Object $Task -Name "validatorCommand" -DefaultValue $StrictCloseValidatorCommand)
    strictCloseValidatorCommand = $StrictCloseValidatorCommand
    nonSubstituteKinds = $NonSubstituteKinds
    missingRealInput = $missing
    mappingState = if ($missing) { "blocked-real-input-required" } else { "real-input-present-needs-strict-validation" }
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$ownerProofPack = Read-JsonOrNull "artifacts\final-release\owner-proof-real-backfill-execution-pack.json"
$ownerProofPackValidation = Read-JsonOrNull "artifacts\final-release\owner-proof-real-backfill-execution-pack-validation.json"
$strictCandidate = Read-JsonOrNull "artifacts\final-release\release-close-strict-record-candidate.json"
$ownerInputTemplate = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-owner-input.template.json"
$finalCloseDecisionTemplate = Read-JsonOrNull "artifacts\final-release\release-issue-final-close-decision.template.json"

if ($null -eq $ownerProofPack) {
  throw "Missing owner proof real backfill execution pack. Run Export-OwnerProofRealBackfillExecutionPack.ps1 first."
}

$strictCloseValidatorCommand = [string](Get-PropertyOrDefault -Object $ownerInputTemplate -Name "strictCloseValidatorCommand" -DefaultValue "")
if ([string]::IsNullOrWhiteSpace($strictCloseValidatorCommand)) {
  $strictCloseValidatorCommand = [string](Get-PropertyOrDefault -Object $finalCloseDecisionTemplate -Name "strictCloseValidatorCommand" -DefaultValue "")
}
if ([string]::IsNullOrWhiteSpace($strictCloseValidatorCommand)) {
  $strictCloseValidatorCommand = [string](Get-PropertyOrDefault -Object $strictCandidate -Name "strictValidatorCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady")
}

$nonSubstituteKinds = @(Get-PropertyOrDefault -Object $ownerProofPack -Name "nonSubstituteProofKinds" -DefaultValue @(
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
))

$ownerInputTasks = @(Get-PropertyOrDefault -Object $ownerProofPack -Name "ownerInputTasks" -DefaultValue @())
$realInputMappings = @($ownerInputTasks | ForEach-Object { New-RealInputMapping -Task $_ -StrictCloseValidatorCommand $strictCloseValidatorCommand -NonSubstituteKinds $nonSubstituteKinds })
$missingRealInputCount = @($realInputMappings | Where-Object { [bool]$_.missingRealInput }).Count
$blockedMappingCount = @($realInputMappings | Where-Object { ([string]$_.mappingState).StartsWith("blocked", [StringComparison]::Ordinal) }).Count
$closeRecordTargetArtifacts = @($realInputMappings | ForEach-Object { $_.targetArtifact } | Sort-Object -Unique)
$firstCommands = @($realInputMappings | ForEach-Object { $_.firstCommand } | Sort-Object -Unique)
$strictValidators = @(
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordRealInputMap.ps1 -Strict",
  $strictCloseValidatorCommand
) | Sort-Object -Unique
$sourceArtifacts = @(
  "artifacts/final-release/owner-proof-real-backfill-execution-pack.json",
  "artifacts/final-release/owner-proof-real-backfill-execution-pack-validation.json",
  "artifacts/final-release/release-close-strict-record-candidate.json",
  "artifacts/final-release/release-issue-close-record-owner-input.template.json",
  "artifacts/final-release/release-issue-final-close-decision.template.json"
)

$map = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-issue-close-record-real-input-map"
  mapState = "blocked-release-close-real-input-required"
  ownerProofRealBackfillExecutionPackState = [string](Get-PropertyOrDefault -Object $ownerProofPack -Name "packState" -DefaultValue "missing-owner-proof-real-backfill-execution-pack")
  ownerProofRealBackfillExecutionPackValidationState = [string](Get-PropertyOrDefault -Object $ownerProofPackValidation -Name "validationState" -DefaultValue "missing-owner-proof-real-backfill-execution-pack-validation")
  ownerInputTaskCount = $ownerInputTasks.Count
  mappedInputCount = $realInputMappings.Count
  missingRealInputCount = $missingRealInputCount
  blockedMappingCount = $blockedMappingCount
  realInputMappings = $realInputMappings
  closeRecordTargetArtifacts = $closeRecordTargetArtifacts
  firstCommands = $firstCommands
  strictValidators = $strictValidators
  nonSubstituteProofKinds = $nonSubstituteKinds
  sourceArtifacts = $sourceArtifacts
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Release issue close record real input map is an owner input mapping only. It does not publish, upload, approve public release, close the release issue, or promote template/draft/candidate/hash-only/local-feed evidence to proof."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-issue-close-record-real-input-map.json"
$markdownPath = Join-Path $artifactRoot "release-issue-close-record-real-input-map.md"
$map | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$mappingRows = $realInputMappings | ForEach-Object {
  "| ``$($_.sourceTaskId)`` | ``$($_.mappingState)`` | ``$($_.targetArtifact)`` | ``$($_.targetField)`` | ``$($_.releaseCloseField)`` | ``$($_.firstCommand)`` |"
}
$nonSubstituteLines = $nonSubstituteKinds | ForEach-Object { "- ``$_``" }
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }

$markdown = @"
# Release Issue Close Record Real Input Map

生成时间：$($map.generatedAtUtc)

该映射把 `owner-proof-real-backfill-execution-pack` 中的 Owner 输入任务落到 release close record 相关字段、目标 artifact、首个命令与严格 validator。它只减少回填歧义，不发布包、不批准公开发布、不关闭 release issue。

| 项目 | 当前值 |
|---|---|
| mapState | ``$($map.mapState)`` |
| ownerProofRealBackfillExecutionPackState | ``$($map.ownerProofRealBackfillExecutionPackState)`` |
| ownerProofRealBackfillExecutionPackValidationState | ``$($map.ownerProofRealBackfillExecutionPackValidationState)`` |
| ownerInputTaskCount | ``$($map.ownerInputTaskCount)`` |
| mappedInputCount | ``$($map.mappedInputCount)`` |
| missingRealInputCount | ``$($map.missingRealInputCount)`` |
| blockedMappingCount | ``$($map.blockedMappingCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Real Input Mappings

| Source Task | State | Target Artifact | Target Field | Release Close Field | First Command |
|---|---|---|---|---|---|
$($mappingRows -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")

## Boundary

$($map.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release issue close record real input map written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "MapState=$($map.mapState) MappedInputs=$($map.mappedInputCount) MissingRealInputs=$($map.missingRealInputCount) BlockedMappings=$($map.blockedMappingCount) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"
