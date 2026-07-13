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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function Get-TargetProofRecordPath {
  param(
    [string]$Id,
    [string]$InputDraftPath
  )

  switch ($Id) {
    "owner-authorization" { return "artifacts/final-release/release-owner-proof-input-record.json" }
    "package-consumer-runtime" { return "artifacts/final-release/external-runtime-proof-record.json" }
    "linux-runner-proof" { return "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record.json" }
    "real-model-runtime" { return "artifacts/user-acceptance/sample-run-evidence-record.json" }
    "post-publish-verification" { return "artifacts/final-release/post-publish-verification-record.json" }
    "release-issue-close-record" { return "artifacts/final-release/release-issue-close-record.json" }
    default {
      if ([string]::IsNullOrWhiteSpace($InputDraftPath)) {
        return "artifacts/final-release/$Id.json"
      }

      return $InputDraftPath -replace "\.input-draft\.json$", ".json"
    }
  }
}

function Get-FirstOwnerCommand {
  param([string]$Id)

  switch ($Id) {
    "owner-authorization" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerProofInputRecordTemplate.ps1" }
    "package-consumer-runtime" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofCollectionBundle.ps1 -RuntimePackageKey $RuntimePackageKey" }
    "linux-runner-proof" { return "Run the Linux runner evidence template and smoke on the target Linux host for $LinuxRuntimePackageKey." }
    "real-model-runtime" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleRunEvidenceRecordTemplate.ps1" }
    "post-publish-verification" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordTemplate.ps1 -RuntimePackageKey $RuntimePackageKey" }
    "release-issue-close-record" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordTemplate.ps1" }
    default { return "Fill the target proof record from the draft spec and run its strict validator." }
  }
}

function New-NextOwnerCommands {
  param(
    [string]$Id,
    [string]$TargetProofRecordPath,
    [string]$StrictValidationCommand
  )

  $commands = New-Object System.Collections.Generic.List[string]
  $commands.Add((Get-FirstOwnerCommand -Id $Id)) | Out-Null

  switch ($Id) {
    "package-consumer-runtime" {
      $commands.Add("Create a clean external consumer root outside this repository; do not use ProjectReference, local feed, or direct .nupkg as public proof.") | Out-Null
      $commands.Add("Run package restore/build/runtime smoke with --runtime-package-key $RuntimePackageKey on a compatible CUDA/TensorRT host.") | Out-Null
      $commands.Add("Record managed/runtime package identity, nupkg SHA256 values, host metadata, smoke command, smoke log path, and smoke log SHA256 in $TargetProofRecordPath.") | Out-Null
    }
    "release-issue-close-record" {
      $commands.Add("Refresh release evidence, release close preflight, stale claims audit, and post-publish proof validation before computing hashes.") | Out-Null
      $commands.Add("Record release evidence bundle SHA256, close preflight hash, stale claims audit hash, post-publish proof validation hash, rollback plan, and owner final close decision in $TargetProofRecordPath.") | Out-Null
    }
    default {
      $commands.Add("Fill $TargetProofRecordPath with real owner input, existing file paths, hashes, and non-placeholder command evidence.") | Out-Null
    }
  }

  $commands.Add($StrictValidationCommand) | Out-Null
  $commands.Add("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerExternalProofInputPreflight.ps1") | Out-Null
  $commands.Add("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1") | Out-Null

  return @($commands.ToArray())
}

function New-BackfillLine {
  param([object]$DraftSpec)

  $id = [string](Get-PropertyOrDefault -Object $DraftSpec -Name "id" -DefaultValue "")
  $inputDraftPath = [string](Get-PropertyOrDefault -Object $DraftSpec -Name "inputDraftPath" -DefaultValue "")
  $targetProofRecordPath = Get-TargetProofRecordPath -Id $id -InputDraftPath $inputDraftPath
  $strictValidationCommand = [string](Get-PropertyOrDefault -Object $DraftSpec -Name "strictValidationCommand" -DefaultValue "")
  $requiredFiles = Convert-ToStringArray (Get-PropertyOrDefault -Object $DraftSpec -Name "fieldsRequiringExistingFiles" -DefaultValue @())
  $requiredSha256Fields = Convert-ToStringArray (Get-PropertyOrDefault -Object $DraftSpec -Name "fieldsRequiringSha256" -DefaultValue @())
  $requiredOwnerDecisions = Convert-ToStringArray (Get-PropertyOrDefault -Object $DraftSpec -Name "fieldsRequiringOwnerDecision" -DefaultValue @())
  $requiredRollbackFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $DraftSpec -Name "fieldsRequiringRollbackPlan" -DefaultValue @())
  $blockedByNonProofMarkers = Convert-ToStringArray (Get-PropertyOrDefault -Object $DraftSpec -Name "blockedByNonProofMarkers" -DefaultValue @())

  [pscustomobject]@{
    id = $id
    backfillState = "blocked-owner-external-proof-required"
    inputDraftPath = $inputDraftPath
    targetProofRecordPath = $targetProofRecordPath
    strictValidationCommand = $strictValidationCommand
    requiredRealInputRules = Convert-ToStringArray (Get-PropertyOrDefault -Object $DraftSpec -Name "requiredRealInputRules" -DefaultValue @())
    requiredFiles = $requiredFiles
    requiredFileCount = $requiredFiles.Count
    requiredSha256Fields = $requiredSha256Fields
    requiredSha256FieldCount = $requiredSha256Fields.Count
    requiredOwnerDecisions = $requiredOwnerDecisions
    requiredOwnerDecisionCount = $requiredOwnerDecisions.Count
    requiredRollbackFields = $requiredRollbackFields
    requiredRollbackFieldCount = $requiredRollbackFields.Count
    blockedByNonProofMarkers = $blockedByNonProofMarkers
    blockedByNonProofMarkerCount = $blockedByNonProofMarkers.Count
    firstOwnerCommand = Get-FirstOwnerCommand -Id $id
    nextOwnerCommands = New-NextOwnerCommands -Id $id -TargetProofRecordPath $targetProofRecordPath -StrictValidationCommand $strictValidationCommand
    canPromoteProof = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    blockedReason = "Real external owner proof has not been collected or validator-promoted for '$id'."
  }
}

$draftPack = Read-JsonOrNull "artifacts\final-release\owner-proof-input-draft-pack.json"
if ($null -eq $draftPack) {
  throw "owner-proof-input-draft-pack.json is missing. Run Export-OwnerProofInputDraftPack.ps1 first."
}

$repairPack = Read-JsonOrNull "artifacts\final-release\owner-proof-input-repair-pack.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"

$draftSpecs = @(Get-PropertyOrDefault -Object $draftPack -Name "draftSpecs" -DefaultValue @())
if ($draftSpecs.Count -eq 0) {
  throw "owner-proof-input-draft-pack.json has no draftSpecs."
}

$backfillLines = @($draftSpecs | ForEach-Object { New-BackfillLine -DraftSpec $_ })
$blockedLineCount = @($backfillLines | Where-Object { $_.backfillState -eq "blocked-owner-external-proof-required" }).Count
$requiredFileCount = ($backfillLines | ForEach-Object { $_.requiredFileCount } | Measure-Object -Sum).Sum
$requiredSha256FieldCount = ($backfillLines | ForEach-Object { $_.requiredSha256FieldCount } | Measure-Object -Sum).Sum
$requiredOwnerDecisionCount = ($backfillLines | ForEach-Object { $_.requiredOwnerDecisionCount } | Measure-Object -Sum).Sum
$requiredRollbackFieldCount = ($backfillLines | ForEach-Object { $_.requiredRollbackFieldCount } | Measure-Object -Sum).Sum

$record = [pscustomobject]@{
  recordKind = "owner-external-proof-backfill-orchestrator"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  orchestratorState = "blocked-owner-external-proof-required"
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  backfillLineCount = $backfillLines.Count
  blockedBackfillLineCount = $blockedLineCount
  requiredFileCount = [int]$requiredFileCount
  requiredSha256FieldCount = [int]$requiredSha256FieldCount
  requiredOwnerDecisionCount = [int]$requiredOwnerDecisionCount
  requiredRollbackFieldCount = [int]$requiredRollbackFieldCount
  releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  repairPackState = [string](Get-PropertyOrDefault -Object $repairPack -Name "repairPackState" -DefaultValue "missing-owner-proof-input-repair-pack")
  draftPackState = [string](Get-PropertyOrDefault -Object $draftPack -Name "draftPackState" -DefaultValue "missing-owner-proof-input-draft-pack")
  backfillLines = $backfillLines
  sourceArtifacts = @(
    "artifacts/final-release/owner-proof-input-draft-pack.json",
    "artifacts/final-release/owner-proof-input-draft-pack-validation.json",
    "artifacts/final-release/owner-proof-input-repair-pack.json",
    "artifacts/final-release/release-evidence-bundle.json"
  )
  safetyBoundary = "owner-external-proof-backfill-orchestrator is owner action guidance only. It does not collect proof by itself, promote proof, publish packages, approve public publication, or close the release issue."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-external-proof-backfill-orchestrator.json"
$markdownPath = Join-Path $artifactRoot "owner-external-proof-backfill-orchestrator.md"

$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $backfillLines | ForEach-Object {
  $strict = ([string]$_.strictValidationCommand).Replace("|", "\|")
  "| ``$($_.id)`` | ``$($_.backfillState)`` | ``$($_.targetProofRecordPath)`` | ``$($_.requiredFileCount)`` | ``$($_.requiredSha256FieldCount)`` | ``$($_.requiredOwnerDecisionCount)`` | ``$($_.requiredRollbackFieldCount)`` | ``$($_.canPromoteProof)`` | ``$strict`` |"
}

$detailSections = $backfillLines | ForEach-Object {
  $commands = $_.nextOwnerCommands | ForEach-Object { "  - ``$_``" }
  $rules = $_.requiredRealInputRules -join "``; ``"
  $markers = $_.blockedByNonProofMarkers -join "``; ``"
  @"
### ``$($_.id)``

- targetProofRecordPath: ``$($_.targetProofRecordPath)``
- inputDraftPath: ``$($_.inputDraftPath)``
- requiredRealInputRules: ``$rules``
- blockedByNonProofMarkers: ``$markers``
- nextOwnerCommands:
$($commands -join "`r`n")
"@
}

$sourceLines = $record.sourceArtifacts | ForEach-Object { "- ``$_``" }

$markdown = @"
# Owner External Proof Backfill Orchestrator

生成时间：$($record.generatedAtUtc)

## 总结

``owner-external-proof-backfill-orchestrator`` 把 draft pack 的 6 条 proof line 转成 owner 可执行的真实外部 proof 回填计划。它只列出 target proof record、draft path、required files、SHA256 fields、owner decisions、rollback fields 和 strict validator；不会执行发布、不会采集或伪造 proof、不会关闭 release issue。

## 当前 Gate

| 项目 | 当前值 |
|---|---|
| orchestratorState | ``$($record.orchestratorState)`` |
| backfillLineCount | ``$($record.backfillLineCount)`` |
| blockedBackfillLineCount | ``$($record.blockedBackfillLineCount)`` |
| requiredFileCount | ``$($record.requiredFileCount)`` |
| requiredSha256FieldCount | ``$($record.requiredSha256FieldCount)`` |
| requiredOwnerDecisionCount | ``$($record.requiredOwnerDecisionCount)`` |
| requiredRollbackFieldCount | ``$($record.requiredRollbackFieldCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Backfill Lines

| ID | State | Target proof record | Required files | SHA256 fields | Owner decisions | Rollback fields | Can promote proof | Strict validator |
|---|---|---|---:|---:|---:|---:|---|---|
$($rows -join "`r`n")

## 明细

$($detailSections -join "`r`n")

## Release Gate Boundary

Orchestrator、draft pack、repair pack、input draft、template、handoff、preflight、runbook、collection package 和 readiness snapshot 都不是 proof。真实 release proof 只能来自 target proof record 中的真实文件、匹配 SHA256、clean consumer evidence、owner decision、rollback plan，以及对应 strict validator 通过。

## Source Artifacts

$($sourceLines -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Owner external proof backfill orchestrator written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "OrchestratorState=$($record.orchestratorState)"
Write-Output "BackfillLineCount=$($record.backfillLineCount)"
Write-Output "BlockedBackfillLineCount=$($record.blockedBackfillLineCount)"
Write-Output "PerformsPublish=False"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
