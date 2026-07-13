[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
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

function Get-ProofLineRules {
  param([string]$Id)

  switch ($Id) {
    "package-consumer-runtime" {
      return [pscustomobject]@{
        requiredRealInputRules = @(
          "cleanExternalConsumerIdentity",
          "noProjectReference",
          "noLocalFeedAsPublicProof",
          "managedNupkgSha256",
          "runtimeNupkgSha256",
          "runtimePackageKeyMatches",
          "compatibleHostMetadata",
          "smokeCommandIncludesRuntimePackageKey",
          "smokeLogPath",
          "smokeLogSha256"
        )
        strictValidationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof"
        blockerHints = @(
          "ProjectReference",
          "local feed",
          "direct .nupkg reference",
          "missing managed/runtime nupkg SHA256",
          "missing compatible host metadata",
          "missing --runtime-package-key smoke command",
          "missing smoke log SHA256",
          "blocked-by-cuda-driver"
        )
      }
    }
    "release-issue-close-record" {
      return [pscustomobject]@{
        requiredRealInputRules = @(
          "releaseEvidenceBundleSha256",
          "releaseClosePreflightPathAndHash",
          "staleClaimsAuditPathAndHash",
          "postPublishProofValidationPathAndHash",
          "rollbackPlan",
          "ownerFinalCloseDecision",
          "strictCloseValidatorCommand"
        )
        strictValidationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
        blockerHints = @(
          "release-issue-close-record-template.json",
          "missing release evidence bundle SHA256",
          "missing stale claims audit hash",
          "missing post-publish proof validation hash",
          "missing rollback plan",
          "missing owner final close decision"
        )
      }
    }
    default {
      return [pscustomobject]@{
        requiredRealInputRules = @()
        strictValidationCommand = ""
        blockerHints = @()
      }
    }
  }
}

function New-DraftSpec {
  param([object]$RepairItem)

  $id = [string](Get-PropertyOrDefault -Object $RepairItem -Name "id" -DefaultValue "")
  $rules = Get-ProofLineRules -Id $id
  $validatorCommand = [string](Get-PropertyOrDefault -Object $RepairItem -Name "validatorCommand" -DefaultValue "")
  $strictValidationCommand = if ([string]::IsNullOrWhiteSpace([string]$rules.strictValidationCommand)) {
    $validatorCommand
  }
  else {
    [string]$rules.strictValidationCommand
  }

  $placeholderFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "placeholderFieldsToReplace" -DefaultValue @())
  $existingFileFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "fieldsRequiringExistingFiles" -DefaultValue @())
  $sha256Fields = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "fieldsRequiringSha256" -DefaultValue @())
  $cleanConsumerFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "fieldsRequiringCleanConsumerEvidence" -DefaultValue @())
  $ownerDecisionFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "fieldsRequiringOwnerDecision" -DefaultValue @())
  $rollbackFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "fieldsRequiringRollbackPlan" -DefaultValue @())
  $cannotUseMarkers = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "cannotUseMarkers" -DefaultValue @())
  $blockerHints = Convert-ToStringArray $rules.blockerHints

  [pscustomobject]@{
    id = $id
    proofClass = [string](Get-PropertyOrDefault -Object $RepairItem -Name "proofClass" -DefaultValue $id)
    draftState = "blocked-draft-non-proof"
    currentCandidateClassification = [string](Get-PropertyOrDefault -Object $RepairItem -Name "currentCandidateClassification" -DefaultValue "missing")
    inputDraftPath = [string](Get-PropertyOrDefault -Object $RepairItem -Name "inputDraftPath" -DefaultValue "")
    inputDraftIsProof = $false
    canPromoteProof = $false
    requiredRealInputs = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "requiredRealInputs" -DefaultValue @())
    requiredRealInputRules = Convert-ToStringArray $rules.requiredRealInputRules
    placeholderFieldsToReplace = $placeholderFields
    fieldsRequiringExistingFiles = $existingFileFields
    fieldsRequiringSha256 = $sha256Fields
    fieldsRequiringCleanConsumerEvidence = $cleanConsumerFields
    fieldsRequiringOwnerDecision = $ownerDecisionFields
    fieldsRequiringRollbackPlan = $rollbackFields
    firstRepairCommand = [string](Get-PropertyOrDefault -Object $RepairItem -Name "firstRepairCommand" -DefaultValue "")
    validatorCommand = $validatorCommand
    strictValidationCommand = $strictValidationCommand
    blockedByNonProofMarkers = @($cannotUseMarkers + $blockerHints | Select-Object -Unique)
    blockedByNonProofMarkerCount = @($cannotUseMarkers + $blockerHints | Select-Object -Unique).Count
    expectedArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "expectedArtifacts" -DefaultValue @())
    sourceArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "sourceArtifacts" -DefaultValue @())
    ownerNextAction = [string](Get-PropertyOrDefault -Object $RepairItem -Name "ownerNextAction" -DefaultValue "")
    blockedReason = [string](Get-PropertyOrDefault -Object $RepairItem -Name "blockedReason" -DefaultValue "blocked-real-input-required")
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$repairPack = Read-JsonOrNull "artifacts\final-release\owner-proof-input-repair-pack.json"
if ($null -eq $repairPack) {
  throw "owner-proof-input-repair-pack.json is missing. Run Export-OwnerProofInputRepairPack.ps1 first."
}

$repairItems = @(Get-PropertyOrDefault -Object $repairPack -Name "repairItems" -DefaultValue @())
if ($repairItems.Count -eq 0) {
  throw "owner-proof-input-repair-pack.json has no repairItems."
}

$draftSpecs = @($repairItems | ForEach-Object { New-DraftSpec -RepairItem $_ })
$blockedDraftSpecCount = @($draftSpecs | Where-Object { $_.draftState -eq "blocked-draft-non-proof" }).Count
$placeholderFieldCount = ($draftSpecs | ForEach-Object { @($_.placeholderFieldsToReplace).Count } | Measure-Object -Sum).Sum
$sha256FieldCount = ($draftSpecs | ForEach-Object { @($_.fieldsRequiringSha256).Count } | Measure-Object -Sum).Sum
$existingFileFieldCount = ($draftSpecs | ForEach-Object { @($_.fieldsRequiringExistingFiles).Count } | Measure-Object -Sum).Sum
$cleanConsumerFieldCount = ($draftSpecs | ForEach-Object { @($_.fieldsRequiringCleanConsumerEvidence).Count } | Measure-Object -Sum).Sum
$ownerDecisionFieldCount = ($draftSpecs | ForEach-Object { @($_.fieldsRequiringOwnerDecision).Count } | Measure-Object -Sum).Sum
$rollbackFieldCount = ($draftSpecs | ForEach-Object { @($_.fieldsRequiringRollbackPlan).Count } | Measure-Object -Sum).Sum

$record = [pscustomobject]@{
  recordKind = "owner-proof-input-draft-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  draftPackState = "blocked-draft-non-proof"
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  draftSpecCount = $draftSpecs.Count
  blockedDraftSpecCount = $blockedDraftSpecCount
  placeholderFieldCount = [int]$placeholderFieldCount
  sha256FieldCount = [int]$sha256FieldCount
  existingFileFieldCount = [int]$existingFileFieldCount
  cleanConsumerFieldCount = [int]$cleanConsumerFieldCount
  ownerDecisionFieldCount = [int]$ownerDecisionFieldCount
  rollbackFieldCount = [int]$rollbackFieldCount
  draftSpecs = $draftSpecs
  sourceArtifacts = @(
    "artifacts/final-release/owner-proof-input-repair-pack.json",
    "artifacts/final-release/owner-proof-input-repair-pack.md",
    "artifacts/final-release/owner-external-proof-input-preflight.json",
    "artifacts/final-release/release-evidence-bundle.json"
  )
  safetyBoundary = "owner-proof-input-draft-pack is a non-proof owner input drafting surface. It does not promote proof, publish packages, approve public publication, or close the release issue."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-proof-input-draft-pack.json"
$markdownPath = Join-Path $artifactRoot "owner-proof-input-draft-pack.md"

Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 16)
$rows = $draftSpecs | ForEach-Object {
  $validator = ([string]$_.strictValidationCommand).Replace("|", "\|")
  "| ``$($_.id)`` | ``$($_.draftState)`` | ``$($_.inputDraftIsProof)`` | ``$($_.canPromoteProof)`` | ``$($_.placeholderFieldsToReplace.Count)`` | ``$($_.fieldsRequiringExistingFiles.Count)`` | ``$($_.fieldsRequiringSha256.Count)`` | ``$($_.fieldsRequiringCleanConsumerEvidence.Count)`` | ``$($_.fieldsRequiringOwnerDecision.Count)`` | ``$($_.fieldsRequiringRollbackPlan.Count)`` | ``$validator`` |"
}

$detailSections = $draftSpecs | ForEach-Object {
  $markers = $_.blockedByNonProofMarkers -join "``; ``"
  $rules = $_.requiredRealInputRules -join "``; ``"
  @"
### ``$($_.id)``

- inputDraftPath: ``$($_.inputDraftPath)``
- inputDraftIsProof: ``False``
- canPromoteProof: ``False``
- requiredRealInputRules: ``$rules``
- strictValidationCommand: ``$($_.strictValidationCommand)``
- blockedByNonProofMarkers: ``$markers``
"@
}

$sourceLines = $record.sourceArtifacts | ForEach-Object { "- ``$_``" }

$markdown = @"
# Owner Proof Input Draft Pack

生成时间：$($record.generatedAtUtc)

## 总结

``owner-proof-input-draft-pack`` 从 ``owner-proof-input-repair-pack`` 生成 6 条 owner proof input draft spec。它只提供真实 proof 输入的填写面和严格校验命令，所有 draft spec 都保持 ``inputDraftIsProof=false``、``canPromoteProof=false``、``performsPublish=false``、``canPublishPublicly=false``、``canCloseReleaseIssue=false``。

## 当前 Gate

| 项目 | 当前值 |
|---|---|
| draftPackState | ``$($record.draftPackState)`` |
| draftSpecCount | ``$($record.draftSpecCount)`` |
| blockedDraftSpecCount | ``$($record.blockedDraftSpecCount)`` |
| placeholderFieldCount | ``$($record.placeholderFieldCount)`` |
| existingFileFieldCount | ``$($record.existingFileFieldCount)`` |
| sha256FieldCount | ``$($record.sha256FieldCount)`` |
| cleanConsumerFieldCount | ``$($record.cleanConsumerFieldCount)`` |
| ownerDecisionFieldCount | ``$($record.ownerDecisionFieldCount)`` |
| rollbackFieldCount | ``$($record.rollbackFieldCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Draft Specs

| ID | Draft state | Input draft is proof | Can promote proof | Placeholder fields | Existing file fields | SHA256 fields | Clean consumer fields | Owner decision fields | Rollback fields | Strict validator |
|---|---|---|---|---|---|---|---|---|---|---|
$($rows -join "`r`n")

## 明细

$($detailSections -join "`r`n")

## Release Gate Boundary

Draft pack、input draft、repair pack、template、handoff、preflight、runbook、collection package 和 readiness snapshot 都不是 proof。只有真实记录带 existing files、匹配 SHA256、clean consumer evidence、owner decision、rollback plan，并通过对应 strict validator 后，才能进入下一轮 preflight/evidence review。

## Source Artifacts

$($sourceLines -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Owner proof input draft pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "DraftPackState=$($record.draftPackState)"
Write-Output "DraftSpecCount=$($record.draftSpecCount)"
Write-Output "BlockedDraftSpecCount=$($record.blockedDraftSpecCount)"
Write-Output "PerformsPublish=False"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
