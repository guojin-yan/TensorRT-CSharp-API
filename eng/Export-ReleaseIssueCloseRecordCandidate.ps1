[CmdletBinding()]
param(
  [string]$OwnerInputPath,
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

function Read-JsonFileOrNull {
  param([AllowNull()][string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return $null
  }

  $resolvedPath = if ([System.IO.Path]::IsPathRooted($Path)) {
    $Path
  }
  else {
    Join-Path $RepositoryRoot $Path
  }

  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
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

function Find-ProofLine {
  param(
    [AllowNull()][object]$Record,
    [string]$Id
  )

  if ($null -eq $Record) {
    return $null
  }

  foreach ($line in @(Get-PropertyOrDefault -Object $Record -Name "backfillLines" -DefaultValue @())) {
    if ([string](Get-PropertyOrDefault -Object $line -Name "id" -DefaultValue "") -eq $Id) {
      return $line
    }
  }

  foreach ($spec in @(Get-PropertyOrDefault -Object $Record -Name "draftSpecs" -DefaultValue @())) {
    if ([string](Get-PropertyOrDefault -Object $spec -Name "id" -DefaultValue "") -eq $Id) {
      return $spec
    }
  }

  return $null
}

function Get-RelativeFileSha256OrPlaceholder {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return "<owner-fill-$($RelativePath.Replace('\','-').Replace('/','-'))-sha256>"
  }

  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$orchestrator = Read-JsonOrNull "artifacts\final-release\owner-external-proof-backfill-orchestrator.json"
$draftPack = Read-JsonOrNull "artifacts\final-release\owner-proof-input-draft-pack.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$ownerInput = Read-JsonFileOrNull -Path $OwnerInputPath

$orchestratorLine = Find-ProofLine -Record $orchestrator -Id "release-issue-close-record"
$draftSpec = Find-ProofLine -Record $draftPack -Id "release-issue-close-record"

$inputDraftPath = [string](Get-PropertyOrDefault -Object $draftSpec -Name "inputDraftPath" -DefaultValue "artifacts/final-release/release-issue-close-record.input-draft.json")
$targetProofRecordPath = [string](Get-PropertyOrDefault -Object $orchestratorLine -Name "targetProofRecordPath" -DefaultValue "artifacts/final-release/release-issue-close-record.json")
$requiredRules = Convert-ToStringArray (Get-PropertyOrDefault -Object $orchestratorLine -Name "requiredRealInputRules" -DefaultValue @(
  "releaseEvidenceBundleSha256",
  "releaseClosePreflightPathAndHash",
  "staleClaimsAuditPathAndHash",
  "postPublishProofValidationPathAndHash",
  "rollbackPlan",
  "ownerFinalCloseDecision",
  "strictCloseValidatorCommand"
))

$releaseEvidenceBundlePath = "artifacts/final-release/release-evidence-bundle.json"
$releaseClosePreflightPath = "artifacts/final-release/release-close-preflight.json"
$staleClaimsAuditPath = "artifacts/final-release/stale-release-claims-audit.json"
$postPublishProofValidationPath = "artifacts/final-release/post-publish-verification-validation.json"
$releaseEvidenceBundleSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseEvidenceBundlePath
$releaseClosePreflightSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseClosePreflightPath
$staleClaimsAuditSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $staleClaimsAuditPath
$postPublishProofValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $postPublishProofValidationPath
$postPublishProofValidationState = "missing-real-post-publish-proof-validation"
$rollbackPlan = "<owner-fill-rollback-plan>"
$rollbackOwner = "<owner-fill-rollback-owner>"
$rollbackTrigger = "<owner-fill-rollback-trigger>"
$ownerFinalCloseDecision = "<owner-fill-final-close-decision>"
$ownerDecisionTimestamp = "<owner-fill-owner-decision-timestamp>"
$releaseIssueId = "<owner-fill-release-issue-id>"
$releaseIssueUrl = "<owner-fill-release-issue-url>"
$strictCloseValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
$ownerInputOverlayApplied = $null -ne $ownerInput

if ($null -ne $ownerInput) {
  $releaseEvidenceBundlePath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "releaseEvidenceBundlePath" -DefaultValue $releaseEvidenceBundlePath)
  $releaseEvidenceBundleSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "releaseEvidenceBundleSha256" -DefaultValue $releaseEvidenceBundleSha256)
  $releaseClosePreflightPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "releaseClosePreflightPath" -DefaultValue $releaseClosePreflightPath)
  $releaseClosePreflightSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "releaseClosePreflightSha256" -DefaultValue $releaseClosePreflightSha256)
  $staleClaimsAuditPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "staleClaimsAuditPath" -DefaultValue $staleClaimsAuditPath)
  $staleClaimsAuditSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "staleClaimsAuditSha256" -DefaultValue $staleClaimsAuditSha256)
  $postPublishProofValidationPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "postPublishProofValidationPath" -DefaultValue $postPublishProofValidationPath)
  $postPublishProofValidationSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "postPublishProofValidationSha256" -DefaultValue $postPublishProofValidationSha256)
  $postPublishProofValidationState = [string](Get-PropertyOrDefault -Object $ownerInput -Name "postPublishProofValidationState" -DefaultValue $postPublishProofValidationState)
  $rollbackPlan = [string](Get-PropertyOrDefault -Object $ownerInput -Name "rollbackPlan" -DefaultValue $rollbackPlan)
  $rollbackOwner = [string](Get-PropertyOrDefault -Object $ownerInput -Name "rollbackOwner" -DefaultValue $rollbackOwner)
  $rollbackTrigger = [string](Get-PropertyOrDefault -Object $ownerInput -Name "rollbackTrigger" -DefaultValue $rollbackTrigger)
  $ownerFinalCloseDecision = [string](Get-PropertyOrDefault -Object $ownerInput -Name "ownerFinalCloseDecision" -DefaultValue $ownerFinalCloseDecision)
  $ownerDecisionTimestamp = [string](Get-PropertyOrDefault -Object $ownerInput -Name "ownerDecisionTimestamp" -DefaultValue $ownerDecisionTimestamp)
  $releaseIssueId = [string](Get-PropertyOrDefault -Object $ownerInput -Name "releaseIssueId" -DefaultValue $releaseIssueId)
  $releaseIssueUrl = [string](Get-PropertyOrDefault -Object $ownerInput -Name "releaseIssueUrl" -DefaultValue $releaseIssueUrl)
  $strictCloseValidatorCommand = [string](Get-PropertyOrDefault -Object $ownerInput -Name "strictCloseValidatorCommand" -DefaultValue $strictCloseValidatorCommand)
}

$blockedBy = @(
  "missing real post-publish proof validation",
  "missing owner final close decision",
  "missing rollback plan",
  "candidate is not release close proof",
  "release-issue-close-record-template.json is not close proof"
)

$record = [pscustomobject]@{
  recordKind = "release-issue-close-record-candidate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  candidateState = "blocked-release-close-real-proof-required"
  proofLineId = "release-issue-close-record"
  inputDraftPath = $inputDraftPath
  orchestratorPath = "artifacts/final-release/owner-external-proof-backfill-orchestrator.json"
  ownerInputPath = if ([string]::IsNullOrWhiteSpace($OwnerInputPath)) { "" } else { $OwnerInputPath }
  ownerInputOverlayApplied = $ownerInputOverlayApplied
  targetProofRecordPath = $targetProofRecordPath
  releaseEvidenceBundlePath = $releaseEvidenceBundlePath
  releaseEvidenceBundleSha256 = $releaseEvidenceBundleSha256
  releaseClosePreflightPath = $releaseClosePreflightPath
  releaseClosePreflightSha256 = $releaseClosePreflightSha256
  staleClaimsAuditPath = $staleClaimsAuditPath
  staleClaimsAuditSha256 = $staleClaimsAuditSha256
  postPublishProofValidationPath = $postPublishProofValidationPath
  postPublishProofValidationSha256 = $postPublishProofValidationSha256
  postPublishProofValidationState = $postPublishProofValidationState
  rollbackPlan = $rollbackPlan
  rollbackOwner = $rollbackOwner
  rollbackTrigger = $rollbackTrigger
  ownerFinalCloseDecision = $ownerFinalCloseDecision
  ownerDecisionTimestamp = $ownerDecisionTimestamp
  strictCloseValidatorCommand = $strictCloseValidatorCommand
  releaseIssueId = $releaseIssueId
  releaseIssueUrl = $releaseIssueUrl
  requiredRealInputRules = $requiredRules
  blockedBy = $blockedBy
  requiredOwnerActions = @(
    "Refresh release evidence bundle, release close preflight, stale claims audit, and post-publish proof validation.",
    "Record all artifact paths and matching SHA256 values.",
    "Fill rollback plan, rollback owner, rollback trigger, release issue identity, and owner final close decision.",
    "Run the strict close validator and keep canCloseReleaseIssue=false until all real proof gates pass."
  )
  strictValidationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordCandidate.ps1 -Strict"
  canCloseReleaseIssue = $false
  performsPublish = $false
  canPublishPublicly = $false
  releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  sourceArtifacts = @(
    "artifacts/final-release/owner-external-proof-backfill-orchestrator.json",
    "artifacts/final-release/owner-proof-input-draft-pack.json",
    "artifacts/final-release/release-issue-close-record-owner-input.template.json",
    "artifacts/final-release/release-issue-close-record-owner-input-validation.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/stale-release-claims-audit.json"
  )
  safetyBoundary = "release-issue-close-record-candidate is an owner input candidate only. It cannot close the release issue and cannot replace real post-publish proof, rollback approval, or owner final close decision."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-issue-close-record-candidate.json"
$markdownPath = Join-Path $artifactRoot "release-issue-close-record-candidate.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$requiredRuleLines = $requiredRules | ForEach-Object { "- ``$_``" }
$blockedLines = $blockedBy | ForEach-Object { "- $_" }
$actionLines = $record.requiredOwnerActions | ForEach-Object { "- $_" }

$markdown = @"
# Release Issue Close Record Candidate

生成时间：$($record.generatedAtUtc)

## 总结

``release-issue-close-record-candidate`` 是 release close 的 owner-fill candidate input record。它聚合 release evidence bundle hash、release close preflight hash、stale claims audit hash、post-publish proof validation hash、rollback plan 和 owner final close decision 的真实输入要求。

它不是 close proof，不关闭 release issue，也不能替代 post-publish proof validation 或 owner final close decision。

## 当前 Gate

| 项目 | 当前值 |
|---|---|
| recordKind | ``$($record.recordKind)`` |
| candidateState | ``$($record.candidateState)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |

## Required Real Input Rules

$($requiredRuleLines -join "`r`n")

## 当前 Blockers

$($blockedLines -join "`r`n")

## Owner Actions

$($actionLines -join "`r`n")

## Strict Validator

```powershell
$($record.strictValidationCommand)
```

## Safety Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release issue close record candidate written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "CandidateState=$($record.candidateState)"
Write-Host "PerformsPublish=$($record.performsPublish)"
Write-Host "CanPublishPublicly=$($record.canPublishPublicly)"
Write-Host "CanCloseReleaseIssue=$($record.canCloseReleaseIssue)"
