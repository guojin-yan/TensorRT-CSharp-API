[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)

  $directory = Split-Path -Parent $LiteralPath
  if ([string]::IsNullOrWhiteSpace($directory)) { $directory = "." }
  New-Item -ItemType Directory -Path $directory -Force | Out-Null

  $lines = New-Object System.Collections.Generic.List[string]
  foreach ($item in @($InputObject)) {
    if ($null -eq $item) {
      $lines.Add("") | Out-Null
    }
    elseif ($item -is [string]) {
      $lines.Add($item) | Out-Null
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { $lines.Add([string]$child) | Out-Null }
    }
    else {
      $lines.Add([string]$item) | Out-Null
    }
  }

  [System.IO.File]::WriteAllText($LiteralPath, (($lines.ToArray() -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Resolve-RepositoryPath -Path $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Test-FileExists {
  param([string]$Path)
  return Test-Path -LiteralPath (Resolve-RepositoryPath -Path $Path) -PathType Leaf
}

function New-FileStatus {
  param([string]$Path, [string]$Role)

  [pscustomobject]@{
    path = $Path
    role = $Role
    exists = (Test-FileExists -Path $Path)
    acceptedAsProof = $false
    boundary = "File presence only; not proof, not publish approval, not release close approval, and not package push."
  }
}

function Get-FailedValidationItems {
  param([AllowNull()][object]$ValidationRecord)

  $items = @(Convert-ToArray (Get-PropertyOrDefault -Object $ValidationRecord -Name "validationItems" -DefaultValue @()))
  $failed = New-Object System.Collections.Generic.List[object]
  foreach ($item in $items) {
    $passed = [bool](Get-PropertyOrDefault -Object $item -Name "passed" -DefaultValue $false)
    if ($passed) { continue }

    $failed.Add([pscustomobject]@{
        id = [string](Get-PropertyOrDefault -Object $item -Name "id" -DefaultValue "")
        severity = [string](Get-PropertyOrDefault -Object $item -Name "severity" -DefaultValue "")
        detail = [string](Get-PropertyOrDefault -Object $item -Name "detail" -DefaultValue "")
      }) | Out-Null
  }

  return @($failed.ToArray())
}

function New-SlotDefinition {
  param(
    [string]$Id,
    [string]$Title,
    [string[]]$RealOwnerInputPaths,
    [string[]]$TemplateOrGuidancePaths,
    [string[]]$GeneratedOutputPaths,
    [string]$ValidationPath,
    [string]$ExpectedReadyState,
    [string]$ReadyProperty,
    [string]$RequiredEvidence
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    realOwnerInputPaths = @($RealOwnerInputPaths)
    templateOrGuidancePaths = @($TemplateOrGuidancePaths)
    generatedOutputPaths = @($GeneratedOutputPaths)
    validationPath = $ValidationPath
    expectedReadyState = $ExpectedReadyState
    readyProperty = $ReadyProperty
    requiredEvidence = $RequiredEvidence
  }
}

function New-LedgerSlot {
  param([object]$Definition)

  $validation = Read-JsonOrNull -RelativePath $Definition.validationPath
  $validationState = [string](Get-PropertyOrDefault -Object $validation -Name "validationState" -DefaultValue "missing-$($Definition.id)-validation")
  $readyByState = -not [string]::IsNullOrWhiteSpace([string]$Definition.expectedReadyState) -and $validationState -eq [string]$Definition.expectedReadyState
  $readyByProperty = if ([string]::IsNullOrWhiteSpace([string]$Definition.readyProperty)) {
    $readyByState
  }
  else {
    [bool](Get-PropertyOrDefault -Object $validation -Name ([string]$Definition.readyProperty) -DefaultValue $false)
  }

  $realInputStatuses = @($Definition.realOwnerInputPaths | ForEach-Object { New-FileStatus -Path ([string]$_) -Role "real-owner-input" })
  $templateStatuses = @($Definition.templateOrGuidancePaths | ForEach-Object { New-FileStatus -Path ([string]$_) -Role "template-or-guidance" })
  $generatedStatuses = @($Definition.generatedOutputPaths | ForEach-Object { New-FileStatus -Path ([string]$_) -Role "generated-output" })
  $validationStatus = New-FileStatus -Path ([string]$Definition.validationPath) -Role "validation-output"
  $failedItems = @(Get-FailedValidationItems -ValidationRecord $validation)
  $realOwnerInputFileCount = @($realInputStatuses | Where-Object { [bool]$_.exists }).Count
  $templateOrGuidanceFileCount = @($templateStatuses | Where-Object { [bool]$_.exists }).Count
  $generatedOutputFileCount = @($generatedStatuses | Where-Object { [bool]$_.exists }).Count
  $proofReady = $realOwnerInputFileCount -gt 0 -and $readyByState -and $readyByProperty
  $missingReasons = New-Object System.Collections.Generic.List[string]
  if ($realOwnerInputFileCount -eq 0) { $missingReasons.Add("missing-real-owner-input-file") | Out-Null }
  if (-not $readyByState) { $missingReasons.Add("validation-state-not-ready") | Out-Null }
  if (-not $readyByProperty) { $missingReasons.Add("ready-property-not-true") | Out-Null }
  if (-not [bool]$validationStatus.exists) { $missingReasons.Add("missing-validation-output") | Out-Null }

  [pscustomobject]@{
    id = [string]$Definition.id
    title = [string]$Definition.title
    slotState = if ($proofReady) { "owner-real-evidence-available" } else { "blocked-owner-real-evidence-required" }
    requiredEvidence = [string]$Definition.requiredEvidence
    realOwnerInputPaths = @($Definition.realOwnerInputPaths)
    templateOrGuidancePaths = @($Definition.templateOrGuidancePaths)
    generatedOutputPaths = @($Definition.generatedOutputPaths)
    validationPath = [string]$Definition.validationPath
    validationState = $validationState
    expectedReadyState = [string]$Definition.expectedReadyState
    readyProperty = [string]$Definition.readyProperty
    readyByValidationState = $readyByState
    readyByValidationProperty = $readyByProperty
    realOwnerInputFileCount = $realOwnerInputFileCount
    templateOrGuidanceFileCount = $templateOrGuidanceFileCount
    generatedOutputFileCount = $generatedOutputFileCount
    validationFileExists = [bool]$validationStatus.exists
    proofReady = $proofReady
    blocked = -not $proofReady
    missingReasonCount = $missingReasons.Count
    missingReasons = @($missingReasons.ToArray())
    blockedValidationItemCount = $failedItems.Count
    blockedValidationItems = @($failedItems | Select-Object -First 80)
    fileStatuses = @($realInputStatuses + $templateStatuses + $generatedStatuses + @($validationStatus))
    notExecutedByAutomation = $true
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner evidence availability slot only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$slotDefinitions = @(
  New-SlotDefinition `
    -Id "owner-public-publish-result" `
    -Title "Owner public publish result" `
    -RealOwnerInputPaths @("artifacts/final-release/public-publish-result-owner-input.json", "artifacts/final-release/owner-public-publish-execution-result-input.json") `
    -TemplateOrGuidancePaths @("artifacts/final-release/public-publish-result-owner-input.template.json", "artifacts/final-release/owner-public-publish-execution-result-input-template.json", "artifacts/final-release/owner-public-publish-execution-result-input-contract.json") `
    -GeneratedOutputPaths @("artifacts/final-release/public-publish-result-import.json", "artifacts/final-release/owner-public-publish-execution-result-candidate.json") `
    -ValidationPath "artifacts/final-release/public-publish-result-owner-input-validation.json" `
    -ExpectedReadyState "public-publish-result-owner-input-ready" `
    -ReadyProperty "" `
    -RequiredEvidence "Owner-filled public package URLs, package ids/versions, publish transcript, SHA256 values, owner review, rollback review, and final close decision fields."
  New-SlotDefinition `
    -Id "github-actions-run-proof" `
    -Title "GitHub Actions run proof" `
    -RealOwnerInputPaths @("artifacts/final-release/github-actions-run-evidence.owner.json", "artifacts/final-release/github-actions-run-evidence-input.json") `
    -TemplateOrGuidancePaths @("artifacts/final-release/github-actions-run-evidence-import.template.json", "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.json") `
    -GeneratedOutputPaths @("artifacts/final-release/github-actions-run-evidence-import.json", "artifacts/final-release/github-publish-and-ci-status-snapshot.json") `
    -ValidationPath "artifacts/final-release/github-actions-run-evidence-import-validation.json" `
    -ExpectedReadyState "github-actions-run-evidence-import-ready" `
    -ReadyProperty "githubActionsRunEvidenceReady" `
    -RequiredEvidence "Completed GitHub Actions run id, URL, head SHA, workflow log SHA256, artifact manifest SHA256, and run conclusion tied to the pushed commit."
  New-SlotDefinition `
    -Id "public-package-download-proof" `
    -Title "Public package download proof" `
    -RealOwnerInputPaths @("artifacts/final-release/public-package-download-proof-input.json") `
    -TemplateOrGuidancePaths @("artifacts/final-release/public-package-download-proof-input.template.json", "artifacts/final-release/public-package-download-proof-owner-execution-pack.json") `
    -GeneratedOutputPaths @("artifacts/final-release/public-package-download-proof-candidate.json") `
    -ValidationPath "artifacts/final-release/public-package-download-proof-input-validation.json" `
    -ExpectedReadyState "public-package-download-proof-input-ready" `
    -ReadyProperty "publicPackageDownloadProofReady" `
    -RequiredEvidence "Public package page/download URLs, downloaded nupkg paths, SHA256 values, source URL, timestamps, owner reviewer, and GitHub Actions/source publish linkage."
  New-SlotDefinition `
    -Id "repository-external-clean-consumer-proof" `
    -Title "Repository-external clean consumer proof" `
    -RealOwnerInputPaths @("artifacts/final-release/external-clean-consumer-execution-result.owner.json", "artifacts/final-release/external-clean-consumer-execution-result.json") `
    -TemplateOrGuidancePaths @("artifacts/final-release/external-clean-consumer-execution-result.template.json", "artifacts/final-release/external-clean-consumer-owner-command-pack.json") `
    -GeneratedOutputPaths @("artifacts/final-release/external-clean-consumer-execution-result-import.json", "artifacts/final-release/external-clean-consumer-execution-result-candidate.json") `
    -ValidationPath "artifacts/final-release/external-clean-consumer-execution-result-validation.json" `
    -ExpectedReadyState "external-clean-consumer-execution-result-proof-ready" `
    -ReadyProperty "proofCandidateReady" `
    -RequiredEvidence "Clean consumer restore/build/runtime smoke logs outside the repository, no ProjectReference/direct nupkg/local feed, host metadata, package metadata, and matching hashes."
  New-SlotDefinition `
    -Id "post-publish-clean-consumer-proof" `
    -Title "Post-publish clean consumer proof" `
    -RealOwnerInputPaths @("artifacts/final-release/post-publish-clean-consumer-proof-result.owner.json", "artifacts/final-release/post-publish-clean-consumer-proof-result.json") `
    -TemplateOrGuidancePaths @("artifacts/final-release/post-publish-clean-consumer-proof-result.template.json", "artifacts/final-release/post-publish-owner-verification-runbook.json") `
    -GeneratedOutputPaths @("artifacts/final-release/post-publish-clean-consumer-proof-result-import.json", "artifacts/final-release/post-publish-clean-consumer-proof-result-candidate.json", "artifacts/final-release/post-publish-clean-consumer-real-proof-from-owner-result.json") `
    -ValidationPath "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json" `
    -ExpectedReadyState "post-publish-clean-consumer-proof-result-ready" `
    -ReadyProperty "proofCandidateReady" `
    -RequiredEvidence "Public-channel clean consumer restore/build/smoke evidence after publish, public package hashes, host metadata, strict validator output, and no local substitutes."
  New-SlotDefinition `
    -Id "post-publish-user-verification" `
    -Title "Post-publish user verification" `
    -RealOwnerInputPaths @("artifacts/final-release/post-publish-verification-owner-input.json", "artifacts/final-release/post-publish-verification-record.json") `
    -TemplateOrGuidancePaths @("artifacts/final-release/post-publish-verification-record-template.json", "artifacts/final-release/post-publish-verification-owner-input-template.json", "artifacts/final-release/post-publish-user-verification-pack.json") `
    -GeneratedOutputPaths @("artifacts/final-release/post-publish-verification-validation.json", "artifacts/final-release/post-publish-proof-owner-confirmation-validation.json") `
    -ValidationPath "artifacts/final-release/post-publish-user-verification-pack-validation.json" `
    -ExpectedReadyState "post-publish-user-verification-pack-ready" `
    -ReadyProperty "canCloseReleaseIssue" `
    -RequiredEvidence "Owner-reviewed post-publish user verification, public package URL/version/hash, clean restore proof, known limitations acknowledgement, and rollback decision."
  New-SlotDefinition `
    -Id "release-issue-close-owner-decision" `
    -Title "Release Issue close owner decision" `
    -RealOwnerInputPaths @("artifacts/final-release/release-issue-close-owner-decision-input.owner.json", "artifacts/final-release/release-issue-close-owner-decision-input.json") `
    -TemplateOrGuidancePaths @("artifacts/final-release/release-issue-close-owner-decision-input.template.json", "artifacts/final-release/release-issue-final-close-decision-template.json") `
    -GeneratedOutputPaths @("artifacts/final-release/release-issue-close-owner-decision-input-validation.json", "artifacts/final-release/release-issue-close-final-owner-decision-audit.json") `
    -ValidationPath "artifacts/final-release/release-issue-close-owner-decision-input-validation.json" `
    -ExpectedReadyState "release-issue-close-owner-decision-input-ready" `
    -ReadyProperty "canCloseReleaseIssue" `
    -RequiredEvidence "Owner final close decision, release issue URL/number, approved proof hashes, strict close output hashes, rollback plan, and owner timestamp."
  New-SlotDefinition `
    -Id "release-issue-close-record" `
    -Title "Release Issue close record" `
    -RealOwnerInputPaths @("artifacts/final-release/release-issue-close-record.json") `
    -TemplateOrGuidancePaths @("artifacts/final-release/release-issue-close-record-template.json", "artifacts/final-release/release-issue-close-record-owner-input.template.json") `
    -GeneratedOutputPaths @("artifacts/final-release/release-issue-close-record-validation.json", "artifacts/final-release/release-issue-close-record-candidate.json") `
    -ValidationPath "artifacts/final-release/release-issue-close-record-validation.json" `
    -ExpectedReadyState "release-issue-close-record-ready" `
    -ReadyProperty "canPromoteReleaseIssueCloseRecord" `
    -RequiredEvidence "Final owner close record backed by real post-publish proof, release close preflight, evidence bundle hash, stale-claim audit, and rollback plan."
  New-SlotDefinition `
    -Id "strict-close-final-convergence" `
    -Title "Strict close final convergence" `
    -RealOwnerInputPaths @("artifacts/final-release/final-owner-close-decision.owner.json", "artifacts/final-release/final-owner-rollback-review.owner.json") `
    -TemplateOrGuidancePaths @("artifacts/final-release/final-public-release-closure-bridge.json", "artifacts/final-release/strict-close-ready-convergence-dashboard.json") `
    -GeneratedOutputPaths @("artifacts/final-release/final-public-release-closure-bridge-validation.json", "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json") `
    -ValidationPath "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json" `
    -ExpectedReadyState "strict-close-ready-convergence-dashboard-ready" `
    -ReadyProperty "canCloseReleaseIssue" `
    -RequiredEvidence "All strict close lanes green with accepted Owner public publish result, public download proof, post-publish clean consumer proof, final close decision, and classification audit."
)

$slots = @($slotDefinitions | ForEach-Object { New-LedgerSlot -Definition $_ })
$slotCount = $slots.Count
$proofReadySlots = @($slots | Where-Object { [bool]$_.proofReady })
$blockedSlots = @($slots | Where-Object { [bool]$_.blocked })
$realOwnerInputFileCount = 0
$templateOrGuidanceFileCount = 0
$generatedOutputFileCount = 0
$blockedValidationItemCount = 0
foreach ($slot in $slots) {
  $realOwnerInputFileCount += [int]$slot.realOwnerInputFileCount
  $templateOrGuidanceFileCount += [int]$slot.templateOrGuidanceFileCount
  $generatedOutputFileCount += [int]$slot.generatedOutputFileCount
  $blockedValidationItemCount += [int]$slot.blockedValidationItemCount
}

$record = [pscustomobject]@{
  recordKind = "owner-real-publish-evidence-availability-ledger"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  ledgerState = if ($proofReadySlots.Count -eq $slotCount) { "owner-real-publish-evidence-available-but-close-still-owner-gated" } else { "blocked-owner-real-publish-evidence-required" }
  slotCount = $slotCount
  proofReadySlotCount = $proofReadySlots.Count
  blockedSlotCount = $blockedSlots.Count
  realOwnerInputFileCount = $realOwnerInputFileCount
  templateOrGuidanceFileCount = $templateOrGuidanceFileCount
  generatedOutputFileCount = $generatedOutputFileCount
  blockedValidationItemCount = $blockedValidationItemCount
  missingRealOwnerInputSlotCount = @($slots | Where-Object { [int]$_.realOwnerInputFileCount -eq 0 }).Count
  realOwnerProofAvailable = $proofReadySlots.Count -gt 0
  allRequiredOwnerProofAvailable = $proofReadySlots.Count -eq $slotCount
  ownerActionRequired = $blockedSlots.Count -gt 0
  releaseIssueCloseCandidate = $false
  slots = @($slots)
  sourceArtifacts = @(
    "artifacts/final-release/public-publish-result-owner-input-validation.json",
    "artifacts/final-release/public-publish-result-import-validation.json",
    "artifacts/final-release/github-actions-run-evidence-import-validation.json",
    "artifacts/final-release/public-package-download-proof-input-validation.json",
    "artifacts/final-release/public-package-download-proof-candidate-validation.json",
    "artifacts/final-release/external-clean-consumer-execution-result-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-real-proof-from-owner-result-validation.json",
    "artifacts/final-release/post-publish-user-verification-pack-validation.json",
    "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json",
    "artifacts/final-release/final-public-release-closure-bridge-validation.json",
    "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json",
    "artifacts/final-release/owner-proof-input-readiness-validation.json",
    "artifacts/final-release/final-owner-real-proof-gap-matrix-validation.json"
  )
  notExecutedByAutomation = $true
  performsPublish = $false
  usesPublishToken = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner real publish evidence availability ledger inventories real Owner input presence and validator readiness only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-real-publish-evidence-availability-ledger.json"
$markdownPath = Join-Path $OutputRoot "owner-real-publish-evidence-availability-ledger.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 24)

$slotRows = foreach ($slot in $slots) {
  "| ``$($slot.id)`` | ``$($slot.slotState)`` | ``$($slot.realOwnerInputFileCount)`` | ``$($slot.validationState)`` | ``$($slot.blockedValidationItemCount)`` | $(ConvertTo-MarkdownCell $slot.requiredEvidence) |"
}

$markdown = @"
# Owner Real Publish Evidence Availability Ledger

Generated: ``$($record.generatedAtUtc)``

| Field | Value |
|---|---|
| ledgerState | ``$($record.ledgerState)`` |
| slotCount | ``$($record.slotCount)`` |
| proofReadySlotCount | ``$($record.proofReadySlotCount)`` |
| blockedSlotCount | ``$($record.blockedSlotCount)`` |
| realOwnerInputFileCount | ``$($record.realOwnerInputFileCount)`` |
| templateOrGuidanceFileCount | ``$($record.templateOrGuidanceFileCount)`` |
| generatedOutputFileCount | ``$($record.generatedOutputFileCount)`` |
| blockedValidationItemCount | ``$($record.blockedValidationItemCount)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Slots

| Slot | State | Real Owner Inputs | Validation State | Blocked Items | Required Evidence |
|---|---|---:|---|---:|---|
$($slotRows -join "`r`n")

## Boundary

$($record.boundary)
"@

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "OwnerRealPublishEvidenceAvailabilityLedgerState=$($record.ledgerState) Slots=$($record.slotCount) Ready=$($record.proofReadySlotCount) Blocked=$($record.blockedSlotCount)"
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
