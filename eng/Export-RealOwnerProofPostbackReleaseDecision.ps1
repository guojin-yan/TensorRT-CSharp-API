[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputPath = "artifacts\final-release\real-owner-proof-postback-release-decision.json",
  [string]$MarkdownOutputPath = "artifacts\final-release\real-owner-proof-postback-release-decision.md"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepoPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Read-Json {
  param([string]$Path)

  $fullPath = Resolve-RepoPath -Path $Path
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $fullPath -Raw | ConvertFrom-Json
}

function New-OwnerEvidenceLane {
  param(
    [string]$Id,
    [string]$EvidenceDirectory,
    [string]$RecordPath,
    [string]$ValidatorCommand
  )

  $evidenceFullPath = Resolve-RepoPath -Path $EvidenceDirectory
  $recordFullPath = Resolve-RepoPath -Path $RecordPath
  $evidenceDirectoryExists = Test-Path -LiteralPath $evidenceFullPath -PathType Container
  $recordExists = Test-Path -LiteralPath $recordFullPath -PathType Leaf
  $evidenceFileCount = 0

  if ($evidenceDirectoryExists) {
    $evidenceFileCount = @(Get-ChildItem -LiteralPath $evidenceFullPath -Recurse -File).Count
  }

  $missingInputs = @()
  if (-not $evidenceDirectoryExists) {
    $missingInputs += "owner evidence directory missing: $EvidenceDirectory"
  } elseif ($evidenceFileCount -eq 0) {
    $missingInputs += "owner evidence directory has no files: $EvidenceDirectory"
  }

  if (-not $recordExists) {
    $missingInputs += "record missing: $RecordPath"
  }

  [pscustomobject]@{
    id = $Id
    evidenceDirectory = $EvidenceDirectory
    evidenceDirectoryExists = [bool]$evidenceDirectoryExists
    evidenceFileCount = [int]$evidenceFileCount
    recordPath = $RecordPath
    recordExists = [bool]$recordExists
    strictValidatorCommand = $ValidatorCommand
    missingInputs = @($missingInputs)
    ownerProofPostbackDetected = [bool]($evidenceDirectoryExists -and $recordExists -and $evidenceFileCount -gt 0)
    strictValidatorExecuted = $false
    strictValidatorPassed = $false
    canPromote = $false
  }
}

$finalSnapshotPath = "artifacts/final-release/final-prepublish-readiness-snapshot.json"
$actionWorklistPath = "artifacts/final-release/owner-real-proof-final-action-worklist.json"
$convergencePath = "artifacts/final-release/release-final-blocker-convergence.json"
$summaryPath = "artifacts/final-release/release-proof-owner-backfill-summary-validation.json"

$finalSnapshot = Read-Json -Path $finalSnapshotPath
$actionWorklist = Read-Json -Path $actionWorklistPath
$convergence = Read-Json -Path $convergencePath
$summary = Read-Json -Path $summaryPath

$lanes = @(
  New-OwnerEvidenceLane `
    -Id "real-model-runtime" `
    -EvidenceDirectory "artifacts/final-release/owner-evidence/real-model-runtime" `
    -RecordPath "artifacts/final-release/real-case-evidence-record.json" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record.json -FailOnNotProof"
  New-OwnerEvidenceLane `
    -Id "package-consumer-runtime" `
    -EvidenceDirectory "artifacts/final-release/owner-evidence/package-consumer-runtime" `
    -RecordPath "artifacts/final-release/package-consumer-runtime-proof-record.json" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -InputPath .\artifacts\final-release\package-consumer-runtime-proof-record.json -Strict -RequireExistingLog -FailOnNotProof"
  New-OwnerEvidenceLane `
    -Id "post-publish-verification" `
    -EvidenceDirectory "artifacts/final-release/owner-evidence/post-publish-verification" `
    -RecordPath "artifacts/final-release/post-publish-verification-record.json" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath .\artifacts\final-release\post-publish-verification-record.json -Strict -RequireExistingLog -FailOnNotProof"
  New-OwnerEvidenceLane `
    -Id "release-issue-close" `
    -EvidenceDirectory "artifacts/final-release/owner-evidence/release-issue-close" `
    -RecordPath "artifacts/final-release/release-issue-close-record.json" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath .\artifacts\final-release\release-issue-close-record.json -FailOnNotCloseReady"
)

$postbackDetectedCount = @($lanes | Where-Object { $_.ownerProofPostbackDetected }).Count
$missingInputCount = ($lanes | ForEach-Object { @($_.missingInputs).Count } | Measure-Object -Sum).Sum
if ($null -eq $missingInputCount) {
  $missingInputCount = 0
}

$result = [pscustomobject]@{
  recordKind = "real-owner-proof-postback-release-decision"
  generatedAtLocal = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
  decisionState = "blocked-owner-proof-not-posted-back"
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  performsPublish = $false
  approvesPublicRelease = $false
  publishDecision = "blocked"
  publishBlockedReason = "owner-real-proof-missing"
  proofBoundary = "This decision snapshot is read-only. It records whether Owner real proof has been posted back; it does not create proof, run publish, promote lanes, or close release issues."
  upstreamGates = [pscustomobject]@{
    finalPrepublishReadinessSnapshot = [pscustomobject]@{
      path = $finalSnapshotPath
      exists = [bool]($null -ne $finalSnapshot)
      recordKind = $(if ($null -ne $finalSnapshot) { [string]$finalSnapshot.recordKind } else { $null })
      state = $(if ($null -ne $finalSnapshot) { [string]$finalSnapshot.snapshotState } else { "missing" })
      canPublishPublicly = $(if ($null -ne $finalSnapshot) { [bool]$finalSnapshot.canPublishPublicly } else { $false })
      canCloseReleaseIssue = $(if ($null -ne $finalSnapshot) { [bool]$finalSnapshot.canCloseReleaseIssue } else { $false })
    }
    ownerRealProofFinalActionWorklist = [pscustomobject]@{
      path = $actionWorklistPath
      exists = [bool]($null -ne $actionWorklist)
      recordKind = $(if ($null -ne $actionWorklist) { [string]$actionWorklist.recordKind } else { $null })
      state = $(if ($null -ne $actionWorklist) { [string]$actionWorklist.worklistState } else { "missing" })
    }
    releaseFinalBlockerConvergence = [pscustomobject]@{
      path = $convergencePath
      exists = [bool]($null -ne $convergence)
      recordKind = $(if ($null -ne $convergence) { [string]$convergence.recordKind } else { $null })
      state = $(if ($null -ne $convergence) { [string]$convergence.convergenceState } else { "missing" })
    }
    releaseProofOwnerBackfillSummaryValidation = [pscustomobject]@{
      path = $summaryPath
      exists = [bool]($null -ne $summary)
      recordKind = $(if ($null -ne $summary) { [string]$summary.recordKind } else { $null })
      state = $(if ($null -ne $summary) { [string]$summary.validationState } else { "missing" })
    }
  }
  ownerProofPostbackDetectedCount = [int]$postbackDetectedCount
  missingInputCount = [int]$missingInputCount
  lanes = @($lanes)
  nextRequiredAction = "Owner must post back real evidence logs, records, hashes, host metadata, and strict validator outputs before public publish can be reconsidered."
  forbiddenProofSubstitutes = @(
    "template",
    "report",
    "matrix",
    "article",
    "command pack",
    "summary pack",
    "dry-run",
    "build-only",
    "sidecar-only",
    "screenshot-only",
    "Skipped=True",
    "blocked-by-cuda-driver",
    "local feed",
    "ProjectReference",
    "direct .nupkg"
  )
}

$outputFullPath = Resolve-RepoPath -Path $OutputPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputFullPath) | Out-Null
$result | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $outputFullPath -Encoding utf8

$markdownFullPath = Resolve-RepoPath -Path $MarkdownOutputPath
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('# Real Owner Proof Postback Release Decision')
$lines.Add('')
$lines.Add('Generated: 2026-07-08')
$lines.Add('')
$lines.Add('- Record kind: `real-owner-proof-postback-release-decision`')
$lines.Add('- Decision state: `blocked-owner-proof-not-posted-back`')
$lines.Add('- Can publish publicly: `false`')
$lines.Add('- Can close release issue: `false`')
$lines.Add('- Publish blocked reason: `owner-real-proof-missing`')
$lines.Add('')
$lines.Add('## Owner Proof Postback')
$lines.Add('')
$lines.Add('| Lane | Evidence directory exists | Record exists | Evidence files | Can promote |')
$lines.Add('|---|---:|---:|---:|---:|')
foreach ($lane in $lanes) {
  $lines.Add(('| `{0}` | `{1}` | `{2}` | {3} | `{4}` |' -f $lane.id, $lane.evidenceDirectoryExists.ToString().ToLowerInvariant(), $lane.recordExists.ToString().ToLowerInvariant(), $lane.evidenceFileCount, $lane.canPromote.ToString().ToLowerInvariant()))
}
$lines.Add('')
$lines.Add('## Boundary')
$lines.Add('')
$lines.Add('This snapshot is not proof and does not publish packages. It only records that Owner real proof has not yet been posted back.')
$lines | Set-Content -LiteralPath $markdownFullPath -Encoding utf8

$result | ConvertTo-Json -Depth 14
