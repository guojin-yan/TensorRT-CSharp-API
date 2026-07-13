[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$ConvergencePath = "artifacts\final-release\release-final-blocker-convergence.json",
  [string]$OutputPath = "artifacts\final-release\owner-real-proof-final-action-worklist.json",
  [string]$MarkdownOutputPath = "artifacts\final-release\owner-real-proof-final-action-worklist.md"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
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

function Get-OwnerCommand {
  param([string]$LaneId)

  switch ($LaneId) {
    "real-model-runtime" {
      return @(
        "Run real model runtime cases on a compatible CUDA/TensorRT host for det, cls, seg, obb, pose, and sem.",
        "Archive stdout/stderr logs under artifacts/final-release/owner-evidence/real-model-runtime/.",
        "Populate artifacts/final-release/real-case-evidence-record.json with real hashes and host metadata.",
        "Run: pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record.json -FailOnNotProof"
      )
    }
    "package-consumer-runtime" {
      return @(
        "Create an external clean consumer project that consumes the public package source, not a local feed, ProjectReference, or direct .nupkg shortcut.",
        "Archive restore/build/runtime smoke logs under artifacts/final-release/owner-evidence/package-consumer-runtime/.",
        "Populate package-consumer-runtime-proof-record.json with real package hashes, smoke hashes, and host metadata.",
        "Run: pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -InputPath .\artifacts\final-release\package-consumer-runtime-proof-record.json -Strict -RequireExistingLog -FailOnNotProof"
      )
    }
    "post-publish-verification" {
      return @(
        "After public publish, create a clean consumer from the public package source.",
        "Archive post-publish restore and runtime smoke logs under artifacts/final-release/owner-evidence/post-publish-verification/.",
        "Populate post-publish-verification-record.json with real package hashes, smoke hashes, and host metadata.",
        "Run: pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath .\artifacts\final-release\post-publish-verification-record.json -Strict -RequireExistingLog -FailOnNotProof"
      )
    }
    "release-issue-close" {
      return @(
        "Wait until real-model-runtime, package-consumer-runtime, and post-publish-verification are strict-validator passed.",
        "Archive upstream validator outputs under artifacts/final-release/owner-evidence/release-issue-close/.",
        "Populate release-issue-close-record.json from real owner release-close decision input.",
        "Run: pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath .\artifacts\final-release\release-issue-close-record.json -FailOnNotCloseReady"
      )
    }
    default {
      return @("Provide missing real evidence and rerun the lane strict validator.")
    }
  }
}

$convergenceFullPath = Resolve-RepoPath -Path $ConvergencePath
if (-not (Test-Path -LiteralPath $convergenceFullPath -PathType Leaf)) {
  throw "Convergence file not found: $ConvergencePath"
}

$convergence = Get-Content -LiteralPath $convergenceFullPath -Raw | ConvertFrom-Json
$lanes = @($convergence.lanes)

$workItems = foreach ($lane in $lanes) {
  $missingInputs = @($lane.missingInputs)
  [pscustomobject]@{
    id = [string]$lane.id
    currentValidationState = [string]$lane.currentValidationState
    canPromote = [bool]$lane.canPromote
    recordExists = [bool]$lane.recordExists
    validatorOutputExists = [bool]$lane.validatorOutputExists
    missingInputs = $missingInputs
    missingInputCount = [int]$lane.missingInputCount
    ownerCommands = Get-OwnerCommand -LaneId ([string]$lane.id)
    strictValidatorCommand = [string]$lane.strictValidatorCommand
    promotionCriteria = @($lane.promotionCriteria)
    ownerActionRequired = $lane.ownerActionRequired
  }
}

$remainingWorkItemCount = @($workItems | Where-Object { -not $_.canPromote }).Count
$missingInputCount = ($workItems | ForEach-Object { $_.missingInputCount } | Measure-Object -Sum).Sum
if ($null -eq $missingInputCount) {
  $missingInputCount = 0
}

$result = [pscustomobject]@{
  recordKind = "owner-real-proof-final-action-worklist"
  generatedAtLocal = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
  sourceConvergencePath = "artifacts/final-release/release-final-blocker-convergence.json"
  sourceConvergenceRecordKind = [string]$convergence.recordKind
  worklistState = "blocked-owner-action-required"
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  performsPublish = $false
  approvesPublicRelease = $false
  proofBoundary = "This worklist is read-only and owner-action-only. It does not create proof, fabricate logs, fabricate hashes, run dotnet nuget publish, promote lanes, or close release issues."
  forbiddenProofSubstitutes = @($convergence.forbiddenProofSubstitutes)
  remainingWorkItemCount = [int]$remainingWorkItemCount
  missingInputCount = [int]$missingInputCount
  workItems = @($workItems)
  nextGateCommands = @(
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseProofOwnerBackfillSummary.ps1 -OutputPath artifacts\final-release\release-proof-owner-backfill-summary-validation.json",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseFinalBlockerConvergence.ps1 -SummaryPath artifacts\final-release\release-proof-owner-backfill-summary-validation.json -OutputPath artifacts\final-release\release-final-blocker-convergence.json",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRealProofFinalActionWorklist.ps1"
  )
}

$outputFullPath = Resolve-RepoPath -Path $OutputPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputFullPath) | Out-Null
Write-Utf8File -LiteralPath $outputFullPath -InputObject ($result | ConvertTo-Json -Depth 14)
$markdownFullPath = Resolve-RepoPath -Path $MarkdownOutputPath
$markdownLines = [System.Collections.Generic.List[string]]::new()
$markdownLines.Add('# Owner Real Proof Final Action Worklist')
$markdownLines.Add('')
$markdownLines.Add('Generated: 2026-07-08')
$markdownLines.Add('')
$markdownLines.Add('This worklist is derived from `artifacts/final-release/release-final-blocker-convergence.json`. It remains blocked until Owner supplies real runtime proof, strict validator output, logs, SHA256 values, and host metadata.')
$markdownLines.Add('')
$markdownLines.Add('- Record kind: `owner-real-proof-final-action-worklist`')
$markdownLines.Add('- Can publish publicly: `false`')
$markdownLines.Add('- Can close release issue: `false`')
$markdownLines.Add(('- Remaining work items: `{0}`' -f $remainingWorkItemCount))
$markdownLines.Add(('- Missing inputs: `{0}`' -f $missingInputCount))
$markdownLines.Add('')
$markdownLines.Add('## Work Items')
$markdownLines.Add('')
$markdownLines.Add('| Lane | Missing inputs | Can promote |')
$markdownLines.Add('|---|---:|---:|')
foreach ($item in $workItems) {
  $markdownLines.Add(('| `{0}` | {1} | `{2}` |' -f $item.id, $item.missingInputCount, $item.canPromote.ToString().ToLowerInvariant()))
}
$markdownLines.Add('')
$markdownLines.Add('## Owner Commands')
$markdownLines.Add('')
foreach ($item in $workItems) {
  $markdownLines.Add(('### {0}' -f $item.id))
  $markdownLines.Add('')
  foreach ($command in @($item.ownerCommands)) {
    $markdownLines.Add(('- {0}' -f $command))
  }
  $markdownLines.Add(('- Strict validator: `{0}`' -f $item.strictValidatorCommand))
  $markdownLines.Add('')
}
$markdownLines.Add('## Forbidden Substitutes')
$markdownLines.Add('')
$markdownLines.Add('The following cannot be treated as proof: template, report, matrix, article, command pack, summary pack, dry-run, build-only, sidecar-only, screenshot-only, `Skipped=True`, `blocked-by-cuda-driver`, local feed, ProjectReference, or direct `.nupkg`.')
$markdownLines.Add('')
$markdownLines.Add('## Regenerate')
$markdownLines.Add('')
$markdownLines.Add('```powershell')
foreach ($command in @($result.nextGateCommands)) {
  $markdownLines.Add($command)
}
$markdownLines.Add('```')
$markdownLines | Set-Content -LiteralPath $markdownFullPath -Encoding utf8

$result | ConvertTo-Json -Depth 14
