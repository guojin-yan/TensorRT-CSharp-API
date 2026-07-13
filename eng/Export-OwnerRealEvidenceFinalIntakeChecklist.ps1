[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\final-release"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot $OutputDirectory
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

function ConvertTo-FlatStringLines {
  param([AllowNull()][object]$Value)

  foreach ($item in @($Value)) {
    if ($null -eq $item) {
      ""
    }
    elseif ($item -is [string]) {
      $item
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { [string]$child }
    }
    else {
      [string]$item
    }
  }
}

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $lines = @(ConvertTo-FlatStringLines -Value $InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ChecklistItem {
  param(
    [int]$Order,
    [string]$Group,
    [string]$Id,
    [string]$RequiredEvidence,
    [string]$TargetArtifact,
    [string]$Validator,
    [string[]]$ForbiddenSubstitutes
  )

  [pscustomobject]@{
    order = $Order
    group = $Group
    id = $Id
    requiredEvidence = $RequiredEvidence
    targetArtifact = $TargetArtifact
    validator = $Validator
    required = $true
    ready = $false
    valueState = "owner-real-evidence-required"
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
  }
}

$commonForbidden = @("local feed", "ProjectReference", "direct .nupkg", "pre-publish smoke", "template", "candidate", "dashboard", "runbook", "command pack", "build-only", "dependency-probe-only", "dry-run-only")
$items = @(
  New-ChecklistItem 1 "external-clean-consumer" "external-clean-consumer-restore-logs" "Repository-external CleanConsumer restore stdout/stderr/merged logs and SHA256 values." "owner-real-proof-staging-workspace/external-clean-consumer/restore.log" "eng/Test-OwnerRealProofStagingWorkspace.ps1 -Strict" $commonForbidden
  New-ChecklistItem 2 "external-clean-consumer" "external-clean-consumer-build-logs" "Repository-external CleanConsumer build stdout/stderr/merged logs, binlog, and SHA256 values." "owner-real-proof-staging-workspace/external-clean-consumer/build.log" "eng/Test-OwnerRealProofStagingWorkspace.ps1 -Strict" $commonForbidden
  New-ChecklistItem 3 "external-clean-consumer" "external-clean-consumer-smoke-logs" "Repository-external CleanConsumer runtime smoke stdout/stderr/report and SHA256 values." "owner-real-proof-staging-workspace/external-clean-consumer/smoke.stdout.log" "eng/Test-OwnerRealProofStagingWorkspace.ps1 -Strict" $commonForbidden
  New-ChecklistItem 4 "post-publish-clean-consumer" "post-publish-restore-build-smoke" "Public package source PostPublish restore/build/smoke stdout/stderr/merged logs and SHA256 values." "owner-real-proof-staging-workspace/post-publish/smoke.stdout.log" "eng/Test-PostPublishCleanConsumerRealProofFromOwnerResult.ps1 -Strict" $commonForbidden
  New-ChecklistItem 5 "public-package" "public-package-url-and-sha256" "Public package URL, package id, version, source channel, downloaded path, and SHA256." "artifacts/final-release/owner-public-publish-execution-result-candidate.json" "eng/Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict" $commonForbidden
  New-ChecklistItem 6 "public-package" "nuget-push-transcripts" "Owner NuGet push command transcript/stdout/stderr/merged transcript and SHA256, or explicit not-performed reason for GitHub-only lane." "artifacts/final-release/owner-public-publish-execution-result-candidate.json" "eng/Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict" $commonForbidden
  New-ChecklistItem 7 "public-package" "github-release-asset" "GitHub release URL, tag, asset URL, downloaded asset path, upload transcript, and SHA256, or explicit not-uploaded reason." "artifacts/final-release/owner-public-publish-execution-result-candidate.json" "eng/Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict" $commonForbidden
  New-ChecklistItem 8 "host-metadata" "host-runtime-metadata" "Host OS, CPU, GPU, driver, CUDA, TensorRT, cuDNN, .NET SDK/runtime, PowerShell version, environment transcript, and SHA256." "owner-real-proof-staging-workspace/external-clean-consumer/host-metadata.json" "eng/Test-OwnerRealProofStagingWorkspace.ps1 -Strict" $commonForbidden
  New-ChecklistItem 9 "package-metadata" "package-identity-and-dependency-graph" "Managed package, native bridge package, runtime package identities, URLs, SHA256 values, source channel, dependency graph, and SHA256." "owner-real-proof-staging-workspace/external-clean-consumer/package-metadata.json" "eng/Test-OwnerRealProofStagingWorkspace.ps1 -Strict" $commonForbidden
  New-ChecklistItem 10 "owner-governance" "rollback-review" "Owner rollback review with reviewer, reviewedAtUtc, decision, acceptedRisk, rollbackPlan, packageVersion, and evidenceBundleSha256." "owner-real-proof-staging-workspace/owner/rollback-review.json" "eng/Test-FinalOwnerRollbackReview.ps1 -Strict" $commonForbidden
  New-ChecklistItem 11 "owner-governance" "final-close-decision" "Owner final close decision with reviewer, reviewedAtUtc, decision, acceptedRisk, rollbackPlan, packageVersion, evidenceBundleSha256, and proof readiness confirmations." "owner-real-proof-staging-workspace/owner/final-close-decision.json" "eng/Test-FinalOwnerCloseDecision.ps1 -Strict" $commonForbidden
  New-ChecklistItem 12 "owner-governance" "final-release-close-approval" "Final release close approval real input after public publish and PostPublish proof acceptance." "artifacts/final-release/final-release-close-approval-real-input-from-owner-result-validation.json" "eng/Test-OwnerRealEvidenceEndToEndReleaseGate.ps1 -Strict" $commonForbidden
  New-ChecklistItem 13 "owner-confirmations" "forbidden-substitute-confirmations" "Owner confirmations that local feed, ProjectReference, direct .nupkg, pre-publish smoke, template, candidate, dashboard, runbook, command pack, build-only, and dependency-probe-only were not used as proof." "owner-real-proof-staging-workspace/owner/owner-confirmations.json" "eng/Test-ReleaseEvidenceClassificationAudit.ps1 -Strict" $commonForbidden
  New-ChecklistItem 14 "final-gates" "final-public-publish-acceptance" "Final public publish acceptance gate ready after real Owner public publish result, PostPublish proof, close approval, and final owner convergence." "artifacts/final-release/final-public-publish-acceptance-gate.json" "eng/Test-FinalPublicPublishAcceptanceGate.ps1 -Strict" $commonForbidden
  New-ChecklistItem 15 "final-gates" "owner-real-evidence-end-to-end-release" "Owner real evidence end-to-end release gate ready with staging, public publish, PostPublish, close approval, acceptance, convergence, and classification audit all passing." "artifacts/final-release/owner-real-evidence-end-to-end-release-gate.json" "eng/Test-OwnerRealEvidenceEndToEndReleaseGate.ps1 -Strict" $commonForbidden
)

$record = [ordered]@{
  recordKind = "owner-real-evidence-final-intake-checklist"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  checklistState = "blocked-owner-real-evidence-final-intake-required"
  checklistItemCount = $items.Count
  blockedChecklistItemCount = $items.Count
  readyChecklistItemCount = 0
  ownerActionRequired = $true
  failedBlockerCountIsNotProof = $true
  checklistItems = @($items)
  nonSubstituteProofKinds = @($commonForbidden + @("owner real evidence final intake checklist", "final intake checklist remains blocked until real Owner evidence"))
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner real evidence final intake checklist is an owner handoff checklist only. It never executes dotnet nuget push, never stores tokens, never uploads packages, and never turns local feed, ProjectReference, direct .nupkg, pre-publish smoke, templates, candidates, dashboards, runbooks, command packs, build-only output, dependency-probe output, or failedBlockerCount=0 into proof. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputDirectory "owner-real-evidence-final-intake-checklist.json"
$markdownPath = Join-Path $OutputDirectory "owner-real-evidence-final-intake-checklist.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 16)

$rows = foreach ($item in $items) {
  "| ``$($item.order)`` | $(ConvertTo-MarkdownCell $item.group) | $(ConvertTo-MarkdownCell $item.id) | $(ConvertTo-MarkdownCell $item.requiredEvidence) |"
}

Write-Utf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Real Evidence Final Intake Checklist",
  "",
  "- checklistState: ``$($record.checklistState)``",
  "- checklistItemCount: ``$($record.checklistItemCount)``",
  "- blockedChecklistItemCount: ``$($record.blockedChecklistItemCount)``",
  "- performsPublish: ``False``",
  "- canPublishPublicly: ``False``",
  "- failedBlockerCountIsNotProof: ``True``",
  "",
  "| Order | Group | ID | Required Evidence |",
  "|---:|---|---|---|",
  @($rows),
  "",
  "## Boundary",
  "",
  $record.boundary
)

Write-Host "OwnerRealEvidenceFinalIntakeChecklistState=$($record.checklistState) Items=$($record.checklistItemCount)"
