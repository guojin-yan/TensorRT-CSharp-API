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

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function New-SubstituteCheck {
  param([string]$Id, [string]$ForbiddenKind, [string]$Scope, [string]$OwnerRequiredEvidence)
  [pscustomobject]@{
    id = $Id
    forbiddenKind = $ForbiddenKind
    scope = $Scope
    ownerRequiredEvidence = $OwnerRequiredEvidence
    resultState = "owner-real-input-required"
    passed = $false
    detected = $null
    ready = $false
  }
}

$checks = @(
  New-SubstituteCheck -Id "no-local-nupkg-public-result" -ForbiddenKind "local .nupkg" -Scope "public publish result" -OwnerRequiredEvidence "Downloaded public package URL and SHA256"
  New-SubstituteCheck -Id "no-local-feed-public-result" -ForbiddenKind "local feed" -Scope "public publish result" -OwnerRequiredEvidence "Public source URL and transcript"
  New-SubstituteCheck -Id "no-project-reference-clean-consumer" -ForbiddenKind "ProjectReference" -Scope "clean consumer proof" -OwnerRequiredEvidence "Repository-external consumer project file or package lock"
  New-SubstituteCheck -Id "no-direct-nupkg-clean-consumer" -ForbiddenKind "direct nupkg" -Scope "clean consumer proof" -OwnerRequiredEvidence "Restore log showing public package source"
  New-SubstituteCheck -Id "no-template-record" -ForbiddenKind "template" -Scope "all owner proof records" -OwnerRequiredEvidence "Owner-filled record with reviewer and log hashes"
  New-SubstituteCheck -Id "no-dry-run-result" -ForbiddenKind "dry-run" -Scope "publish and smoke commands" -OwnerRequiredEvidence "Real command transcript and smoke exit code"
  New-SubstituteCheck -Id "no-dashboard-substitute" -ForbiddenKind "dashboard" -Scope "release evidence" -OwnerRequiredEvidence "Validator-passing public publish and clean consumer records"
  New-SubstituteCheck -Id "no-audit-pack-substitute" -ForbiddenKind "audit pack" -Scope "release evidence" -OwnerRequiredEvidence "Real owner proof records, not an audit summary"
  New-SubstituteCheck -Id "no-local-only-artifact-scan" -ForbiddenKind "local-only artifact scan" -Scope "package proof" -OwnerRequiredEvidence "Public channel package URL and downloaded package hash"
  New-SubstituteCheck -Id "no-manual-approval-only" -ForbiddenKind "manual approval" -Scope "release close evidence" -OwnerRequiredEvidence "Strict validator-passing owner authorization plus public publish/post-publish proofs"
  New-SubstituteCheck -Id "no-queued-workflow-public-result" -ForbiddenKind "queued GitHub Actions run" -Scope "publish execution" -OwnerRequiredEvidence "Completed run transcript and real public package URL/hash"
  New-SubstituteCheck -Id "no-missing-self-hosted-runner" -ForbiddenKind "missing self-hosted runner" -Scope "publish and package-consumer proof" -OwnerRequiredEvidence "Available runner and completed restore/build/smoke logs"
  New-SubstituteCheck -Id "no-sidecar-only-proof" -ForbiddenKind "sidecar-only" -Scope "runtime/package proof" -OwnerRequiredEvidence "Clean consumer runtime smoke with package download hashes"
  New-SubstituteCheck -Id "no-tensorrtexec-report-only" -ForbiddenKind "TensorRtExec report" -Scope "runtime/package proof" -OwnerRequiredEvidence "Strict PostPublish clean consumer validator output"
)

$record = [pscustomobject]@{
  recordKind = "public-publish-forbidden-substitute-scan"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  scanState = "blocked-public-publish-forbidden-substitute-scan-owner-proof-required"
  substituteCheckCount = $checks.Count
  blockedSubstituteCheckCount = $checks.Count
  passedSubstituteCheckCount = 0
  substituteChecks = @($checks)
  sourceArtifacts = @(
    "artifacts/final-release/public-publish-real-result-record-draft-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.json",
    "artifacts/final-release/public-publish-real-result-owner-input-contract-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json"
  )
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  boundary = "This scan lists forbidden substitutes for owner review only. It covers local feed, ProjectReference, direct nupkg, manual approval, queued workflow, missing self-hosted runner, sidecar-only, and TensorRtExec report substitutes. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "public-publish-forbidden-substitute-scan.json"
$markdownPath = Join-Path $OutputRoot "public-publish-forbidden-substitute-scan.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Public Publish Forbidden Substitute Scan",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| scanState | ``$($record.scanState)`` |",
  "| substituteCheckCount | ``$($record.substituteCheckCount)`` |",
  "| blockedSubstituteCheckCount | ``$($record.blockedSubstituteCheckCount)`` |",
  "| canPublishPublicly | ``$($record.canPublishPublicly)`` |",
  "| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |",
  "",
  "## Substitute Checks",
  "",
  "| ID | Forbidden Kind | Scope | Required Owner Evidence |",
  "| --- | --- | --- | --- |"
)

foreach ($check in $checks) {
  $markdown += "| $($check.id) | $($check.forbiddenKind) | $($check.scope) | $($check.ownerRequiredEvidence) |"
}

$markdown += @("", "## Boundary", "", $record.boundary)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public publish forbidden substitute scan written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ScanState=$($record.scanState) Checks=$($record.substituteCheckCount) Blocked=$($record.blockedSubstituteCheckCount)"
