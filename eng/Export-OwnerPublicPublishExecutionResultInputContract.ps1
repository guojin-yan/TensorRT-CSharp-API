[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPublicPublishExecutionResultCommon.ps1")

$ctx = Initialize-OwnerPublicPublishContext -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
$requiredFields = @(Get-OwnerPublicPublishRequiredFields)
$fieldGroups = @($requiredFields | Group-Object group | ForEach-Object { [pscustomobject]@{ group = $_.Name; requiredFieldCount = $_.Count } })

$record = [ordered]@{
  recordKind = "owner-public-publish-execution-result-input-contract"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  contractState = "blocked-owner-public-publish-execution-result-input-required"
  requiredFieldCount = $requiredFields.Count
  blockedRequiredFieldCount = $requiredFields.Count
  readyRequiredFieldCount = 0
  requiredGroupCount = $fieldGroups.Count
  requiredGroups = @($fieldGroups)
  requiredFields = @($requiredFields)
  requiredOwnerEvidenceAreas = @(
    "managed/runtime package id/version/public source/url/sha256",
    "GitHub release asset url/sha256 or explicit not-uploaded reason",
    "NuGet push transcript/stdout/stderr/hash, publish command plan hash, managed/runtime publish command hash",
    "clean external consumer restore/build/runtime smoke stdout/stderr/merged transcript/hash",
    "strict validator output/hash",
    "host identity plus source runner queue/infrastructure/owner action status",
    "package identity: managed/native bridge/runtime/source channel",
    "release notes, rollback, release issue, final package approval",
    "Owner reviewer/timestamp/signature/approval id plus owner authorization id/scope",
    "non-substitute confirmations and forbidden substitute scan path/hash"
  )
  forbiddenSubstituteMarkers = @(
    "local feed",
    "ProjectReference",
    "direct .nupkg",
    "build-only",
    "dependency-probe-only",
    "dry-run-only",
    "candidate",
    "dashboard",
    "runbook",
    "template",
    "draft",
    "manual approval",
    "queued GitHub Actions run",
    "missing self-hosted runner",
    "sidecar-only",
    "TensorRtExec report"
  )
  strictProofRequirements = @(
    "owner authorization",
    "public publish result",
    "post-publish proof",
    "clean consumer runtime proof",
    "release close owner decision"
  )
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner public publish execution result input contract only. It is not proof, not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release. Local feed, ProjectReference, direct .nupkg, dashboard, dry-run, manual approval, queued workflow, missing runner, sidecar-only, and TensorRtExec report are non-proof substitutes."
}

$jsonPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-input-contract.json"
$markdownPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-input-contract.md"
Write-OwnerUtf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 32)

$rows = foreach ($group in $fieldGroups) {
  "| $(ConvertTo-OwnerMarkdownCell $group.group) | ``$($group.requiredFieldCount)`` |"
}

Write-OwnerUtf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Public Publish Execution Result Input Contract",
  "",
  "- contractState: ``$($record.contractState)``",
  "- requiredFieldCount: ``$($record.requiredFieldCount)``",
  "- blockedRequiredFieldCount: ``$($record.blockedRequiredFieldCount)``",
  "- performsPublish: ``$($record.performsPublish)``",
  "- canPublishPublicly: ``$($record.canPublishPublicly)``",
  "- canCloseReleaseIssue: ``$($record.canCloseReleaseIssue)``",
  "",
  "| Group | Required Fields |",
  "| --- | --- |",
  @($rows),
  "",
  "## Boundary",
  "",
  $record.boundary
)

Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
Write-Host "RequiredFieldCount=$($record.requiredFieldCount) ContractState=$($record.contractState)"
