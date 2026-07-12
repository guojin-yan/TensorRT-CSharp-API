[CmdletBinding()]
param(
  [string]$OwnerStagingRoot = "",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$RequireExistingFiles,
  [switch]$RequireHashMatch,
  [switch]$FailOnNotProof
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

function New-StagingMapping {
  param([string]$Lane, [string]$SourcePath, [string]$TargetJson, [string]$TargetField, [string]$TargetHashField)
  [pscustomobject]@{
    lane = $Lane
    sourcePath = $SourcePath
    targetJson = $TargetJson
    targetField = $TargetField
    targetHashField = $TargetHashField
    ownerActionRequired = $true
    passed = $false
  }
}

$contract = Read-JsonOrNull $RepositoryRoot "artifacts\final-release\owner-real-proof-staging-workspace-contract.json"
if ($null -eq $contract) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerRealProofStagingWorkspaceContract.ps1") -RepositoryRoot $RepositoryRoot
  $contract = Read-JsonOrNull $RepositoryRoot "artifacts\final-release\owner-real-proof-staging-workspace-contract.json"
}

$findings = New-Object System.Collections.Generic.List[object]
if ([string]::IsNullOrWhiteSpace($OwnerStagingRoot)) {
  $findings.Add((New-OwnerFinding "owner-staging-root-missing" "action-required" "missing-field" "OwnerStagingRoot was not supplied.")) | Out-Null
  $resolvedRoot = ""
} else {
  $resolvedRoot = Resolve-OwnerPath $RepositoryRoot $OwnerStagingRoot
  if ($RequireExistingFiles.IsPresent -and -not (Test-Path -LiteralPath $resolvedRoot -PathType Container)) {
    $findings.Add((New-OwnerFinding "owner-staging-root-exists" "action-required" "missing-file" "Owner staging root does not exist.")) | Out-Null
  }
}

$mappings = @(
  New-StagingMapping "external-clean-consumer" "external-clean-consumer/restore.log" "external-clean-consumer-execution-result.owner.json" "restoreLogPath" "restoreLogSha256"
  New-StagingMapping "external-clean-consumer" "external-clean-consumer/build.log" "external-clean-consumer-execution-result.owner.json" "buildLogPath" "buildLogSha256"
  New-StagingMapping "external-clean-consumer" "external-clean-consumer/smoke.stdout.log" "external-clean-consumer-execution-result.owner.json" "smokeStdoutPath" "smokeStdoutSha256"
  New-StagingMapping "external-clean-consumer" "external-clean-consumer/smoke.stderr.log" "external-clean-consumer-execution-result.owner.json" "smokeStderrPath" "smokeStderrSha256"
  New-StagingMapping "external-clean-consumer" "external-clean-consumer/native-assets.json" "external-clean-consumer-execution-result.owner.json" "nativeAssetListingPath" "nativeAssetListingSha256"
  New-StagingMapping "external-clean-consumer" "external-clean-consumer/host-metadata.json" "external-clean-consumer-execution-result.owner.json" "hostMetadataPath" "hostMetadataSha256"
  New-StagingMapping "external-clean-consumer" "external-clean-consumer/package-metadata.json" "external-clean-consumer-execution-result.owner.json" "packageMetadataPath" "packageMetadataSha256"
  New-StagingMapping "post-publish" "post-publish/downloaded-packages.json" "post-publish-clean-consumer-proof-result.owner.json" "downloadedPackagesPath" "downloadedPackagesSha256"
  New-StagingMapping "post-publish" "post-publish/install.log" "post-publish-clean-consumer-proof-result.owner.json" "installLogPath" "installLogSha256"
  New-StagingMapping "post-publish" "post-publish/smoke.stdout.log" "post-publish-clean-consumer-proof-result.owner.json" "smokeStdoutPath" "smokeStdoutSha256"
  New-StagingMapping "post-publish" "post-publish/smoke.stderr.log" "post-publish-clean-consumer-proof-result.owner.json" "smokeStderrPath" "smokeStderrSha256"
  New-StagingMapping "post-publish" "post-publish/host-metadata.json" "post-publish-clean-consumer-proof-result.owner.json" "hostMetadataPath" "hostMetadataSha256"
  New-StagingMapping "owner" "owner/rollback-review.json" "final-owner-rollback-review.owner.json" "rollbackReviewPath" "rollbackReviewSha256"
  New-StagingMapping "owner" "owner/final-close-decision.json" "final-owner-close-decision.owner.json" "finalCloseDecisionPath" "finalCloseDecisionSha256"
  New-StagingMapping "owner" "owner/owner-confirmations.json" "owner-real-proof-confirmations.owner.json" "ownerConfirmationsPath" "ownerConfirmationsSha256"
)

foreach ($mapping in $mappings) {
  $fullPath = if ([string]::IsNullOrWhiteSpace($resolvedRoot)) { "" } else { Resolve-OwnerPath $resolvedRoot $mapping.sourcePath }
  if ([string]::IsNullOrWhiteSpace($fullPath)) {
    $findings.Add((New-OwnerFinding "$($mapping.targetField)-missing-root" "action-required" "missing-file" "Cannot resolve $($mapping.sourcePath) without OwnerStagingRoot.")) | Out-Null
    continue
  }
  if ($RequireExistingFiles.IsPresent -and -not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    $findings.Add((New-OwnerFinding "$($mapping.targetField)-exists" "action-required" "missing-file" "Required staging file is missing: $($mapping.sourcePath)")) | Out-Null
    continue
  }
  if ($RequireHashMatch.IsPresent -and (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    $hash = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash
    if (-not (Test-Sha256Text $hash)) {
      $findings.Add((New-OwnerFinding "$($mapping.targetHashField)-sha256" "action-required" "missing-sha256" "SHA256 could not be computed for $($mapping.sourcePath).")) | Out-Null
    }
  }
}

foreach ($required in @("external-clean-consumer/host-metadata.json", "post-publish/host-metadata.json", "owner/owner-confirmations.json")) {
  if ([string]::IsNullOrWhiteSpace($resolvedRoot)) { continue }
  $path = Resolve-OwnerPath $resolvedRoot $required
  if ($RequireExistingFiles.IsPresent -and -not (Test-Path -LiteralPath $path -PathType Leaf)) {
    $category = if ($required.Contains("host-metadata")) { "missing-host-metadata" } else { "missing-owner-confirmation" }
    $findings.Add((New-OwnerFinding ($required.Replace("/", "-") + "-required") "action-required" $category "Required owner staging metadata is missing: $required")) | Out-Null
  }
}

$failedBlockers = @($findings | Where-Object { [string]$_.severity -eq "blocker" })
$failedActionRequired = @($findings | Where-Object { [string]$_.severity -eq "action-required" })
$proofReady = $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0 -and -not [string]::IsNullOrWhiteSpace($OwnerStagingRoot)
$state = if ($proofReady) { "owner-real-proof-staging-workspace-import-ready" } else { "blocked-owner-real-proof-staging-workspace-required" }

$candidate = [pscustomobject]@{
  recordKind = "owner-real-proof-staging-workspace-candidate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  candidateState = if ($proofReady) { "owner-real-proof-staging-workspace-candidate-ready-for-strict-import" } else { "blocked-owner-real-proof-staging-workspace-candidate" }
  ownerStagingRoot = $OwnerStagingRoot
  resolvedOwnerStagingRoot = $resolvedRoot
  mappingCount = $mappings.Count
  mappings = @($mappings)
  proofCandidateReady = $false
  readyForStrictImport = $proofReady
  ownerActionRequired = -not $proofReady
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner staging workspace candidate only maps files into owner input candidates. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$import = [pscustomobject]@{
  recordKind = "owner-real-proof-staging-workspace-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = $state
  ownerStagingRoot = $OwnerStagingRoot
  resolvedOwnerStagingRoot = $resolvedRoot
  requireExistingFiles = $RequireExistingFiles.IsPresent
  requireHashMatch = $RequireHashMatch.IsPresent
  failOnNotProof = $FailOnNotProof.IsPresent
  findingCount = $findings.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  findings = @($findings.ToArray())
  candidatePath = "artifacts/final-release/owner-real-proof-staging-workspace-candidate.json"
  readyForStrictImport = $proofReady
  proofCandidateReady = $false
  ownerActionRequired = -not $proofReady
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner staging workspace import validates local owner file layout only. Strict External CleanConsumer and PostPublish import validators must still accept real evidence; this import is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$importPath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-import.json"
$importMdPath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-import.md"
$candidatePath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-candidate.json"
$candidateMdPath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-candidate.md"
$import | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $importPath -Encoding utf8
$candidate | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $candidatePath -Encoding utf8

$findingRows = foreach ($finding in $findings) {
  "| ``$(ConvertTo-MarkdownCell $finding.id)`` | ``$(ConvertTo-MarkdownCell $finding.severity)`` | ``$(ConvertTo-MarkdownCell $finding.category)`` | $(ConvertTo-MarkdownCell $finding.message) |"
}
Write-Utf8File -LiteralPath $importMdPath -InputObject @("# Owner Real Proof Staging Workspace Import", "", "- importState: ``$state``", "- readyForStrictImport: ``$proofReady``", "- failedActionRequiredCount: ``$($failedActionRequired.Count)``", "", "| ID | Severity | Category | Message |", "|---|---|---|---|", @($findingRows), "", "## Boundary", "", $import.boundary)
Write-Utf8File -LiteralPath $candidateMdPath -InputObject @("# Owner Real Proof Staging Workspace Candidate", "", "- candidateState: ``$($candidate.candidateState)``", "- readyForStrictImport: ``$($candidate.readyForStrictImport)``", "- proofCandidateReady: ``False``", "", "## Boundary", "", $candidate.boundary)

Write-Host "OwnerRealProofStagingWorkspaceImportState=$state ReadyForStrictImport=$proofReady FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"
if ($FailOnNotProof.IsPresent -and -not $proofReady) { throw "Owner real proof staging workspace is not ready for strict import." }
