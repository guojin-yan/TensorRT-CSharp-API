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

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

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

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function New-FinalProofStep {
  param(
    [int]$Order,
    [string]$Id,
    [string]$EvidenceItemId,
    [string]$SourceArtifact,
    [string]$RequiredReadyState,
    [string]$CurrentState,
    [string]$OwnerAction
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    evidenceItemId = $EvidenceItemId
    sourceArtifact = $SourceArtifact
    requiredReadyState = $RequiredReadyState
    currentState = $CurrentState
    blocked = $true
    proofAccepted = $false
    ownerAction = $OwnerAction
    ownerActionRequired = $true
    performsPublish = $false
    performsRuntimeExecution = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "not runtime proof; not post-publish proof; not publish approval; not release close approval; not package push"
  }
}

$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$oneScreenPack = Read-JsonOrNull "artifacts\final-release\final-owner-execution-one-screen-pack.json"
$finalBridgeValidation = Read-JsonOrNull "artifacts\final-release\final-public-release-closure-bridge-validation.json"
$releaseIssueDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-owner-decision-input-validation.json"

$bundleEvidenceItems = Convert-ToArray (Get-PropertyOrDefault -Object $releaseEvidence -Name "evidenceItems" -DefaultValue @())
$bundleEvidenceItemIds = @($bundleEvidenceItems | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$bundleSourceArtifacts = @(Convert-ToArray (Get-PropertyOrDefault -Object $releaseEvidence -Name "sourceArtifacts" -DefaultValue @()) | ForEach-Object { [string]$_ })
$bundleNonSubstitutes = @(Convert-ToArray (Get-PropertyOrDefault -Object $releaseEvidence -Name "nonSubstituteProofKinds" -DefaultValue @()) | ForEach-Object { [string]$_ })
$oneScreenProofPath = Convert-ToArray (Get-PropertyOrDefault -Object $oneScreenPack -Name "finalPublicProofPath" -DefaultValue @())
$oneScreenProofPathIds = @($oneScreenProofPath | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })

$requiredSourceArtifacts = @(
  "artifacts/final-release/github-actions-run-evidence-import-validation.json",
  "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json",
  "artifacts/final-release/public-package-download-proof-candidate-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json",
  "artifacts/final-release/final-public-release-closure-bridge.json",
  "artifacts/final-release/final-public-release-closure-bridge-validation.json",
  "artifacts/final-release/release-issue-close-owner-decision-input.template.json",
  "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
  "artifacts/final-release/final-owner-execution-one-screen-pack.json",
  "artifacts/final-release/final-owner-execution-one-screen-pack-validation.json"
)

$requiredEvidenceItemIds = @(
  "remote-ci-and-public-publish-proof-backfill-gate",
  "owner-public-publish-execution-result-candidate",
  "post-publish-clean-consumer-proof-result-import",
  "post-publish-clean-consumer-proof-result-candidate",
  "final-public-release-closure-bridge",
  "release-issue-close-owner-decision-input",
  "final-owner-execution-one-screen-pack"
)

$requiredOneScreenProofStepIds = @(
  "github-actions-run-evidence",
  "owner-public-publish-result",
  "public-package-download-proof",
  "post-publish-clean-consumer-proof-result",
  "final-public-release-closure-bridge",
  "release-issue-close-owner-decision-input"
)

$sourceStates = [ordered]@{
  githubActionsRunEvidence = [string](Get-PropertyOrDefault -Object (Read-JsonOrNull "artifacts\final-release\github-actions-run-evidence-import-validation.json") -Name "validationState" -DefaultValue "missing-github-actions-run-evidence-import-validation")
  ownerPublicPublishResult = [string](Get-PropertyOrDefault -Object (Read-JsonOrNull "artifacts\final-release\owner-public-publish-execution-result-candidate-validation.json") -Name "validationState" -DefaultValue "missing-owner-public-publish-execution-result-candidate-validation")
  publicPackageDownloadProof = [string](Get-PropertyOrDefault -Object (Read-JsonOrNull "artifacts\final-release\public-package-download-proof-candidate-validation.json") -Name "validationState" -DefaultValue "missing-public-package-download-proof-candidate-validation")
  postPublishCleanConsumerProofResult = [string](Get-PropertyOrDefault -Object (Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-validation.json") -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-result-validation")
  finalPublicReleaseClosureBridge = [string](Get-PropertyOrDefault -Object $finalBridgeValidation -Name "validationState" -DefaultValue "missing-final-public-release-closure-bridge-validation")
  releaseIssueCloseOwnerDecisionInput = [string](Get-PropertyOrDefault -Object $releaseIssueDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-close-owner-decision-input-validation")
}

$finalProofChain = @(
  New-FinalProofStep -Order 1 -Id "github-actions-run-evidence" -EvidenceItemId "remote-ci-and-public-publish-proof-backfill-gate" -SourceArtifact "artifacts/final-release/github-actions-run-evidence-import-validation.json" -RequiredReadyState "github-actions-run-evidence-ready" -CurrentState $sourceStates.githubActionsRunEvidence -OwnerAction "Import real GitHub Actions run URL, run id, head SHA, conclusion, workflow log SHA256, and artifact manifest SHA256."
  New-FinalProofStep -Order 2 -Id "owner-public-publish-result" -EvidenceItemId "owner-public-publish-execution-result-candidate" -SourceArtifact "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json" -RequiredReadyState "owner-public-publish-execution-result-candidate-ready" -CurrentState $sourceStates.ownerPublicPublishResult -OwnerAction "Backfill public package URL/version/SHA, publish transcript hash, GitHub release asset URL/hash, rollback review, and GitHub Actions linkage."
  New-FinalProofStep -Order 3 -Id "public-package-download-proof" -EvidenceItemId "remote-ci-and-public-publish-proof-backfill-gate" -SourceArtifact "artifacts/final-release/public-package-download-proof-candidate-validation.json" -RequiredReadyState "public-package-download-proof-candidate-ready" -CurrentState $sourceStates.publicPackageDownloadProof -OwnerAction "Download public managed/runtime packages and record public URLs, package identities, SHA256 hashes, and GitHub release asset linkage."
  New-FinalProofStep -Order 4 -Id "post-publish-clean-consumer-proof-result" -EvidenceItemId "post-publish-clean-consumer-proof-result-import" -SourceArtifact "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json" -RequiredReadyState "post-publish-clean-consumer-proof-result-validation-ready plus proofCandidateReady=true and sourceProofLinkageReady=true" -CurrentState $sourceStates.postPublishCleanConsumerProofResult -OwnerAction "Run repository-external clean consumer restore/build/runtime smoke after publication and link ready upstream public proof records."
  New-FinalProofStep -Order 5 -Id "final-public-release-closure-bridge" -EvidenceItemId "final-public-release-closure-bridge" -SourceArtifact "artifacts/final-release/final-public-release-closure-bridge-validation.json" -RequiredReadyState "final-public-release-closure-bridge-ready-for-owner-close-review" -CurrentState $sourceStates.finalPublicReleaseClosureBridge -OwnerAction "Refresh final bridge and verify URL/version/SHA/source-linkage consistency across all public proof lanes."
  New-FinalProofStep -Order 6 -Id "release-issue-close-owner-decision-input" -EvidenceItemId "release-issue-close-owner-decision-input" -SourceArtifact "artifacts/final-release/release-issue-close-owner-decision-input-validation.json" -RequiredReadyState "release-issue-close-owner-decision-input-ready" -CurrentState $sourceStates.releaseIssueCloseOwnerDecisionInput -OwnerAction "Record final release issue close decision only after final bridge, post-publish proof linkage, evidence bundle hash, and rollback review are ready."
)

$evidenceCoverage = foreach ($id in $requiredEvidenceItemIds) {
  $item = $bundleEvidenceItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq $id } | Select-Object -First 1
  $boundary = [string](Get-PropertyOrDefault -Object $item -Name "boundary" -DefaultValue "")
  $hasNonProofBoundary = -not [string]::IsNullOrWhiteSpace($boundary) -and
    $boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and
    $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and
    $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and
    $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and
    $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)
  [pscustomobject]@{
    id = $id
    present = $bundleEvidenceItemIds -contains $id
    passed = [bool](Get-PropertyOrDefault -Object $item -Name "passed" -DefaultValue $false)
    hasNonProofBoundary = $hasNonProofBoundary
    boundary = $boundary
  }
}

$forbiddenSubstitutes = @(
  "local feed",
  "direct .nupkg",
  "ProjectReference",
  "dry-run",
  "package-managed-dry-run",
  "dashboard-only",
  "artifact-only",
  "queued workflow",
  "missing runner",
  "sidecar-only",
  "local test"
)

$forbiddenSubstituteAudit = foreach ($marker in $forbiddenSubstitutes) {
  [pscustomobject]@{
    marker = $marker
    accepted = $false
    severity = "blocker"
    rule = "forbidden substitute cannot satisfy final public proof, publish, post-publish, or close gates"
  }
}

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "release-evidence-bundle-present" -Passed ($null -ne $releaseEvidence) -Severity "blocker" -Detail "Release evidence bundle must be available.")) | Out-Null
$items.Add((New-ValidationItem -Id "one-screen-pack-present" -Passed ($null -ne $oneScreenPack) -Severity "blocker" -Detail "Final owner one-screen pack must be available.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-source-artifacts-present" -Passed (@($requiredSourceArtifacts | Where-Object { $bundleSourceArtifacts -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Release evidence bundle sourceArtifacts must include every final public proof chain artifact.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-evidence-items-present" -Passed (@($requiredEvidenceItemIds | Where-Object { $bundleEvidenceItemIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Release evidence bundle must include all final close proof-chain evidence items.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-evidence-items-non-proof" -Passed (@($evidenceCoverage | Where-Object { -not $_.present -or $_.passed -or -not $_.hasNonProofBoundary }).Count -eq 0) -Severity "blocker" -Detail "Required final proof-chain evidence items must remain blocked/non-proof in the release candidate audit.")) | Out-Null
$items.Add((New-ValidationItem -Id "one-screen-final-public-proof-path-present" -Passed (@($requiredOneScreenProofStepIds | Where-Object { $oneScreenProofPathIds -notcontains $_ }).Count -eq 0 -and $oneScreenProofPath.Count -ge 6) -Severity "blocker" -Detail "Final owner one-screen pack must expose the six-step final public proof path.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-substitute-markers-present" -Passed (@($forbiddenSubstitutes | Where-Object { $bundleNonSubstitutes -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Release evidence bundle must propagate forbidden substitute markers.")) | Out-Null
$items.Add((New-ValidationItem -Id "final-bridge-no-forbidden-substitute-findings" -Passed ([int](Get-PropertyOrDefault -Object $finalBridgeValidation -Name "forbiddenSubstituteFindingCount" -DefaultValue 0) -eq 0) -Severity "blocker" -Detail "Final bridge validation must not carry forbidden substitute findings.")) | Out-Null

$failedBlockers = @($items.ToArray() | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedActionRequiredCount = $finalProofChain.Count
$auditState = if ($failedBlockers.Count -eq 0) { "blocked-release-candidate-public-proof-final-audit-owner-proof-required" } else { "invalid-release-candidate-public-proof-final-audit" }

$record = [pscustomobject]@{
  recordKind = "release-candidate-public-proof-final-audit"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditState = $auditState
  finalProofChainCount = $finalProofChain.Count
  blockedFinalProofChainCount = $finalProofChain.Count
  finalProofChain = @($finalProofChain)
  evidenceCoverage = @($evidenceCoverage)
  forbiddenSubstituteAudit = @($forbiddenSubstituteAudit)
  sourceArtifacts = @($requiredSourceArtifacts)
  requiredEvidenceItemIds = @($requiredEvidenceItemIds)
  requiredOneScreenProofStepIds = @($requiredOneScreenProofStepIds)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequiredCount
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  usesPublishToken = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  nonSubstituteProofKinds = @($forbiddenSubstitutes + @("release candidate public proof final audit", "final public proof path", "owner one-screen execution guidance"))
  boundary = "This release candidate public proof final audit checks coverage and non-proof boundaries only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot substitute real Owner public proof inputs."
}

$jsonPath = Join-Path $OutputRoot "release-candidate-public-proof-final-audit.json"
$markdownPath = Join-Path $OutputRoot "release-candidate-public-proof-final-audit.md"
$record | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$chainRows = foreach ($step in $finalProofChain) {
  "| ``$(ConvertTo-MarkdownCell $step.id)`` | ``$($step.order)`` | $(ConvertTo-MarkdownCell $step.currentState) | $(ConvertTo-MarkdownCell $step.requiredReadyState) | ``$($step.proofAccepted)`` | $(ConvertTo-MarkdownCell $step.ownerAction) |"
}
$coverageRows = foreach ($coverage in $evidenceCoverage) {
  "| ``$(ConvertTo-MarkdownCell $coverage.id)`` | ``$($coverage.present)`` | ``$($coverage.passed)`` | ``$($coverage.hasNonProofBoundary)`` |"
}
$validationRows = foreach ($item in $items) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Release Candidate Public Proof Final Audit

| Field | Value |
|---|---|
| auditState | ``$($record.auditState)`` |
| finalProofChainCount | ``$($record.finalProofChainCount)`` |
| blockedFinalProofChainCount | ``$($record.blockedFinalProofChainCount)`` |
| failedBlockerCount | ``$($record.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($record.failedActionRequiredCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Final Proof Chain

| Step | Order | Current State | Required Ready State | Proof Accepted | Owner Action |
|---|---:|---|---|---:|---|
$($chainRows -join "`r`n")

## Evidence Coverage

| Evidence Item | Present | Passed | Non-Proof Boundary |
|---|---:|---:|---:|
$($coverageRows -join "`r`n")

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($validationRows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate public proof final audit written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "AuditState=$auditState FailedBlockers=$($record.failedBlockerCount) FailedActionRequired=$($record.failedActionRequiredCount) CanPublish=False CanClose=False"
