[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-candidate-public-proof-final-audit.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
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

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release candidate public proof final audit not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$finalProofChain = Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "finalProofChain" -DefaultValue @())
$evidenceCoverage = Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "evidenceCoverage" -DefaultValue @())
$forbiddenSubstituteAudit = Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteAudit" -DefaultValue @())
$sourceArtifacts = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @()) | ForEach-Object { [string]$_ })
$nonSubstitutes = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "nonSubstituteProofKinds" -DefaultValue @()) | ForEach-Object { [string]$_ })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$requiredFinalProofStepIds = @(
  "github-actions-run-evidence",
  "owner-public-publish-result",
  "public-package-download-proof",
  "post-publish-clean-consumer-proof-result",
  "final-public-release-closure-bridge",
  "release-issue-close-owner-decision-input"
)
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
$requiredForbiddenMarkers = @(
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

$finalProofStepIds = @($finalProofChain | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$coverageIds = @($evidenceCoverage | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$forbiddenMarkers = @($forbiddenSubstituteAudit | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "marker" -DefaultValue "") })

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-candidate-public-proof-final-audit") -Severity "blocker" -Detail "recordKind must be release-candidate-public-proof-final-audit.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "auditState" -DefaultValue "") -eq "blocked-release-candidate-public-proof-final-audit-owner-proof-required") -Severity "blocker" -Detail "Default audit state must remain blocked until real Owner public proof is accepted.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-final-proof-chain" -Passed (@($requiredFinalProofStepIds | Where-Object { $finalProofStepIds -notcontains $_ }).Count -eq 0 -and $finalProofChain.Count -ge 6) -Severity "blocker" -Detail "Audit must include the six-step final public proof chain.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-source-artifacts" -Passed (@($requiredSourceArtifacts | Where-Object { $sourceArtifacts -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Audit sourceArtifacts must include GitHub Actions, Owner publish, public download, post-publish, final bridge, release close, and one-screen pack artifacts.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-evidence-coverage" -Passed (@($requiredEvidenceItemIds | Where-Object { $coverageIds -notcontains $_ }).Count -eq 0 -and @($evidenceCoverage | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "present" -DefaultValue $false) -or [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $true) -or -not [bool](Get-PropertyOrDefault -Object $_ -Name "hasNonProofBoundary" -DefaultValue $false) }).Count -eq 0) -Severity "blocker" -Detail "Required evidence items must be present, blocked, and carry non-proof boundaries.")) | Out-Null
$items.Add((New-ValidationItem -Id "final-proof-chain-blocked-non-proof" -Passed (@($finalProofChain | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "blocked" -DefaultValue $false) -or [bool](Get-PropertyOrDefault -Object $_ -Name "proofAccepted" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "performsRuntimeExecution" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "isRuntimeExecutionProof" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "isPostPublishProof" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "isReleaseCloseProof" -DefaultValue $true) }).Count -eq 0) -Severity "blocker" -Detail "Final proof chain steps must remain blocked, proofAccepted=false, non-publish, non-runtime, and non-close.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes-not-accepted" -Passed (@($forbiddenSubstituteAudit | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "accepted" -DefaultValue $true) }).Count -eq 0 -and @($requiredForbiddenMarkers | Where-Object { $forbiddenMarkers -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Forbidden substitutes must be listed and accepted=false.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-substitute-markers" -Passed (@($requiredForbiddenMarkers | Where-Object { $nonSubstitutes -notcontains $_ }).Count -eq 0 -and $nonSubstitutes -contains "release candidate public proof final audit" -and $nonSubstitutes -contains "final public proof path") -Severity "blocker" -Detail "Audit must propagate forbidden substitute and final public proof path markers.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-non-proof-flags" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsRuntimeExecution" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Audit must not publish, execute runtime, promote proof, or close release.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Audit boundary must explicitly reject proof, publish, close, and package push substitution.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "blocked-release-candidate-public-proof-final-audit-owner-proof-required" } else { "invalid-release-candidate-public-proof-final-audit" }

$validation = [pscustomobject]@{
  recordKind = "release-candidate-public-proof-final-audit-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  inputPath = $resolvedInputPath
  finalProofChainCount = $finalProofChain.Count
  evidenceCoverageCount = $evidenceCoverage.Count
  forbiddenSubstituteAuditCount = $forbiddenSubstituteAudit.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $finalProofChain.Count
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
  validationItems = $validationItems
  boundary = "Validation checks release candidate public proof final audit structure and non-proof boundaries only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-candidate-public-proof-final-audit-validation.json"
$markdownPath = Join-Path $OutputRoot "release-candidate-public-proof-final-audit-validation.md"
$validation | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validationItems) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Release Candidate Public Proof Final Audit Validation

| Field | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| finalProofChainCount | ``$($validation.finalProofChainCount)`` |
| evidenceCoverageCount | ``$($validation.evidenceCoverageCount)`` |
| forbiddenSubstituteAuditCount | ``$($validation.forbiddenSubstituteAuditCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate public proof final audit validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) CanPublish=False CanClose=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Release candidate public proof final audit validation failed with $($failedBlockers.Count) blocker(s)."
}
