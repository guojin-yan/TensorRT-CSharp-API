[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\clean-consumer-proof-execution-bundle.json",
  [string]$OutputRoot = "artifacts\final-release",
  [switch]$Strict,
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

function ConvertTo-Array {
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

  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function Get-StringArrayProperty {
  param([object]$Record, [string]$Name)

  return @((ConvertTo-Array (Get-PropertyOrDefault -Object $Record -Name $Name -DefaultValue @())) | ForEach-Object { [string]$_ })
}

function Get-IdArrayProperty {
  param([object]$Record, [string]$Name)

  return @((ConvertTo-Array (Get-PropertyOrDefault -Object $Record -Name $Name -DefaultValue @())) | ForEach-Object {
    [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")
  })
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Clean consumer proof execution bundle not found: $resolvedInputPath"
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @())
$laneIds = @($lanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$forbiddenSubstitutes = Get-StringArrayProperty -Record $record -Name "forbiddenSubstitutes"
$nonSubstituteProofKinds = Get-StringArrayProperty -Record $record -Name "nonSubstituteProofKinds"
$promotionRequirementIds = Get-IdArrayProperty -Record $record -Name "promotionRequirements"
$executionCommandIds = Get-IdArrayProperty -Record $record -Name "executionCommands"
$sourceArtifacts = Get-StringArrayProperty -Record $record -Name "sourceArtifacts"
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$allText = (($record | ConvertTo-Json -Depth 16) -join "`n")

$requiredLaneIds = @(
  "local-smoke",
  "local-feed",
  "clean-external-package-consumer",
  "compatible-cuda-host-runtime",
  "post-publish-clean-consumer"
)
$requiredForbidden = @(
  "Skipped=True",
  "local-smoke",
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "build-only",
  "dependency-probe",
  "blocked-by-cuda-driver",
  "article roadmap",
  "dashboard",
  "runbook",
  "candidate",
  "draft",
  "template",
  "preflight-only",
  "dry-run-only",
  "schema-only",
  "owner input without strict validator pass"
)
$requiredPromotionRequirements = @(
  "repository-external-clean-consumer",
  "no-project-reference",
  "public-or-approved-package-source",
  "native-assets-and-sha256",
  "runtime-smoke-log-sha256",
  "host-metadata",
  "strict-external-runtime-proof-validator",
  "post-publish-separate-gate"
)
$requiredExecutionCommands = @(
  "export-local-smoke-classification",
  "validate-local-smoke-classification",
  "prepare-owner-clean-consumer",
  "strict-runtime-proof-validation",
  "forbidden-substitute-scan",
  "refresh-release-evidence",
  "post-publish-owner-gate"
)
$requiredSources = @(
  "artifacts/final-release/cuda-device-initialization-local-smoke-classification.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json",
  "artifacts/final-release/external-runtime-proof-record.input-template.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json",
  "artifacts/final-release/final-post-publish-clean-consumer-proof-record-contract.json",
  "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json",
  "artifacts/final-release/clean-consumer-proof-owner-execution-pack.json"
)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "clean-consumer-proof-execution-bundle") -Severity "blocker" -Detail "recordKind must be clean-consumer-proof-execution-bundle.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "bundleState" -DefaultValue "") -eq "blocked-owner-clean-consumer-runtime-and-post-publish-proof-required") -Severity "blocker" -Detail "Bundle must remain blocked until owner supplies real clean consumer runtime and post-publish proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-runtime-or-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsRuntimeExecution" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Bundle must not run runtime, publish, close, or promote proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-lanes" -Passed ($lanes.Count -ge 5 -and @($requiredLaneIds | Where-Object { $laneIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Bundle must include local-smoke, local-feed, clean external package consumer, compatible host, and post-publish clean consumer lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-boundaries" -Passed (@($lanes | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "boundary" -DefaultValue "")) }).Count -eq 0) -Severity "blocker" -Detail "Every lane must be non-publish, non-promoting, non-closing, and boundary-tagged.")) | Out-Null
$items.Add((New-ValidationItem -Id "clean-external-owner-action" -Passed (@($lanes | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "clean-external-package-consumer" -and [bool](Get-PropertyOrDefault -Object $_ -Name "ownerActionRequired" -DefaultValue $false) -and [string](Get-PropertyOrDefault -Object $_ -Name "state" -DefaultValue "") -eq "owner-action-required" -and [string](Get-PropertyOrDefault -Object $_ -Name "evidenceKind" -DefaultValue "") -eq "candidate-runtime-proof-after-strict-validation" }).Count -eq 1) -Severity "blocker" -Detail "Clean external package consumer lane must be owner-action-required and candidate-only until strict validation.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-separate-owner-gate" -Passed (@($lanes | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "post-publish-clean-consumer" -and [string](Get-PropertyOrDefault -Object $_ -Name "state" -DefaultValue "") -eq "owner-action-required-after-publication" }).Count -eq 1) -Severity "blocker" -Detail "Post-publish clean consumer must remain a separate owner gate after publication.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes" -Passed (@($requiredForbidden | Where-Object { $forbiddenSubstitutes -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Bundle must enumerate all forbidden substitutes, including skipped/local/build/runbook/draft/candidate records.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-substitute-proof-kinds" -Passed (@($requiredForbidden | Where-Object { $nonSubstituteProofKinds -notcontains $_ }).Count -eq 0 -and $nonSubstituteProofKinds -contains "clean-consumer-proof-execution-bundle" -and $nonSubstituteProofKinds -contains "owner-action-required") -Severity "blocker" -Detail "Bundle must propagate forbidden substitute markers into nonSubstituteProofKinds.")) | Out-Null
$items.Add((New-ValidationItem -Id "promotion-requirements" -Passed (@($requiredPromotionRequirements | Where-Object { $promotionRequirementIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Bundle must define repository-external, no ProjectReference, package source, asset/hash/log/host, strict external validator, and post-publish requirements.")) | Out-Null
$items.Add((New-ValidationItem -Id "execution-commands" -Passed (@($requiredExecutionCommands | Where-Object { $executionCommandIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Bundle must include export, validate, owner preparation, strict proof, forbidden scan, release refresh, and post-publish command lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifacts" -Passed (@($requiredSources | Where-Object { $sourceArtifacts -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Bundle must source local smoke classification, release evidence, preflight/template/contract, checklist, and owner execution pack artifacts.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-validators" -Passed ($allText.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-ExternalRuntimeProofRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-PostPublishCleanConsumerProofRecordContract.ps1", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Bundle must reference strict runtime, external runtime, and post-publish validators.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-real-proof-fields" -Passed ($allText.Contains("native asset listing", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("SHA256", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("runtime smoke stdout/stderr logs", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("owner review", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Bundle must require native asset hashes, runtime logs, host metadata, and owner review.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed ($boundary.Contains("execution map", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("does not run runtime smoke", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("does not publish packages", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("strict validators with FailOnNotProof", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must state map-only status and strict FailOnNotProof requirements.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "clean-consumer-proof-execution-bundle-ready-non-proof-boundaries-intact" } else { "blocked-clean-consumer-proof-execution-bundle-invalid" }

$validation = [pscustomobject]@{
  recordKind = "clean-consumer-proof-execution-bundle-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  inputPath = $resolvedInputPath
  validationItemCount = $validationItems.Count
  failedBlockerCount = $failedBlockers.Count
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  validationItems = $validationItems
  boundary = "Validation confirms execution-bundle shape and non-proof boundaries only; it is not package-consumer-runtime proof, post-publish proof, publication approval, release close approval, or package push."
}

$jsonPath = Join-Path $OutputRoot "clean-consumer-proof-execution-bundle-validation.json"
$markdownPath = Join-Path $OutputRoot "clean-consumer-proof-execution-bundle-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# Clean Consumer Proof Execution Bundle Validation

| Field | Value |
| --- | --- |
| validationState | ``$($validation.validationState)`` |
| validationItemCount | ``$($validation.validationItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| ownerActionRequired | ``$($validation.ownerActionRequired)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| performsRuntimeExecution | ``$($validation.performsRuntimeExecution)`` |
| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| isRuntimeExecutionProof | ``$($validation.isRuntimeExecutionProof)`` |
| isPackageConsumerRuntimeProof | ``$($validation.isPackageConsumerRuntimeProof)`` |
| isPostPublishProof | ``$($validation.isPostPublishProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Clean consumer proof execution bundle validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  foreach ($failure in $failedBlockers) {
    Write-Error "$($failure.id): $($failure.detail)"
  }

  exit 1
}
