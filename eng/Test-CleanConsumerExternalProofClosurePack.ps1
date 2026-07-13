[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\clean-consumer-external-proof-closure-pack.json",
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
  throw "Clean consumer external proof closure pack not found: $resolvedInputPath"
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "closureLanes" -DefaultValue @())
$laneIds = Get-IdArrayProperty -Record $record -Name "closureLanes"
$fieldIds = Get-IdArrayProperty -Record $record -Name "requiredOwnerFields"
$stepIds = Get-IdArrayProperty -Record $record -Name "executionSteps"
$sourceArtifacts = Get-StringArrayProperty -Record $record -Name "sourceArtifacts"
$strictValidators = Get-StringArrayProperty -Record $record -Name "strictValidatorCommands"
$forbiddenSubstitutes = Get-StringArrayProperty -Record $record -Name "forbiddenSubstitutes"
$nonSubstituteProofKinds = Get-StringArrayProperty -Record $record -Name "nonSubstituteProofKinds"
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$allText = (($record | ConvertTo-Json -Depth 16) -join "`n")

$requiredLaneIds = @(
  "owner-input-contracts",
  "clean-external-package-consumer-runtime",
  "compatible-cuda-host-runtime",
  "owner-external-proof-result-import",
  "post-publish-clean-consumer-proof",
  "strict-close-admission"
)
$requiredFieldIds = @(
  "clean-consumer-root",
  "package-source",
  "managed-runtime-package-identity",
  "native-assets",
  "runtime-smoke-logs",
  "host-metadata",
  "owner-review",
  "post-publish-evidence"
)
$requiredStepIds = @(
  "select-runtime-proof-preflight-option",
  "create-clean-consumer-outside-repository",
  "restore-managed-runtime-packages",
  "run-clean-consumer-runtime-smoke",
  "hash-logs-packages-native-assets",
  "import-owner-external-result",
  "refresh-release-evidence",
  "post-publish-clean-consumer-after-publication"
)
$requiredSources = @(
  "artifacts/final-release/clean-consumer-proof-execution-bundle.json",
  "artifacts/final-release/owner-external-proof-execution-bundle.json",
  "artifacts/final-release/external-clean-consumer-proof-kit.json",
  "artifacts/final-release/owner-external-real-proof-input-contract.json",
  "artifacts/final-release/owner-external-real-proof-import-validator.json",
  "artifacts/final-release/runtime-compatible-host-real-proof-gate.json",
  "artifacts/final-release/post-publish-clean-consumer-real-proof-gate.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json",
  "artifacts/final-release/final-post-publish-clean-consumer-proof-record-contract.json",
  "artifacts/final-release/release-close-real-proof-readiness-gate.json",
  "artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json",
  "artifacts/final-release/external-runtime-proof-record.input-template.json",
  "artifacts/final-release/release-evidence-bundle.json"
)
$requiredForbidden = @(
  "Skipped=True",
  "local smoke",
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "build-only",
  "dependency-probe",
  "blocked-by-cuda-driver",
  "dashboard",
  "runbook",
  "candidate",
  "draft",
  "template",
  "preflight-only",
  "dry-run-only",
  "schema-only",
  "owner input without strict validator pass",
  "host metadata without runtime smoke",
  "package hash without existing log validation",
  "native asset listing without runtime smoke",
  "pre-publish smoke reused as post-publish proof"
)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "clean-consumer-external-proof-closure-pack") -Severity "blocker" -Detail "recordKind must be clean-consumer-external-proof-closure-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "closureState" -DefaultValue "") -eq "blocked-owner-external-clean-consumer-proof-closure-required") -Severity "blocker" -Detail "Closure pack must remain owner-action blocked until real external proof is supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-runtime-or-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsRuntimeExecution" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Closure pack must not run runtime, publish, close, or promote proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-lanes" -Passed ($lanes.Count -ge 6 -and @($requiredLaneIds | Where-Object { $laneIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Closure pack must include all six convergence lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-boundaries" -Passed (@($lanes | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "performsRuntimeExecution" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) -or -not ([string](Get-PropertyOrDefault -Object $_ -Name "boundary" -DefaultValue "")).Contains("not", [StringComparison]::OrdinalIgnoreCase) }).Count -eq 0) -Severity "blocker" -Detail "Every closure lane must be non-publish, non-runtime, non-promoting, non-closing, and explicit about non-proof boundaries.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-owner-fields" -Passed (@($requiredFieldIds | Where-Object { $fieldIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Closure pack must require clean consumer, package, native asset, runtime log, host, owner, and post-publish fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "execution-steps" -Passed (@($requiredStepIds | Where-Object { $stepIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Closure pack must provide the full owner execution sequence.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifacts" -Passed (@($requiredSources | Where-Object { $sourceArtifacts -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Closure pack must source the existing clean consumer, external proof, real proof, post-publish, close, preflight, and release evidence artifacts.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-validators" -Passed ($strictValidators.Count -ge 12 -and $allText.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-ExternalRuntimeProofRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-PostPublishCleanConsumerProofRecordDraft.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Test-ReleaseEvidenceClassificationAudit.ps1", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Closure pack must reference strict runtime, external runtime, post-publish, close, and classification validators.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes" -Passed (@($requiredForbidden | Where-Object { $forbiddenSubstitutes -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Closure pack must enumerate all forbidden substitutes and weak proof-like artifacts.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-substitute-proof-kinds" -Passed (@($requiredForbidden | Where-Object { $nonSubstituteProofKinds -notcontains $_ }).Count -eq 0 -and $nonSubstituteProofKinds -contains "clean-consumer-external-proof-closure-pack" -and $nonSubstituteProofKinds -contains "external proof closure guidance" -and $nonSubstituteProofKinds -contains "owner-action closure pack") -Severity "blocker" -Detail "Closure pack must propagate forbidden substitute markers.")) | Out-Null
$items.Add((New-ValidationItem -Id "real-proof-evidence-fields" -Passed ($allText.Contains("repository-external", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("native asset listing", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("SHA256", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("host metadata", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("owner review", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("post-publish", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Closure pack must require repository-external execution, hashes, native assets, host metadata, owner review, and post-publish proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed ($boundary.Contains("blocked owner-action convergence layer", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("does not run runtime smoke", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("does not publish packages", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("does not close the release issue", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("strict validators with FailOnNotProof", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must explicitly keep the closure pack non-proof and owner-action blocked.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "clean-consumer-external-proof-closure-pack-ready-non-proof-boundaries-intact" } else { "blocked-clean-consumer-external-proof-closure-pack-invalid" }

$validation = [pscustomobject]@{
  recordKind = "clean-consumer-external-proof-closure-pack-validation"
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
  isReleaseCloseProof = $false
  validationItems = $validationItems
  boundary = "Validation confirms closure-pack shape and non-proof boundaries only; it is not runtime proof, post-publish proof, publication approval, release close approval, or package push."
}

$jsonPath = Join-Path $OutputRoot "clean-consumer-external-proof-closure-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "clean-consumer-external-proof-closure-pack-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# Clean Consumer External Proof Closure Pack Validation

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
| isReleaseCloseProof | ``$($validation.isReleaseCloseProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Clean consumer external proof closure pack validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  foreach ($failure in $failedBlockers) {
    Write-Error "$($failure.id): $($failure.detail)"
  }

  exit 1
}
