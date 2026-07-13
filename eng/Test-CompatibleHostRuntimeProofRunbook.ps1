[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\compatible-host-runtime-proof-runbook.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)
  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Compatible host runtime proof runbook not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$allText = $record | ConvertTo-Json -Depth 16
$commands = Get-PropertyOrDefault -Object $record -Name "commands" -DefaultValue $null
$steps = @(Get-PropertyOrDefault -Object $record -Name "steps" -DefaultValue @())
$requiredOwnerInputFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "requiredOwnerInputFields" -DefaultValue @())
$forbiddenSubstitutes = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "forbiddenRuntimeSmokeSubstitutes" -DefaultValue @())

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "compatible-host-runtime-proof-runbook") -Severity "blocker" -Detail "recordKind must be compatible-host-runtime-proof-runbook.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-runtime-smoke-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "ownerRuntimeSmokeRunbookState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke") -Severity "blocker" -Detail "Current owner runtime smoke blocker must be explicit as blocked-owner-compatible-host-runtime-smoke.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-or-close-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "approvesPublicRelease" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionEvidence" -DefaultValue $true)) -Severity "blocker" -Detail "Runbook must not publish, approve, close, promote proof, or claim runtime execution evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "package-consumer-smoke-command" -Passed ($allText.Contains("Test-PackageConsumer.ps1", [StringComparison]::Ordinal) -and $allText.Contains("-RunSmoke", [StringComparison]::Ordinal) -and $allText.Contains("-KeepConsumerOutput", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "Runbook must include Test-PackageConsumer.ps1 -RunSmoke -KeepConsumerOutput.")) | Out-Null
$items.Add((New-ValidationItem -Id "external-runtime-proof-validator" -Passed ($allText.Contains("Test-ExternalRuntimeProofRecord.ps1", [StringComparison]::Ordinal) -and $allText.Contains("-RequireExistingLog", [StringComparison]::Ordinal) -and $allText.Contains("-FailOnNotProof", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "Runbook must include Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-input-strict-chain" -Passed ($allText.Contains("Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1", [StringComparison]::Ordinal) -and $allText.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", [StringComparison]::Ordinal) -and $allText.Contains("Import-PackageConsumerRuntimeProofOwnerInput.ps1", [StringComparison]::Ordinal) -and $allText.Contains("-Strict", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "Runbook must include export, strict validation, and import commands for package-consumer runtime owner input.")) | Out-Null
$items.Add((New-ValidationItem -Id "step-coverage" -Passed ($steps.Count -ge 12 -and $allText.Contains("export-owner-input-template", [StringComparison]::Ordinal) -and $allText.Contains("validate-owner-input-strict", [StringComparison]::Ordinal) -and $allText.Contains("import-owner-input-strict", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "Runbook must include external proof and owner input steps in one sequence.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-owner-input-fields" -Passed (($requiredOwnerInputFields -contains "consumerProjectPath") -and ($requiredOwnerInputFields -contains "publicPackageSource") -and ($requiredOwnerInputFields -contains "managedNupkgSha256") -and ($requiredOwnerInputFields -contains "runtimeNupkgSha256") -and ($requiredOwnerInputFields -contains "smokeLogSha256") -and ($requiredOwnerInputFields -contains "stdoutSummary") -and ($requiredOwnerInputFields -contains "stderrSummary") -and ($requiredOwnerInputFields -contains "gpuName") -and ($requiredOwnerInputFields -contains "cudaRuntimeVersion") -and ($requiredOwnerInputFields -contains "tensorRtVersion")) -Severity "blocker" -Detail "Runbook must list clean consumer, package/hash, log, stdout/stderr, and host metadata owner fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes" -Passed (($forbiddenSubstitutes -contains "local feed") -and ($forbiddenSubstitutes -contains "ProjectReference") -and ($forbiddenSubstitutes -contains "direct .nupkg") -and ($forbiddenSubstitutes -contains "Smoke=not-requested") -and ($forbiddenSubstitutes -contains "dependency-probe-only") -and ($forbiddenSubstitutes -contains "blocked-by-cuda-driver")) -Severity "blocker" -Detail "Runbook must forbid local feed, ProjectReference, direct nupkg, Smoke=not-requested, dependency-probe-only, and blocked-by-cuda-driver substitutes.")) | Out-Null
$items.Add((New-ValidationItem -Id "artifact-paths" -Passed ($allText.Contains("package-consumer-runtime-proof-owner-input.template.json", [StringComparison]::Ordinal) -and $allText.Contains("package-consumer-runtime-proof-owner-input-validation.json", [StringComparison]::Ordinal) -and $allText.Contains("external-runtime-proof-record.json", [StringComparison]::Ordinal) -and $allText.Contains("external-runtime-proof-validation.json", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "Runbook must point to owner input and external proof artifacts.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) {
  "blocked-owner-compatible-host-runtime-smoke"
}
else {
  "invalid-compatible-host-runtime-proof-runbook"
}

$validation = [pscustomobject]@{
  recordKind = "compatible-host-runtime-proof-runbook-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  ownerRuntimeSmokeRunbookState = [string](Get-PropertyOrDefault -Object $record -Name "ownerRuntimeSmokeRunbookState" -DefaultValue "")
  stepCount = $steps.Count
  requiredOwnerInputFieldCount = $requiredOwnerInputFields.Count
  forbiddenRuntimeSmokeSubstituteCount = $forbiddenSubstitutes.Count
  failedBlockerCount = $failedBlockers.Count
  performsPublish = $false
  approvesPublicRelease = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionEvidence = $false
  validationItems = @($items.ToArray())
  boundary = "This validation checks compatible-host owner runbook shape only. It is not runtime proof, owner approval, package publish, or release-close authorization."
}

$jsonPath = Join-Path $OutputRoot "compatible-host-runtime-proof-runbook-validation.json"
$markdownPath = Join-Path $OutputRoot "compatible-host-runtime-proof-runbook-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validation.validationItems) {
  "| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Compatible Host Runtime Proof Runbook Validation

| Field | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| ownerRuntimeSmokeRunbookState | ``$($validation.ownerRuntimeSmokeRunbookState)`` |
| stepCount | ``$($validation.stepCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |
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

Write-Host "Compatible host runtime proof runbook validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) CanPromoteRuntimeProof=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Compatible host runtime proof runbook validation failed with $($failedBlockers.Count) blocker(s)."
}
