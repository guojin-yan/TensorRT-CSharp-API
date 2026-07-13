[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-proof-owner-handoff-pack.json",
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
  throw "Final proof owner handoff pack not found: $resolvedInputPath"
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$requiredOwnerInputs = ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "requiredOwnerInputs" -DefaultValue @())
$requiredCommands = ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "requiredCommands" -DefaultValue @())
$requiredValidators = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "requiredValidators" -DefaultValue @())) | ForEach-Object { [string]$_ })
$forbiddenSubstitutes = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())) | ForEach-Object { [string]$_ })
$sourceArtifacts = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())) | ForEach-Object { [string]$_ })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$requiredInputIds = @(
  "public-package-source-url",
  "package-id-version",
  "package-sha256",
  "clean-consumer-repo",
  "restore-build-smoke-command",
  "smoke-exit-code",
  "smoke-log-hash",
  "host-os",
  "gpu-driver-cuda-trt",
  "runtime-package-metadata",
  "owner-review",
  "post-publish-verification"
)
$inputIds = @($requiredOwnerInputs | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$requiredForbidden = @(
  "template",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "dry-run",
  "build-only",
  "preflight-only",
  "GUI screenshot",
  "TensorRtExec report",
  "YoloVision matrix",
  "OnnxToEngine report",
  "sample manifest",
  "sidecar-only",
  "readonly diagnostics"
)
$requiredSources = @(
  "artifacts/final-release/final-release-pre-publish-audit-matrix.json",
  "artifacts/final-release/final-release-pre-publish-audit-matrix-validation.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/final-release-dry-run-summary.json",
  "artifacts/final-release/final-release-close-blocker-dashboard.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json",
  "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json",
  "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json",
  "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json"
)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-proof-owner-handoff-pack") -Severity "blocker" -Detail "recordKind must be final-proof-owner-handoff-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "handoff-state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "handoffState" -DefaultValue "") -eq "blocked-owner-real-proof-required") -Severity "blocker" -Detail "Handoff pack must remain blocked until real owner proof is supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-or-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Handoff pack must not publish, close issue, or promote proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-owner-inputs" -Passed (@($requiredInputIds | Where-Object { $inputIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Handoff pack must enumerate all required owner input fields for clean consumer and post-publish proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-commands" -Passed ($requiredCommands.Count -ge 6 -and @($requiredCommands | Where-Object { ([string](Get-PropertyOrDefault -Object $_ -Name "command" -DefaultValue "")).Contains("dotnet nuget push", [StringComparison]::OrdinalIgnoreCase) }).Count -eq 0) -Severity "blocker" -Detail "Handoff pack must include safe validation commands and no dotnet nuget push command.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-validators" -Passed ($requiredValidators -contains "eng/Test-FinalReleasePrePublishAuditMatrix.ps1 -Strict" -and @($requiredValidators | Where-Object { $_.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-Strict", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1 -and @($requiredValidators | Where-Object { $_.Contains("Test-PostPublishVerificationRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1) -Severity "blocker" -Detail "Handoff pack must point to pre-publish, package consumer, and post-publish strict proof validators with existing-log and FailOnNotProof gates.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes" -Passed (@($requiredForbidden | Where-Object { $forbiddenSubstitutes -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Handoff pack must list every forbidden substitute proof source.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifacts" -Passed (@($requiredSources | Where-Object { $sourceArtifacts -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Handoff pack must source pre-publish matrix, release bundle, dry run, close dashboard, owner input schema, and forbidden scan.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-runtime-smoke-field-alignment" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke-field-alignment" -and [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke-field-alignment-valid" -and [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus" -DefaultValue "") -eq "Smoke=not-requested" -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount" -DefaultValue 0) -ge 30 -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount" -DefaultValue -1) -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Handoff pack must project owner runtime smoke field alignment as valid, missingRequiredFields=0, and Smoke=not-requested without promoting proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "prepublish-matrix-ready" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "prePublishAuditMatrixReady" -DefaultValue $false)) -Severity "blocker" -Detail "Final proof handoff must consume a validated pre-publish matrix.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-proof-still-required" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "cleanOwnerInputReady" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "ownerInputCanPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "packageConsumerRuntimeProofRecordReady" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "postPublishVerificationReady" -DefaultValue $true) -and [int](Get-PropertyOrDefault -Object $record -Name "remainingOwnerActionCount" -DefaultValue 0) -ge 1) -Severity "blocker" -Detail "Current handoff must still require real clean owner input, package consumer proof, and post-publish verification.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-boundary" -Passed ($boundary.Contains("does not publish packages", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("clean external package-consumer-runtime", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must explicitly block publication and proof substitution.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "final-proof-owner-handoff-pack-ready" } else { "blocked-final-proof-owner-handoff-pack-invalid" }

$validation = [pscustomobject]@{
  recordKind = "final-proof-owner-handoff-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  inputPath = $resolvedInputPath
  validationItemCount = $validationItems.Count
  failedBlockerCount = $failedBlockers.Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  validationItems = $validationItems
  boundary = "Validation confirms owner handoff shape and non-proof boundaries only; it is not runtime proof, publish approval, or release close approval."
}

$jsonPath = Join-Path $OutputRoot "final-proof-owner-handoff-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "final-proof-owner-handoff-pack-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# Final Proof Owner Handoff Pack Validation

| Field | Value |
| --- | --- |
| validationState | ``$($validation.validationState)`` |
| validationItemCount | ``$($validation.validationItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
| --- | ---: | --- | --- |
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final proof owner handoff pack validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  foreach ($failure in $failedBlockers) {
    Write-Error "$($failure.id): $($failure.detail)"
  }

  exit 1
}
