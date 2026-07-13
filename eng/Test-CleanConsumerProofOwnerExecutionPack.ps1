[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\clean-consumer-proof-owner-execution-pack.json",
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
  throw "Clean consumer proof owner execution pack not found: $resolvedInputPath"
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$commandGroups = ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "ownerCommandGroups" -DefaultValue @())
$commandGroupIds = @($commandGroups | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$commandText = (($commandGroups | ConvertTo-Json -Depth 10) -join "`n")
$requiredFiles = Get-IdArrayProperty -Record $record -Name "requiredOwnerFiles"
$requiredInputs = Get-IdArrayProperty -Record $record -Name "requiredOwnerInputs"
$projectionIds = Get-IdArrayProperty -Record $record -Name "proofRecordProjection"
$strictValidators = Get-StringArrayProperty -Record $record -Name "strictValidatorCommands"
$forbiddenRules = Get-StringArrayProperty -Record $record -Name "forbiddenSubstituteRules"
$refreshCommands = Get-StringArrayProperty -Record $record -Name "releaseRefreshCommands"
$sourceArtifacts = Get-StringArrayProperty -Record $record -Name "sourceArtifacts"
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$allText = (($record | ConvertTo-Json -Depth 16) -join "`n")

$requiredGroupIds = @(
  "prepare-clean-consumer-workspace",
  "configure-public-source-and-install-packages",
  "restore-build-smoke-and-capture-logs",
  "hash-logs-packages-and-project",
  "capture-host-package-and-owner-metadata",
  "fill-import-and-validate-owner-input",
  "scan-forbidden-substitutes-and-refresh-release-state",
  "post-publish-remains-separate-owner-gate"
)
$requiredFileIds = @(
  "clean-consumer-project",
  "restore-log",
  "build-log",
  "smoke-stdout-log",
  "smoke-stderr-log",
  "managed-nupkg",
  "runtime-nupkg",
  "owner-input"
)
$requiredInputIds = @(
  "public-package-source-url",
  "package-ids-and-versions",
  "consumer-project-identity",
  "commands",
  "logs-and-hashes",
  "host-metadata",
  "runtime-status",
  "owner-review"
)
$requiredProjectionIds = @(
  "classification",
  "runtime-execution",
  "package-consumer",
  "publishability",
  "release-close"
)
$requiredSources = @(
  "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json",
  "artifacts/final-release/final-proof-owner-handoff-pack.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/final-release-close-blocker-dashboard.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json",
  "artifacts/final-release/package-consumer-runtime-proof-record-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json"
)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "clean-consumer-proof-owner-execution-pack") -Severity "blocker" -Detail "recordKind must be clean-consumer-proof-owner-execution-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "packState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-proof-required") -Severity "blocker" -Detail "Pack must remain blocked until owner runs real compatible-host proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-close-or-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Pack must not publish, close, or promote proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-checklist" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "sourceChecklist" -DefaultValue "") -eq "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json") -Severity "blocker" -Detail "Pack must source the clean consumer execution checklist.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifacts" -Passed (@($requiredSources | Where-Object { $sourceArtifacts -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Pack must source checklist, handoff, release evidence, close dashboard, owner schema, proof validation, and forbidden scan artifacts.")) | Out-Null
$items.Add((New-ValidationItem -Id "command-groups" -Passed ($commandGroups.Count -ge 8 -and @($requiredGroupIds | Where-Object { $commandGroupIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Pack must contain the full owner command groups from workspace setup through post-publish separation.")) | Out-Null
$items.Add((New-ValidationItem -Id "command-group-boundaries" -Passed (@($commandGroups | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ownerActionRequired" -DefaultValue $false) -or [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $true) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "proofBoundary" -DefaultValue "")) }).Count -eq 0) -Severity "blocker" -Detail "Every command group must be owner-action required, non-publish, non-promoting, and boundary-tagged.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-dotnet-nuget-push" -Passed (-not $commandText.Contains("dotnet nuget push", [StringComparison]::OrdinalIgnoreCase) -and -not $allText.Contains("dotnet nuget push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Pack must never include dotnet nuget push.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-owner-files" -Passed (@($requiredFileIds | Where-Object { $requiredFiles -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Pack must require clean consumer, restore/build/smoke logs, managed/runtime nupkg, and owner input files.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-owner-inputs" -Passed (@($requiredInputIds | Where-Object { $requiredInputs -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Pack must require package source, ids, project identity, commands, logs/hashes, host metadata, runtime status, and owner review.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-record-projection" -Passed (@($requiredProjectionIds | Where-Object { $projectionIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Pack must project classification, runtime proof, package consumer proof, publishability, and release close fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-validators" -Passed (@($strictValidators | Where-Object { $_.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-Strict", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1 -and @($strictValidators | Where-Object { $_.Contains("Test-PostPublishVerificationRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1 -and @($strictValidators | Where-Object { $_.Contains("Import-PackageConsumerRuntimeProofOwnerInput.ps1", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1 -and @($strictValidators | Where-Object { $_.Contains("Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1 -and @($strictValidators | Where-Object { $_.Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-FailOnNotCloseReady", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1) -Severity "blocker" -Detail "Pack must include strict owner input, import, runtime proof, post-publish proof, forbidden scan, and final close validators.")) | Out-Null
$items.Add((New-ValidationItem -Id "release-refresh-commands" -Passed (@($refreshCommands | Where-Object { $_.Contains("Export-ReleaseEvidenceBundle.ps1", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1 -and @($refreshCommands | Where-Object { $_.Contains("Export-FinalReleaseCloseBlockerDashboard.ps1", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1 -and @($refreshCommands | Where-Object { $_.Contains("Test-FinalReleasePrePublishAuditMatrix.ps1", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1) -Severity "blocker" -Detail "Pack must include release evidence, close dashboard, and pre-publish audit refresh commands.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitute-rules" -Passed ($forbiddenRules.Count -ge 12 -and $allText.Contains("local feed is not clean public package source proof", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("ProjectReference is not package-consumer-runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("TensorRtExec report is not package-consumer-runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("owner input without strict validator pass is not proof", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Pack must list forbidden substitutes and reject owner input without strict validation.")) | Out-Null
$items.Add((New-ValidationItem -Id "handoff-summary" -Passed ($allText.Contains("nextOwnerAction", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("forbiddenSubstituteScanState", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("detectedForbiddenSubstituteCount", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("cleanOwnerInputReady", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Pack must carry current owner handoff blockers and next action.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed ($boundary.Contains("repository-external clean consumer smoke", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("public package source", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("strict validation", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("does not execute package publish", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must require real clean public-source smoke and block publish/close/proof substitution.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "clean-consumer-proof-owner-execution-pack-ready" } else { "blocked-clean-consumer-proof-owner-execution-pack-invalid" }

$validation = [pscustomobject]@{
  recordKind = "clean-consumer-proof-owner-execution-pack-validation"
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
  boundary = "Validation confirms owner execution pack shape and non-proof boundaries only; it is not package publish, post-publish verification, release close approval, or package-consumer-runtime proof."
}

$jsonPath = Join-Path $OutputRoot "clean-consumer-proof-owner-execution-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "clean-consumer-proof-owner-execution-pack-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# Clean Consumer Proof Owner Execution Pack Validation

| Field | Value |
| --- | --- |
| validationState | ``$($validation.validationState)`` |
| validationItemCount | ``$($validation.validationItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Clean consumer proof owner execution pack validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  foreach ($failure in $failedBlockers) {
    Write-Error "$($failure.id): $($failure.detail)"
  }

  exit 1
}
