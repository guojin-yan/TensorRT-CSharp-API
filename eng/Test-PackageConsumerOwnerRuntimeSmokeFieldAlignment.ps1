[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\package-consumer-owner-runtime-smoke-field-alignment.json",
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
  throw "Package consumer owner runtime smoke field alignment not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$fields = @(Get-PropertyOrDefault -Object $record -Name "fields" -DefaultValue @())
$forbiddenSubstitutes = @((Get-PropertyOrDefault -Object $record -Name "forbiddenRuntimeSmokeSubstitutes" -DefaultValue @()) | ForEach-Object { [string]$_ })
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "package-consumer-owner-runtime-smoke-field-alignment") -Severity "blocker" -Detail "recordKind must be package-consumer-owner-runtime-smoke-field-alignment.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "alignmentState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke-field-alignment") -Severity "blocker" -Detail "Alignment matrix must remain blocked until real owner runtime smoke proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-runtime-smoke-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "ownerRuntimeSmokeRunbookState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke") -Severity "blocker" -Detail "Owner runtime smoke runbook state must be visible.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-or-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "approvesPublicRelease" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -Severity "blocker" -Detail "Field alignment must not publish, approve, close, promote proof, or claim runtime proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "field-count" -Passed ($fields.Count -ge 30) -Severity "blocker" -Detail "Alignment matrix must cover the main owner runtime smoke fields.")) | Out-Null
$criticalFieldMissingCount = @("cleanExternalConsumerRoot","consumerProjectPath","publicPackageSource","managedNupkgSha256","runtimeNupkgSha256","smokeLogSha256","stdoutSummary","stderrSummary","gpuName","cudaRuntimeVersion","tensorRtVersion","smokeStatus") |
  ForEach-Object {
    $name = $_
    @($fields | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "") -eq $name }).Count -eq 1
  } |
  Where-Object { -not $_ } |
  Measure-Object |
  Select-Object -ExpandProperty Count
$items.Add((New-ValidationItem -Id "critical-fields-present" -Passed ($criticalFieldMissingCount -eq 0) -Severity "blocker" -Detail "Critical clean consumer, package hash, smoke log, stdout/stderr, and host fields must be listed.")) | Out-Null

$coverageFields = @($fields | Where-Object {
    [bool](Get-PropertyOrDefault -Object $_ -Name "presentInOwnerSchema" -DefaultValue $false) -and
    [bool](Get-PropertyOrDefault -Object $_ -Name "presentInRunbookRequiredOwnerInputFields" -DefaultValue $false) -and
    [bool](Get-PropertyOrDefault -Object $_ -Name "presentInCollectionBundle" -DefaultValue $false)
  })
$items.Add((New-ValidationItem -Id "schema-runbook-collection-coverage" -Passed ($coverageFields.Count -eq $fields.Count) -Severity "blocker" -Detail "Every field must be covered by owner schema, compatible-host runbook, and collection bundle.")) | Out-Null
$items.Add((New-ValidationItem -Id "final-owner-surface-coverage" -Passed (@($fields | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "presentInFinalOwnerOneScreenPack" -DefaultValue $false) -or [bool](Get-PropertyOrDefault -Object $_ -Name "presentInCleanConsumerChecklist" -DefaultValue $false) }).Count -ge 24) -Severity "blocker" -Detail "Final owner one-screen pack or clean consumer checklist must cover most owner runtime smoke fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes" -Passed (($forbiddenSubstitutes -contains "local feed") -and ($forbiddenSubstitutes -contains "ProjectReference") -and ($forbiddenSubstitutes -contains "direct .nupkg") -and ($forbiddenSubstitutes -contains "Smoke=not-requested") -and ($forbiddenSubstitutes -contains "dependency-probe-only") -and ($forbiddenSubstitutes -contains "blocked-by-cuda-driver")) -Severity "blocker" -Detail "Alignment matrix must carry runtime smoke forbidden substitutes.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) {
  "blocked-owner-compatible-host-runtime-smoke-field-alignment-valid"
}
else {
  "invalid-package-consumer-owner-runtime-smoke-field-alignment"
}

$validation = [pscustomobject]@{
  recordKind = "package-consumer-owner-runtime-smoke-field-alignment-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  fieldCount = $fields.Count
  failedBlockerCount = $failedBlockers.Count
  performsPublish = $false
  approvesPublicRelease = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validation checks field alignment only. It is not runtime proof, package publish approval, public channel proof, or release close approval."
}

$jsonPath = Join-Path $OutputRoot "package-consumer-owner-runtime-smoke-field-alignment-validation.json"
$markdownPath = Join-Path $OutputRoot "package-consumer-owner-runtime-smoke-field-alignment-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validation.validationItems) {
  "| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Package Consumer Owner Runtime Smoke Field Alignment Validation

| Field | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| fieldCount | ``$($validation.fieldCount)`` |
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

Write-Host "Package consumer owner runtime smoke field alignment validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) CanPromoteRuntimeProof=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Package consumer owner runtime smoke field alignment validation failed with $($failedBlockers.Count) blocker(s)."
}
