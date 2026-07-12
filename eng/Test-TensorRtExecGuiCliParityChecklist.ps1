[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\tensor-rt-exec-gui-cli-parity-checklist.json",
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
  throw "TensorRtExec GUI/CLI parity checklist not found: $resolvedInputPath"
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$itemsRaw = ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "items" -DefaultValue @())
$items = @($itemsRaw)
$ids = @($items | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "optionId" -DefaultValue "") })
$requiredIds = @(
  "onnx",
  "save-engine",
  "load-engine",
  "shape-profiles",
  "precision",
  "int8-calibration",
  "workspace-memory-pool",
  "timing-cache",
  "plugins",
  "profiling",
  "layer-info",
  "report-export",
  "runtime-benchmark",
  "binding-output",
  "safety-cache-policy",
  "device-dla"
)
$sourceArtifacts = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())) | ForEach-Object { [string]$_ })
$requiredSources = @(
  "applications/TensorRtExec/tensor-rt-exec-feature-matrix.json",
  "applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json",
  "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json",
  "applications/TensorRtExec/README.md",
  "applications/TensorRtExec/Core/TensorRtExecOptions.cs",
  "applications/TensorRtExec/Console/TensorRtExecCommand.cs",
  "applications/TensorRtExec/WinForms/MainForm.cs"
)
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$badProofItems = @($items | Where-Object {
  [bool](Get-PropertyOrDefault -Object $_ -Name "isRuntimeProof" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "isPackageConsumerRuntimeProof" -DefaultValue $true)
})
$parseOnlyIds = @("timing-cache", "safety-cache-policy")
$parseOnlyItems = @($items | Where-Object { $parseOnlyIds -contains [string](Get-PropertyOrDefault -Object $_ -Name "optionId" -DefaultValue "") })

$validationItems = New-Object System.Collections.Generic.List[object]
$validationItems.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "tensor-rt-exec-gui-cli-parity-checklist") -Severity "blocker" -Detail "recordKind must be tensor-rt-exec-gui-cli-parity-checklist.")) | Out-Null
$validationItems.Add((New-ValidationItem -Id "checklist-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "checklistState" -DefaultValue "") -eq "release-candidate-gui-cli-parity-non-proof") -Severity "blocker" -Detail "Checklist state must remain non-proof.")) | Out-Null
$validationItems.Add((New-ValidationItem -Id "required-options-present" -Passed (@($requiredIds | Where-Object { $ids -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Checklist must cover the core trtexec-like CLI/GUI options.")) | Out-Null
$validationItems.Add((New-ValidationItem -Id "source-artifacts-present" -Passed (@($requiredSources | Where-Object { $sourceArtifacts -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Checklist must cite TensorRtExec matrices, README, options, CLI, and WinForms source.")) | Out-Null
$validationItems.Add((New-ValidationItem -Id "no-runtime-proof-items" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "runtimeProofItems" -DefaultValue 1) -eq 0 -and $badProofItems.Count -eq 0) -Severity "blocker" -Detail "Checklist must not mark any option as runtime proof or package-consumer proof.")) | Out-Null
$validationItems.Add((New-ValidationItem -Id "no-publish-or-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -Severity "blocker" -Detail "Checklist must not publish, close release, or promote proof.")) | Out-Null
$validationItems.Add((New-ValidationItem -Id "command-preview-covered" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "commandPreviewSupportedCount" -DefaultValue 0) -eq $items.Count) -Severity "blocker" -Detail "WinForms command preview must be tied to shared ToArgumentLine for every checklist row.")) | Out-Null
$validationItems.Add((New-ValidationItem -Id "cli-and-winforms-surface" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "cliSupportedCount" -DefaultValue 0) -ge 10 -and [int](Get-PropertyOrDefault -Object $record -Name "winFormsSupportedCount" -DefaultValue 0) -ge 10) -Severity "blocker" -Detail "Checklist must show broad CLI and WinForms coverage.")) | Out-Null
$validationItems.Add((New-ValidationItem -Id "parse-only-not-implemented-runtime" -Passed (@($parseOnlyItems | Where-Object { ([string](Get-PropertyOrDefault -Object $_ -Name "status" -DefaultValue "")).Contains("parse", [StringComparison]::OrdinalIgnoreCase) }).Count -eq $parseOnlyItems.Count) -Severity "blocker" -Detail "Parse/report-only options must not be labeled as implemented runtime behavior.")) | Out-Null
$validationItems.Add((New-ValidationItem -Id "boundary-text" -Passed ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package-consumer-runtime proof", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary text must block runtime and package-consumer proof substitution.")) | Out-Null

$validationArray = @($validationItems.ToArray())
$failedBlockers = @($validationArray | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "tensor-rt-exec-gui-cli-parity-checklist-ready" } else { "blocked-tensor-rt-exec-gui-cli-parity-checklist-invalid" }

$validation = [pscustomobject]@{
  recordKind = "tensor-rt-exec-gui-cli-parity-checklist-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  inputPath = $resolvedInputPath
  validationItemCount = $validationArray.Count
  failedBlockerCount = $failedBlockers.Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  validationItems = $validationArray
  boundary = "Validation confirms TensorRtExec GUI/CLI parity checklist shape and non-proof boundaries only."
}

$jsonPath = Join-Path $OutputRoot "tensor-rt-exec-gui-cli-parity-checklist-validation.json"
$markdownPath = Join-Path $OutputRoot "tensor-rt-exec-gui-cli-parity-checklist-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validationArray | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# TensorRtExec GUI/CLI Parity Checklist Validation

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

Write-Host "TensorRtExec GUI/CLI parity checklist validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  foreach ($failure in $failedBlockers) {
    Write-Error "$($failure.id): $($failure.detail)"
  }

  exit 1
}
