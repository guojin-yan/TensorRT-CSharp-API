[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-external-proof-execution-result.input.template.json",
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

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner external proof execution result input template not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$resultInputs = @(Get-PropertyOrDefault -Object $record -Name "resultInputs" -DefaultValue @())
$forbiddenSubstitutes = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())
$requiredReadyConditions = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "requiredReadyConditions" -DefaultValue @())
$requiredCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "requiredImportCommands" -DefaultValue @())

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-external-proof-execution-result-input-template") -Severity "blocker" -Detail "recordKind must be owner-external-proof-execution-result-input-template.")) | Out-Null
$items.Add((New-ValidationItem -Id "template-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "templateState" -DefaultValue "") -eq "blocked-owner-external-proof-execution-result-required") -Severity "blocker" -Detail "Template must remain blocked until owner fills real external execution results.")) | Out-Null
$items.Add((New-ValidationItem -Id "import-target" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "importTarget" -DefaultValue "") -eq "artifacts/final-release/owner-external-proof-execution-result.input.json") -Severity "blocker" -Detail "Template must point owner to the real import target.")) | Out-Null
$items.Add((New-ValidationItem -Id "result-input-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "resultInputCount" -DefaultValue 0) -eq 6 -and $resultInputs.Count -eq 6) -Severity "blocker" -Detail "Template must carry all six external proof result input lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-and-not-ready" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedResultInputCount" -DefaultValue -1) -eq 6 -and [int](Get-PropertyOrDefault -Object $record -Name "readyForRealProofRecordImportCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Default template lanes must remain blocked and not ready for real proof record import.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-or-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Template must not publish, promote proof, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-conditions-and-forbidden-substitutes" -Passed (($requiredReadyConditions -join "`n").Contains("SHA256", [StringComparison]::OrdinalIgnoreCase) -and ($requiredReadyConditions -join "`n").Contains("nonSubstituteConfirmations", [StringComparison]::OrdinalIgnoreCase) -and ($forbiddenSubstitutes -contains "ProjectReference") -and ($forbiddenSubstitutes -contains "direct .nupkg") -and ($forbiddenSubstitutes -contains "dry-run") -and ($forbiddenSubstitutes -contains "template-only")) -Severity "blocker" -Detail "Template must spell out hash, confirmation, and forbidden substitute boundaries.")) | Out-Null
$items.Add((New-ValidationItem -Id "import-command-chain" -Passed (($requiredCommands -join "`n").Contains("Import-OwnerExternalProofExecutionResult.ps1", [StringComparison]::OrdinalIgnoreCase) -and ($requiredCommands -join "`n").Contains("Test-OwnerExternalProofExecutionResultImport.ps1 -Strict", [StringComparison]::OrdinalIgnoreCase) -and ($requiredCommands -join "`n").Contains("Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Template must include the import, strict validation, and candidate bridge command chain.")) | Out-Null

foreach ($resultInput in $resultInputs) {
  $resultInputId = [string](Get-PropertyOrDefault -Object $resultInput -Name "resultInputId" -DefaultValue "unknown-result-input")
  $packageIdentity = Get-PropertyOrDefault -Object $resultInput -Name "packageIdentity" -DefaultValue $null
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $resultInput -Name "executionInputId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $resultInput -Name "candidateId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $resultInput -Name "proofLane" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $resultInput -Name "runtimePackageKey" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $packageIdentity -Name "nupkgPath" -DefaultValue "")) -and
    ($resultInput.PSObject.Properties.Name -contains "passed") -and
    @((Get-PropertyOrDefault -Object $resultInput -Name "nonSubstituteConfirmations" -DefaultValue @())).Count -ge 10 -and
    @((Get-PropertyOrDefault -Object $resultInput -Name "requiredResultFields" -DefaultValue @())).Count -ge 20 -and
    [bool](Get-PropertyOrDefault -Object $resultInput -Name "strictValidatorInputOnly" -DefaultValue $false) -and
    -not [bool](Get-PropertyOrDefault -Object $resultInput -Name "canPromoteLaneResult" -DefaultValue $true)
  $items.Add((New-ValidationItem -Id "$resultInputId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each resultInputs[] item must carry lane identity, package path/hash fields, passed field, confirmations, required fields, and strict-validator-only boundary.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-owner-external-proof-execution-result-input-template" } else { "blocked-owner-external-proof-execution-result-required" }

$validation = [pscustomobject]@{
  recordKind = "owner-external-proof-execution-result-input-template-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  resultInputCount = $resultInputs.Count
  failedBlockerCount = $failedBlockers.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  boundary = "This validates owner input template shape only. It is not proof and cannot publish or close release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-external-proof-execution-result-input-template-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-external-proof-execution-result-input-template-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Owner External Proof Execution Result Input Template Validation

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| resultInputCount | ``$($validation.resultInputCount)`` |
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

Write-Host "Owner external proof execution result input template validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Owner external proof execution result input template validation failed with $($failedBlockers.Count) blocker(s)."
}
