[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-issue-close-record-real-input-map.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

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

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release issue close record real input map not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$mapState = [string](Get-PropertyOrDefault -Object $record -Name "mapState" -DefaultValue "")
$mappedInputCount = [int](Get-PropertyOrDefault -Object $record -Name "mappedInputCount" -DefaultValue -1)
$missingRealInputCount = [int](Get-PropertyOrDefault -Object $record -Name "missingRealInputCount" -DefaultValue -1)
$blockedMappingCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedMappingCount" -DefaultValue -1)
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$strictValidators = @(Get-PropertyOrDefault -Object $record -Name "strictValidators" -DefaultValue @())
$nonSubstituteKinds = @(Get-PropertyOrDefault -Object $record -Name "nonSubstituteProofKinds" -DefaultValue @())
$mappings = @(Get-PropertyOrDefault -Object $record -Name "realInputMappings" -DefaultValue @())

$strictValidatorText = [string]::Join("`n", $strictValidators)
$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "release-issue-close-record-real-input-map") -Severity "blocker" -Detail "recordKind must be release-issue-close-record-real-input-map.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-map-state" -Passed ($mapState -eq "blocked-release-close-real-input-required") -Severity "blocker" -Detail "Map must remain blocked until real release close inputs are filled.")) | Out-Null
$items.Add((New-ValidationItem -Id "mapped-input-count" -Passed ($mappedInputCount -ge 8 -and $mappings.Count -ge 8) -Severity "blocker" -Detail "Map must include all owner input tasks.")) | Out-Null
$items.Add((New-ValidationItem -Id "missing-real-input-count" -Passed ($missingRealInputCount -eq 0) -Severity "action-required" -Detail "Real owner inputs are still missing.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-mapping-count" -Passed ($blockedMappingCount -eq 0) -Severity "action-required" -Detail "At least one mapping must stay blocked until real owner input exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-close-validator" -Passed ($strictValidatorText.Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $strictValidatorText.Contains("-FailOnNotCloseReady", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "Strict close validator must include Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Map must not publish, approve publication, or close release issue.")) | Out-Null

foreach ($mapping in $mappings) {
  $mappingId = [string](Get-PropertyOrDefault -Object $mapping -Name "id" -DefaultValue "unknown")
  $targetArtifact = [string](Get-PropertyOrDefault -Object $mapping -Name "targetArtifact" -DefaultValue "")
  $targetField = [string](Get-PropertyOrDefault -Object $mapping -Name "targetField" -DefaultValue "")
  $releaseCloseField = [string](Get-PropertyOrDefault -Object $mapping -Name "releaseCloseField" -DefaultValue "")
  $firstCommand = [string](Get-PropertyOrDefault -Object $mapping -Name "firstCommand" -DefaultValue "")
  $validatorCommand = [string](Get-PropertyOrDefault -Object $mapping -Name "validatorCommand" -DefaultValue "")
  $strictCloseValidatorCommand = [string](Get-PropertyOrDefault -Object $mapping -Name "strictCloseValidatorCommand" -DefaultValue "")
  $hasRequiredShape = -not [string]::IsNullOrWhiteSpace($targetArtifact) -and
    -not [string]::IsNullOrWhiteSpace($targetField) -and
    -not [string]::IsNullOrWhiteSpace($releaseCloseField) -and
    -not [string]::IsNullOrWhiteSpace($firstCommand) -and
    -not [string]::IsNullOrWhiteSpace($validatorCommand) -and
    $strictCloseValidatorCommand.Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and
    $strictCloseValidatorCommand.Contains("-FailOnNotCloseReady", [StringComparison]::Ordinal)

  $items.Add((New-ValidationItem -Id "mapping-shape-$mappingId" -Passed $hasRequiredShape -Severity "blocker" -Detail "$mappingId must include target artifact/field, release close field, first command, validator command, and strict close validator.")) | Out-Null
}

foreach ($requiredKind in @("local feed", "ProjectReference", "direct .nupkg", "template", "draft", "candidate", "schema-only", "preflight-only", "dependency-probe-only", "blocked-by-cuda-driver")) {
  $items.Add((New-ValidationItem -Id "non-substitute-$($requiredKind.Replace(' ', '-').Replace('.', ''))" -Passed ($nonSubstituteKinds -contains $requiredKind) -Severity "blocker" -Detail "nonSubstituteProofKinds must include $requiredKind.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "release-close-real-input-map-ready"
}
elseif ($failedBlockers.Count -eq 0) {
  "blocked-release-close-real-input-required"
}
else {
  "invalid-release-issue-close-record-real-input-map"
}

$validation = [pscustomobject]@{
  recordKind = "release-issue-close-record-real-input-map-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  mapState = $mapState
  isValidRealInputMapShape = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  mappedInputCount = $mappedInputCount
  missingRealInputCount = $missingRealInputCount
  blockedMappingCount = $blockedMappingCount
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates release close real input mapping shape only. It cannot publish packages, approve public release, promote proof, or close release issue."
}

$jsonPath = Join-Path $OutputRoot "release-issue-close-record-real-input-map-validation.json"
$markdownPath = Join-Path $OutputRoot "release-issue-close-record-real-input-map-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Release Issue Close Record Real Input Map Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| mapState | ``$($validation.mapState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| mappedInputCount | ``$($validation.mappedInputCount)`` |
| missingRealInputCount | ``$($validation.missingRealInputCount)`` |
| blockedMappingCount | ``$($validation.blockedMappingCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release issue close record real input map validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) MissingRealInputs=$missingRealInputCount BlockedMappings=$blockedMappingCount PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Release issue close record real input map validation failed with $($failedBlockers.Count) blocker(s)."
}
