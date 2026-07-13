[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-external-proof-overlay-pack.json",
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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)
  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Real external proof overlay pack not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$overlayLines = @((Get-PropertyOrDefault -Object $record -Name "overlayLines" -DefaultValue @()))
$missingOwnerInputFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "missingOwnerInputFields" -DefaultValue @())
$rules = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "requiredGlobalRules" -DefaultValue @())
$sourceArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "real-external-proof-overlay-pack") -Severity "blocker" -Detail "recordKind must be real-external-proof-overlay-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Overlay pack must not publish, approve publication, or close the release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "overlay-lines-present" -Passed ($overlayLines.Count -ge 4) -Severity "blocker" -Detail "Overlay pack must include package-consumer, post-publish, release-close, and final decision lines.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-fields-required" -Passed $false -Severity "action-required" -Detail "Owner fields are intentionally still required until real external proof is supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "global-rules-present" -Passed (($rules -join "`n").Contains("ProjectReference", [StringComparison]::OrdinalIgnoreCase) -and ($rules -join "`n").Contains("direct .nupkg", [StringComparison]::OrdinalIgnoreCase) -and ($rules -join "`n").Contains("blocked-by-cuda-driver", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Overlay rules must reject ProjectReference, direct .nupkg, and blocked-by-cuda-driver substitutes.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifacts-present" -Passed ($sourceArtifacts.Count -ge 8) -Severity "blocker" -Detail "Overlay pack must preserve source artifact references.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "real-external-proof-overlay-ready"
}
elseif ($failedBlockers.Count -eq 0) {
  "blocked-real-owner-input-required"
}
else {
  "invalid-real-external-proof-overlay-pack"
}

$validation = [pscustomobject]@{
  recordKind = "real-external-proof-overlay-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  isValidOverlayShape = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  overlayLineCount = $overlayLines.Count
  missingOwnerInputFieldCount = $missingOwnerInputFields.Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates owner overlay guidance only. It cannot publish packages, promote proof, or close release issue."
}

$jsonPath = Join-Path $OutputRoot "real-external-proof-overlay-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "real-external-proof-overlay-pack-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Real External Proof Overlay Pack Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| overlayLineCount | ``$($validation.overlayLineCount)`` |
| missingOwnerInputFieldCount | ``$($validation.missingOwnerInputFieldCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real external proof overlay pack validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Real external proof overlay pack validation failed with $($failedBlockers.Count) blocker(s)."
}
