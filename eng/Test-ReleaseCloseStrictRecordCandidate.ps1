[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-strict-record-candidate.json",
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

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-FileHashMatches {
  param([AllowNull()][object]$Path, [AllowNull()][object]$Sha256)
  $pathText = [string]$Path
  $shaText = [string]$Sha256
  if ((Test-IsPlaceholder -Value $pathText) -or -not (Test-Sha256Format -Value $shaText)) { return $false }
  $resolvedPath = Resolve-RepositoryPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) { return $false }
  $actual = (Get-FileHash -LiteralPath $resolvedPath -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release close strict record candidate not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$candidateState = [string](Get-PropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$mismatchedHashCount = [int](Get-PropertyOrDefault -Object $record -Name "mismatchedHashCount" -DefaultValue -1)
$missingOwnerInputCount = [int](Get-PropertyOrDefault -Object $record -Name "missingOwnerInputCount" -DefaultValue -1)
$missingRealProofCount = [int](Get-PropertyOrDefault -Object $record -Name "missingRealProofCount" -DefaultValue -1)
$strictCommand = [string](Get-PropertyOrDefault -Object $record -Name "strictValidatorCommand" -DefaultValue "")

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "release-close-strict-record-candidate") -Severity "blocker" -Detail "recordKind must be release-close-strict-record-candidate.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-candidate-state" -Passed ($candidateState -eq "blocked-release-close-strict-record-owner-input-required") -Severity "blocker" -Detail "Strict close record candidate must remain blocked until real owner proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Strict candidate must not publish, approve publication, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-close-validator-command" -Passed ($strictCommand.Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $strictCommand.Contains("-FailOnNotCloseReady", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "strict close validator command must remain Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifacts-present" -Passed (@(Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @()).Count -ge 9) -Severity "blocker" -Detail "Strict candidate must preserve all source artifacts.")) | Out-Null
$items.Add((New-ValidationItem -Id "hash-line-count" -Passed (@(Get-PropertyOrDefault -Object $record -Name "hashLines" -DefaultValue @()).Count -ge 9) -Severity "blocker" -Detail "Strict candidate must include all hash lines.")) | Out-Null

foreach ($line in @(Get-PropertyOrDefault -Object $record -Name "hashLines" -DefaultValue @())) {
  $lineId = [string](Get-PropertyOrDefault -Object $line -Name "id" -DefaultValue "unknown")
  $linePath = [string](Get-PropertyOrDefault -Object $line -Name "path" -DefaultValue "")
  $lineSha = [string](Get-PropertyOrDefault -Object $line -Name "actualSha256" -DefaultValue "")
  $items.Add((New-ValidationItem -Id "hash-$lineId" -Passed (Test-FileHashMatches -Path $linePath -Sha256 $lineSha) -Severity "blocker" -Detail "$lineId path and current SHA256 must match.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "mismatched-hash-count" -Passed ($mismatchedHashCount -eq 0) -Severity "blocker" -Detail "All strict candidate hashes must match current local files.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-input-still-required" -Passed ($missingOwnerInputCount -eq 0) -Severity "action-required" -Detail "Owner placeholders must be replaced with real close input before ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "real-proof-still-required" -Passed ($missingRealProofCount -eq 0) -Severity "action-required" -Detail "Real post-publish proof, final decision, and close validation must pass before ready.")) | Out-Null

foreach ($field in @(Get-PropertyOrDefault -Object $record -Name "requiredOwnerFields" -DefaultValue @())) {
  $fieldId = [string](Get-PropertyOrDefault -Object $field -Name "id" -DefaultValue "unknown")
  $fieldValue = [string](Get-PropertyOrDefault -Object $field -Name "value" -DefaultValue "")
  $items.Add((New-ValidationItem -Id "owner-field-$fieldId" -Passed (-not (Test-IsPlaceholder -Value $fieldValue)) -Severity "action-required" -Detail "$fieldId must be real owner input before close.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "release-close-strict-record-candidate-ready"
}
elseif ($failedBlockers.Count -eq 0) {
  "blocked-release-close-strict-record-owner-input-required"
}
else {
  "invalid-release-close-strict-record-candidate"
}

$validation = [pscustomobject]@{
  recordKind = "release-close-strict-record-candidate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  candidateState = $candidateState
  isValidStrictCandidateShape = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  mismatchedHashCount = $mismatchedHashCount
  missingOwnerInputCount = $missingOwnerInputCount
  missingRealProofCount = $missingRealProofCount
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates strict close record candidate shape and hashes only. It cannot publish packages, approve publication, or close release issue."
}

$jsonPath = Join-Path $OutputRoot "release-close-strict-record-candidate-validation.json"
$markdownPath = Join-Path $OutputRoot "release-close-strict-record-candidate-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Release Close Strict Record Candidate Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| candidateState | ``$($validation.candidateState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| mismatchedHashCount | ``$($validation.mismatchedHashCount)`` |
| missingOwnerInputCount | ``$($validation.missingOwnerInputCount)`` |
| missingRealProofCount | ``$($validation.missingRealProofCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close strict record candidate validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) MismatchedHashCount=$mismatchedHashCount PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Release close strict record candidate validation failed with $($failedBlockers.Count) blocker(s)."
}
