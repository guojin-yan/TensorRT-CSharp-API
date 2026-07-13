[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-proof-real-input-convergence.json",
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
  throw "Owner proof real input convergence not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$convergenceState = [string](Get-PropertyOrDefault -Object $record -Name "convergenceState" -DefaultValue "")
$inputMappingCount = [int](Get-PropertyOrDefault -Object $record -Name "inputMappingCount" -DefaultValue -1)
$missingRealInputCount = [int](Get-PropertyOrDefault -Object $record -Name "missingRealInputCount" -DefaultValue -1)
$blockedValidatorCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedValidatorCount" -DefaultValue -1)
$blockedProofCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedProofCount" -DefaultValue -1)
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$rows = @(Get-PropertyOrDefault -Object $record -Name "convergenceRows" -DefaultValue @())
$sequence = @(Get-PropertyOrDefault -Object $record -Name "recommendedOwnerSequence" -DefaultValue @())
$sequenceText = [string]::Join("`n", $sequence)
$nonSubstituteKinds = @(Get-PropertyOrDefault -Object $record -Name "nonSubstituteProofKinds" -DefaultValue @())

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-proof-real-input-convergence") -Severity "blocker" -Detail "recordKind must be owner-proof-real-input-convergence.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-convergence-state" -Passed ($convergenceState -eq "blocked-owner-real-input-convergence-required") -Severity "blocker" -Detail "Convergence must stay blocked until real owner input and proof exist.")) | Out-Null
$items.Add((New-ValidationItem -Id "input-mapping-count" -Passed ($inputMappingCount -ge 8 -and @($rows | Where-Object { $_.category -eq "input" }).Count -ge 8) -Severity "blocker" -Detail "Convergence must include at least 8 input mapping rows.")) | Out-Null
$items.Add((New-ValidationItem -Id "missing-real-input-count" -Passed ($missingRealInputCount -eq 0) -Severity "action-required" -Detail "Real owner inputs are still missing.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-validator-count" -Passed ($blockedValidatorCount -eq 0) -Severity "action-required" -Detail "Validators remain blocked/action-required.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-proof-count" -Passed ($blockedProofCount -eq 0) -Severity "action-required" -Detail "Real proof blockers remain unresolved.")) | Out-Null
$items.Add((New-ValidationItem -Id "recommended-sequence-present" -Passed ($sequence.Count -ge 4) -Severity "blocker" -Detail "Recommended owner sequence must not be empty.")) | Out-Null
$items.Add((New-ValidationItem -Id "recommended-sequence-keywords" -Passed ($sequenceText.Contains("post-publish", [StringComparison]::OrdinalIgnoreCase) -and $sequenceText.Contains("clean consumer runtime smoke", [StringComparison]::OrdinalIgnoreCase) -and $sequenceText.Contains("final close decision", [StringComparison]::OrdinalIgnoreCase) -and $sequenceText.Contains("strict close validator", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Recommended owner sequence must mention post-publish, clean consumer runtime smoke, final close decision, and strict close validator.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Convergence must not publish, approve publication, or close release issue.")) | Out-Null

foreach ($requiredKind in @("local feed", "ProjectReference", "direct .nupkg", "template", "draft", "candidate", "schema-only", "preflight-only", "dependency-probe-only", "blocked-by-cuda-driver")) {
  $items.Add((New-ValidationItem -Id "non-substitute-$($requiredKind.Replace(' ', '-').Replace('.', ''))" -Passed ($nonSubstituteKinds -contains $requiredKind) -Severity "blocker" -Detail "nonSubstituteProofKinds must include $requiredKind.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "owner-proof-real-input-convergence-ready"
}
elseif ($failedBlockers.Count -eq 0) {
  "blocked-owner-real-input-convergence-required"
}
else {
  "invalid-owner-proof-real-input-convergence"
}

$validation = [pscustomobject]@{
  recordKind = "owner-proof-real-input-convergence-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  convergenceState = $convergenceState
  isValidConvergenceShape = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  inputMappingCount = $inputMappingCount
  missingRealInputCount = $missingRealInputCount
  blockedValidatorCount = $blockedValidatorCount
  blockedProofCount = $blockedProofCount
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates owner proof real input convergence matrix shape only. It cannot publish packages, approve public release, promote proof, or close release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-proof-real-input-convergence-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-proof-real-input-convergence-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$validationRows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Owner Proof Real Input Convergence Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| convergenceState | ``$($validation.convergenceState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| inputMappingCount | ``$($validation.inputMappingCount)`` |
| missingRealInputCount | ``$($validation.missingRealInputCount)`` |
| blockedValidatorCount | ``$($validation.blockedValidatorCount)`` |
| blockedProofCount | ``$($validation.blockedProofCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($validationRows -join "`r`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner proof real input convergence validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) MissingRealInputs=$missingRealInputCount BlockedValidators=$blockedValidatorCount BlockedProofs=$blockedProofCount PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Owner proof real input convergence validation failed with $($failedBlockers.Count) blocker(s)."
}
