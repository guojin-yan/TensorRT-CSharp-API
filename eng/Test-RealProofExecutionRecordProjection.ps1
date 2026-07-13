[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-proof-execution-record-projection.json",
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

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-ArrayHasItems {
  param([AllowNull()][object]$Value)

  return @($Value).Count -gt 0
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Real proof execution record projection not found: $resolvedInputPath"
}

$projection = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $projection -Name "recordKind" -DefaultValue "")
$records = @(Get-PropertyOrDefault -Object $projection -Name "proofExecutionRecords" -DefaultValue @())
$forbiddenSubstitutes = @(Get-PropertyOrDefault -Object $projection -Name "forbiddenSubstitutes" -DefaultValue @())

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "real-proof-execution-record-projection") -Severity "blocker" -Detail "recordKind must be real-proof-execution-record-projection.")) | Out-Null
$items.Add((New-ValidationItem -Id "record-count" -Passed ($records.Count -eq 6) -Severity "blocker" -Detail "Projection must include exactly 6 proof execution records.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-shape" -Passed ([string](Get-PropertyOrDefault -Object $projection -Name "projectionState" -DefaultValue "") -eq "blocked-real-proof-execution-record-input-required") -Severity "blocker" -Detail "Projection must remain blocked until owner fills real proof execution records.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $projection -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $projection -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $projection -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $projection -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Projection must not promote runtime proof, release close proof, or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $projection -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $projection -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Projection must not publish or approve public publication.")) | Out-Null

foreach ($required in @("local feed", "ProjectReference", "direct .nupkg", "DependencyProbe", "build-only", "template", "Windows handoff for Linux proof", "hash-only audit")) {
  $items.Add((New-ValidationItem -Id "forbidden-$($required.Replace(' ', '-').Replace('.', '').ToLowerInvariant())" -Passed ($forbiddenSubstitutes -contains $required) -Severity "blocker" -Detail "Forbidden substitute must be listed: $required.")) | Out-Null
}

foreach ($record in $records) {
  $recordId = [string](Get-PropertyOrDefault -Object $record -Name "recordId" -DefaultValue "unknown-record")
  $hostMetadata = Get-PropertyOrDefault -Object $record -Name "hostMetadata" -DefaultValue $null
  $execution = Get-PropertyOrDefault -Object $record -Name "execution" -DefaultValue $null
  $logs = @(Get-PropertyOrDefault -Object $record -Name "logs" -DefaultValue @())
  $hashes = @(Get-PropertyOrDefault -Object $record -Name "hashes" -DefaultValue @())
  $validators = @(Get-PropertyOrDefault -Object $record -Name "validatorOutputs" -DefaultValue @())
  $forbiddenChecks = @(Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteChecks" -DefaultValue @())
  $promotionFlags = Get-PropertyOrDefault -Object $record -Name "promotionFlags" -DefaultValue $null
  $commands = @(Get-PropertyOrDefault -Object $execution -Name "commands" -DefaultValue @())

  $items.Add((New-ValidationItem -Id "$recordId-source-track" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "sourceTrackId" -DefaultValue ""))) -Severity "blocker" -Detail "Each record must have a source track id.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$recordId-host-metadata-shape" -Passed ($null -ne $hostMetadata) -Severity "blocker" -Detail "Each record must include hostMetadata.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$recordId-execution-shape" -Passed ($null -ne $execution -and (Test-ArrayHasItems -Value $commands)) -Severity "blocker" -Detail "Each record must include execution commands.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$recordId-log-shape" -Passed (Test-ArrayHasItems -Value $logs) -Severity "blocker" -Detail "Each record must include logs.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$recordId-hash-shape" -Passed (Test-ArrayHasItems -Value $hashes) -Severity "blocker" -Detail "Each record must include hashes.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$recordId-validator-shape" -Passed (Test-ArrayHasItems -Value $validators) -Severity "blocker" -Detail "Each record must include validator outputs.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$recordId-forbidden-checks-shape" -Passed (Test-ArrayHasItems -Value $forbiddenChecks) -Severity "blocker" -Detail "Each record must include forbidden substitute checks.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$recordId-promotion-flags-shape" -Passed ($null -ne $promotionFlags) -Severity "blocker" -Detail "Each record must include promotionFlags.")) | Out-Null

  foreach ($field in @("hostOs", "hostArchitecture", "cudaDriverVersion", "cudaRuntimeVersion", "tensorRtVersion", "runtimePackageKey")) {
    $items.Add((New-ValidationItem -Id "$recordId-$field-input-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hostMetadata -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "Owner must fill real host metadata field: $field.")) | Out-Null
  }

  foreach ($command in $commands) {
    $items.Add((New-ValidationItem -Id "$recordId-command-log-input-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $command -Name "logPath" -DefaultValue ""))) -Severity "action-required" -Detail "Owner must fill command log path for real proof execution.")) | Out-Null
    $items.Add((New-ValidationItem -Id "$recordId-command-sha-input-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $command -Name "logSha256" -DefaultValue ""))) -Severity "action-required" -Detail "Owner must fill command log SHA256 for real proof execution.")) | Out-Null
  }

  foreach ($log in $logs) {
    $items.Add((New-ValidationItem -Id "$recordId-log-path-input-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $log -Name "path" -DefaultValue ""))) -Severity "action-required" -Detail "Owner must fill log path.")) | Out-Null
    $items.Add((New-ValidationItem -Id "$recordId-log-sha-input-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $log -Name "sha256" -DefaultValue ""))) -Severity "action-required" -Detail "Owner must fill log SHA256.")) | Out-Null
    $items.Add((New-ValidationItem -Id "$recordId-log-hash-match-required" -Passed ([bool](Get-PropertyOrDefault -Object $log -Name "matches" -DefaultValue $false)) -Severity "action-required" -Detail "Validator must confirm log SHA256 match before promotion.")) | Out-Null
  }

  foreach ($hash in $hashes) {
    $items.Add((New-ValidationItem -Id "$recordId-hash-path-input-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hash -Name "path" -DefaultValue ""))) -Severity "action-required" -Detail "Owner must fill hash path.")) | Out-Null
    $items.Add((New-ValidationItem -Id "$recordId-hash-sha-input-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hash -Name "sha256" -DefaultValue ""))) -Severity "action-required" -Detail "Owner must fill expected SHA256.")) | Out-Null
    $items.Add((New-ValidationItem -Id "$recordId-hash-match-required" -Passed ([bool](Get-PropertyOrDefault -Object $hash -Name "matches" -DefaultValue $false)) -Severity "action-required" -Detail "Validator must confirm hash match before promotion.")) | Out-Null
  }

  foreach ($validator in $validators) {
    $items.Add((New-ValidationItem -Id "$recordId-validator-command-input-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $validator -Name "command" -DefaultValue ""))) -Severity "action-required" -Detail "Owner must fill validator command.")) | Out-Null
    $items.Add((New-ValidationItem -Id "$recordId-validator-output-pass-required" -Passed ([bool](Get-PropertyOrDefault -Object $validator -Name "passed" -DefaultValue $false)) -Severity "action-required" -Detail "Validator output must pass before promotion.")) | Out-Null
  }

  foreach ($check in $forbiddenChecks) {
    $checkName = [string](Get-PropertyOrDefault -Object $check -Name "name" -DefaultValue "unknown")
    $items.Add((New-ValidationItem -Id "$recordId-forbidden-$($checkName.Replace(' ', '-').Replace('.', '').ToLowerInvariant())-blocked" -Passed ([bool](Get-PropertyOrDefault -Object $check -Name "checked" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $check -Name "present" -DefaultValue $true) -and [bool](Get-PropertyOrDefault -Object $check -Name "passed" -DefaultValue $false)) -Severity "action-required" -Detail "Forbidden substitute must be checked absent before promotion: $checkName.")) | Out-Null
  }

  $items.Add((New-ValidationItem -Id "$recordId-no-proof-flags" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Record-level proof flags must remain false in projection.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-real-proof-execution-record-projection"
}
else {
  "blocked-real-proof-execution-record-input-required"
}

$validation = [pscustomobject]@{
  recordKind = "real-proof-execution-record-projection-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  recordCount = $records.Count
  blockedRecordCount = [int](Get-PropertyOrDefault -Object $projection -Name "blockedRecordCount" -DefaultValue $records.Count)
  readyRecordCount = [int](Get-PropertyOrDefault -Object $projection -Name "readyRecordCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates projected execution record shape only. It is not runtime proof, not publication approval, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "real-proof-execution-record-projection-validation.json"
$markdownPath = Join-Path $OutputRoot "real-proof-execution-record-projection-validation.md"
$validation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace('|', '\|')) |"
}

$markdown = @"
# Real Proof Execution Record Projection Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| recordCount | ``$($validation.recordCount)`` |
| blockedRecordCount | ``$($validation.blockedRecordCount)`` |
| readyRecordCount | ``$($validation.readyRecordCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canPromoteRuntimeProof | ``False`` |
| canCloseReleaseIssue | ``False`` |
| isRuntimeExecutionProof | ``False`` |
| isReleaseCloseProof | ``False`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join [Environment]::NewLine)

## Boundary

$($validation.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real proof execution record projection validation written to $jsonPath"
Write-Host "Real proof execution record projection validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState Records=$($validation.recordCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Real proof execution record projection validation failed with $($failedBlockers.Count) blocker(s)."
}
