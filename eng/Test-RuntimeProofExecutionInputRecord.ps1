[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\runtime-proof-execution-input-record.json",
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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
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

function Test-FilledValue {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return $false }
  $text = ([string]$Value).Trim()
  if ([string]::IsNullOrWhiteSpace($text)) { return $false }
  if ($text.StartsWith("<", [StringComparison]::Ordinal) -and $text.EndsWith(">", [StringComparison]::Ordinal)) { return $false }
  if ($text.IndexOf("owner-fill", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("placeholder", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  return $true
}

function Test-HashValue {
  param([AllowNull()][object]$Value)

  if (-not (Test-FilledValue -Value $Value)) { return $false }
  return ([string]$Value) -match "^[a-fA-F0-9]{64}$"
}

function Test-PathField {
  param([AllowNull()][object]$Value)

  if (-not (Test-FilledValue -Value $Value)) { return $false }
  return -not (([string]$Value).IndexOf("DependencyProbe", [StringComparison]::OrdinalIgnoreCase) -ge 0)
}

function Test-TextContainsForbiddenSubstitute {
  param([AllowNull()][string]$Text)

  if ([string]::IsNullOrWhiteSpace($Text)) { return $false }

  $markers = @(
    "local feed",
    "ProjectReference",
    "direct nupkg",
    "DependencyProbe",
    "sidecar-only",
    "build-only",
    "precheck-only"
  )

  foreach ($marker in $markers) {
    if ($Text.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { return $true }
  }

  return $false
}

function Get-FieldValue {
  param(
    [object]$Input,
    [string]$FieldPath
  )

  $current = $Input
  foreach ($part in $FieldPath.Split(".")) {
    if ($null -eq $current) { return $null }
    if (-not ($current.PSObject.Properties.Name -contains $part)) { return $null }
    $current = $current.PSObject.Properties[$part].Value
  }

  return $current
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Runtime proof execution input record not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$executionInputs = @(Get-PropertyOrDefault -Object $record -Name "executionInputs" -DefaultValue @())
$items = New-Object System.Collections.Generic.List[object]
$blockedInputs = @($executionInputs | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "inputState" -DefaultValue "") -eq "blocked-runtime-proof-execution-input-required" })
$readyInputs = @($executionInputs | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "readyForRuntimeProofValidation" -DefaultValue $false) })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "runtime-proof-execution-input-record") -Severity "blocker" -Detail "recordKind must be runtime-proof-execution-input-record.")) | Out-Null
$items.Add((New-ValidationItem -Id "input-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "inputState" -DefaultValue "") -eq "blocked-runtime-proof-execution-input-required") -Severity "blocker" -Detail "Default runtime proof execution input record must remain blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "execution-input-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "executionInputCount" -DefaultValue 0) -eq 6) -Severity "blocker" -Detail "Execution input record must cover 6 proof lanes from the closure pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-input-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedExecutionInputCount" -DefaultValue -1) -eq $blockedInputs.Count -and $blockedInputs.Count -ge 6) -Severity "blocker" -Detail "Default runtime proof execution input record must keep all inputs blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-input-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "readyExecutionInputCount" -DefaultValue -1) -eq $readyInputs.Count -and $readyInputs.Count -eq 0) -Severity "blocker" -Detail "Default runtime proof execution input record must not claim ready inputs.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Execution input record must not promote proof or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Execution input record must not publish or approve public publication.")) | Out-Null

$requiredFields = @(
  "hostMetadata.os",
  "hostMetadata.arch",
  "hostMetadata.gpu",
  "hostMetadata.driverVersion",
  "hostMetadata.cudaVersion",
  "hostMetadata.tensorRtVersion",
  "hostMetadata.dotnetVersion",
  "packageIdentity.packageId",
  "packageIdentity.packageVersion",
  "packageIdentity.nupkgPath",
  "packageIdentity.nupkgSha256",
  "packageIdentity.packageSource",
  "commandLine",
  "workingDirectory",
  "stdoutPath",
  "stderrPath",
  "stdoutSha256",
  "stderrSha256",
  "mergedTranscriptPath",
  "mergedTranscriptSha256",
  "validatorOutputPath",
  "validatorOutputSha256",
  "ownerReviewer",
  "reviewTimestampUtc"
)

foreach ($input in $executionInputs) {
  $executionInputId = [string](Get-PropertyOrDefault -Object $input -Name "executionInputId" -DefaultValue "unknown-execution-input")
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $input -Name "candidateId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $input -Name "proofLane" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $input -Name "runtimePackageKey" -DefaultValue "")) -and
    @((Get-PropertyOrDefault -Object $input -Name "forbiddenSubstituteChecks" -DefaultValue @())).Count -ge 7 -and
    @((Get-PropertyOrDefault -Object $input -Name "expectedArtifacts" -DefaultValue @())).Count -gt 0 -and
    @((Get-PropertyOrDefault -Object $input -Name "validatorCommands" -DefaultValue @())).Count -gt 0 -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $input -Name "nonSubstituteBoundary" -DefaultValue ""))
  $items.Add((New-ValidationItem -Id "$executionInputId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each execution input must include lane, package key, forbidden substitutes, expected artifacts, validator commands, and non-substitute boundary.")) | Out-Null

  $missingFields = New-Object System.Collections.Generic.List[string]
  foreach ($field in $requiredFields) {
    $value = Get-FieldValue -Input $input -FieldPath $field
    $isPresent = switch -Regex ($field) {
      "Sha256$" { Test-HashValue -Value $value; break }
      "Path$|workingDirectory|nupkgPath" { Test-PathField -Value $value; break }
      default { Test-FilledValue -Value $value; break }
    }

    if (-not $isPresent) { $missingFields.Add($field) | Out-Null }
  }

  $commandLine = [string](Get-PropertyOrDefault -Object $input -Name "commandLine" -DefaultValue "")
  $forbiddenSubstituteTriggered = Test-TextContainsForbiddenSubstitute -Text $commandLine
  $items.Add((New-ValidationItem -Id "$executionInputId-owner-fields-filled" -Passed ($missingFields.Count -eq 0) -Severity "action-required" -Detail ("Owner must fill runtime proof fields before validation can promote this input. Missing: {0}" -f (($missingFields.ToArray()) -join ", ")))) | Out-Null
  $items.Add((New-ValidationItem -Id "$executionInputId-forbidden-substitute-command-check" -Passed (-not $forbiddenSubstituteTriggered) -Severity "action-required" -Detail "Command line must not rely on local feed, ProjectReference, direct nupkg, DependencyProbe-only, sidecar-only, build-only, or precheck-only substitute paths.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-runtime-proof-execution-input-record" } else { "blocked-runtime-proof-execution-input-required" }

$validation = [pscustomobject]@{
  recordKind = "runtime-proof-execution-input-record-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  executionInputCount = [int](Get-PropertyOrDefault -Object $record -Name "executionInputCount" -DefaultValue 0)
  blockedExecutionInputCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedExecutionInputCount" -DefaultValue 0)
  readyExecutionInputCount = [int](Get-PropertyOrDefault -Object $record -Name "readyExecutionInputCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates owner-filled execution input fields only. It is not runtime proof, package publish, post-publish verification, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "runtime-proof-execution-input-record-validation.json"
$markdownPath = Join-Path $OutputRoot "runtime-proof-execution-input-record-validation.md"
$validation | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Runtime Proof Execution Input Record Validation")
$lines.Add("")
$lines.Add("该验证器严格拒绝 placeholder、缺失日志、缺失 SHA256、local feed、ProjectReference、direct nupkg、DependencyProbe-only、sidecar-only、build-only、precheck-only 等替代 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| validationState | ``$(ConvertTo-MarkdownCell $validation.validationState)`` |")
$lines.Add("| executionInputCount | ``$($validation.executionInputCount)`` |")
$lines.Add("| blockedExecutionInputCount | ``$($validation.blockedExecutionInputCount)`` |")
$lines.Add("| readyExecutionInputCount | ``$($validation.readyExecutionInputCount)`` |")
$lines.Add("| failedBlockerCount | ``$($validation.failedBlockerCount)`` |")
$lines.Add("| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Failed Action Required")
$lines.Add("")
$lines.Add("| ID | Detail |")
$lines.Add("| --- | --- |")
foreach ($item in @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })) {
  $lines.Add("| $(ConvertTo-MarkdownCell $item.id) | $(ConvertTo-MarkdownCell $item.detail) |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($validation.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Runtime proof execution input record validation written to $jsonPath"
Write-Host "Runtime proof execution input record validation markdown written to $markdownPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Runtime proof execution input record validation has blocker failures: $($failedBlockers.Count)"
}
