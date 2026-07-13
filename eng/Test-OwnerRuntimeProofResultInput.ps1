[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-runtime-proof-result-input.template.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

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
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail, [string]$Category = "shape")

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    category = $Category
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

function Test-ExistingPathField {
  param([AllowNull()][object]$Value)

  if (-not (Test-FilledValue -Value $Value)) { return $false }
  $text = [string]$Value
  if ($text.IndexOf("DependencyProbe", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  $resolved = if ([System.IO.Path]::IsPathRooted($text)) { $text } else { Join-Path $RepositoryRoot $text }
  return Test-Path -LiteralPath $resolved -PathType Leaf
}

function Get-FieldValue {
  param(
    [object]$InputObject,
    [string]$FieldPath
  )

  $current = $InputObject
  foreach ($part in $FieldPath.Split(".")) {
    if ($null -eq $current) { return $null }
    if (-not ($current.PSObject.Properties.Name -contains $part)) { return $null }
    $current = $current.PSObject.Properties[$part].Value
  }

  return $current
}

function Get-SubstituteMarkers {
  param([object]$ResultInput)

  $texts = @(
    [string](Get-PropertyOrDefault -Object $ResultInput -Name "executedCommandLine" -DefaultValue ""),
    [string](Get-PropertyOrDefault -Object $ResultInput -Name "workingDirectory" -DefaultValue ""),
    [string](Get-PropertyOrDefault -Object (Get-PropertyOrDefault -Object $ResultInput -Name "packageIdentity" -DefaultValue $null) -Name "packageSource" -DefaultValue ""),
    [string](Get-PropertyOrDefault -Object (Get-PropertyOrDefault -Object $ResultInput -Name "packageIdentity" -DefaultValue $null) -Name "nupkgPath" -DefaultValue "")
  )
  $joined = ($texts -join " ")
  $markers = @(
    "local feed",
    "ProjectReference",
    "direct nupkg",
    "DependencyProbe",
    "sidecar-only",
    "build-only",
    "precheck-only",
    "Skipped=True",
    "skipped run"
  )

  @($markers | Where-Object { $joined.IndexOf($_, [StringComparison]::OrdinalIgnoreCase) -ge 0 })
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner runtime proof result input template not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$resultInputs = @(Get-PropertyOrDefault -Object $record -Name "resultInputs" -DefaultValue @())
$items = New-Object System.Collections.Generic.List[object]
$blockedInputs = @($resultInputs | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "resultInputState" -DefaultValue "") -eq "blocked-owner-runtime-proof-result-input-required" })
$readyInputs = @($resultInputs | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "readyForRuntimeProofValidation" -DefaultValue $false) })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-runtime-proof-result-input-template") -Severity "blocker" -Detail "recordKind must be owner-runtime-proof-result-input-template.")) | Out-Null
$items.Add((New-ValidationItem -Id "template-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "templateState" -DefaultValue "") -eq "blocked-owner-runtime-proof-result-input-required") -Severity "blocker" -Detail "Default owner runtime proof result input must remain blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "result-input-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "resultInputCount" -DefaultValue 0) -eq 6) -Severity "blocker" -Detail "Result input template must cover 6 proof lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-result-input-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedResultInputCount" -DefaultValue -1) -eq $blockedInputs.Count -and $blockedInputs.Count -ge 6) -Severity "blocker" -Detail "Default result inputs must remain blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-result-input-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "readyResultInputCount" -DefaultValue -1) -eq $readyInputs.Count -and $readyInputs.Count -eq 0) -Severity "blocker" -Detail "Default result inputs must not claim proof readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Result input template must not promote proof or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Result input template must not publish or approve public publication.")) | Out-Null

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
  "executedCommandLine",
  "workingDirectory",
  "stdoutPath",
  "stderrPath",
  "stdoutSha256",
  "stderrSha256",
  "mergedTranscriptPath",
  "mergedTranscriptSha256",
  "validatorOutputPath",
  "validatorOutputSha256",
  "exitCode",
  "startedAtUtc",
  "endedAtUtc",
  "ownerReviewer",
  "ownerReviewTimestampUtc"
)

$missingRealInputTotal = 0
$substituteBlockerTotal = 0

foreach ($resultInput in $resultInputs) {
  $resultInputId = [string](Get-PropertyOrDefault -Object $resultInput -Name "resultInputId" -DefaultValue "unknown-result-input")
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $resultInput -Name "executionInputId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $resultInput -Name "candidateId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $resultInput -Name "proofLane" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $resultInput -Name "runtimePackageKey" -DefaultValue "")) -and
    @((Get-PropertyOrDefault -Object $resultInput -Name "nonSubstituteConfirmations" -DefaultValue @())).Count -ge 8 -and
    @((Get-PropertyOrDefault -Object $resultInput -Name "validatorCommands" -DefaultValue @())).Count -gt 0 -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $resultInput -Name "nonSubstituteBoundary" -DefaultValue ""))
  $items.Add((New-ValidationItem -Id "$resultInputId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each result input must include lane identity, confirmations, validators, and non-substitute boundary.")) | Out-Null

  $missingFields = New-Object System.Collections.Generic.List[string]
  foreach ($field in $requiredFields) {
    $value = Get-FieldValue -InputObject $resultInput -FieldPath $field
    $present = switch -Regex ($field) {
      "Sha256$" { Test-HashValue -Value $value; break }
      "stdoutPath|stderrPath|mergedTranscriptPath|validatorOutputPath|nupkgPath" { Test-ExistingPathField -Value $value; break }
      "exitCode" {
        if (-not (Test-FilledValue -Value $value)) { $false }
        else {
          $number = 0
          [int]::TryParse([string]$value, [ref]$number)
        }
        break
      }
      default { Test-FilledValue -Value $value; break }
    }
    if (-not $present) { $missingFields.Add($field) | Out-Null }
  }

  $substitutes = @(Get-SubstituteMarkers -ResultInput $resultInput)
  $missingRealInputTotal += $missingFields.Count
  $substituteBlockerTotal += $substitutes.Count

  $items.Add((New-ValidationItem -Id "$resultInputId-real-fields-filled" -Passed ($missingFields.Count -eq 0) -Severity "action-required" -Category "missing-real-input" -Detail ("Owner must fill existing files, hashes, exit code, timestamps, package identity, host metadata, and reviewer fields. Missing: {0}" -f (($missingFields.ToArray()) -join ", ")))) | Out-Null
  $substituteDetail = if ($substitutes.Count -eq 0) { "none" } else { ($substitutes -join ", ") }
  $items.Add((New-ValidationItem -Id "$resultInputId-substitute-checks-clear" -Passed ($substitutes.Count -eq 0) -Severity "action-required" -Category "substitute-blocker" -Detail ("Owner result input must avoid proof substitutes. Detected: {0}" -f $substituteDetail))) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-owner-runtime-proof-result-input" } else { "blocked-owner-runtime-proof-result-input-required" }

$validation = [pscustomobject]@{
  recordKind = "owner-runtime-proof-result-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  resultInputCount = [int](Get-PropertyOrDefault -Object $record -Name "resultInputCount" -DefaultValue 0)
  blockedResultInputCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedResultInputCount" -DefaultValue 0)
  readyResultInputCount = [int](Get-PropertyOrDefault -Object $record -Name "readyResultInputCount" -DefaultValue 0)
  missingRealInputCount = $missingRealInputTotal
  substituteBlockerCount = $substituteBlockerTotal
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validation checks owner result input shape and missing real evidence only. It is not runtime proof, package publish, post-publish verification, rollback approval, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-runtime-proof-result-input-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-runtime-proof-result-input-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 18)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner Runtime Proof Result Input Validation")
$lines.Add("")
$lines.Add("该验证器拒绝 placeholder、缺失文件、缺失 SHA256、缺失 exit code、缺失 owner review，以及 local feed / ProjectReference / direct nupkg / DependencyProbe-only / sidecar-only / build-only / precheck-only / skipped run 等替代 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| validationState | ``$(ConvertTo-MarkdownCell $validation.validationState)`` |")
$lines.Add("| resultInputCount | ``$($validation.resultInputCount)`` |")
$lines.Add("| blockedResultInputCount | ``$($validation.blockedResultInputCount)`` |")
$lines.Add("| readyResultInputCount | ``$($validation.readyResultInputCount)`` |")
$lines.Add("| missingRealInputCount | ``$($validation.missingRealInputCount)`` |")
$lines.Add("| substituteBlockerCount | ``$($validation.substituteBlockerCount)`` |")
$lines.Add("| failedBlockerCount | ``$($validation.failedBlockerCount)`` |")
$lines.Add("| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($validation.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner runtime proof result input validation written to $jsonPath"
Write-Host "Owner runtime proof result input validation markdown written to $markdownPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount) MissingRealInputs=$($validation.missingRealInputCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Owner runtime proof result input validation has blocker failures: $($failedBlockers.Count)"
}
