[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\package-consumer-runtime-proof-candidate.json",
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

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail
  )

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

function Test-Sha256Format {
  param([AllowNull()][object]$Value)

  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-IsOutsideRepository {
  param([AllowNull()][object]$Path)

  $text = [string]$Path
  if (Test-IsPlaceholder -Value $text) {
    return $false
  }

  try {
    $repositoryFullPath = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $candidateFullPath = [IO.Path]::GetFullPath($text).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    return -not $candidateFullPath.StartsWith($repositoryFullPath, [StringComparison]::OrdinalIgnoreCase)
  }
  catch {
    return $false
  }
}

function Test-OptionalFileHash {
  param(
    [AllowNull()][object]$Path,
    [AllowNull()][object]$Sha256
  )

  $pathText = [string]$Path
  $shaText = [string]$Sha256
  if ((Test-IsPlaceholder -Value $pathText) -or -not (Test-Path -LiteralPath $pathText -PathType Leaf)) {
    return $false
  }

  if (-not (Test-Sha256Format -Value $shaText)) {
    return $false
  }

  $actual = (Get-FileHash -LiteralPath $pathText -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
}

$resolvedInputPath = if ([System.IO.Path]::IsPathRooted($InputPath)) {
  $InputPath
}
else {
  Join-Path $RepositoryRoot $InputPath
}

if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Package consumer runtime proof candidate not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$candidateState = [string](Get-PropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "")
$proofLineId = [string](Get-PropertyOrDefault -Object $record -Name "proofLineId" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$canPromoteProof = [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteProof" -DefaultValue $true)
$rules = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "requiredRealInputRules" -DefaultValue @())
$smokeCommand = [string](Get-PropertyOrDefault -Object $record -Name "smokeCommand" -DefaultValue "")
$runtimePackageKey = [string](Get-PropertyOrDefault -Object $record -Name "runtimePackageKey" -DefaultValue "")

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "package-consumer-runtime-proof-candidate") -Severity "blocker" -Detail "recordKind must be package-consumer-runtime-proof-candidate.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-line-id" -Passed ($proofLineId -eq "package-consumer-runtime") -Severity "blocker" -Detail "proofLineId must be package-consumer-runtime.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-state" -Passed ($candidateState -eq "blocked-real-package-consumer-smoke-required") -Severity "blocker" -Detail "Candidate must remain blocked until real package consumer smoke proof is available.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue -and -not $canPromoteProof) -Severity "blocker" -Detail "Candidate must not publish, approve publication, close the issue, or promote proof.")) | Out-Null

foreach ($requiredRule in @(
  "cleanExternalConsumerIdentity",
  "noProjectReference",
  "noLocalFeedAsPublicProof",
  "managedNupkgSha256",
  "runtimeNupkgSha256",
  "runtimePackageKeyMatches",
  "compatibleHostMetadata",
  "smokeCommandIncludesRuntimePackageKey",
  "smokeLogPath",
  "smokeLogSha256"
)) {
  $items.Add((New-ValidationItem -Id "rule-$requiredRule" -Passed ($rules -contains $requiredRule) -Severity "blocker" -Detail "Candidate must contain rule '$requiredRule'.")) | Out-Null
}

$cleanRoot = [string](Get-PropertyOrDefault -Object $record -Name "cleanExternalConsumerRoot" -DefaultValue "")
$cleanRootOutside = [bool](Get-PropertyOrDefault -Object $record -Name "cleanExternalConsumerRootIsOutsideRepository" -DefaultValue $false)
$usesProjectReference = [bool](Get-PropertyOrDefault -Object $record -Name "consumerProjectUsesProjectReference" -DefaultValue $true)
$usesLocalFeed = [bool](Get-PropertyOrDefault -Object $record -Name "consumerProjectUsesLocalFeed" -DefaultValue $true)
$usesDirectNupkg = [bool](Get-PropertyOrDefault -Object $record -Name "consumerProjectUsesDirectNupkg" -DefaultValue $true)
$publicPackageSourceIsLocal = [bool](Get-PropertyOrDefault -Object $record -Name "publicPackageSourceIsLocal" -DefaultValue $true)
$managedNupkgSha256 = [string](Get-PropertyOrDefault -Object $record -Name "managedNupkgSha256" -DefaultValue "")
$runtimeNupkgSha256 = [string](Get-PropertyOrDefault -Object $record -Name "runtimeNupkgSha256" -DefaultValue "")
$smokeLogSha256 = [string](Get-PropertyOrDefault -Object $record -Name "smokeLogSha256" -DefaultValue "")

$computedOutside = Test-IsOutsideRepository -Path $cleanRoot
$items.Add((New-ValidationItem -Id "clean-consumer-root-outside-repo" -Passed ($cleanRootOutside -and $computedOutside) -Severity "action-required" -Detail "Clean consumer root must exist as a real non-placeholder path outside this repository.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-project-reference" -Passed (-not $usesProjectReference) -Severity "action-required" -Detail "ProjectReference cannot be used as public package proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-local-feed" -Passed (-not $usesLocalFeed -and -not $publicPackageSourceIsLocal) -Severity "action-required" -Detail "Local feed cannot be used as public package proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-direct-nupkg" -Passed (-not $usesDirectNupkg) -Severity "action-required" -Detail "Direct .nupkg reference cannot be used as public package proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-nupkg-sha256-format" -Passed (Test-Sha256Format -Value $managedNupkgSha256) -Severity "action-required" -Detail "Managed nupkg SHA256 must be a 64-character hex hash.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-nupkg-sha256-format" -Passed (Test-Sha256Format -Value $runtimeNupkgSha256) -Severity "action-required" -Detail "Runtime nupkg SHA256 must be a 64-character hex hash.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-log-sha256-format" -Passed (Test-Sha256Format -Value $smokeLogSha256) -Severity "action-required" -Detail "Smoke log SHA256 must be a 64-character hex hash.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-command-runtime-key" -Passed ($smokeCommand.Contains("--runtime-package-key", [StringComparison]::Ordinal) -and $smokeCommand.Contains($runtimePackageKey, [StringComparison]::Ordinal)) -Severity "blocker" -Detail "Smoke command must include --runtime-package-key and the candidate runtime key.")) | Out-Null

$hostMetadata = Get-PropertyOrDefault -Object $record -Name "compatibleHostMetadata" -DefaultValue $null
$hostFieldsReady = $false
if ($null -ne $hostMetadata) {
  $hostFieldsReady = -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hostMetadata -Name "hostOs" -DefaultValue "")) -and
    -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hostMetadata -Name "hostArchitecture" -DefaultValue "")) -and
    -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hostMetadata -Name "cudaDriverVersion" -DefaultValue "")) -and
    -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hostMetadata -Name "cudaRuntimeVersion" -DefaultValue "")) -and
    -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hostMetadata -Name "tensorRtVersion" -DefaultValue ""))
}

$items.Add((New-ValidationItem -Id "compatible-host-metadata" -Passed $hostFieldsReady -Severity "action-required" -Detail "Compatible host metadata must include OS, architecture, CUDA driver/runtime, and TensorRT version.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-nupkg-hash-match" -Passed (Test-OptionalFileHash -Path (Get-PropertyOrDefault -Object $record -Name "managedNupkgPath" -DefaultValue "") -Sha256 $managedNupkgSha256) -Severity "action-required" -Detail "Managed nupkg file must exist and match its SHA256 before proof promotion.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-nupkg-hash-match" -Passed (Test-OptionalFileHash -Path (Get-PropertyOrDefault -Object $record -Name "runtimeNupkgPath" -DefaultValue "") -Sha256 $runtimeNupkgSha256) -Severity "action-required" -Detail "Runtime nupkg file must exist and match its SHA256 before proof promotion.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-log-hash-match" -Passed (Test-OptionalFileHash -Path (Get-PropertyOrDefault -Object $record -Name "smokeLogPath" -DefaultValue "") -Sha256 $smokeLogSha256) -Severity "action-required" -Detail "Smoke log file must exist and match its SHA256 before proof promotion.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "candidate-ready-for-real-proof-review"
}
else {
  "blocked-real-package-consumer-smoke-required"
}

$validation = [pscustomobject]@{
  recordKind = "package-consumer-runtime-proof-candidate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  isValidCandidate = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  promotedProofItemCount = 0
  canPromoteProof = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validation reviews owner input readiness only. It cannot publish packages, close release issues, or promote proof without real external consumer smoke evidence."
}

$jsonPath = Join-Path $OutputRoot "package-consumer-runtime-proof-candidate-validation.json"
$markdownPath = Join-Path $OutputRoot "package-consumer-runtime-proof-candidate-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Package Consumer Runtime Proof Candidate Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| isValidCandidate | ``$($validation.isValidCandidate)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canPromoteProof | ``$($validation.canPromoteProof)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Safety Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Package consumer runtime proof candidate validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Package consumer runtime proof candidate validation written to $jsonPath"
Write-Host "Package consumer runtime proof candidate validation written to $markdownPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount) CanPromoteProof=$($validation.canPromoteProof) PerformsPublish=$($validation.performsPublish) CanPublishPublicly=$($validation.canPublishPublicly) CanCloseReleaseIssue=$($validation.canCloseReleaseIssue)"
