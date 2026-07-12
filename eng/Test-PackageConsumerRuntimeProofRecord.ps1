[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\package-consumer-runtime-proof-record.template.json",
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$RequireExistingLog,
  [switch]$FailOnNotProof,
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

function Resolve-InputPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

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

function Test-IntZero {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $parsed = 0
  return [int]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed -eq 0
}

function Test-BoolTrue {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $parsed = $false
  return [bool]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed
}

function Test-DateTimeOffsetFormat {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse(([string]$Value).Trim(), [ref]$parsed)
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

function Test-PublicPackageSourceIsLocal {
  param([AllowNull()][object]$Value)

  $text = ([string]$Value).Trim()
  if (Test-IsPlaceholder -Value $text) {
    return $true
  }

  if ($text -match "^[a-zA-Z]:[\\/]" -or $text.StartsWith("\\", [StringComparison]::Ordinal) -or $text.StartsWith("./", [StringComparison]::Ordinal) -or $text.StartsWith("../", [StringComparison]::Ordinal)) {
    return $true
  }

  return $text.Contains("local", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase)
}

function Test-FileHashMatches {
  param(
    [AllowNull()][object]$Path,
    [AllowNull()][object]$Sha256,
    [bool]$RequireFile
  )

  $pathText = [string]$Path
  $shaText = [string]$Sha256
  if ((Test-IsPlaceholder -Value $pathText) -or -not (Test-Sha256Format -Value $shaText)) {
    return $false
  }

  $resolvedPath = Resolve-InputPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return -not $RequireFile
  }

  $actual = (Get-FileHash -LiteralPath $resolvedPath -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Package consumer runtime proof record not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$proofClassification = [string](Get-PropertyOrDefault -Object $record -Name "proofClassification" -DefaultValue "")
$templateOnly = [bool](Get-PropertyOrDefault -Object $record -Name "templateOnly" -DefaultValue $true)
$runtimePackageKey = [string](Get-PropertyOrDefault -Object $record -Name "runtimePackageKey" -DefaultValue "")
$scan = Get-PropertyOrDefault -Object $record -Name "consumerProjectScan" -DefaultValue $null
$hostInfo = Get-PropertyOrDefault -Object $record -Name "host" -DefaultValue $null
$command = Get-PropertyOrDefault -Object $record -Name "command" -DefaultValue $null
$results = Get-PropertyOrDefault -Object $record -Name "results" -DefaultValue $null
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)

$publicPackageSource = [string](Get-PropertyOrDefault -Object $record -Name "publicPackageSource" -DefaultValue "")
$managedNupkgPath = [string](Get-PropertyOrDefault -Object $record -Name "managedNupkgPath" -DefaultValue "")
$runtimeNupkgPath = [string](Get-PropertyOrDefault -Object $record -Name "runtimeNupkgPath" -DefaultValue "")
$managedNupkgSha256 = [string](Get-PropertyOrDefault -Object $record -Name "managedNupkgSha256" -DefaultValue "")
$runtimeNupkgSha256 = [string](Get-PropertyOrDefault -Object $record -Name "runtimeNupkgSha256" -DefaultValue "")
$smokeCommand = [string](Get-PropertyOrDefault -Object $command -Name "smokeCommand" -DefaultValue "")
$logPath = [string](Get-PropertyOrDefault -Object $command -Name "logPath" -DefaultValue "")
$logSha256 = [string](Get-PropertyOrDefault -Object $command -Name "logSha256" -DefaultValue "")
$exitCode = Get-PropertyOrDefault -Object $command -Name "exitCode" -DefaultValue $null
$smokeStatus = [string](Get-PropertyOrDefault -Object $results -Name "smokeStatus" -DefaultValue "")
$nativeAssetsCopied = Get-PropertyOrDefault -Object $results -Name "nativeAssetsCopied" -DefaultValue $false

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -in @("package-consumer-runtime-proof-record", "package-consumer-runtime-proof-record-template")) -Severity "blocker" -Detail "recordKind must identify a package consumer runtime proof record or template.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Proof record validation must not publish, approve publication, or close the release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-key" -Passed ($runtimePackageKey.Equals($RuntimePackageKey, [StringComparison]::OrdinalIgnoreCase)) -Severity "proof-required" -Detail "runtimePackageKey must match the target runtime package key.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-classification" -Passed ($proofClassification -eq "package-consumer-runtime") -Severity "proof-required" -Detail "Promotable proof requires proofClassification=package-consumer-runtime.")) | Out-Null
$items.Add((New-ValidationItem -Id "real-record-kind" -Passed ($recordKind -eq "package-consumer-runtime-proof-record" -and -not $templateOnly) -Severity "proof-required" -Detail "Real proof requires recordKind=package-consumer-runtime-proof-record and templateOnly=false.")) | Out-Null
$items.Add((New-ValidationItem -Id "clean-root-outside-repository" -Passed (Test-IsOutsideRepository -Path (Get-PropertyOrDefault -Object $record -Name "cleanExternalConsumerRoot" -DefaultValue "")) -Severity "proof-required" -Detail "cleanExternalConsumerRoot must be outside this repository.")) | Out-Null
$items.Add((New-ValidationItem -Id "consumer-project-exists" -Passed ([bool](Get-PropertyOrDefault -Object $scan -Name "projectExists" -DefaultValue $false)) -Severity "proof-required" -Detail "Consumer project must exist.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-project-reference" -Passed ([bool](Get-PropertyOrDefault -Object $scan -Name "noProjectReference" -DefaultValue $false)) -Severity "proof-required" -Detail "Consumer project must not ProjectReference this repository.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-local-feed" -Passed ([bool](Get-PropertyOrDefault -Object $scan -Name "noLocalFeed" -DefaultValue $false) -and -not (Test-PublicPackageSourceIsLocal -Value $publicPackageSource)) -Severity "proof-required" -Detail "Public package source and restore sources must not be local feed/folder evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-direct-nupkg" -Passed ([bool](Get-PropertyOrDefault -Object $scan -Name "noDirectNupkg" -DefaultValue $false)) -Severity "proof-required" -Detail "Direct .nupkg references cannot be public proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "package-sha256-format" -Passed ((Test-Sha256Format -Value $managedNupkgSha256) -and (Test-Sha256Format -Value $runtimeNupkgSha256)) -Severity "proof-required" -Detail "Managed/runtime nupkg SHA256 values must be 64-character hashes.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-nupkg-hash-match" -Passed (Test-FileHashMatches -Path $managedNupkgPath -Sha256 $managedNupkgSha256 -RequireFile $false) -Severity "proof-required" -Detail "Managed nupkg hash must match when the file is available.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-nupkg-hash-match" -Passed (Test-FileHashMatches -Path $runtimeNupkgPath -Sha256 $runtimeNupkgSha256 -RequireFile $false) -Severity "proof-required" -Detail "Runtime nupkg hash must match when the file is available.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-command-runtime-key" -Passed ($smokeCommand.Contains("--runtime-package-key", [StringComparison]::Ordinal) -and $smokeCommand.Contains($RuntimePackageKey, [StringComparison]::OrdinalIgnoreCase)) -Severity "proof-required" -Detail "smokeCommand must include --runtime-package-key and target runtime key.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-exit-code" -Passed (Test-IntZero -Value $exitCode) -Severity "proof-required" -Detail "Smoke command exitCode must be 0.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-status" -Passed ($smokeStatus -eq "passed") -Severity "proof-required" -Detail "Smoke status must be passed.")) | Out-Null
$items.Add((New-ValidationItem -Id "native-assets-copied" -Passed (Test-BoolTrue -Value $nativeAssetsCopied) -Severity "proof-required" -Detail "Native assets must be copied in the clean consumer.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-log-hash-match" -Passed (Test-FileHashMatches -Path $logPath -Sha256 $logSha256 -RequireFile $RequireExistingLog.IsPresent) -Severity "proof-required" -Detail "Smoke log SHA256 must be valid and match when required.")) | Out-Null

foreach ($field in @("ownerName", "machineName", "osDescription", "hostArchitecture", "gpuName", "driverVersion", "cudaDriverSupportedRuntime", "cudaRuntimeVersion", "tensorRtRuntimeVersion", "cudnnVersion", "tensorRtLine")) {
  $items.Add((New-ValidationItem -Id "host-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hostInfo -Name $field -DefaultValue ""))) -Severity "proof-required" -Detail "$field host metadata must be real.")) | Out-Null
}

foreach ($field in @("restoreCommand", "buildCommand")) {
  $items.Add((New-ValidationItem -Id "command-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $command -Name $field -DefaultValue ""))) -Severity "proof-required" -Detail "$field must be real command evidence.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "command-startedAtUtc" -Passed (Test-DateTimeOffsetFormat -Value (Get-PropertyOrDefault -Object $command -Name "startedAtUtc" -DefaultValue "")) -Severity "proof-required" -Detail "startedAtUtc must be real parseable command evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "command-finishedAtUtc" -Passed (Test-DateTimeOffsetFormat -Value (Get-PropertyOrDefault -Object $command -Name "finishedAtUtc" -DefaultValue "")) -Severity "proof-required" -Detail "finishedAtUtc must be real parseable command evidence.")) | Out-Null

foreach ($field in @("stdoutSummary", "stderrSummary")) {
  $items.Add((New-ValidationItem -Id "result-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $results -Name $field -DefaultValue ""))) -Severity "proof-required" -Detail "$field must be reviewed and non-placeholder.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedProofItems = @($items | Where-Object { -not $_.passed -and $_.severity -eq "proof-required" })
$passedItemIds = @($items | Where-Object { $_.passed } | ForEach-Object { [string]$_.id })
$cleanOwnerInputReady = $failedBlockers.Count -eq 0 -and
  $passedItemIds -contains "clean-root-outside-repository" -and
  $passedItemIds -contains "consumer-project-exists" -and
  $passedItemIds -contains "no-project-reference" -and
  $passedItemIds -contains "no-local-feed" -and
  $passedItemIds -contains "no-direct-nupkg" -and
  $passedItemIds -contains "package-sha256-format"
$ownerInputForbiddenSubstituteFree = $passedItemIds -contains "no-project-reference" -and
  $passedItemIds -contains "no-local-feed" -and
  $passedItemIds -contains "no-direct-nupkg"
$ownerInputHashFieldsReady = $passedItemIds -contains "package-sha256-format"
$ownerInputPackageHashFilesMatch = $passedItemIds -contains "managed-nupkg-hash-match" -and
  $passedItemIds -contains "runtime-nupkg-hash-match"
$ownerInputSmokeLogReady = $passedItemIds -contains "smoke-command-runtime-key" -and
  $passedItemIds -contains "smoke-exit-code" -and
  $passedItemIds -contains "smoke-status" -and
  $passedItemIds -contains "native-assets-copied" -and
  $passedItemIds -contains "smoke-log-hash-match"
$ownerInputBlockedReasons = @($items | Where-Object { -not $_.passed } | ForEach-Object { [string]$_.id })
$canPromoteRuntimeProof = $failedBlockers.Count -eq 0 -and $failedProofItems.Count -eq 0
$validationState = if ($canPromoteRuntimeProof) {
  "real-package-consumer-runtime-proof"
}
elseif ($failedBlockers.Count -gt 0) {
  "invalid-record"
}
elseif ($recordKind -eq "package-consumer-runtime-proof-record-template" -or $templateOnly) {
  "template-only"
}
else {
  "incomplete-package-consumer-runtime-proof"
}

$validation = [pscustomobject]@{
  recordKind = "package-consumer-runtime-proof-record-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  proofClassification = $proofClassification
  canPromoteRuntimeProof = $canPromoteRuntimeProof
  isRuntimeExecutionEvidence = $canPromoteRuntimeProof
  cleanOwnerInputReady = $cleanOwnerInputReady
  ownerInputForbiddenSubstituteFree = $ownerInputForbiddenSubstituteFree
  ownerInputHashFieldsReady = $ownerInputHashFieldsReady
  ownerInputPackageHashFilesMatch = $ownerInputPackageHashFilesMatch
  ownerInputSmokeLogReady = $ownerInputSmokeLogReady
  ownerInputCanPromoteRuntimeProof = $canPromoteRuntimeProof
  ownerInputBlockedReason = if ($ownerInputBlockedReasons.Count -eq 0) { "none" } else { $ownerInputBlockedReasons -join "; " }
  failedBlockerCount = $failedBlockers.Count
  failedProofItemCount = $failedProofItems.Count
  failedActionRequiredCount = $failedProofItems.Count
  failOnNotProofRequested = $FailOnNotProof.IsPresent
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates package consumer runtime proof only. It does not publish packages or close the release issue. -FailOnNotProof fails unless every proof-required item passes."
}

$jsonPath = Join-Path $OutputRoot "package-consumer-runtime-proof-record-validation.json"
$markdownPath = Join-Path $OutputRoot "package-consumer-runtime-proof-record-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Package Consumer Runtime Proof Record Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| proofClassification | ``$($validation.proofClassification)`` |
| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |
| cleanOwnerInputReady | ``$($validation.cleanOwnerInputReady)`` |
| ownerInputForbiddenSubstituteFree | ``$($validation.ownerInputForbiddenSubstituteFree)`` |
| ownerInputHashFieldsReady | ``$($validation.ownerInputHashFieldsReady)`` |
| ownerInputPackageHashFilesMatch | ``$($validation.ownerInputPackageHashFilesMatch)`` |
| ownerInputSmokeLogReady | ``$($validation.ownerInputSmokeLogReady)`` |
| ownerInputCanPromoteRuntimeProof | ``$($validation.ownerInputCanPromoteRuntimeProof)`` |
| failOnNotProofRequested | ``$($validation.failOnNotProofRequested)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedProofItemCount | ``$($validation.failedProofItemCount)`` |
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
  throw "Package consumer runtime proof record validation failed with $($failedBlockers.Count) blocker(s)."
}

if ($FailOnNotProof -and -not $canPromoteRuntimeProof) {
  throw "Package consumer runtime proof record is not promotable. FailedBlockers=$($failedBlockers.Count) FailedProofItems=$($failedProofItems.Count) BlockedReason=$($validation.ownerInputBlockedReason)"
}

Write-Host "Package consumer runtime proof record validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) CanPromoteRuntimeProof=$($validation.canPromoteRuntimeProof) FailedBlockers=$($validation.failedBlockerCount) FailedProofItems=$($validation.failedProofItemCount)"
