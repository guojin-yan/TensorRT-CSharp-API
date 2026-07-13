[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-publish-execution-result-input.template.json",
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

function Get-BoolPropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [bool]$DefaultValue)
  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue $DefaultValue
  if ($value -is [bool]) { return [bool]$value }
  $parsed = $false
  if ([bool]::TryParse(([string]$value).Trim(), [ref]$parsed)) { return $parsed }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [AllowNull()][object]$Passed, [string]$Severity, [string]$Detail)
  $normalizedPassed = $false
  if ($Passed -is [bool]) {
    $normalizedPassed = [bool]$Passed
  }
  elseif ($null -ne $Passed) {
    $parsed = $false
    $text = ([string]$Passed).Trim()
    if ([bool]::TryParse($text, [ref]$parsed)) {
      $normalizedPassed = $parsed
    }
    elseif ($text -match "^-?\d+$") {
      $normalizedPassed = ([int]$text) -ne 0
    }
  }
  [pscustomobject]@{ id = $Id; passed = $normalizedPassed; severity = $Severity; detail = $Detail }
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

function Test-DateTimeOffsetFormat {
  param([AllowNull()][object]$Value)
  if (Test-IsPlaceholder -Value $Value) { return $false }
  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse(([string]$Value).Trim(), [ref]$parsed)
}

function Test-IntZero {
  param([AllowNull()][object]$Value)
  if (Test-IsPlaceholder -Value $Value) { return $false }
  $parsed = 0
  return [int]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed -eq 0
}

function Test-BoolTrue {
  param([AllowNull()][object]$Value)
  if (Test-IsPlaceholder -Value $Value) { return $false }
  $parsed = $false
  return [bool]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed
}

function Test-HttpsNonLocalUrl {
  param([AllowNull()][object]$Value)
  $text = ([string]$Value).Trim()
  if (Test-IsPlaceholder -Value $text) { return $false }
  if (-not $text.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase)) { return $false }
  $isPublic = -not $text.Contains("localhost", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.Contains("127.0.0.1", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.Contains("package-managed-dry-run", [StringComparison]::OrdinalIgnoreCase)
  return [bool]$isPublic
}

function Test-PathNotForbiddenSubstitute {
  param([AllowNull()][object]$Value)
  $text = ([string]$Value).Trim()
  if (Test-IsPlaceholder -Value $text) { return $false }
  $isAllowed = -not $text.Contains("package-managed-dry-run", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.Contains("github-actions-runs", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.Contains("\bin\", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.Contains("\obj\", [StringComparison]::OrdinalIgnoreCase)
  return [bool]$isAllowed
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

function Test-TranscriptHasNoTokenLikeContent {
  param([AllowNull()][object]$Path)
  $pathText = [string]$Path
  if (Test-IsPlaceholder -Value $pathText) { return $false }
  $resolvedPath = Resolve-RepositoryPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) { return $false }
  $content = [string](Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8)
  return $content -notmatch "(?i)(NUGET_AUTH_TOKEN|ghp_[A-Za-z0-9_]+|pat_[A-Za-z0-9_]+|github[_-]?token\s*[:=]\s*\S+|publish[_-]?token\s*[:=]\s*\S+|nuget[_-]?key\s*[:=]\s*\S+|api[_-]?key\s+(?!REDACTED\b)\S+|--api-key\s+(?!REDACTED\b)\S+|[A-Za-z0-9_]{32,}\.[A-Za-z0-9_]{16,}\.[A-Za-z0-9_]{16,})"
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner publish execution result input not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$preReleaseReadinessMatrixPath = [string](Get-PropertyOrDefault -Object $record -Name "preReleaseReadinessMatrixPath" -DefaultValue "artifacts\final-release\pre-release-package-proof-readiness-matrix.json")
$resolvedPreReleaseReadinessMatrixPath = Resolve-RepositoryPath -Path $preReleaseReadinessMatrixPath
$preReleaseReadinessMatrix = if (Test-Path -LiteralPath $resolvedPreReleaseReadinessMatrixPath -PathType Leaf) {
  Get-Content -LiteralPath $resolvedPreReleaseReadinessMatrixPath -Raw -Encoding utf8 | ConvertFrom-Json
}
else {
  $null
}
$preReleaseReadinessMatrixState = [string](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "matrixState" -DefaultValue "missing-pre-release-package-proof-readiness-matrix")
$preReleaseReadinessMatrixReady = $preReleaseReadinessMatrixState.Equals("pre-release-package-proof-ready", [StringComparison]::OrdinalIgnoreCase)
$preReleaseReadinessBlockedLaneCount = [int](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "blockedLaneCount" -DefaultValue 999)
$preReleaseReadinessLanes = @((Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "lanes" -DefaultValue @()))
$preReleaseRequiredLaneIds = @(
  "source-quality-ci",
  "current-head-package-dry-run",
  "owner-dispatch-pack",
  "public-package-download",
  "clean-external-package-consumer-runtime",
  "post-publish-clean-consumer-proof"
)
$preReleaseLaneIds = @($preReleaseReadinessLanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$preReleaseMissingLaneIds = @($preReleaseRequiredLaneIds | Where-Object { $preReleaseLaneIds -notcontains $_ })
$preReleaseMetadataMissing = @($preReleaseReadinessLanes | Where-Object {
    [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "requiredEvidence" -DefaultValue "")) -or
    [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "validatorPath" -DefaultValue ""))
  } | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$preReleasePrematurePromoteFindings = @($preReleaseReadinessLanes | Where-Object {
    $ready = Get-BoolPropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false
    $canPromotePublic = Get-BoolPropertyOrDefault -Object $_ -Name "canPromotePublicProof" -DefaultValue $false
    $canPromoteRuntime = Get-BoolPropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $false
    $canPromotePostPublish = Get-BoolPropertyOrDefault -Object $_ -Name "canPromotePostPublishProof" -DefaultValue $false
    (-not $ready -and ($canPromotePublic -or $canPromoteRuntime -or $canPromotePostPublish)) -or
    (@("source-quality-ci", "current-head-package-dry-run", "owner-dispatch-pack") -contains [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -and ($canPromotePublic -or $canPromoteRuntime -or $canPromotePostPublish))
  } | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-publish-execution-result-input") -Severity "blocker" -Detail "recordKind must be owner-publish-execution-result-input.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-automation-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not (Get-BoolPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not (Get-BoolPropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not (Get-BoolPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not (Get-BoolPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not (Get-BoolPropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -Severity "blocker" -Detail "Validation must not publish, use tokens, promote runtime proof, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "pre-release-readiness-matrix-present" -Passed ($null -ne $preReleaseReadinessMatrix) -Severity "action-required" -Detail "preReleaseReadinessMatrixPath must point to the generated readiness matrix.")) | Out-Null
$items.Add((New-ValidationItem -Id "pre-release-readiness-lanes-present" -Passed ($preReleaseMissingLaneIds.Count -eq 0) -Severity "blocker" -Detail $(if ($preReleaseMissingLaneIds.Count -eq 0) { "All required pre-release readiness lanes are present." } else { "Missing pre-release lane(s): $($preReleaseMissingLaneIds -join ', ')" }))) | Out-Null
$items.Add((New-ValidationItem -Id "pre-release-readiness-lane-metadata-present" -Passed ($preReleaseMetadataMissing.Count -eq 0) -Severity "blocker" -Detail $(if ($preReleaseMetadataMissing.Count -eq 0) { "All pre-release readiness lanes expose requiredEvidence and validatorPath." } else { "Missing lane metadata for: $($preReleaseMetadataMissing -join ', ')" }))) | Out-Null
$items.Add((New-ValidationItem -Id "pre-release-readiness-no-premature-promote-flags" -Passed ($preReleasePrematurePromoteFindings.Count -eq 0) -Severity "blocker" -Detail $(if ($preReleasePrematurePromoteFindings.Count -eq 0) { "No blocked or non-proof readiness lane exposes promote flags." } else { "Premature promote lane(s): $($preReleasePrematurePromoteFindings -join ', ')" }))) | Out-Null
$items.Add((New-ValidationItem -Id "pre-release-readiness-ready-for-owner-publish-execution" -Passed ($preReleaseReadinessMatrixReady -and $preReleaseReadinessBlockedLaneCount -eq 0) -Severity "action-required" -Detail "Owner publish execution result cannot be ready until pre-release package proof readiness matrix is ready and has no blocked lanes.")) | Out-Null

foreach ($field in @("managedPackageVersion", "runtimePackageVersion", "ownerName", "ownerApprovalReference", "rollbackDecision")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be supplied by owner execution result evidence.")) | Out-Null
}

foreach ($field in @("ownerReviewedAtUtc", "publishStartedAtUtc", "publishCompletedAtUtc")) {
  $items.Add((New-ValidationItem -Id "$field-parseable" -Passed (Test-DateTimeOffsetFormat -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must be a parseable timestamp.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "publish-executed-by-owner" -Passed (Test-BoolTrue -Value (Get-PropertyOrDefault -Object $record -Name "publishExecutedByOwner" -DefaultValue "")) -Severity "action-required" -Detail "publishExecutedByOwner must be true for a real owner execution result.")) | Out-Null
$items.Add((New-ValidationItem -Id "publish-command-reviewed" -Passed (Test-BoolTrue -Value (Get-PropertyOrDefault -Object $record -Name "publishCommandReviewed" -DefaultValue "")) -Severity "action-required" -Detail "publishCommandReviewed must be true.")) | Out-Null
$items.Add((New-ValidationItem -Id "publish-exit-code-zero" -Passed (Test-IntZero -Value (Get-PropertyOrDefault -Object $record -Name "publishExitCode" -DefaultValue "")) -Severity "action-required" -Detail "publishExitCode must be 0.")) | Out-Null

$publishCommand = [string](Get-PropertyOrDefault -Object $record -Name "publishCommand" -DefaultValue "")
$publishCommandHasSecret = $publishCommand -match "(?i)(NUGET_AUTH_TOKEN|ghp_[A-Za-z0-9_]+|pat_[A-Za-z0-9_]+|github[_-]?token\s*[:=]\s*\S+|publish[_-]?token\s*[:=]\s*\S+|nuget[_-]?key\s*[:=]\s*\S+|api[_-]?key\s+(?!REDACTED\b)\S+|--api-key\s+(?!REDACTED\b)\S+|[A-Za-z0-9_]{32,}\.[A-Za-z0-9_]{16,}\.[A-Za-z0-9_]{16,})"
$items.Add((New-ValidationItem -Id "publish-command-shape" -Passed (-not (Test-IsPlaceholder -Value $publishCommand) -and $publishCommand.Contains("nuget", [StringComparison]::OrdinalIgnoreCase) -and $publishCommand.Contains("push", [StringComparison]::OrdinalIgnoreCase) -and -not $publishCommandHasSecret) -Severity "action-required" -Detail "publishCommand must be a redacted nuget push command with no token-like content.")) | Out-Null

foreach ($field in @("publicManagedPackageUrl", "publicRuntimePackageUrl", "nugetPackageMetadataUrl", "githubPackagesMetadataUrl")) {
  $items.Add((New-ValidationItem -Id "$field-https-public" -Passed (Test-HttpsNonLocalUrl -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must be an HTTPS public non-local URL.")) | Out-Null
}

foreach ($field in @("pushTranscript", "pushStdout", "pushStderr", "downloadedManagedNupkg", "downloadedRuntimeNupkg", "releaseNotes", "rollbackPlan")) {
  $pathName = "${field}Path"
  $shaName = "${field}Sha256"
  $items.Add((New-ValidationItem -Id "$field-path-not-forbidden-substitute" -Passed (Test-PathNotForbiddenSubstitute -Value (Get-PropertyOrDefault -Object $record -Name $pathName -DefaultValue "")) -Severity "action-required" -Detail "$pathName must not come from bin/obj/package-managed-dry-run/github-actions-runs.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$field-hash-match" -Passed (Test-FileHashMatches -Path (Get-PropertyOrDefault -Object $record -Name $pathName -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name $shaName -DefaultValue "")) -Severity "action-required" -Detail "$pathName must exist and match $shaName.")) | Out-Null
}

foreach ($field in @("pushTranscriptPath", "pushStdoutPath", "pushStderrPath")) {
  $items.Add((New-ValidationItem -Id "$field-no-token-like-content" -Passed (Test-TranscriptHasNoTokenLikeContent -Path (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must not contain API key, token, PAT, or credential-like content.")) | Out-Null
}

foreach ($field in @("confirmsPreReleaseReadinessMatrixReviewed", "confirmsNoTokenPersisted", "confirmsNoTokenInTranscripts", "confirmsNoDryRunArtifactSubstitution", "confirmsNoLocalFeedSubstitution", "confirmsNoDirectNupkgSubstitution", "confirmsNoGitHubActionsArtifactSubstitution", "confirmsPublicPackageDownloadProofStillRequired", "confirmsPostPublishProofStillRequired")) {
  $items.Add((New-ValidationItem -Id "$field-true" -Passed (Test-BoolTrue -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must be true.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$ready = $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0
$validationState = if ($ready) { "owner-publish-execution-result-input-ready" } else { "blocked-owner-publish-execution-result-required" }

$validation = [pscustomobject]@{
  recordKind = "owner-publish-execution-result-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  ownerExecutionResultReady = $ready
  preReleaseReadinessMatrixPath = $preReleaseReadinessMatrixPath
  preReleaseReadinessMatrixState = $preReleaseReadinessMatrixState
  preReleaseReadinessMatrixReady = $preReleaseReadinessMatrixReady
  preReleaseReadinessBlockedLaneCount = $preReleaseReadinessBlockedLaneCount
  preReleaseMissingLaneIds = @($preReleaseMissingLaneIds)
  preReleaseMetadataMissingLaneIds = @($preReleaseMetadataMissing)
  preReleasePrematurePromoteFindings = @($preReleasePrematurePromoteFindings)
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Owner publish execution result validation checks owner-supplied evidence shape only. It does not publish, does not use tokens, does not prove runtime smoke, does not prove post-publish clean consumer execution, and cannot close the release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-publish-execution-result-input-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-publish-execution-result-input-validation.md"
$validation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object { "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |" }
$markdown = @"
# Owner Publish Execution Result Input Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| ownerExecutionResultReady | ``$($validation.ownerExecutionResultReady)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| preReleaseReadinessMatrixPath | ``$($validation.preReleaseReadinessMatrixPath)`` |
| preReleaseReadinessMatrixState | ``$($validation.preReleaseReadinessMatrixState)`` |
| preReleaseReadinessMatrixReady | ``$($validation.preReleaseReadinessMatrixReady)`` |
| preReleaseReadinessBlockedLaneCount | ``$($validation.preReleaseReadinessBlockedLaneCount)`` |
| preReleaseMissingLaneIds | ``$($validation.preReleaseMissingLaneIds -join ", ")`` |
| preReleaseMetadataMissingLaneIds | ``$($validation.preReleaseMetadataMissingLaneIds -join ", ")`` |
| preReleasePrematurePromoteFindings | ``$($validation.preReleasePrematurePromoteFindings -join ", ")`` |
| performsPublish | ``$($validation.performsPublish)`` |
| usesPublishToken | ``$($validation.usesPublishToken)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner publish execution result input validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Owner publish execution result input validation failed with $($failedBlockers.Count) blocker(s)."
}
