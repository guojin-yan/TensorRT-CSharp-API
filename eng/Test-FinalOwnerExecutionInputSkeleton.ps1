[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-input-skeleton.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final owner execution input skeleton not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$groups = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "fieldGroups" -DefaultValue @()))
$fields = @($groups | ForEach-Object { Convert-ToArray (Get-PropertyOrDefault -Object $_ -Name "fields" -DefaultValue @()) })
$fieldIds = @($fields | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$nonSubstitutes = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "nonSubstituteProofKinds" -DefaultValue @()) | ForEach-Object { [string]$_ })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$requiredFieldIds = @(
  "clean-consumer-project-root",
  "package-source-url",
  "managed-package-id",
  "managed-package-version",
  "managed-package-sha256",
  "runtime-package-id",
  "runtime-package-version",
  "runtime-package-key",
  "runtime-package-sha256",
  "native-asset-listing-path",
  "native-asset-listing-sha256",
  "stdout-path",
  "stderr-path",
  "merged-transcript-path",
  "stdout-sha256",
  "stderr-sha256",
  "merged-transcript-sha256",
  "exit-code",
  "smoke-status",
  "host-os",
  "host-arch",
  "host-rid",
  "gpu-name",
  "nvidia-driver",
  "cuda-runtime-toolkit",
  "tensorrt-version",
  "cudnn-version",
  "owner-name",
  "owner-machine",
  "reviewed-at-utc",
  "owner-note",
  "post-publish-downloaded-package-hash",
  "post-publish-proof-log-path",
  "post-publish-proof-log-sha256",
  "dual-package-nuget-owner-authorization-url",
  "dual-package-nuget-public-download-url",
  "dual-package-nuget-clean-consumer-log-path",
  "dual-package-nuget-post-publish-proof-log-sha256",
  "dual-package-github-owner-authorization-url",
  "dual-package-github-restore-source-url",
  "dual-package-github-runtime-dll-resolution-report-path",
  "dual-package-github-clean-runtime-smoke-log-sha256",
  "rollback-review",
  "final-close-decision",
  "strict-validator-output-path",
  "strict-validator-output-sha256",
  "strict-validator-chain-state"
)

$requiredMarkers = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "pre-publish smoke reused as post-publish proof",
  "final owner execution input skeleton",
  "owner fillable input skeleton",
  "placeholder owner input",
  "dual package route proof",
  "dual package final close lanes"
)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-execution-input-skeleton") -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "skeletonState" -DefaultValue "") -eq "blocked-final-owner-real-input-required") -Severity "blocker" -Detail "Skeleton must remain blocked until real Owner input exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-fields" -Passed (@($requiredFieldIds | Where-Object { $fieldIds -notcontains $_ }).Count -eq 0 -and $fields.Count -ge $requiredFieldIds.Count) -Severity "blocker" -Detail "Skeleton must expose all required Owner input fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "all-placeholders" -Passed (@($fields | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "placeholder" -DefaultValue $false) -or [bool](Get-PropertyOrDefault -Object $_ -Name "readyForImport" -DefaultValue $true) -or [string](Get-PropertyOrDefault -Object $_ -Name "status" -DefaultValue "") -ne "missing owner input" }).Count -eq 0) -Severity "blocker" -Detail "Every field must default to placeholder/missing and not ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "counts" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "requiredFieldCount" -DefaultValue 0) -eq $fields.Count -and [int](Get-PropertyOrDefault -Object $record -Name "missingFieldCount" -DefaultValue 0) -eq $fields.Count -and [int](Get-PropertyOrDefault -Object $record -Name "placeholderFieldCount" -DefaultValue 0) -eq $fields.Count -and [int](Get-PropertyOrDefault -Object $record -Name "readyForImportFieldCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Summary counts must match blocked placeholder state.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-runtime-smoke-field-alignment-projected" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke-field-alignment" -and [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke-field-alignment-valid" -and [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus" -DefaultValue "") -eq "Smoke=not-requested" -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount" -DefaultValue 0) -ge 30 -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount" -DefaultValue -1) -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Skeleton must project owner runtime smoke field alignment as zero-missing non-proof coverage while keeping every owner field placeholder-only.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsRuntimeExecution" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "Skeleton must remain non-proof and non-publish.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must exclude proof, publish, close, and package push.")) | Out-Null

foreach ($marker in $requiredMarkers) {
  $items.Add((New-ValidationItem -Id "marker-$($marker.Replace(' ', '-').Replace('.', 'dot'))" -Passed ($nonSubstitutes -contains $marker -or $raw.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Required non-substitute marker must be present: $marker")) | Out-Null
}

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "blocked-final-owner-real-input-required" } else { "invalid-final-owner-execution-input-skeleton" }

$validation = [pscustomobject]@{
  recordKind = "final-owner-execution-input-skeleton-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  inputPath = $resolvedInputPath
  fieldGroupCount = $groups.Count
  requiredFieldCount = $fields.Count
  missingFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "missingFieldCount" -DefaultValue 0)
  placeholderFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "placeholderFieldCount" -DefaultValue 0)
  readyForImportFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "readyForImportFieldCount" -DefaultValue 0)
  packageConsumerOwnerRuntimeSmokeFieldAlignmentState = [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentState" -DefaultValue "")
  packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState = [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState" -DefaultValue "")
  packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount" -DefaultValue -1)
  failedBlockerCount = $failedBlockers.Count
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = $validationItems
  boundary = "Validation confirms the fillable Owner input skeleton remains placeholder-only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-input-skeleton-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-input-skeleton-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validationItems) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Final Owner Execution Input Skeleton Validation

| Field | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| fieldGroupCount | ``$($validation.fieldGroupCount)`` |
| requiredFieldCount | ``$($validation.requiredFieldCount)`` |
| readyForImportFieldCount | ``$($validation.readyForImportFieldCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@
Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown

Write-Host "Final owner execution input skeleton validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final owner execution input skeleton validation failed with $($failedBlockers.Count) blocker(s)."
}
