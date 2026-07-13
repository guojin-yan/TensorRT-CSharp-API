[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-docs-and-nuget-metadata-audit.json",
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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

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

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release docs and NuGet metadata audit not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$auditItems = @((Get-PropertyOrDefault -Object $record -Name "validationItems" -DefaultValue @()))
$auditItemIds = @($auditItems | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$requiredItemIds = @(
  "readme-managed-package-id",
  "readme-zh-managed-package-id",
  "samples-readme-current-samples",
  "yolovision-family-matrix-complete",
  "yolovision-task-matrix-complete",
  "onnx-to-engine-boundary",
  "tensorrtexec-cli-winforms-modes",
  "managed-pack-csproj-package-id",
  "runtime-split-package-roles",
  "runtime-split-version-lines",
  "no-live-yolodet-name",
  "public-docs-gate-companion-passed"
)
$missingItemIds = @($requiredItemIds | Where-Object { $auditItemIds -notcontains $_ })
$failedAuditBlockers = @($auditItems | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) -and [string](Get-PropertyOrDefault -Object $_ -Name "severity" -DefaultValue "") -eq "blocker" })

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-docs-and-nuget-metadata-audit") "blocker" "recordKind must be release-docs-and-nuget-metadata-audit.")) | Out-Null
$items.Add((New-ValidationItem "audit-ready-non-proof" ([string](Get-PropertyOrDefault -Object $record -Name "auditState" -DefaultValue "") -eq "release-docs-and-nuget-metadata-audit-ready-non-proof") "blocker" "Audit must pass as ready-non-proof, not as publish/proof readiness.")) | Out-Null
$items.Add((New-ValidationItem "required-items-present" ($missingItemIds.Count -eq 0) "blocker" ("Missing required audit items: " + ($missingItemIds -join ", ")))) | Out-Null
$items.Add((New-ValidationItem "audit-items-no-blockers" ($failedAuditBlockers.Count -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "failedBlockerCount" -DefaultValue -1) -eq 0) "blocker" "Audit validation items must not have blocker failures.")) | Out-Null
$items.Add((New-ValidationItem "minimum-audit-surface" ([int](Get-PropertyOrDefault -Object $record -Name "auditItemCount" -DefaultValue 0) -ge 18) "blocker" "Audit must cover docs, samples, applications, package metadata, runtime split packages, and stale-name scan.")) | Out-Null
$items.Add((New-ValidationItem "runtime-lines-present" ((@((Get-PropertyOrDefault -Object $record -Name "splitRuntimeTensorRtLines" -DefaultValue @())) -contains "8") -and (@((Get-PropertyOrDefault -Object $record -Name "splitRuntimeTensorRtLines" -DefaultValue @())) -contains "10") -and (@((Get-PropertyOrDefault -Object $record -Name "splitRuntimeTensorRtLines" -DefaultValue @())) -contains "11")) "blocker" "Split runtime manifest must cover TensorRT 8/10/11 lines.")) | Out-Null
$items.Add((New-ValidationItem "no-yolodet-blocked-matches" ([int](Get-PropertyOrDefault -Object $record -Name "yoloDetBlockedMatchCount" -DefaultValue -1) -eq 0) "blocker" "Audit must not find live YoloDet names outside boundary context.")) | Out-Null
$items.Add((New-ValidationItem "no-side-effects" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) "blocker" "Audit must not publish, use tokens, promote proof, claim post-publish proof, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) {
  "release-docs-and-nuget-metadata-audit-ready-non-proof"
}
else {
  "invalid-release-docs-and-nuget-metadata-audit"
}

$validation = [ordered]@{
  recordKind = "release-docs-and-nuget-metadata-audit-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  auditState = [string](Get-PropertyOrDefault -Object $record -Name "auditState" -DefaultValue "")
  auditItemCount = [int](Get-PropertyOrDefault -Object $record -Name "auditItemCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  auditFailedBlockerCount = $failedAuditBlockers.Count
  yoloDetBlockedMatchCount = [int](Get-PropertyOrDefault -Object $record -Name "yoloDetBlockedMatchCount" -DefaultValue 0)
  splitRuntimePackageCount = [int](Get-PropertyOrDefault -Object $record -Name "splitRuntimePackageCount" -DefaultValue 0)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  canPromotePublicProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Release docs and NuGet metadata audit validation is non-publishing metadata validation only. It cannot prove public package availability, runtime smoke, post-publish clean consumer execution, or release close readiness."
}

$jsonPath = Join-Path $OutputRoot "release-docs-and-nuget-metadata-audit-validation.json"
$markdownPath = Join-Path $OutputRoot "release-docs-and-nuget-metadata-audit-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Release Docs And NuGet Metadata Audit Validation

Generated at: ``$($validation.generatedAtUtc)``

| Item | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| auditState | ``$($validation.auditState)`` |
| auditItemCount | ``$($validation.auditItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| yoloDetBlockedMatchCount | ``$($validation.yoloDetBlockedMatchCount)`` |
| splitRuntimePackageCount | ``$($validation.splitRuntimePackageCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Boundary

$($validation.safetyBoundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release docs and NuGet metadata audit validation written to $jsonPath"
Write-Output "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Release docs and NuGet metadata audit has blocker validation failures."
}
