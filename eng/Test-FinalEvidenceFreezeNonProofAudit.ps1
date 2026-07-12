[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-evidence-freeze-non-proof-audit.json",
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

function Resolve-RepositoryPath { param([string]$Path) if ([System.IO.Path]::IsPathRooted($Path)) { return $Path } return Join-Path $RepositoryRoot $Path }
function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}
function New-ValidationItem { param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail) [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail } }

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) { throw "Final evidence freeze non-proof audit not found: $resolvedInputPath" }

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$lanes = @((Get-PropertyOrDefault -Object $record -Name "auditLanes" -DefaultValue @()))
$boundaryFailures = @($lanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "boundaryOk" -DefaultValue $false) })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-evidence-freeze-non-proof-audit") -Severity "blocker" -Detail "recordKind must be final-evidence-freeze-non-proof-audit.")) | Out-Null
$items.Add((New-ValidationItem -Id "lanes-present" -Passed ($lanes.Count -ge 8) -Severity "blocker" -Detail "Audit must include all final publish/close non-proof lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary-ok" -Passed ($boundaryFailures.Count -eq 0) -Severity "blocker" -Detail "All audited lanes must keep non-proof and no-side-effect flags false.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Audit must not publish, approve, promote proof, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-final-evidence-freeze-non-proof-audit" } else { "blocked-final-evidence-freeze-non-proof-audit" }

$validation = [pscustomobject]@{
  recordKind = "final-evidence-freeze-non-proof-audit-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  auditLaneCount = $lanes.Count
  blockedAuditLaneCount = @($lanes | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "blocked" -DefaultValue $false) }).Count
  boundaryFailureCount = $boundaryFailures.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Validation confirms final freeze non-proof boundaries only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-evidence-freeze-non-proof-audit-validation.json"
$markdownPath = Join-Path $OutputRoot "final-evidence-freeze-non-proof-audit-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$markdown = @"
# Final Evidence Freeze Non-Proof Audit Validation

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| auditLaneCount | ``$($validation.auditLaneCount)`` |
| blockedAuditLaneCount | ``$($validation.blockedAuditLaneCount)`` |
| boundaryFailureCount | ``$($validation.boundaryFailureCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) { throw "Final evidence freeze non-proof audit validation failed with $($failedBlockers.Count) blocker(s)." }

Write-Host "Final evidence freeze non-proof audit validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) Lanes=$($validation.auditLaneCount) BoundaryFailures=$($validation.boundaryFailureCount) FailedBlockers=$($validation.failedBlockerCount)"
