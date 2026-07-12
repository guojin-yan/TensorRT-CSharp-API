[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-public-proof-bridge.json",
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
  throw "Release close public proof bridge not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$gates = @((Get-PropertyOrDefault -Object $record -Name "publicProofGates" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-close-public-proof-bridge") -Severity "blocker" -Detail "recordKind must be release-close-public-proof-bridge.")) | Out-Null
$items.Add((New-ValidationItem -Id "gate-count" -Passed ($gates.Count -ge 6) -Severity "blocker" -Detail "Bridge must include post-publish confirmation, close owner bridge, candidate, final decision, strict close, and classification gates.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-close-command" -Passed (([string](Get-PropertyOrDefault -Object $record -Name "strictCloseCommand" -DefaultValue "")).Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and ([string](Get-PropertyOrDefault -Object $record -Name "strictCloseCommand" -DefaultValue "")).Contains("-FailOnNotCloseReady", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "Bridge must point at strict release issue close validation.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Bridge must not publish, promote proof, prove post-publish state, or close release issue.")) | Out-Null

foreach ($gate in $gates) {
  $ready = [bool](Get-PropertyOrDefault -Object $gate -Name "ready" -DefaultValue $false)
  $gateId = [string](Get-PropertyOrDefault -Object $gate -Name "gateId" -DefaultValue "unknown-gate")
  $items.Add((New-ValidationItem -Id "$gateId-owner-action-required" -Passed $ready -Severity "action-required" -Detail "Gate $gateId must reach its required state before release close public proof bridge is ready.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-release-close-public-proof-bridge"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-release-close-public-proof-required"
}
else {
  "release-close-public-proof-ready"
}

$validation = [pscustomobject]@{
  recordKind = "release-close-public-proof-bridge-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  publicProofGateCount = $gates.Count
  blockedPublicProofGateCount = @($gates | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) }).Count
  readyPublicProofGateCount = @($gates | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) }).Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Release close public proof bridge validation only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$jsonPath = Join-Path $OutputRoot "release-close-public-proof-bridge-validation.json"
$markdownPath = Join-Path $OutputRoot "release-close-public-proof-bridge-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Release Close Public Proof Bridge Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| publicProofGateCount | ``$($validation.publicProofGateCount)`` |
| blockedPublicProofGateCount | ``$($validation.blockedPublicProofGateCount)`` |
| readyPublicProofGateCount | ``$($validation.readyPublicProofGateCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close public proof bridge validation written to $jsonPath"
Write-Host "ValidationState=$validationState Gates=$($validation.publicProofGateCount) Blocked=$($validation.blockedPublicProofGateCount) FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Release close public proof bridge has blocker validation failures."
}
