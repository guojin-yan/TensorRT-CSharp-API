[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-proof-owner-confirmation.json",
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
  throw "Post-publish proof owner confirmation not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$gates = @((Get-PropertyOrDefault -Object $record -Name "confirmationGates" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "post-publish-proof-owner-confirmation") -Severity "blocker" -Detail "recordKind must be post-publish-proof-owner-confirmation.")) | Out-Null
$items.Add((New-ValidationItem -Id "gate-count" -Passed ($gates.Count -ge 5) -Severity "blocker" -Detail "Confirmation must include public package, owner input, post-publish validation, result import, and close bridge gates.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Confirmation must not publish, promote proof, prove post-publish state, or close the issue.")) | Out-Null

foreach ($gate in $gates) {
  $ready = [bool](Get-PropertyOrDefault -Object $gate -Name "ready" -DefaultValue $false)
  $gateId = [string](Get-PropertyOrDefault -Object $gate -Name "gateId" -DefaultValue "unknown-gate")
  $items.Add((New-ValidationItem -Id "$gateId-owner-action-required" -Passed $ready -Severity "action-required" -Detail "Gate $gateId must reach its required state before post-publish proof owner confirmation is ready.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-post-publish-proof-owner-confirmation"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-post-publish-proof-owner-confirmation-required"
}
else {
  "post-publish-proof-owner-confirmation-ready"
}

$validation = [pscustomobject]@{
  recordKind = "post-publish-proof-owner-confirmation-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  confirmationGateCount = $gates.Count
  blockedConfirmationGateCount = @($gates | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) }).Count
  readyConfirmationGateCount = @($gates | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) }).Count
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
  safetyBoundary = "Post-publish proof owner confirmation validation only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$jsonPath = Join-Path $OutputRoot "post-publish-proof-owner-confirmation-validation.json"
$markdownPath = Join-Path $OutputRoot "post-publish-proof-owner-confirmation-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Post-Publish Proof Owner Confirmation Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| confirmationGateCount | ``$($validation.confirmationGateCount)`` |
| blockedConfirmationGateCount | ``$($validation.blockedConfirmationGateCount)`` |
| readyConfirmationGateCount | ``$($validation.readyConfirmationGateCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish proof owner confirmation validation written to $jsonPath"
Write-Host "ValidationState=$validationState Gates=$($validation.confirmationGateCount) Blocked=$($validation.blockedConfirmationGateCount) FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Post-publish proof owner confirmation has blocker validation failures."
}
