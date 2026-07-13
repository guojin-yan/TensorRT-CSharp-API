[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-real-proof-execution-closure-pack.json",
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

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Test-NonEmptyArray {
  param([AllowNull()][object[]]$Items)

  return @($Items).Count -gt 0
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner real proof execution closure pack not found: $resolvedInputPath"
}

$pack = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$closureItems = @(Get-PropertyOrDefault -Object $pack -Name "closureItems" -DefaultValue @())
$blockedClosureItems = @($closureItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "closureState" -DefaultValue "") -eq "blocked-owner-real-proof-execution-closure-required" })
$readyClosureItems = @($closureItems | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "readyForExecutionClosure" -DefaultValue $false) })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $pack -Name "recordKind" -DefaultValue "") -eq "owner-real-proof-execution-closure-pack") -Severity "blocker" -Detail "recordKind must be owner-real-proof-execution-closure-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "closure-state" -Passed ([string](Get-PropertyOrDefault -Object $pack -Name "closureState" -DefaultValue "") -eq "blocked-owner-real-proof-execution-closure-required") -Severity "blocker" -Detail "Closure pack must remain blocked until Owner executes real proof inputs.")) | Out-Null
$items.Add((New-ValidationItem -Id "closure-item-count" -Passed ([int](Get-PropertyOrDefault -Object $pack -Name "closureItemCount" -DefaultValue 0) -eq 6) -Severity "blocker" -Detail "Closure pack must cover 6 closure items.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-closure-item-count" -Passed ([int](Get-PropertyOrDefault -Object $pack -Name "blockedClosureItemCount" -DefaultValue -1) -eq $blockedClosureItems.Count -and $blockedClosureItems.Count -ge 6) -Severity "blocker" -Detail "Default closure pack must keep all closure items blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-closure-item-count" -Passed ([int](Get-PropertyOrDefault -Object $pack -Name "readyClosureItemCount" -DefaultValue -1) -eq $readyClosureItems.Count -and $readyClosureItems.Count -eq 0) -Severity "blocker" -Detail "Default closure pack must not claim ready closure items.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $pack -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $pack -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $pack -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $pack -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Closure pack must not promote proof or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $pack -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $pack -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Closure pack must not publish or approve public publication.")) | Out-Null

foreach ($closureItem in $closureItems) {
  $candidateId = [string](Get-PropertyOrDefault -Object $closureItem -Name "candidateId" -DefaultValue "unknown-candidate")
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $closureItem -Name "proofLane" -DefaultValue "")) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $closureItem -Name "ownerDeltaIds" -DefaultValue @())) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $closureItem -Name "firstCommand" -DefaultValue "")) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $closureItem -Name "expectedArtifacts" -DefaultValue @())) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $closureItem -Name "requiredLogs" -DefaultValue @())) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $closureItem -Name "requiredSha256" -DefaultValue @())) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $closureItem -Name "validatorCommands" -DefaultValue @())) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $closureItem -Name "promotionGuardRequirements" -DefaultValue @())) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $closureItem -Name "releaseCloseFollowUp" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $closureItem -Name "notProofBoundary" -DefaultValue ""))
  $items.Add((New-ValidationItem -Id "$candidateId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each closure item must include owner delta ids, first command, artifacts, logs, hashes, validators, guard requirements, release close follow-up, and non-proof boundary.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$candidateId-owner-action-required" -Passed ([bool](Get-PropertyOrDefault -Object $closureItem -Name "readyForExecutionClosure" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must execute real proof closure before this item can become ready.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-owner-real-proof-execution-closure-pack" } else { "blocked-owner-real-proof-execution-closure-required" }

$validation = [pscustomobject]@{
  recordKind = "owner-real-proof-execution-closure-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  closureItemCount = [int](Get-PropertyOrDefault -Object $pack -Name "closureItemCount" -DefaultValue 0)
  readyClosureItemCount = [int](Get-PropertyOrDefault -Object $pack -Name "readyClosureItemCount" -DefaultValue 0)
  blockedClosureItemCount = [int](Get-PropertyOrDefault -Object $pack -Name "blockedClosureItemCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates owner execution closure pack shape only. It is not runtime proof, not publication approval, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-real-proof-execution-closure-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-real-proof-execution-closure-pack-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 14)
$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace('|', '\|')) |"
}

$markdown = @"
# Owner Real Proof Execution Closure Pack Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| closureItemCount | ``$($validation.closureItemCount)`` |
| readyClosureItemCount | ``$($validation.readyClosureItemCount)`` |
| blockedClosureItemCount | ``$($validation.blockedClosureItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canPromoteRuntimeProof | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |
| isRuntimeExecutionProof | ``False`` |
| isReleaseCloseProof | ``False`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join [Environment]::NewLine)

## Boundary

$($validation.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner real proof execution closure pack validation written to $jsonPath"
Write-Host "Owner real proof execution closure pack validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState Items=$($validation.closureItemCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Owner real proof execution closure pack validation failed with $($failedBlockers.Count) blocker(s)."
}
