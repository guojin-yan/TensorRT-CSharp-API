[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath { param([string]$Path) if ([System.IO.Path]::IsPathRooted($Path)) { return $Path } return Join-Path $RepositoryRoot $Path }
function Read-JsonOrNull { param([string]$Path) $resolved = Resolve-RepositoryPath -Path $Path; if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }; return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json }
function Get-PropertyOrDefault { param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue) if ($null -eq $Object) { return $DefaultValue }; if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }; return $DefaultValue }
function ConvertTo-MarkdownCell { param([AllowNull()][object]$Value) if ($null -eq $Value) { return "" }; return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ") }
function New-ConsistencyItem { param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail) [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail } }
function Get-Text { param([AllowNull()][object]$Value) return ([string]$Value).Trim() }

$finalExecutionPack = Read-JsonOrNull "artifacts\final-release\public-publish-final-owner-execution-pack.json"
$authorizationInput = Read-JsonOrNull "artifacts\final-release\owner-public-publish-authorization-input-import.json"
$authorizationInputValidation = Read-JsonOrNull "artifacts\final-release\owner-public-publish-authorization-input-validation.json"
$authorizationGate = Read-JsonOrNull "artifacts\final-release\owner-public-publish-authorization-gate.json"
$commandCrossCheck = Read-JsonOrNull "artifacts\final-release\public-publish-command-cross-check.json"
$publishResultOwnerInput = Read-JsonOrNull "artifacts\final-release\public-publish-result-owner-input.template.json"
$publishResultImport = Read-JsonOrNull "artifacts\final-release\public-publish-result-import.json"
$publishResultConvergence = Read-JsonOrNull "artifacts\final-release\public-publish-result-authorization-convergence-gate.json"

$managedPackageIdAuth = Get-Text (Get-PropertyOrDefault -Object $authorizationInput -Name "managedPackageId" -DefaultValue "")
$managedPackageIdResult = Get-Text (Get-PropertyOrDefault -Object $publishResultImport -Name "packageId" -DefaultValue "")
$managedPackageVersionAuth = Get-Text (Get-PropertyOrDefault -Object $authorizationInput -Name "managedPackageVersion" -DefaultValue "")
$managedPackageVersionResult = Get-Text (Get-PropertyOrDefault -Object $publishResultOwnerInput -Name "packageVersion" -DefaultValue "")
$managedHashAuth = Get-Text (Get-PropertyOrDefault -Object $authorizationInput -Name "managedPackageSha256" -DefaultValue "")
$managedHashResult = Get-Text (Get-PropertyOrDefault -Object $publishResultImport -Name "managedNupkgSha256" -DefaultValue "")
$runtimePackageIdAuth = Get-Text (Get-PropertyOrDefault -Object $authorizationInput -Name "runtimePackageId" -DefaultValue "")
$runtimePackageVersionAuth = Get-Text (Get-PropertyOrDefault -Object $authorizationInput -Name "runtimePackageVersion" -DefaultValue "")
$runtimeHashAuth = Get-Text (Get-PropertyOrDefault -Object $authorizationInput -Name "runtimePackageSha256" -DefaultValue "")
$runtimeHashesResult = @((Get-PropertyOrDefault -Object $publishResultImport -Name "runtimePackageSha256" -DefaultValue @()))

$forbidden = @("local feed", "ProjectReference", "direct .nupkg", "dry-run", "dashboard", "manual approval", "queued workflow", "missing runner", "sidecar-only", "TensorRtExec report")
$recordsForText = @($finalExecutionPack, $authorizationInput, $authorizationInputValidation, $authorizationGate, $commandCrossCheck, $publishResultOwnerInput, $publishResultImport, $publishResultConvergence)
$combinedText = @($recordsForText | ForEach-Object { if ($null -ne $_) { $_ | ConvertTo-Json -Depth 12 } }) -join "`n"

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ConsistencyItem -Id "managed-package-id-present" -Passed (-not [string]::IsNullOrWhiteSpace($managedPackageIdAuth)) -Severity "action-required" -Detail "Authorization input must carry managedPackageId.")) | Out-Null
$items.Add((New-ConsistencyItem -Id "managed-package-id-consistency" -Passed (-not [string]::IsNullOrWhiteSpace($managedPackageIdAuth) -and $managedPackageIdAuth -eq $managedPackageIdResult) -Severity "action-required" -Detail "Authorized managedPackageId must match imported public publish packageId.")) | Out-Null
$items.Add((New-ConsistencyItem -Id "managed-package-version-present" -Passed (-not [string]::IsNullOrWhiteSpace($managedPackageVersionAuth) -and $managedPackageVersionAuth -notmatch "<owner-fill") -Severity "action-required" -Detail "Authorization input must carry real managedPackageVersion.")) | Out-Null
$items.Add((New-ConsistencyItem -Id "managed-package-version-cross-surface" -Passed (-not [string]::IsNullOrWhiteSpace($managedPackageVersionAuth) -and $managedPackageVersionAuth -eq $managedPackageVersionResult) -Severity "action-required" -Detail "Authorized managedPackageVersion must match public publish owner input packageVersion.")) | Out-Null
$items.Add((New-ConsistencyItem -Id "managed-package-hash-consistency" -Passed ($managedHashAuth -match "^[a-fA-F0-9]{64}$" -and $managedHashAuth -eq $managedHashResult) -Severity "action-required" -Detail "Authorized managed nupkg SHA256 must match imported public publish result.")) | Out-Null
$items.Add((New-ConsistencyItem -Id "runtime-package-id-present" -Passed (-not [string]::IsNullOrWhiteSpace($runtimePackageIdAuth) -and $runtimePackageIdAuth -notmatch "<owner-fill") -Severity "action-required" -Detail "Authorization input must carry real runtimePackageId.")) | Out-Null
$items.Add((New-ConsistencyItem -Id "runtime-package-version-present" -Passed (-not [string]::IsNullOrWhiteSpace($runtimePackageVersionAuth) -and $runtimePackageVersionAuth -notmatch "<owner-fill") -Severity "action-required" -Detail "Authorization input must carry real runtimePackageVersion.")) | Out-Null
$items.Add((New-ConsistencyItem -Id "runtime-package-hash-consistency" -Passed ($runtimeHashAuth -match "^[a-fA-F0-9]{64}$" -and $runtimeHashesResult -contains $runtimeHashAuth) -Severity "action-required" -Detail "Authorized runtime nupkg SHA256 must be present in imported public publish result.")) | Out-Null
foreach ($marker in $forbidden) {
  $items.Add((New-ConsistencyItem -Id ("forbidden-substitute-" + ($marker -replace "[^A-Za-z0-9]+", "-")) -Passed (($forbidden -contains $marker) -or $combinedText.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Forbidden substitute marker must remain documented in the consistency gate contract: $marker")) | Out-Null
}
foreach ($recordName in @("finalExecutionPack", "authorizationInput", "authorizationGate", "commandCrossCheck", "publishResultConvergence")) {
  $recordObject = Get-Variable -Name $recordName -ValueOnly
  $items.Add((New-ConsistencyItem -Id "non-proof-flags-$recordName" -Passed (-not [bool](Get-PropertyOrDefault -Object $recordObject -Name "performsPublish" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $recordObject -Name "canPublishPublicly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $recordObject -Name "canCloseReleaseIssue" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $recordObject -Name "isRuntimeExecutionProof" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $recordObject -Name "isPostPublishProof" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $recordObject -Name "isReleaseCloseProof" -DefaultValue $false)) -Severity "blocker" -Detail "$recordName must remain side-effect free and non-proof.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$gateState = if ($failedBlockers.Count -gt 0) { "invalid-owner-public-publish-execution-consistency" } elseif ($failedActionRequired.Count -gt 0) { "blocked-owner-public-publish-execution-consistency-owner-input-required" } else { "owner-public-publish-execution-consistency-ready" }

$record = [pscustomobject]@{
  recordKind = "owner-public-publish-execution-consistency-gate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  gateState = $gateState
  consistencyItemCount = $items.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  managedPackageIdAuthorized = $managedPackageIdAuth
  managedPackageIdImported = $managedPackageIdResult
  runtimePackageIdAuthorized = $runtimePackageIdAuth
  forbiddenSubstitutes = $forbidden
  consistencyItems = @($items.ToArray())
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  safetyBoundary = "Owner public publish execution consistency gate checks local field alignment only; it does not publish, dispatch workflows, close releases, or promote local artifacts to proof."
}

$jsonPath = Join-Path $OutputRoot "owner-public-publish-execution-consistency-gate.json"
$markdownPath = Join-Path $OutputRoot "owner-public-publish-execution-consistency-gate.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$rows = $items | ForEach-Object { "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.passed)`` | $(ConvertTo-MarkdownCell $_.severity) | $(ConvertTo-MarkdownCell $_.detail) |" }
$markdown = @"
# Owner Public Publish Execution Consistency Gate

| Item | Value |
|---|---|
| gateState | ``$($record.gateState)`` |
| consistencyItemCount | ``$($record.consistencyItemCount)`` |
| failedBlockerCount | ``$($record.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($record.failedActionRequiredCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Consistency Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner public publish execution consistency gate written: $jsonPath"
Write-Host "GateState=$($record.gateState) FailedBlockers=$($record.failedBlockerCount) FailedActionRequired=$($record.failedActionRequiredCount)"
