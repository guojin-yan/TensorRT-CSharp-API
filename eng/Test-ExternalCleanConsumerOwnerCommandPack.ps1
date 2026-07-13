[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\external-clean-consumer-owner-command-pack.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

foreach ($pathName in @("InputPath", "OutputRoot")) {
  if (-not [System.IO.Path]::IsPathRooted((Get-Variable $pathName).Value)) {
    Set-Variable -Name $pathName -Value (Join-Path $RepositoryRoot (Get-Variable $pathName).Value)
  }
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-ExternalCleanConsumerOwnerCommandPack.ps1") -RepositoryRoot $RepositoryRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$recordText = $record | ConvertTo-Json -Depth 18
$steps = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "steps" -DefaultValue @()))
$stepIds = @($steps | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$forbidden = @((Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())) | ForEach-Object { [string]$_ })

$recordKindOk = [string](Get-PropertyOrDefault $record "recordKind" "") -eq "external-clean-consumer-owner-command-pack"
$blockedNonProofOk = [string](Get-PropertyOrDefault $record "packState" "") -eq "blocked-external-clean-consumer-owner-command-pack-required" -and -not [bool](Get-PropertyOrDefault $record "passed" $true) -and [bool](Get-PropertyOrDefault $record "ownerActionRequired" $false)
$nonProofFlagsOk = -not [bool](Get-PropertyOrDefault $record "performsPublish" $true) -and -not [bool](Get-PropertyOrDefault $record "performsRuntimeExecution" $true) -and -not [bool](Get-PropertyOrDefault $record "canPromoteRuntimeProof" $true) -and -not [bool](Get-PropertyOrDefault $record "canPublishPublicly" $true) -and -not [bool](Get-PropertyOrDefault $record "canCloseReleaseIssue" $true) -and -not [bool](Get-PropertyOrDefault $record "isRuntimeExecutionProof" $true) -and -not [bool](Get-PropertyOrDefault $record "isPostPublishProof" $true)
$requiredStepsOk = @(@("create-external-root", "create-clean-consumer-project", "add-managed-package", "add-runtime-package", "restore-with-log", "build-with-log", "run-smoke-with-stdout-stderr", "collect-native-assets", "compute-sha256", "capture-host-metadata", "run-strict-import") | Where-Object { $stepIds -notcontains $_ }).Count -eq 0 -and $steps.Count -ge 10
$forbiddenSubstitutesOk = @(@("local feed", "ProjectReference", "direct .nupkg", "direct nupkg", "pre-publish smoke reused as post-publish proof") | Where-Object { $forbidden -notcontains $_ }).Count -eq 0
$commandContentOk = $recordText.Contains("dotnet new", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("dotnet restore", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("dotnet build", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("dotnet run", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("Get-FileHash", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("native-assets", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("host-metadata", [StringComparison]::OrdinalIgnoreCase)
$boundary = [string](Get-PropertyOrDefault $record "boundary" "")
$boundaryOk = $boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed $recordKindOk -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-non-proof" -Passed $blockedNonProofOk -Severity "blocker" -Detail "Command pack must remain blocked, failed, and owner-action-required.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed $nonProofFlagsOk -Severity "blocker" -Detail "Command pack must not publish, execute, close, or claim proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-steps" -Passed $requiredStepsOk -Severity "blocker" -Detail "Command pack must include external project, restore, build, smoke, logs, native assets, hash, host metadata, and strict import steps.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes" -Passed $forbiddenSubstitutesOk -Severity "blocker" -Detail "Command pack must list forbidden substitute markers.")) | Out-Null
$items.Add((New-ValidationItem -Id "command-content" -Passed $commandContentOk -Severity "blocker" -Detail "Command content must cover restore/build/run/hash/native assets/host metadata.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed $boundaryOk -Severity "blocker" -Detail "Boundary must preserve non-proof classification.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "external-clean-consumer-owner-command-pack-ready-non-proof" } else { "blocked-external-clean-consumer-owner-command-pack-invalid" }

$validation = [pscustomobject]@{
  recordKind = "external-clean-consumer-owner-command-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  validationItemCount = $validationItems.Count
  failedBlockerCount = $failedBlockers.Count
  stepCount = $steps.Count
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($validationItems)
  boundary = "Validation checks owner command pack shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "external-clean-consumer-owner-command-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "external-clean-consumer-owner-command-pack-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validationItems) {
  "| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# External CleanConsumer Owner Command Pack Validation

| Field | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| stepCount | ``$($validation.stepCount)`` |
| ownerActionRequired | ``$($validation.ownerActionRequired)`` |
| passed | ``$($validation.passed)`` |

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "OwnerCommandPackValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "External CleanConsumer owner command pack validation failed."
}
