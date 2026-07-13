[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\external-clean-consumer-execution-workspace-contract.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-ExternalCleanConsumerExecutionWorkspaceContract.ps1") -RepositoryRoot $RepositoryRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$recordText = $record | ConvertTo-Json -Depth 18
$rules = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "requiredWorkspaceRules" -DefaultValue @()))
$evidence = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "requiredEvidence" -DefaultValue @()))
$forbidden = @((Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())) | ForEach-Object { [string]$_ })
$shaFields = @((Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "requiredSha256Inputs" -DefaultValue @())) | ForEach-Object { [string]$_ })
$hostFields = @((Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "requiredHostMetadata" -DefaultValue @())) | ForEach-Object { [string]$_ })

$recordKindOk = [string](Get-PropertyOrDefault $record "recordKind" "") -eq "external-clean-consumer-execution-workspace-contract"
$blockedNonProofOk = [string](Get-PropertyOrDefault $record "contractState" "") -eq "blocked-external-clean-consumer-workspace-contract-required" -and -not [bool](Get-PropertyOrDefault $record "passed" $true) -and [bool](Get-PropertyOrDefault $record "ownerActionRequired" $false)
$nonProofFlagsOk = -not [bool](Get-PropertyOrDefault $record "performsPublish" $true) -and -not [bool](Get-PropertyOrDefault $record "performsRuntimeExecution" $true) -and -not [bool](Get-PropertyOrDefault $record "canPromoteRuntimeProof" $true) -and -not [bool](Get-PropertyOrDefault $record "canPublishPublicly" $true) -and -not [bool](Get-PropertyOrDefault $record "canCloseReleaseIssue" $true) -and -not [bool](Get-PropertyOrDefault $record "isRuntimeExecutionProof" $true) -and -not [bool](Get-PropertyOrDefault $record "isPostPublishProof" $true)
$repositoryExternalRuleOk = $recordText.Contains("repository-external", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("outside this repository", [StringComparison]::OrdinalIgnoreCase)
$forbiddenSubstitutesOk = @(@("local feed", "ProjectReference", "direct .nupkg", "direct nupkg", "pre-publish smoke reused as post-publish proof", "build-only", "dependency-probe", "dashboard", "runbook", "template", "candidate") | Where-Object { $forbidden -notcontains $_ }).Count -eq 0
$requiredEvidenceFieldsOk = @(@("repositoryExternalWorkspaceRoot", "cleanConsumerCsprojPath", "packageSourceUrl", "restoreLog", "buildLog", "runLog", "smokeStdout", "smokeStderr", "nativeAssetListing", "managedPackage", "runtimePackage", "hostMetadata") | Where-Object { $recordText.IndexOf($_, [StringComparison]::OrdinalIgnoreCase) -lt 0 }).Count -eq 0
$sha256FieldsOk = @(@("restoreLogSha256", "buildLogSha256", "runLogSha256", "smokeStdoutSha256", "smokeStderrSha256", "nativeAssetListingSha256", "managedPackageSha256", "runtimePackageSha256") | Where-Object { $shaFields -notcontains $_ }).Count -eq 0
$hostMetadataOk = @(@("os", "rid", "gpuName", "nvidiaDriver", "cudaRuntimeToolkit", "tensorrt", "cudnn") | Where-Object { $hostFields -notcontains $_ }).Count -eq 0
$boundary = [string](Get-PropertyOrDefault $record "boundary" "")
$boundaryOk = $boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed $recordKindOk -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-non-proof" -Passed $blockedNonProofOk -Severity "blocker" -Detail "Contract must remain blocked, failed, and owner-action-required.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed $nonProofFlagsOk -Severity "blocker" -Detail "Contract must not publish, execute, close, or claim proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "repository-external-rule" -Passed $repositoryExternalRuleOk -Severity "blocker" -Detail "Contract must require a repository-external workspace.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes" -Passed $forbiddenSubstitutesOk -Severity "blocker" -Detail "Contract must list all forbidden substitutes.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-evidence-fields" -Passed $requiredEvidenceFieldsOk -Severity "blocker" -Detail "Contract must require workspace, package source, logs, stdout/stderr, native assets, packages, and host metadata.")) | Out-Null
$items.Add((New-ValidationItem -Id "sha256-fields" -Passed $sha256FieldsOk -Severity "blocker" -Detail "Contract must require SHA256 fields for all proof-carrying files.")) | Out-Null
$items.Add((New-ValidationItem -Id "host-metadata" -Passed $hostMetadataOk -Severity "blocker" -Detail "Contract must require host metadata fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed $boundaryOk -Severity "blocker" -Detail "Boundary must preserve non-proof classification.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "external-clean-consumer-execution-workspace-contract-ready-non-proof" } else { "blocked-external-clean-consumer-execution-workspace-contract-invalid" }

$validation = [pscustomobject]@{
  recordKind = "external-clean-consumer-execution-workspace-contract-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  validationItemCount = $validationItems.Count
  failedBlockerCount = $failedBlockers.Count
  requiredWorkspaceRuleCount = $rules.Count
  requiredEvidenceCount = $evidence.Count
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
  boundary = "Validation checks contract shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "external-clean-consumer-execution-workspace-contract-validation.json"
$markdownPath = Join-Path $OutputRoot "external-clean-consumer-execution-workspace-contract-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validationItems) {
  "| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# External CleanConsumer Execution Workspace Contract Validation

| Field | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| ownerActionRequired | ``$($validation.ownerActionRequired)`` |
| passed | ``$($validation.passed)`` |

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "WorkspaceContractValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "External CleanConsumer execution workspace contract validation failed."
}
