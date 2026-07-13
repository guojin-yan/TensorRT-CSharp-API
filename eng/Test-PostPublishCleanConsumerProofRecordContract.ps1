[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-clean-consumer-proof-record-contract.json",
  [string]$OutputRoot = "artifacts\final-release",
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  throw "Post-publish clean consumer proof record contract not found: $InputPath"
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$fields = @((Get-PropertyOrDefault -Object $record -Name "requiredFields" -DefaultValue @()))
$blockedFields = @($fields | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$fieldNames = @($fields | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "") })
$requiredCanonicalFields = @(
  "cleanExternalConsumer.root",
  "cleanExternalConsumer.projectPath",
  "cleanExternalConsumer.restoreLogPath",
  "cleanExternalConsumer.restoreLogSha256",
  "cleanExternalConsumer.buildLogPath",
  "cleanExternalConsumer.buildLogSha256",
  "cleanExternalConsumer.smokeLogPath",
  "cleanExternalConsumer.smokeLogSha256",
  "cleanExternalConsumer.stdoutLogPath",
  "cleanExternalConsumer.stdoutLogSha256",
  "cleanExternalConsumer.stderrLogPath",
  "cleanExternalConsumer.stderrLogSha256",
  "hostMetadata.osDescription",
  "hostMetadata.architecture",
  "hostMetadata.gpuName",
  "hostMetadata.cudaDriverVersion",
  "hostMetadata.cudaRuntimeVersion",
  "hostMetadata.cudnnVersion",
  "hostMetadata.tensorRtVersion",
  "hostMetadata.tensorRtLine",
  "ownerReview.reviewer",
  "ownerReview.reviewedAtUtc",
  "ownerReview.approvalState",
  "forbiddenSubstituteCounts.projectReferenceCount",
  "forbiddenSubstituteCounts.localFeedReferenceCount",
  "forbiddenSubstituteCounts.directNupkgReferenceCount",
  "forbiddenSubstituteCounts.buildOnlyCount",
  "forbiddenSubstituteCounts.dependencyProbeOnlyCount",
  "forbiddenSubstituteCounts.blockedByDriverOnlyCount",
  "sourceProofs.githubActionsRunEvidenceReady",
  "sourceProofs.githubActionsRunId",
  "sourceProofs.githubActionsRunUrl",
  "sourceProofs.githubActionsHeadSha",
  "sourceProofs.ownerPublicPublishResultReady",
  "sourceProofs.publicPackageDownloadProofReady",
  "sourceProofs.publicPackageUrl",
  "sourceProofs.publicPackageVersion",
  "sourceProofs.publicPackageSha256",
  "sourceProofs.managedPackageDownloadUrl",
  "sourceProofs.runtimePackageDownloadUrl",
  "sourceProofs.githubReleaseAssetUrl",
  "sourceProofs.githubReleaseAssetSha256"
)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "post-publish-clean-consumer-proof-record-contract") -Severity "blocker" -Detail "recordKind must be post-publish-clean-consumer-proof-record-contract.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "contractState" -DefaultValue "") -eq "blocked-post-publish-clean-consumer-proof-record-required") -Severity "blocker" -Detail "Contract must stay blocked until owner supplies real clean consumer proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-field-shape" -Passed ($fields.Count -ge 40) -Severity "blocker" -Detail "Contract must expose canonical clean consumer, host metadata, owner review, and forbidden-substitute fields.")) | Out-Null
foreach ($requiredField in $requiredCanonicalFields) {
  $items.Add((New-ValidationItem -Id "required-field-$($requiredField.Replace('.','-'))" -Passed ($fieldNames -contains $requiredField) -Severity "blocker" -Detail "Contract must include required field: $requiredField")) | Out-Null
}
$items.Add((New-ValidationItem -Id "owner-input-required" -Passed ($blockedFields.Count -eq 0) -Severity "action-required" -Detail "Owner must fill real clean consumer proof fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Contract must not publish, approve, promote proof, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-post-publish-clean-consumer-proof-record-contract" } else { "blocked-post-publish-clean-consumer-proof-record-required" }

$validation = [pscustomobject]@{
  recordKind = "post-publish-clean-consumer-proof-record-contract-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $InputPath
  validationState = $validationState
  requiredFieldCount = $fields.Count
  canonicalRequiredFieldCount = $requiredCanonicalFields.Count
  blockedRequiredFieldCount = $blockedFields.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  validationItems = @($items.ToArray())
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  boundary = "Validation checks the clean consumer proof contract shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-record-contract-validation.json"
$markdownPath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-record-contract-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Post-Publish Clean Consumer Proof Record Contract Validation",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| validationState | ``$($validation.validationState)`` |",
  "| requiredFieldCount | ``$($validation.requiredFieldCount)`` |",
  "| blockedRequiredFieldCount | ``$($validation.blockedRequiredFieldCount)`` |",
  "| failedBlockerCount | ``$($validation.failedBlockerCount)`` |",
  "| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |",
  "| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |",
  "",
  "## Boundary",
  "",
  $validation.boundary
)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish clean consumer proof record contract validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) RequiredFields=$($validation.requiredFieldCount) Blocked=$($validation.blockedRequiredFieldCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Post-publish clean consumer proof record contract validation failed with $($failedBlockers.Count) blocker(s)."
}
