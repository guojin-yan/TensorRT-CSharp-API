[CmdletBinding()]
param(
  [string]$ImportPath = "artifacts\final-release\post-publish-clean-consumer-proof-result-import.json",
  [string]$CandidatePath = "artifacts\final-release\post-publish-clean-consumer-proof-result-candidate.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict,
  [switch]$RequireExistingFiles,
  [switch]$RequireHashMatch,
  [switch]$FailOnNotProof
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

foreach ($pathName in @("ImportPath", "CandidatePath", "OutputRoot")) {
  if (-not [System.IO.Path]::IsPathRooted((Get-Variable $pathName).Value)) {
    Set-Variable -Name $pathName -Value (Join-Path $RepositoryRoot (Get-Variable $pathName).Value)
  }
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function ConvertTo-FlatStringLines {
  param([AllowNull()][object]$Value)

  foreach ($item in @($Value)) {
    if ($null -eq $item) {
      ""
    }
    elseif ($item -is [string]) {
      $item
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { [string]$child }
    }
    else {
      [string]$item
    }
  }
}
function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $lines = @(ConvertTo-FlatStringLines -Value $InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
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

$defaultPostPublishCleanConsumerProofRequiredFields = @(
  "publicPackageSourceUrl",
  "publicPackageUrl",
  "publicPackageSourceKind",
  "managedPackageId",
  "managedPackageVersion",
  "runtimePackageId",
  "runtimePackageVersion",
  "runtimePackageKey",
  "downloadedManagedPackagePath",
  "downloadedManagedPackageSha256",
  "downloadedRuntimePackagePath",
  "downloadedRuntimePackageSha256",
  "cleanConsumerRoot",
  "consumerProjectPath",
  "restoreCommand",
  "restoreLogPath",
  "restoreLogSha256",
  "buildCommand",
  "buildLogPath",
  "buildLogSha256",
  "runCommand",
  "runLogPath",
  "runLogSha256",
  "smokeStdoutPath",
  "smokeStdoutSha256",
  "smokeStderrPath",
  "smokeStderrSha256",
  "nativeAssetListingPath",
  "nativeAssetListingSha256",
  "dotnetInfoPath",
  "dotnetInfoSha256",
  "exitCode",
  "hostMetadata.os",
  "hostMetadata.arch",
  "hostMetadata.rid",
  "hostMetadata.gpuName",
  "hostMetadata.nvidiaDriver",
  "hostMetadata.cudaRuntimeToolkit",
  "hostMetadata.tensorrt",
  "hostMetadata.cudnn",
  "sourceGitHubActionsRunEvidenceReady",
  "sourceOwnerPublicPublishResultReady",
  "sourcePublicPackageDownloadProofReady",
  "sourceGitHubActionsRunId",
  "sourceGitHubActionsRunUrl",
  "sourceGitHubActionsHeadSha",
  "sourceOwnerPublicPackageUrl",
  "sourceOwnerPublicPackageVersion",
  "sourceOwnerPublicPackageSha256",
  "sourcePublicDownloadManagedPackageDownloadUrl",
  "sourcePublicDownloadRuntimePackageDownloadUrl",
  "ownerReviewer",
  "ownerReviewedAtUtc"
)

$defaultPostPublishCleanConsumerProofRejectedSubstitutes = @(
  "project-reference",
  "local-feed",
  "direct-local-nupkg",
  "repo-internal-consumer",
  "dependency-probe-only",
  "skipped-smoke",
  "blocked-by-cuda-driver",
  "tensorrtexec-report-only",
  "dashboard-only",
  "runbook-only",
  "template-or-candidate-only"
)

$defaultPostPublishCleanConsumerProofSourceReadinessSignals = @(
  "sourceGitHubActionsRunEvidenceReady",
  "sourceGitHubActionsRunId",
  "sourceGitHubActionsRunUrl",
  "sourceGitHubActionsHeadSha",
  "sourceOwnerPublicPublishResultReady",
  "sourceOwnerPublicPackageUrl",
  "sourceOwnerPublicPackageVersion",
  "sourceOwnerPublicPackageSha256",
  "sourcePublicPackageDownloadProofReady",
  "sourcePublicDownloadManagedPackageDownloadUrl",
  "sourcePublicDownloadRuntimePackageDownloadUrl"
)

if (-not (Test-Path -LiteralPath $ImportPath -PathType Leaf) -or -not (Test-Path -LiteralPath $CandidatePath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Import-PostPublishCleanConsumerProofResult.ps1") -RepositoryRoot $RepositoryRoot
}

$import = Get-Content -LiteralPath $ImportPath -Raw -Encoding utf8 | ConvertFrom-Json
$candidate = Get-Content -LiteralPath $CandidatePath -Raw -Encoding utf8 | ConvertFrom-Json
$proofReady = [bool](Get-PropertyOrDefault -Object $import -Name "proofCandidateReady" -DefaultValue $false)
$postPublishCleanConsumerProofRequiredFields = @((Get-PropertyOrDefault -Object $import -Name "postPublishCleanConsumerProofRequiredFields" -DefaultValue $defaultPostPublishCleanConsumerProofRequiredFields))
$postPublishCleanConsumerProofRejectedSubstitutes = @((Get-PropertyOrDefault -Object $import -Name "postPublishCleanConsumerProofRejectedSubstitutes" -DefaultValue $defaultPostPublishCleanConsumerProofRejectedSubstitutes))
$postPublishCleanConsumerProofSourceReadinessSignals = @((Get-PropertyOrDefault -Object $import -Name "postPublishCleanConsumerProofSourceReadinessSignals" -DefaultValue $defaultPostPublishCleanConsumerProofSourceReadinessSignals))
$importFailedBlockerCount = [int](Get-PropertyOrDefault -Object $import -Name "failedBlockerCount" -DefaultValue 0)
$importFailedActionRequiredCount = [int](Get-PropertyOrDefault -Object $import -Name "failedActionRequiredCount" -DefaultValue 0)
$postPublishCleanConsumerProofBlockedRealInputCount = [int](Get-PropertyOrDefault -Object $import -Name "postPublishCleanConsumerProofBlockedRealInputCount" -DefaultValue ($importFailedBlockerCount + $importFailedActionRequiredCount))

$recordKindOk = [string](Get-PropertyOrDefault $import "recordKind" "") -eq "post-publish-clean-consumer-proof-result-import" -and [string](Get-PropertyOrDefault $candidate "recordKind" "") -eq "post-publish-clean-consumer-proof-result-candidate"
$defaultBlockedOk = ([string](Get-PropertyOrDefault $import "importState" "")).Contains("blocked", [StringComparison]::OrdinalIgnoreCase) -or $proofReady
$nonProofFlagsOk = -not [bool](Get-PropertyOrDefault $import "canPromoteRuntimeProof" $true) -and
  -not [bool](Get-PropertyOrDefault $import "isRuntimeExecutionProof" $true) -and
  -not [bool](Get-PropertyOrDefault $import "isPackageConsumerRuntimeProof" $true) -and
  -not [bool](Get-PropertyOrDefault $import "isPostPublishProof" $true) -and
  -not [bool](Get-PropertyOrDefault $candidate "canPromoteRuntimeProof" $true) -and
  -not [bool](Get-PropertyOrDefault $candidate "isRuntimeExecutionProof" $true) -and
  -not [bool](Get-PropertyOrDefault $candidate "isPackageConsumerRuntimeProof" $true) -and
  -not [bool](Get-PropertyOrDefault $candidate "isPostPublishProof" $true)
$noPublishCloseOk = -not [bool](Get-PropertyOrDefault $import "performsPublish" $true) -and -not [bool](Get-PropertyOrDefault $import "canPublishPublicly" $true) -and -not [bool](Get-PropertyOrDefault $import "canCloseReleaseIssue" $true)
$findingsPresentOk = [int](Get-PropertyOrDefault $import "failedActionRequiredCount" 0) -gt 0 -or [int](Get-PropertyOrDefault $import "failedBlockerCount" 0) -gt 0 -or $proofReady
$sourceProofLinkageReady = [bool](Get-PropertyOrDefault $import "sourceProofLinkageReady" $false)
$boundary = [string](Get-PropertyOrDefault $import "boundary" "")
$boundaryOk = $boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed $recordKindOk -Severity "blocker" -Detail "Import and candidate recordKind values must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "default-blocked" -Passed $defaultBlockedOk -Severity "blocker" -Detail "Default post-publish import must remain blocked unless real proof is supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed $nonProofFlagsOk -Severity "blocker" -Detail "Import and candidate must keep runtime/post-publish proof classification flags false even when proofCandidateReady is true.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-close" -Passed $noPublishCloseOk -Severity "blocker" -Detail "Import must never publish or close.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-evidence-findings-present" -Passed $findingsPresentOk -Severity "blocker" -Detail "Blocked imports must report blocker or action-required owner evidence findings.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed $boundaryOk -Severity "blocker" -Detail "Boundary must preserve non-proof classification.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-required-field-contract" -Passed ($postPublishCleanConsumerProofRequiredFields.Count -ge 16) -Severity "blocker" -Detail "Post-publish CleanConsumer proof result must expose a stable required-field contract.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-rejected-substitute-contract" -Passed ($postPublishCleanConsumerProofRejectedSubstitutes.Count -ge 10) -Severity "blocker" -Detail "Post-publish CleanConsumer proof result must expose a stable rejected-substitute contract.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-candidate-ready" -Passed $proofReady -Severity "action-required" -Detail "Real owner evidence must make proofCandidateReady true before the remote proof lane can become ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-proof-linkage-ready" -Passed $sourceProofLinkageReady -Severity "action-required" -Detail "Post-publish proof must link to ready GitHub Actions, Owner public publish, and public package download proof records.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedActionRequired = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0) { "post-publish-clean-consumer-proof-result-validation-ready" } else { "blocked-post-publish-clean-consumer-proof-result-validation-invalid" }

$validation = [pscustomobject]@{
  recordKind = "post-publish-clean-consumer-proof-result-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  validationItemCount = $validationItems.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  proofCandidateReady = $proofReady
  postPublishCleanConsumerProofRequiredFields = @($postPublishCleanConsumerProofRequiredFields)
  postPublishCleanConsumerProofRequiredFieldCount = $postPublishCleanConsumerProofRequiredFields.Count
  postPublishCleanConsumerProofRejectedSubstitutes = @($postPublishCleanConsumerProofRejectedSubstitutes)
  postPublishCleanConsumerProofRejectedSubstituteCount = $postPublishCleanConsumerProofRejectedSubstitutes.Count
  postPublishCleanConsumerProofBlockedRealInputCount = $postPublishCleanConsumerProofBlockedRealInputCount
  postPublishCleanConsumerProofSourceReadinessSignals = @($postPublishCleanConsumerProofSourceReadinessSignals)
  postPublishCleanConsumerProofSourceReadinessSignalCount = $postPublishCleanConsumerProofSourceReadinessSignals.Count
  sourceProofLinkageReady = $sourceProofLinkageReady
  sourceGitHubActionsRunEvidenceReady = [bool](Get-PropertyOrDefault $import "sourceGitHubActionsRunEvidenceReady" $false)
  sourceGitHubActionsRunId = [string](Get-PropertyOrDefault $import "sourceGitHubActionsRunId" "")
  sourceGitHubActionsRunUrl = [string](Get-PropertyOrDefault $import "sourceGitHubActionsRunUrl" "")
  sourceGitHubActionsHeadSha = [string](Get-PropertyOrDefault $import "sourceGitHubActionsHeadSha" "")
  sourceOwnerPublicPublishResultReady = [bool](Get-PropertyOrDefault $import "sourceOwnerPublicPublishResultReady" $false)
  sourceOwnerPublicPackageUrl = [string](Get-PropertyOrDefault $import "sourceOwnerPublicPackageUrl" "")
  sourceOwnerPublicPackageVersion = [string](Get-PropertyOrDefault $import "sourceOwnerPublicPackageVersion" "")
  sourceOwnerPublicPackageSha256 = [string](Get-PropertyOrDefault $import "sourceOwnerPublicPackageSha256" "")
  sourcePublicPackageDownloadProofReady = [bool](Get-PropertyOrDefault $import "sourcePublicPackageDownloadProofReady" $false)
  sourcePublicDownloadManagedPackageUrl = [string](Get-PropertyOrDefault $import "sourcePublicDownloadManagedPackageUrl" "")
  sourcePublicDownloadManagedPackageDownloadUrl = [string](Get-PropertyOrDefault $import "sourcePublicDownloadManagedPackageDownloadUrl" "")
  sourcePublicDownloadRuntimePackageUrl = [string](Get-PropertyOrDefault $import "sourcePublicDownloadRuntimePackageUrl" "")
  sourcePublicDownloadRuntimePackageDownloadUrl = [string](Get-PropertyOrDefault $import "sourcePublicDownloadRuntimePackageDownloadUrl" "")
  sourceGitHubReleaseAssetUrl = [string](Get-PropertyOrDefault $import "sourceGitHubReleaseAssetUrl" "")
  sourceGitHubReleaseAssetSha256 = [string](Get-PropertyOrDefault $import "sourceGitHubReleaseAssetSha256" "")
  publicPackageUrl = [string](Get-PropertyOrDefault $import "publicPackageUrl" "")
  managedPackageVersion = [string](Get-PropertyOrDefault $import "managedPackageVersion" "")
  downloadedManagedPackageSha256 = [string](Get-PropertyOrDefault $import "downloadedManagedPackageSha256" "")
  ownerActionRequired = -not $proofReady
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
  boundary = "Validation checks post-publish CleanConsumer proof import only. proofCandidateReady=true means the owner evidence can satisfy the remote lane, but this validation record is not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-result-validation.json"
$markdownPath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-result-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validationItems) {
  "| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}
Write-Utf8File -LiteralPath $markdownPath -InputObject @(
  "# Post-Publish CleanConsumer Proof Result Validation",
  "",
  "- validationState: ``$validationState``",
  "- failedBlockerCount: ``$($failedBlockers.Count)``",
  "- failedActionRequiredCount: ``$($failedActionRequired.Count)``",
  "- proofCandidateReady: ``$proofReady``",
  "- sourceProofLinkageReady: ``$sourceProofLinkageReady``",
  "- postPublishCleanConsumerProofRequiredFieldCount: ``$($postPublishCleanConsumerProofRequiredFields.Count)``",
  "- postPublishCleanConsumerProofRejectedSubstituteCount: ``$($postPublishCleanConsumerProofRejectedSubstitutes.Count)``",
  "- postPublishCleanConsumerProofBlockedRealInputCount: ``$postPublishCleanConsumerProofBlockedRealInputCount``",
  "- postPublishCleanConsumerProofSourceReadinessSignalCount: ``$($postPublishCleanConsumerProofSourceReadinessSignals.Count)``",
  "",
  "| ID | Passed | Severity | Detail |",
  "|---|---:|---|---|",
  @($rows),
  "",
  "## Boundary",
  "",
  $validation.boundary
)

Write-Host "PostPublishCleanConsumerProofResultValidationState=$validationState FailedBlockers=$($failedBlockers.Count) ProofCandidateReady=$proofReady"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Post-publish CleanConsumer proof result validation failed."
}
if ($FailOnNotProof.IsPresent -and -not $proofReady) {
  throw "Post-publish CleanConsumer proof result is not proof-ready."
}
