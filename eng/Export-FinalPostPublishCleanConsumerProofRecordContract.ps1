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

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
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

function New-CleanConsumerContractItem {
  param([string]$Id, [string]$Description)

  [pscustomobject]@{
    id = $Id
    description = $Description
    proofState = "blocked-post-publish-clean-consumer-proof-required"
    packageIdentity = [pscustomobject]@{
      packageId = "<owner-fill-public-package-id>"
      packageVersion = "<owner-fill-public-package-version>"
      packageSource = "<owner-fill-public-package-source>"
      publicPackageUrl = "<owner-fill-public-package-url>"
      nupkgSha256 = "<owner-fill-public-nupkg-sha256>"
      managedPackageDownloadUrl = "<owner-fill-public-managed-package-download-url>"
      runtimePackageUrl = "<owner-fill-public-runtime-package-url>"
      runtimePackageDownloadUrl = "<owner-fill-public-runtime-package-download-url>"
    }
    sourceProofLinkage = [pscustomobject]@{
      githubActionsRunEvidenceReady = $false
      githubActionsRunId = "<owner-fill-github-actions-run-id>"
      githubActionsRunUrl = "<owner-fill-github-actions-run-url>"
      githubActionsHeadSha = "<owner-fill-github-actions-head-sha>"
      ownerPublicPublishResultReady = $false
      publicPackageDownloadProofReady = $false
      sourceOwnerPublicPackageUrl = "<owner-fill-source-owner-public-package-url>"
      sourceOwnerPublicPackageVersion = "<owner-fill-source-owner-public-package-version>"
      sourceOwnerPublicPackageSha256 = "<owner-fill-source-owner-public-package-sha256>"
      githubReleaseUrl = "<owner-fill-github-release-url>"
      githubReleaseAssetUrl = "<owner-fill-github-release-asset-url>"
      githubReleaseAssetSha256 = "<owner-fill-github-release-asset-sha256>"
    }
    cleanConsumerProjectRoot = "<owner-fill-clean-consumer-project-root>"
    cleanConsumerProjectSha256Manifest = "<owner-fill-clean-consumer-project-sha256-manifest>"
    cleanConsumerRestoreLogPath = "<owner-fill-clean-consumer-restore-log-path>"
    cleanConsumerRestoreLogSha256 = "<owner-fill-clean-consumer-restore-log-sha256>"
    cleanConsumerBuildLogPath = "<owner-fill-clean-consumer-build-log-path>"
    cleanConsumerBuildLogSha256 = "<owner-fill-clean-consumer-build-log-sha256>"
    cleanConsumerRunLogPath = "<owner-fill-clean-consumer-run-log-path>"
    cleanConsumerRunLogSha256 = "<owner-fill-clean-consumer-run-log-sha256>"
    cleanConsumerMergedTranscriptPath = "<owner-fill-clean-consumer-merged-transcript-path>"
    cleanConsumerMergedTranscriptSha256 = "<owner-fill-clean-consumer-merged-transcript-sha256>"
    cleanConsumerValidatorOutputPath = "<owner-fill-clean-consumer-validator-output-path>"
    cleanConsumerValidatorOutputSha256 = "<owner-fill-clean-consumer-validator-output-sha256>"
    executedCommand = "<owner-fill-clean-consumer-executed-command>"
    exitCode = "<owner-fill-clean-consumer-exit-code>"
    executedAtUtc = "<owner-fill-clean-consumer-executed-at-utc>"
    hostIdentity = [pscustomobject]@{
      machineName = "<owner-fill-clean-consumer-machine-name>"
      os = "<owner-fill-clean-consumer-os>"
      architecture = "<owner-fill-clean-consumer-architecture>"
      cudaVersion = "<owner-fill-clean-consumer-cuda-version-or-unavailable-reason>"
      tensorrtVersion = "<owner-fill-clean-consumer-tensorrt-version-or-unavailable-reason>"
      driverVersion = "<owner-fill-clean-consumer-driver-version-or-unavailable-reason>"
    }
    noProjectReferenceConfirmation = $false
    noLocalFeedConfirmation = $false
    noDirectNupkgConfirmation = $false
    noSourceCheckoutReferenceConfirmation = $false
    ownerReviewer = "<owner-fill-clean-consumer-owner-reviewer>"
    ownerReviewTimestampUtc = "<owner-fill-clean-consumer-owner-review-timestamp-utc>"
    forbiddenSubstituteMarkers = @(
      "local feed",
      "ProjectReference",
      "direct .nupkg",
      "source checkout reference",
      "template",
      "draft",
      "dry-run",
      "dashboard",
      "candidate",
      "build-only",
      "dependency-probe-only",
      "blocked-by-cuda-driver"
    )
    readyForPreflight = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Final post-publish clean consumer proof record contract item only. It is not runtime proof, not post-publish proof, not release close approval, not publish approval, and not package push."
  }
}

$ownerExternalCandidate = Read-JsonOrNull "artifacts\final-release\final-owner-execution-external-result-candidate.json"
$ownerExternalPreflight = Read-JsonOrNull "artifacts\final-release\final-owner-execution-external-result-input-preflight.json"

$contractItems = @(
  New-CleanConsumerContractItem -Id "public-package-clean-consumer-restore" -Description "Owner supplies public package source and clean consumer restore evidence."
  New-CleanConsumerContractItem -Id "public-package-clean-consumer-build" -Description "Owner supplies clean consumer build evidence from a repository-external project."
  New-CleanConsumerContractItem -Id "public-package-clean-consumer-run" -Description "Owner supplies clean consumer runtime smoke evidence from the published package."
)

$record = [ordered]@{
  recordKind = "final-post-publish-clean-consumer-proof-record-contract"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  contractState = "blocked-post-publish-clean-consumer-proof-required"
  sourceOwnerExternalCandidateState = [string](Get-PropertyOrDefault -Object $ownerExternalCandidate -Name "candidateState" -DefaultValue "missing-final-owner-execution-external-result-candidate")
  sourceOwnerExternalPreflightState = [string](Get-PropertyOrDefault -Object $ownerExternalPreflight -Name "preflightState" -DefaultValue "missing-final-owner-execution-external-result-input-preflight")
  contractItemCount = $contractItems.Count
  blockedContractItemCount = $contractItems.Count
  readyForPreflightCount = 0
  requiredEvidenceFieldCount = 27 * $contractItems.Count
  contractItems = @($contractItems)
  requiredExternalEvidenceFields = @(
    "packageIdentity",
    "publicPackageUrl",
    "packageSource",
    "packageVersion",
    "nupkgSha256",
    "managedPackageDownloadUrl",
    "runtimePackageUrl",
    "runtimePackageDownloadUrl",
    "sourceProofLinkage",
    "githubActionsRunEvidenceReady",
    "githubActionsRunId",
    "githubActionsRunUrl",
    "githubActionsHeadSha",
    "ownerPublicPublishResultReady",
    "publicPackageDownloadProofReady",
    "sourceOwnerPublicPackageUrl",
    "sourceOwnerPublicPackageVersion",
    "sourceOwnerPublicPackageSha256",
    "githubReleaseUrl",
    "githubReleaseAssetUrl",
    "githubReleaseAssetSha256",
    "cleanConsumerProjectRoot",
    "cleanConsumerProjectSha256Manifest",
    "cleanConsumerRestoreLogPath",
    "cleanConsumerRestoreLogSha256",
    "cleanConsumerBuildLogPath",
    "cleanConsumerBuildLogSha256",
    "cleanConsumerRunLogPath",
    "cleanConsumerRunLogSha256",
    "cleanConsumerMergedTranscriptPath",
    "cleanConsumerMergedTranscriptSha256",
    "cleanConsumerValidatorOutputPath",
    "cleanConsumerValidatorOutputSha256",
    "executedCommand",
    "exitCode",
    "executedAtUtc",
    "hostIdentity",
    "noProjectReferenceConfirmation",
    "noLocalFeedConfirmation",
    "noDirectNupkgConfirmation",
    "noSourceCheckoutReferenceConfirmation",
    "ownerReviewer",
    "ownerReviewTimestampUtc"
  )
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    "artifacts/final-release/final-owner-execution-external-result-candidate.json",
    "artifacts/final-release/final-owner-execution-external-result-candidate-validation.json",
    "artifacts/final-release/final-owner-execution-external-result-input-preflight.json",
    "artifacts/final-release/github-actions-run-evidence-import-validation.json",
    "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json",
    "artifacts/final-release/public-package-download-proof-candidate-validation.json"
  )
  boundary = "Final post-publish clean consumer proof record contract only. It waits for real public package clean consumer evidence; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $OutputRoot "final-post-publish-clean-consumer-proof-record-contract.json"
$markdownPath = Join-Path $OutputRoot "final-post-publish-clean-consumer-proof-record-contract.md"
$record | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $contractItems) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$(ConvertTo-MarkdownCell $item.proofState)`` | ``$($item.readyForPreflight)`` |"
}

$markdown = @(
  "# Final Post-Publish Clean Consumer Proof Record Contract",
  "",
  "- contractState: ``$($record.contractState)``",
  "- contractItemCount: ``$($record.contractItemCount)``",
  "- blockedContractItemCount: ``$($record.blockedContractItemCount)``",
  "- readyForPreflightCount: ``0``",
  "- requiredEvidenceFieldCount: ``$($record.requiredEvidenceFieldCount)``",
  "- boundary: $($record.boundary)",
  "",
  "| Contract Item | State | Ready For Preflight |",
  "|---|---|---|",
  @($rows)
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
