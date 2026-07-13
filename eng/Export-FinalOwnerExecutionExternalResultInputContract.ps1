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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)
  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ExternalResultContractItem {
  param([object]$DraftItem)

  $sourceDraftItemId = [string](Get-PropertyOrDefault -Object $DraftItem -Name "sourceSkeletonItemId" -DefaultValue "")
  $sourceExecutionStepId = [string](Get-PropertyOrDefault -Object $DraftItem -Name "sourceExecutionStepId" -DefaultValue "")
  $laneId = [string](Get-PropertyOrDefault -Object $DraftItem -Name "laneId" -DefaultValue "")
  $confirmations = @((Get-PropertyOrDefault -Object $DraftItem -Name "ownerProvidedNonSubstituteConfirmations" -DefaultValue @()))
  $forbiddenMarkers = Convert-ToStringArray (Get-PropertyOrDefault -Object $DraftItem -Name "cannotUseMarkers" -DefaultValue @())
  $validatorCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $DraftItem -Name "expectedValidatorCommands" -DefaultValue @())
  $expectedArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $DraftItem -Name "expectedResultArtifacts" -DefaultValue @())

  [pscustomobject]@{
    sourceDraftItemId = $sourceDraftItemId
    sourceExecutionStepId = $sourceExecutionStepId
    laneId = $laneId
    ownerInputState = "blocked-owner-real-external-result-required"
    realExecutionRoot = "<owner-fill-real-execution-root>"
    stdoutPath = "<owner-fill-real-stdout-path>"
    stderrPath = "<owner-fill-real-stderr-path>"
    mergedTranscriptPath = "<owner-fill-real-merged-transcript-path>"
    validatorOutputPath = "<owner-fill-real-validator-output-path>"
    stdoutSha256 = "<owner-fill-real-stdout-sha256>"
    stderrSha256 = "<owner-fill-real-stderr-sha256>"
    mergedTranscriptSha256 = "<owner-fill-real-merged-transcript-sha256>"
    validatorOutputSha256 = "<owner-fill-real-validator-output-sha256>"
    exitCode = "<owner-fill-real-exit-code>"
    executedCommand = "<owner-fill-real-executed-command>"
    executedAtUtc = "<owner-fill-real-executed-at-utc>"
    hostIdentity = [pscustomobject]@{
      machineName = "<owner-fill-real-machine-name>"
      os = "<owner-fill-real-os>"
      architecture = "<owner-fill-real-architecture>"
      cudaVersion = "<owner-fill-real-cuda-version-or-unavailable-reason>"
      tensorrtVersion = "<owner-fill-real-tensorrt-version-or-unavailable-reason>"
      driverVersion = "<owner-fill-real-driver-version-or-unavailable-reason>"
    }
    packageIdentity = [pscustomobject]@{
      packageId = "<owner-fill-real-package-id>"
      packageVersion = "<owner-fill-real-package-version>"
      packageSource = "<owner-fill-real-package-source>"
      nupkgPath = "<owner-fill-real-nupkg-path>"
      nupkgSha256 = "<owner-fill-real-nupkg-sha256>"
      publishedPackageUrl = "<owner-fill-real-published-package-url-or-not-yet-published>"
    }
    ownerReviewer = "<owner-fill-real-owner-reviewer>"
    ownerReviewTimestampUtc = "<owner-fill-real-owner-review-timestamp-utc>"
    ownerProvidedNonSubstituteConfirmations = @($confirmations)
    forbiddenSubstituteMarkers = @($forbiddenMarkers)
    expectedValidatorCommands = @($validatorCommands)
    expectedResultArtifacts = @($expectedArtifacts)
    readyForImport = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Final Owner external result input contract only. Owner must replace every placeholder with real external execution evidence; this contract is not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and not a release-ready signal."
  }
}

$draft = Read-JsonOrNull "artifacts\final-release\final-owner-execution-owner-input-draft.json"
$draftValidation = Read-JsonOrNull "artifacts\final-release\final-owner-execution-owner-input-draft-validation.json"
$draftItems = @()
if ($null -ne $draft) {
  $draftItems = @((Get-PropertyOrDefault -Object $draft -Name "draftItems" -DefaultValue @()))
}

$contractItems = @($draftItems | ForEach-Object { New-ExternalResultContractItem -DraftItem $_ } | Sort-Object sourceExecutionStepId)
$contractItemCount = $contractItems.Count
$placeholderFieldCount = $contractItemCount * 24
$blockedContractItemCount = $contractItemCount

$record = [ordered]@{
  recordKind = "final-owner-execution-external-result-input-contract"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  contractState = "blocked-owner-real-external-result-required"
  sourceDraftState = [string](Get-PropertyOrDefault -Object $draft -Name "draftState" -DefaultValue "missing-final-owner-execution-owner-input-draft")
  sourceDraftValidationState = [string](Get-PropertyOrDefault -Object $draftValidation -Name "validationState" -DefaultValue "missing-final-owner-execution-owner-input-draft-validation")
  contractItemCount = $contractItemCount
  blockedContractItemCount = $blockedContractItemCount
  readyContractItemCount = 0
  readyForImportCount = 0
  placeholderFieldCount = $placeholderFieldCount
  contractItems = @($contractItems)
  forbiddenSubstituteMarkers = @(
    "local feed",
    "ProjectReference",
    "direct .nupkg",
    "template",
    "draft",
    "dry-run",
    "dashboard",
    "candidate",
    "build-only",
    "dependency-probe-only",
    "blocked-by-cuda-driver"
  )
  requiredExternalEvidenceFields = @(
    "realExecutionRoot",
    "stdoutPath",
    "stderrPath",
    "mergedTranscriptPath",
    "validatorOutputPath",
    "stdoutSha256",
    "stderrSha256",
    "mergedTranscriptSha256",
    "validatorOutputSha256",
    "exitCode",
    "executedCommand",
    "executedAtUtc",
    "hostIdentity",
    "packageIdentity",
    "ownerReviewer",
    "ownerReviewTimestampUtc",
    "ownerProvidedNonSubstituteConfirmations"
  )
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    "artifacts/final-release/final-owner-execution-owner-input-draft.json",
    "artifacts/final-release/final-owner-execution-owner-input-draft.md",
    "artifacts/final-release/final-owner-execution-owner-input-draft-validation.json",
    "artifacts/final-release/final-owner-execution-owner-input-draft-validation.md"
  )
  boundary = "Final Owner external result input contract only. It defines real Owner evidence fields and placeholders; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-external-result-input-contract.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-external-result-input-contract.md"
$record | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $contractItems) {
  "| ``$(ConvertTo-MarkdownCell $item.sourceExecutionStepId)`` | ``$(ConvertTo-MarkdownCell $item.laneId)`` | ``$($item.ownerInputState)`` | ``$($item.readyForImport)`` |"
}

$markdown = @(
  "# Final Owner Execution External Result Input Contract",
  "",
  "- contractState: ``$($record.contractState)``",
  "- contractItemCount: ``$contractItemCount``",
  "- blockedContractItemCount: ``$blockedContractItemCount``",
  "- readyForImportCount: ``0``",
  "- placeholderFieldCount: ``$placeholderFieldCount``",
  "- boundary: $($record.boundary)",
  "",
  "> Owner must replace every placeholder with real external execution evidence before preflight can promote any lane to candidate.",
  "",
  "| Execution Step | Lane | State | Ready For Import |",
  "|---|---|---|---|",
  @($rows)
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
