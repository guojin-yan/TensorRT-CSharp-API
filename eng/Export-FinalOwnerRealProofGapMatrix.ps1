[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
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

function New-Gap {
  param([string]$Category, [string]$Id, [string]$SourceArtifact, [string]$FieldOrArtifact, [string]$GapType, [string]$RequiredOwnerEvidence)
  [pscustomobject]@{
    category = $Category
    id = $Id
    sourceArtifact = $SourceArtifact
    fieldOrArtifact = $FieldOrArtifact
    gapType = $GapType
    requiredOwnerEvidence = $RequiredOwnerEvidence
    ownerActionRequired = $true
    passed = $false
  }
}

function Add-FindingsAsGaps {
  param(
    [System.Collections.Generic.List[object]]$GapList,
    [AllowNull()][object]$Record,
    [string]$Category,
    [string]$SourceArtifact,
    [int]$Limit
  )

  $findings = @(Convert-ToArray (Get-PropertyOrDefault -Object $Record -Name "findings" -DefaultValue @()))
  $index = 0
  foreach ($finding in $findings) {
    $index++
    $findingId = [string](Get-PropertyOrDefault -Object $finding -Name "id" -DefaultValue "finding-$index")
    $findingCategory = [string](Get-PropertyOrDefault -Object $finding -Name "category" -DefaultValue "owner input")
    $message = [string](Get-PropertyOrDefault -Object $finding -Name "message" -DefaultValue "Owner evidence required.")
    $gapType = if ($findingId.Contains("Sha256", [StringComparison]::OrdinalIgnoreCase) -or $findingId.Contains("sha256", [StringComparison]::OrdinalIgnoreCase) -or $findingCategory.Contains("sha256", [StringComparison]::OrdinalIgnoreCase)) {
      "missing-sha256"
    } elseif ($findingId.Contains("Path", [StringComparison]::OrdinalIgnoreCase) -or $findingCategory.Contains("path", [StringComparison]::OrdinalIgnoreCase)) {
      "missing-file"
    } elseif ($findingId.Contains("host", [StringComparison]::OrdinalIgnoreCase) -or $findingCategory.Contains("host", [StringComparison]::OrdinalIgnoreCase)) {
      "missing-host-metadata"
    } elseif ($findingId.Contains("confirmation", [StringComparison]::OrdinalIgnoreCase) -or $findingCategory.Contains("confirmation", [StringComparison]::OrdinalIgnoreCase)) {
      "missing-owner-confirmation"
    } else {
      "missing-field"
    }
    $GapList.Add((New-Gap $Category "$Category-$findingId" $SourceArtifact $findingId $gapType $message)) | Out-Null
    if ($index -ge $Limit) { break }
  }
}

$externalImport = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-execution-result-import.json"
$externalCandidate = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-execution-result-candidate.json"
$externalValidation = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-execution-result-validation.json"
$postPublishImport = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-import.json"
$postPublishCandidate = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-candidate.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-validation.json"
$closeReadiness = Read-JsonOrNull "artifacts\final-release\final-owner-execution-close-readiness-from-real-input.json"
$closeReadinessValidation = Read-JsonOrNull "artifacts\final-release\final-owner-execution-close-readiness-from-real-input-validation.json"

$gaps = New-Object System.Collections.Generic.List[object]

Add-FindingsAsGaps $gaps $externalImport "external-clean-consumer-runtime" "artifacts/final-release/external-clean-consumer-execution-result-import.json" 80
Add-FindingsAsGaps $gaps $postPublishImport "post-publish-clean-consumer-proof" "artifacts/final-release/post-publish-clean-consumer-proof-result-import.json" 80

if (-not [bool](Get-PropertyOrDefault -Object $externalCandidate -Name "proofCandidateReady" -DefaultValue $false)) {
  $gaps.Add((New-Gap "external-clean-consumer-runtime" "external-clean-consumer-proof-candidate-not-ready" "artifacts/final-release/external-clean-consumer-execution-result-candidate.json" "proofCandidateReady" "missing-owner-confirmation" "Owner must provide real external CleanConsumer restore/build/run/smoke logs, hashes, package source, native asset listing, host metadata, and confirmations.")) | Out-Null
}

if (-not [bool](Get-PropertyOrDefault -Object $postPublishCandidate -Name "proofCandidateReady" -DefaultValue $false)) {
  $gaps.Add((New-Gap "post-publish-clean-consumer-proof" "post-publish-proof-candidate-not-ready" "artifacts/final-release/post-publish-clean-consumer-proof-result-candidate.json" "proofCandidateReady" "missing-owner-confirmation" "Owner must provide public-source package download evidence, hashes, clean consumer logs, host metadata, and confirmations after public publish.")) | Out-Null
}

$closeChecks = @(Convert-ToArray (Get-PropertyOrDefault -Object $closeReadiness -Name "checks" -DefaultValue @()))
foreach ($check in $closeChecks) {
  $id = [string](Get-PropertyOrDefault -Object $check -Name "id" -DefaultValue "")
  $passed = [bool](Get-PropertyOrDefault -Object $check -Name "passed" -DefaultValue $false)
  if ($passed) { continue }
  $category = if ($id -eq "rollback-review-present") {
    "rollback-review"
  } elseif ($id -eq "final-close-decision-present") {
    "final-close-decision"
  } elseif ($id -eq "release-evidence-classification-clean") {
    "release-evidence-refresh"
  } elseif ($id -eq "post-publish-proof-present") {
    "post-publish-clean-consumer-proof"
  } else {
    "external-clean-consumer-runtime"
  }
  $ownerAction = [string](Get-PropertyOrDefault -Object $check -Name "ownerAction" -DefaultValue "Owner action required.")
  $gaps.Add((New-Gap $category "close-readiness-$id" "artifacts/final-release/final-owner-execution-close-readiness-from-real-input.json" $id "missing-owner-confirmation" $ownerAction)) | Out-Null
}

foreach ($required in @(
    @("external-clean-consumer-runtime", "external-import-validation", $externalValidation, "artifacts/final-release/external-clean-consumer-execution-result-validation.json", "proofCandidateReady"),
    @("post-publish-clean-consumer-proof", "post-publish-validation", $postPublishValidation, "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json", "proofCandidateReady"),
    @("final-close-decision", "close-readiness-validation", $closeReadinessValidation, "artifacts/final-release/final-owner-execution-close-readiness-from-real-input-validation.json", "canCloseReleaseIssue")
  )) {
  $category = [string]$required[0]
  $id = [string]$required[1]
  $record = $required[2]
  $artifact = [string]$required[3]
  $field = [string]$required[4]
  if (-not [bool](Get-PropertyOrDefault -Object $record -Name $field -DefaultValue $false)) {
    $gaps.Add((New-Gap $category "$id-$field-false" $artifact $field "missing-owner-confirmation" "Strict validation output is present but not proof-ready; do not use failedBlockerCount=0 as a substitute.")) | Out-Null
  }
}

$gapArray = @($gaps.ToArray())
$categoryCounts = @($gapArray | Group-Object category | ForEach-Object { [pscustomobject]@{ category = $_.Name; count = $_.Count } })
$gapTypeCounts = @($gapArray | Group-Object gapType | ForEach-Object { [pscustomobject]@{ gapType = $_.Name; count = $_.Count } })

$record = [pscustomobject]@{
  recordKind = "final-owner-real-proof-gap-matrix"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  matrixState = "blocked-final-owner-real-proof-gaps-remain"
  gapCount = $gapArray.Count
  categoryCounts = @($categoryCounts)
  gapTypeCounts = @($gapTypeCounts)
  externalCleanConsumerProofReady = [bool](Get-PropertyOrDefault -Object $externalCandidate -Name "proofCandidateReady" -DefaultValue $false)
  postPublishProofReady = [bool](Get-PropertyOrDefault -Object $postPublishCandidate -Name "proofCandidateReady" -DefaultValue $false)
  closeReadinessState = [string](Get-PropertyOrDefault -Object $closeReadiness -Name "readinessState" -DefaultValue "missing-final-owner-execution-close-readiness-from-real-input")
  closeReadinessBlockedCount = [int](Get-PropertyOrDefault -Object $closeReadiness -Name "blockedReadinessCheckCount" -DefaultValue 0)
  gaps = @($gapArray)
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
  boundary = "Final Owner real proof gap matrix is a blocked owner-action gap view only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-real-proof-gap-matrix.json"
$markdownPath = Join-Path $OutputRoot "final-owner-real-proof-gap-matrix.md"
$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$gapRows = foreach ($gap in ($gapArray | Select-Object -First 120)) {
  "| ``$($gap.category)`` | ``$($gap.id)`` | ``$($gap.gapType)`` | $(ConvertTo-MarkdownCell $gap.requiredOwnerEvidence) |"
}
$categoryRows = foreach ($count in $categoryCounts) {
  "| ``$($count.category)`` | $($count.count) |"
}

$markdown = @"
# Final Owner Real Proof Gap Matrix

| Field | Value |
|---|---|
| matrixState | ``$($record.matrixState)`` |
| gapCount | ``$($record.gapCount)`` |
| externalCleanConsumerProofReady | ``$($record.externalCleanConsumerProofReady)`` |
| postPublishProofReady | ``$($record.postPublishProofReady)`` |
| closeReadinessState | ``$($record.closeReadinessState)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Category Counts

| Category | Count |
|---|---:|
$($categoryRows -join "`r`n")

## Gaps

| Category | ID | Gap Type | Required Owner Evidence |
|---|---|---|---|
$($gapRows -join "`r`n")

## Boundary

$($record.boundary)
"@

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
