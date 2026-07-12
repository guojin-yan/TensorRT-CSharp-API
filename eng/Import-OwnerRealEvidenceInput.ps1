[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/final-release/owner-input/owner-real-evidence-input.json",
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

. (Join-Path $PSScriptRoot "OwnerRealEvidenceInput.Common.ps1")

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$inputFullPath = Resolve-OwnerRepositoryPath -RepositoryRoot $RepositoryRoot -Path $InputPath
$items = @()
$laneResults = @()
$input = $null
$inputExists = Test-Path -LiteralPath $inputFullPath -PathType Leaf
$recordKind = "missing"

if ($inputExists) {
  try {
    $input = Get-Content -LiteralPath $inputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
    $recordKind = [string]$input.recordKind
  }
  catch {
    $items += New-OwnerValidationItem -Id "input-json-parse" -Passed $false -Severity "blocker" -Detail "Owner input JSON could not be parsed: $($_.Exception.Message)"
  }
}
else {
  $items += New-OwnerValidationItem -Id "owner-input-file-present" -Passed $false -Severity "action-required" -Detail "Owner input file is required at $InputPath."
}

if ($null -ne $input) {
  $lanes = @($input.lanes)
  $laneIds = @($lanes | ForEach-Object { [string]$_.laneId })
  $missingLaneIds = @($script:OwnerRealEvidenceRequiredLaneIds | Where-Object { $laneIds -notcontains $_ })
  $duplicateLaneIds = @($laneIds | Group-Object | Where-Object { $_.Count -gt 1 } | ForEach-Object { $_.Name })

  $items += New-OwnerValidationItem -Id "record-kind" -Passed ([string]$input.recordKind -eq "owner-real-evidence-input") -Severity "blocker" -Detail "Input recordKind must be owner-real-evidence-input."
  $items += New-OwnerValidationItem -Id "all-six-lanes-present" -Passed ($missingLaneIds.Count -eq 0 -and $lanes.Count -eq 6) -Severity "blocker" -Detail "Input must contain exactly the six final action-required lanes."
  $items += New-OwnerValidationItem -Id "no-duplicate-lanes" -Passed ($duplicateLaneIds.Count -eq 0) -Severity "blocker" -Detail "Input lane IDs must be unique."
  $items += New-OwnerValidationItem -Id "flags-remain-false" -Passed (-not [bool]$input.performsPublish -and -not [bool]$input.canPublishPublicly -and -not [bool]$input.canCloseReleaseIssue -and -not [bool]$input.canPromotePackageConsumerRuntime -and -not [bool]$input.canPromoteRuntimeProof) -Severity "blocker" -Detail "Owner input import must not publish, close, or promote proof flags."

  foreach ($lane in $lanes) {
    $laneId = [string]$lane.laneId
    $artifactPath = [string]$lane.artifactPath
    $logPath = [string]$lane.logPath
    $recordPath = [string]$lane.recordPath
    $artifactFullPath = if (Test-OwnerStringFilled $artifactPath) { Resolve-OwnerRepositoryPath -RepositoryRoot $RepositoryRoot -Path $artifactPath } else { "" }
    $logFullPath = if (Test-OwnerStringFilled $logPath) { Resolve-OwnerRepositoryPath -RepositoryRoot $RepositoryRoot -Path $logPath } else { "" }
    $recordFullPath = if (Test-OwnerStringFilled $recordPath) { Resolve-OwnerRepositoryPath -RepositoryRoot $RepositoryRoot -Path $recordPath } else { "" }

    $artifactExists = -not [string]::IsNullOrWhiteSpace($artifactFullPath) -and (Test-Path -LiteralPath $artifactFullPath -PathType Leaf)
    $logExists = -not [string]::IsNullOrWhiteSpace($logFullPath) -and (Test-Path -LiteralPath $logFullPath -PathType Leaf)
    $recordExists = -not [string]::IsNullOrWhiteSpace($recordFullPath) -and (Test-Path -LiteralPath $recordFullPath -PathType Leaf)

    $artifactActualSha = if ($artifactExists) { Get-OwnerFileSha256 -Path $artifactFullPath } else { "" }
    $logActualSha = if ($logExists) { Get-OwnerFileSha256 -Path $logFullPath } else { "" }
    $recordActualSha = if ($recordExists) { Get-OwnerFileSha256 -Path $recordFullPath } else { "" }

    $artifactShaMatches = $artifactExists -and ($artifactActualSha -eq ([string]$lane.artifactSha256).ToLowerInvariant())
    $logShaMatches = $logExists -and ($logActualSha -eq ([string]$lane.logSha256).ToLowerInvariant())
    $recordShaMatches = $recordExists -and ($recordActualSha -eq ([string]$lane.recordSha256).ToLowerInvariant())

    $logText = if ($logExists) { Get-Content -LiteralPath $logFullPath -Raw -Encoding utf8 } else { "" }
    $recordText = if ($recordExists) { Get-Content -LiteralPath $recordFullPath -Raw -Encoding utf8 } else { "" }
    $laneText = $lane | ConvertTo-Json -Depth 10
    $hasForbiddenText = (Test-OwnerTextHasForbiddenSubstitute -Text $logText) -or (Test-OwnerTextHasForbiddenSubstitute -Text $recordText) -or (Test-OwnerTextHasForbiddenSubstitute -Text $laneText)

    $hostMetadata = $lane.hostMetadata
    $hostMetadataFilled = $null -ne $hostMetadata -and
      (Test-OwnerStringFilled ([string]$hostMetadata.hostName)) -and
      (Test-OwnerStringFilled ([string]$hostMetadata.os)) -and
      (Test-OwnerStringFilled ([string]$hostMetadata.cudaVersion)) -and
      (Test-OwnerStringFilled ([string]$hostMetadata.tensorRtVersion)) -and
      (Test-OwnerStringFilled ([string]$hostMetadata.gpuName))

    $ownerReview = $lane.ownerReview
    $ownerReviewAccepted = $null -ne $ownerReview -and
      (Test-OwnerStringFilled ([string]$ownerReview.reviewer)) -and
      ([string]$ownerReview.decision -eq "accepted-real-proof") -and
      ([string]$lane.decision -eq "accepted-real-proof")

    $publicRequirementSatisfied = $true
    if ($laneId -in @("package-consumer-runtime-owner-proof-required", "post-publish-verification-owner-proof-required")) {
      $publicRequirementSatisfied = $null -ne $lane.publicPackageIdentity -and
        $null -ne $lane.publicChannel -and
        (Test-OwnerStringFilled ([string]$lane.publicPackageIdentity.packageId)) -and
        (Test-OwnerStringFilled ([string]$lane.publicPackageIdentity.version)) -and
        (Test-OwnerStringFilled ([string]$lane.publicPackageIdentity.sourceName)) -and
        (Test-OwnerStringFilled ([string]$lane.publicChannel.installSource)) -and
        (Test-OwnerStringFilled ([string]$lane.publicChannel.publicChannelUrlOrFeedIdentity))
    }

    if ($laneId -eq "post-publish-verification-owner-proof-required") {
      $rollback = $lane.rollbackPlan
      $rollbackPath = if ($null -ne $rollback) { [string]$rollback.path } else { "" }
      $rollbackFullPath = if (Test-OwnerStringFilled $rollbackPath) { Resolve-OwnerRepositoryPath -RepositoryRoot $RepositoryRoot -Path $rollbackPath } else { "" }
      $rollbackExists = -not [string]::IsNullOrWhiteSpace($rollbackFullPath) -and (Test-Path -LiteralPath $rollbackFullPath -PathType Leaf)
      $rollbackShaMatches = $rollbackExists -and ((Get-OwnerFileSha256 -Path $rollbackFullPath) -eq ([string]$rollback.sha256).ToLowerInvariant())
      $publicRequirementSatisfied = $publicRequirementSatisfied -and $rollbackShaMatches -and (Test-OwnerStringFilled ([string]$rollback.ownerDecision))
    }

    $lanePassed = $artifactShaMatches -and $logShaMatches -and $recordShaMatches -and ([int]$lane.exitCode -eq 0) -and $hostMetadataFilled -and $ownerReviewAccepted -and -not $hasForbiddenText -and $publicRequirementSatisfied
    $laneResults += [pscustomobject]@{
      laneId = $laneId
      proofLane = [string]$lane.proofLane
      laneState = if ($lanePassed) { "accepted-real-owner-evidence" } else { "blocked-owner-real-evidence-required" }
      artifactPath = $artifactPath
      artifactExists = [bool]$artifactExists
      artifactSha256Matches = [bool]$artifactShaMatches
      logPath = $logPath
      logExists = [bool]$logExists
      logSha256Matches = [bool]$logShaMatches
      recordPath = $recordPath
      recordExists = [bool]$recordExists
      recordSha256Matches = [bool]$recordShaMatches
      exitCode = [int]$lane.exitCode
      hostMetadataFilled = [bool]$hostMetadataFilled
      ownerReviewAccepted = [bool]$ownerReviewAccepted
      publicPackageFieldsSatisfied = [bool]$publicRequirementSatisfied
      forbiddenSubstituteDetected = [bool]$hasForbiddenText
      performsPublish = $false
      canPublishPublicly = $false
      canCloseReleaseIssue = $false
      canPromotePackageConsumerRuntime = $false
      canPromoteRuntimeProof = $false
    }
  }
}

$laneCount = @($laneResults).Count
$acceptedLaneCount = @($laneResults | Where-Object { [string]$_.laneState -eq "accepted-real-owner-evidence" }).Count
$blockedLaneCount = @($laneResults | Where-Object { [string]$_.laneState -ne "accepted-real-owner-evidence" }).Count
if (-not $inputExists) {
  $blockedLaneCount = 6
}
$failedBlockerCount = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" }).Count
$failedActionRequiredCount = if (-not $inputExists) { 6 } else { $blockedLaneCount }

$importState = if (-not $inputExists) {
  "blocked-owner-input-file-required"
}
elseif ($failedBlockerCount -gt 0) {
  "failed-owner-input-shape"
}
elseif ($acceptedLaneCount -eq 6) {
  "accepted-real-owner-evidence-input"
}
else {
  "blocked-owner-real-evidence-required"
}

$report = [pscustomobject]@{
  recordKind = "owner-real-evidence-input-import"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  importState = $importState
  inputPath = $InputPath
  inputExists = [bool]$inputExists
  sourceRecordKind = $recordKind
  laneCount = [int]$laneCount
  acceptedLaneCount = [int]$acceptedLaneCount
  blockedLaneCount = [int]$blockedLaneCount
  failedBlockerCount = [int]$failedBlockerCount
  failedActionRequiredCount = [int]$failedActionRequiredCount
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  executesDotnetNugetPush = $false
  uploadsGitHubReleaseAssets = $false
  forbiddenSubstitutes = @($script:OwnerRealEvidenceForbiddenSubstitutes)
  validationItems = @($items)
  lanes = @($laneResults)
  boundary = "This import validates owner-submitted real evidence. It does not publish, upload assets, close release issues, or promote proof unless all real evidence validators pass."
}

$jsonPath = Join-Path $OutputRoot "owner-real-evidence-input-import.json"
$markdownPath = Join-Path $OutputRoot "owner-real-evidence-input-import.md"
$report | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($lane in $laneResults) {
  "| ``$(ConvertTo-OwnerMarkdownCell $lane.laneId)`` | ``$(ConvertTo-OwnerMarkdownCell $lane.laneState)`` | ``$($lane.artifactSha256Matches)`` | ``$($lane.logSha256Matches)`` | ``$($lane.recordSha256Matches)`` | ``$($lane.forbiddenSubstituteDetected)`` |"
}

if ($rows.Count -eq 0) {
  $rows = @("| owner-input-file | ``blocked-owner-input-file-required`` | ``False`` | ``False`` | ``False`` | ``False`` |")
}

$markdown = @"
# Owner Real Evidence Input Import

Generated at: ``$($report.generatedAtUtc)``

## Summary

- importState: ``$($report.importState)``
- inputPath: ``$($report.inputPath)``
- inputExists: ``$($report.inputExists)``
- laneCount: ``$($report.laneCount)``
- acceptedLaneCount: ``$($report.acceptedLaneCount)``
- blockedLaneCount: ``$($report.blockedLaneCount)``
- failedBlockerCount: ``$($report.failedBlockerCount)``
- failedActionRequiredCount: ``$($report.failedActionRequiredCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Lane Results

| Lane | State | Artifact SHA | Log SHA | Record SHA | Forbidden Substitute |
| --- | --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($report.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Owner real evidence input import written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ImportState=$($report.importState) AcceptedLaneCount=$acceptedLaneCount BlockedLaneCount=$blockedLaneCount"
