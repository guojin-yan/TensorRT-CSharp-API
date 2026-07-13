[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-release-close-owner-approval-contract.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

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

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function Test-Placeholder {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text.StartsWith("<owner-fill-", [StringComparison]::OrdinalIgnoreCase)
}

function Get-FileSha256 {
  param([string]$Path)
  try {
    return (Get-FileHash -LiteralPath (Resolve-RepositoryPath -Path $Path) -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
  } catch {
    return ""
  }
}

function Test-Sha256 {
  param([string]$Path, [string]$Expected)
  if (Test-Placeholder $Path) { return $false }
  if (Test-Placeholder $Expected) { return $false }
  if ($Expected -notmatch '^[0-9a-fA-F]{64}$') { return $false }
  return (Get-FileSha256 -Path $Path) -eq $Expected.ToLowerInvariant()
}

function Test-IdentityFilled {
  param([AllowNull()][object]$Identity, [string[]]$Fields)
  if ($null -eq $Identity) { return $false }
  foreach ($field in $Fields) {
    if (-not ($Identity.PSObject.Properties.Name -contains $field)) { return $false }
    if (Test-Placeholder (Get-PropertyOrDefault -Object $Identity -Name $field -DefaultValue "")) { return $false }
  }
  return $true
}

function Test-AllConfirmationsTrue {
  param([AllowNull()][object]$Confirmations)
  if ($null -eq $Confirmations) { return $false }
  $names = @($Confirmations.PSObject.Properties.Name)
  if ($names.Count -lt 8) { return $false }
  foreach ($name in $names) {
    if (-not [bool]$Confirmations.PSObject.Properties[$name].Value) { return $false }
  }
  return $true
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final release close Owner approval contract not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$approvalResults = New-Object System.Collections.Generic.List[object]
$approvalLanes = @((Get-PropertyOrDefault -Object $record -Name "approvalLanes" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-release-close-owner-approval-contract") -Severity "blocker" -Detail "Preflight input must be final-release-close-owner-approval-contract.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-top-level" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "Input contract must stay non-proof and non-publish.")) | Out-Null

foreach ($lane in $approvalLanes) {
  $id = [string](Get-PropertyOrDefault -Object $lane -Name "id" -DefaultValue "")
  if ([string]::IsNullOrWhiteSpace($id)) { $id = "unknown-owner-approval-$($approvalResults.Count)" }

  $checks = New-Object System.Collections.Generic.List[object]
  foreach ($field in @("ownerReviewer", "ownerReviewTimestampUtc", "ownerApprovalDecision", "ownerApprovalRationale", "releaseIssueUrl", "releaseIssueCloseDecision", "rollbackDecision", "rollbackRationale", "finalPublicPackageUrl", "finalPublicPackageUrlReviewDecision", "finalPublicPackageSha256", "postPublishCleanConsumerProofCandidateId")) {
    $checks.Add((New-ValidationItem -Id "$id-$field-filled" -Passed (-not (Test-Placeholder (Get-PropertyOrDefault -Object $lane -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$id $field must be filled with real Owner input.")) | Out-Null
  }

  foreach ($pathPair in @(
    @{ Name = "releaseNotes"; PathName = "releaseNotesPath"; HashName = "releaseNotesSha256" },
    @{ Name = "postPublishCandidateValidation"; PathName = "postPublishCleanConsumerProofCandidateValidationPath"; HashName = "postPublishCleanConsumerProofCandidateValidationSha256" },
    @{ Name = "classificationAudit"; PathName = "classificationAuditPath"; HashName = "classificationAuditSha256" },
    @{ Name = "releaseEvidenceBundle"; PathName = "releaseEvidenceBundlePath"; HashName = "releaseEvidenceBundleSha256" }
  )) {
    $name = [string]$pathPair.Name
    $pathName = [string]$pathPair.PathName
    $hashName = [string]$pathPair.HashName
    $path = [string](Get-PropertyOrDefault -Object $lane -Name $pathName -DefaultValue "")
    $expectedHash = [string](Get-PropertyOrDefault -Object $lane -Name $hashName -DefaultValue "")
    $checks.Add((New-ValidationItem -Id "$id-$name-path-filled" -Passed (-not (Test-Placeholder $path)) -Severity "action-required" -Detail "$id $name path must be filled.")) | Out-Null
    $checks.Add((New-ValidationItem -Id "$id-$name-path-exists" -Passed ((-not (Test-Placeholder $path)) -and (Test-Path -LiteralPath (Resolve-RepositoryPath -Path $path) -PathType Leaf)) -Severity "action-required" -Detail "$id $name path must exist.")) | Out-Null
    $checks.Add((New-ValidationItem -Id "$id-$name-sha256-match" -Passed (Test-Sha256 -Path $path -Expected $expectedHash) -Severity "action-required" -Detail "$id $name SHA256 must match real file.")) | Out-Null
  }

  $finalPackageIdentity = Get-PropertyOrDefault -Object $lane -Name "finalPackageIdentity" -DefaultValue $null
  $checks.Add((New-ValidationItem -Id "$id-final-package-identity-filled" -Passed (Test-IdentityFilled -Identity $finalPackageIdentity -Fields @("packageId", "packageVersion", "packageSource", "publicPackageUrl", "nupkgSha256")) -Severity "action-required" -Detail "$id finalPackageIdentity must be complete.")) | Out-Null

  $packageSource = [string](Get-PropertyOrDefault -Object $finalPackageIdentity -Name "packageSource" -DefaultValue "")
  $checks.Add((New-ValidationItem -Id "$id-package-source-not-local-feed" -Passed ((-not (Test-Placeholder $packageSource)) -and $packageSource.IndexOf("local", [StringComparison]::OrdinalIgnoreCase) -lt 0 -and $packageSource.IndexOf("feed", [StringComparison]::OrdinalIgnoreCase) -lt 0) -Severity "action-required" -Detail "$id final package source must be public, not local feed.")) | Out-Null

  $confirmations = Get-PropertyOrDefault -Object $lane -Name "nonSubstituteConfirmations" -DefaultValue $null
  $checks.Add((New-ValidationItem -Id "$id-non-substitute-confirmations" -Passed (Test-AllConfirmationsTrue -Confirmations $confirmations) -Severity "action-required" -Detail "$id all non-substitute confirmations must be true.")) | Out-Null

  $failedChecks = @($checks | Where-Object { -not $_.passed })
  $ready = $failedChecks.Count -eq 0
  foreach ($check in $checks) { $items.Add($check) | Out-Null }

  $approvalResults.Add([pscustomobject]@{
    id = $id
    preflightState = if ($ready) { "final-release-close-owner-approval-candidate-ready" } else { "blocked-final-release-close-owner-approval-required" }
    readyForCloseCandidate = $ready
    failedCheckCount = $failedChecks.Count
    checks = @($checks.ToArray())
    boundary = "Final release close Owner approval preflight result only. Ready status is still not package push and not release close proof until final validator consumes real Owner approval."
  }) | Out-Null
}

$readyCount = @($approvalResults | Where-Object { $_.readyForCloseCandidate }).Count
$blockedCount = $approvalResults.Count - $readyCount
$failedItems = @($items | Where-Object { -not $_.passed })
$failedBlockers = @($failedItems | Where-Object { $_.severity -eq "blocker" }).Count
$failedActionRequired = @($failedItems | Where-Object { $_.severity -eq "action-required" }).Count
$preflightState = if ($failedBlockers -gt 0) { "invalid-final-release-close-owner-approval-preflight" } elseif ($readyCount -gt 0) { "final-release-close-owner-approval-candidate-ready" } else { "blocked-final-release-close-owner-approval-required" }

$validation = [ordered]@{
  recordKind = "final-release-close-owner-approval-preflight"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  preflightState = $preflightState
  sourceInputPath = $InputPath
  approvalLaneCount = $approvalLanes.Count
  readyCloseCandidateCount = $readyCount
  blockedCloseCandidateCount = $blockedCount
  failedBlockerCount = $failedBlockers
  failedActionRequiredCount = $failedActionRequired
  findingCount = $failedItems.Count
  approvalResults = @($approvalResults.ToArray())
  validationItems = @($items.ToArray())
  failedItems = @($failedItems)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final release close Owner approval preflight only. It screens Owner approval evidence; it is not publish approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $OutputRoot "final-release-close-owner-approval-preflight.json"
$markdownPath = Join-Path $OutputRoot "final-release-close-owner-approval-preflight.md"
$validation | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Final Release Close Owner Approval Preflight",
  "",
  "- preflightState: ``$preflightState``",
  "- approvalLaneCount: ``$($approvalLanes.Count)``",
  "- readyCloseCandidateCount: ``$readyCount``",
  "- blockedCloseCandidateCount: ``$blockedCount``",
  "- failedBlockerCount: ``$failedBlockers``",
  "- failedActionRequiredCount: ``$failedActionRequired``",
  "- boundary: $($validation.boundary)"
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"

if ($Strict -and $failedBlockers -gt 0) {
  throw "Final release close Owner approval preflight failed with $failedBlockers blocker(s)."
}
