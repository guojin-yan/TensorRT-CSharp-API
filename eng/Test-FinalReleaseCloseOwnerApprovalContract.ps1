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

  $directory = Split-Path -Parent $LiteralPath
  if ([string]::IsNullOrWhiteSpace($directory)) {
    $directory = "."
  }
  New-Item -ItemType Directory -Path $directory -Force | Out-Null

  $lines = New-Object System.Collections.Generic.List[string]
  foreach ($item in @($InputObject)) {
    if ($null -eq $item) {
      $lines.Add("") | Out-Null
    }
    elseif ($item -is [string]) {
      $lines.Add($item) | Out-Null
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { $lines.Add([string]$child) | Out-Null }
    }
    else {
      $lines.Add([string]$item) | Out-Null
    }
  }

  $content = (($lines.ToArray() -join [Environment]::NewLine) + [Environment]::NewLine)
  $fileName = Split-Path -Leaf $LiteralPath
  $tempPath = Join-Path $directory (".{0}.{1}.tmp" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  $backupPath = Join-Path $directory (".{0}.{1}.bak" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  try {
    [System.IO.File]::WriteAllText($tempPath, $content, $script:utf8)
    for ($attempt = 1; $attempt -le 10; $attempt++) {
      try {
        if (Test-Path -LiteralPath $LiteralPath -PathType Leaf) {
          [System.IO.File]::Replace($tempPath, $LiteralPath, $backupPath)
          Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
        }
        else {
          [System.IO.File]::Move($tempPath, $LiteralPath)
        }

        return
      }
      catch {
        if ($attempt -eq 10) { throw }
        Start-Sleep -Milliseconds ([Math]::Min(250, 25 * $attempt))
      }
    }
  }
  finally {
    if (Test-Path -LiteralPath $tempPath -PathType Leaf) {
      Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $backupPath -PathType Leaf) {
      Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
    }
  }
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

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final release close Owner approval contract not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$approvalLanes = @((Get-PropertyOrDefault -Object $record -Name "approvalLanes" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-release-close-owner-approval-contract") -Severity "blocker" -Detail "recordKind must be final-release-close-owner-approval-contract.")) | Out-Null
$items.Add((New-ValidationItem -Id "contract-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "contractState" -DefaultValue "") -eq "blocked-final-release-close-owner-approval-required") -Severity "blocker" -Detail "Contract must stay blocked until real Owner approval input is supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "approval-lanes" -Passed ($approvalLanes.Count -ge 4 -and [int](Get-PropertyOrDefault -Object $record -Name "readyForPreflightCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Contract must expose at least four approval lanes and zero ready lanes by default.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-field-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "requiredOwnerInputFieldCount" -DefaultValue 0) -ge 80) -Severity "blocker" -Detail "Contract must expose a broad Owner approval evidence surface.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "Contract must stay non-proof and non-publish.")) | Out-Null

foreach ($marker in @("ownerReviewer", "ownerReviewTimestampUtc", "ownerApprovalDecision", "ownerApprovalRationale", "releaseIssueUrl", "releaseIssueCloseDecision", "rollbackDecision", "rollbackRationale", "releaseNotesPath", "releaseNotesSha256", "finalPublicPackageUrl", "finalPublicPackageUrlReviewDecision", "finalPublicPackageSha256", "finalPackageIdentity", "postPublishCleanConsumerProofCandidateId", "postPublishCleanConsumerProofCandidateValidationPath", "postPublishCleanConsumerProofCandidateValidationSha256", "classificationAuditPath", "classificationAuditSha256", "releaseEvidenceBundlePath", "releaseEvidenceBundleSha256", "nonSubstituteConfirmations", "local feed", "ProjectReference", "direct .nupkg", "draft", "dry-run", "dashboard", "candidate", "blocked-by-driver")) {
  $items.Add((New-ValidationItem -Id "raw-marker-$($marker.Replace(' ', '-').Replace('.', 'dot'))" -Passed ($raw.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Contract raw JSON must contain marker $marker.")) | Out-Null
}

foreach ($marker in @("github-actions-run-proof", "owner-public-publish-result", "public-package-download-proof", "post-publish-clean-consumer-proof", "githubActionsRunProofPath", "ownerPublicPublishResultPath", "publicPackageDownloadProofPath", "postPublishCleanConsumerProofResultPath", "remote-ci-and-public-publish-proof-backfill-gate-validation.json")) {
  $items.Add((New-ValidationItem -Id "remote-proof-marker-$($marker.Replace(' ', '-').Replace('.', 'dot'))" -Passed ($raw.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Contract must preserve remote proof dependency marker $marker.")) | Out-Null
}

foreach ($lane in $approvalLanes) {
  $id = [string](Get-PropertyOrDefault -Object $lane -Name "id" -DefaultValue "")
  $finalPackageIdentity = Get-PropertyOrDefault -Object $lane -Name "finalPackageIdentity" -DefaultValue $null
  $confirmations = Get-PropertyOrDefault -Object $lane -Name "nonSubstituteConfirmations" -DefaultValue $null
  $boundary = [string](Get-PropertyOrDefault -Object $lane -Name "boundary" -DefaultValue "")

  $items.Add((New-ValidationItem -Id "$id-state" -Passed ([string](Get-PropertyOrDefault -Object $lane -Name "approvalState" -DefaultValue "") -eq "blocked-final-release-close-owner-approval-required") -Severity "blocker" -Detail "$id must remain blocked.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-final-package-identity" -Passed ($null -ne $finalPackageIdentity -and $finalPackageIdentity.PSObject.Properties.Name -contains "packageId" -and $finalPackageIdentity.PSObject.Properties.Name -contains "packageVersion" -and $finalPackageIdentity.PSObject.Properties.Name -contains "publicPackageUrl" -and $finalPackageIdentity.PSObject.Properties.Name -contains "nupkgSha256") -Severity "blocker" -Detail "$id must expose final package identity fields.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-confirmations" -Passed ($null -ne $confirmations -and $confirmations.PSObject.Properties.Name.Count -ge 8) -Severity "blocker" -Detail "$id must expose non-substitute confirmations.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-not-ready" -Passed (-not [bool](Get-PropertyOrDefault -Object $lane -Name "readyForPreflight" -DefaultValue $true)) -Severity "blocker" -Detail "$id must not be ready by default.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-boundary" -Passed ($boundary.IndexOf("not package push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("cannot close the release", [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "$id must state non-proof close boundary.")) | Out-Null
}

$failedItems = @($items | Where-Object { -not $_.passed })
$failedBlockers = @($failedItems | Where-Object { $_.severity -eq "blocker" }).Count
$failedActionRequired = $approvalLanes.Count
$validationState = if ($failedBlockers -eq 0) { "blocked-final-release-close-owner-approval-required" } else { "invalid-final-release-close-owner-approval-contract" }

$validation = [ordered]@{
  recordKind = "final-release-close-owner-approval-contract-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  contractState = [string](Get-PropertyOrDefault -Object $record -Name "contractState" -DefaultValue "")
  approvalLaneCount = $approvalLanes.Count
  blockedApprovalLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedApprovalLaneCount" -DefaultValue 0)
  readyForPreflightCount = [int](Get-PropertyOrDefault -Object $record -Name "readyForPreflightCount" -DefaultValue 0)
  requiredOwnerInputFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "requiredOwnerInputFieldCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers
  failedActionRequiredCount = $failedActionRequired
  findingCount = $failedItems.Count
  performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $false)
  canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $false)
  canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $false)
  canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $false)
  isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $false)
  isPostPublishProof = [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $false)
  isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $false)
  validationItems = @($items.ToArray())
  failedItems = @($failedItems)
}

$jsonPath = Join-Path $OutputRoot "final-release-close-owner-approval-contract-validation.json"
$markdownPath = Join-Path $OutputRoot "final-release-close-owner-approval-contract-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 32)
$markdown = @(
  "# Final Release Close Owner Approval Contract Validation",
  "",
  "- validationState: ``$validationState``",
  "- failedBlockerCount: ``$failedBlockers``",
  "- failedActionRequiredCount: ``$failedActionRequired``",
  "- approvalLaneCount: ``$($approvalLanes.Count)``",
  "- requiredOwnerInputFieldCount: ``$($validation.requiredOwnerInputFieldCount)``"
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"

if ($Strict -and $failedBlockers -gt 0) {
  throw "Final release close Owner approval contract validation failed with $failedBlockers blocker(s)."
}
