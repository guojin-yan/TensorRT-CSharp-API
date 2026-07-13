[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-post-publish-clean-consumer-proof-record-contract.json",
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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)
  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function Test-RawContains {
  param([string]$Raw, [string]$Needle)
  return $Raw.IndexOf($Needle, [StringComparison]::OrdinalIgnoreCase) -ge 0
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final post-publish clean consumer proof record contract not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$contractItems = @((Get-PropertyOrDefault -Object $record -Name "contractItems" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-post-publish-clean-consumer-proof-record-contract") -Severity "blocker" -Detail "recordKind must be final-post-publish-clean-consumer-proof-record-contract.")) | Out-Null
$items.Add((New-ValidationItem -Id "contract-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "contractState" -DefaultValue "") -eq "blocked-post-publish-clean-consumer-proof-required") -Severity "blocker" -Detail "Contract must remain blocked until real post-publish clean consumer proof is supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "contract-items" -Passed ($contractItems.Count -ge 3 -and [int](Get-PropertyOrDefault -Object $record -Name "readyForPreflightCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Contract must expose restore/build/run clean consumer proof items and zero ready items by default.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-field-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "requiredEvidenceFieldCount" -DefaultValue 0) -ge 80) -Severity "blocker" -Detail "Contract must expose a broad post-publish clean consumer evidence surface.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "Contract must stay non-proof and non-publish.")) | Out-Null

foreach ($marker in @("packageIdentity", "publicPackageUrl", "packageSource", "packageVersion", "nupkgSha256", "managedPackageDownloadUrl", "runtimePackageUrl", "runtimePackageDownloadUrl", "sourceProofLinkage", "githubActionsRunEvidenceReady", "githubActionsRunId", "githubActionsRunUrl", "githubActionsHeadSha", "ownerPublicPublishResultReady", "publicPackageDownloadProofReady", "sourceOwnerPublicPackageUrl", "sourceOwnerPublicPackageVersion", "sourceOwnerPublicPackageSha256", "githubReleaseAssetUrl", "githubReleaseAssetSha256", "cleanConsumerProjectRoot", "cleanConsumerProjectSha256Manifest", "cleanConsumerRestoreLogPath", "cleanConsumerBuildLogPath", "cleanConsumerRunLogPath", "cleanConsumerMergedTranscriptPath", "cleanConsumerValidatorOutputPath", "executedCommand", "exitCode", "executedAtUtc", "hostIdentity", "noProjectReferenceConfirmation", "noLocalFeedConfirmation", "noDirectNupkgConfirmation", "noSourceCheckoutReferenceConfirmation", "ownerReviewer", "ownerReviewTimestampUtc", "local feed", "ProjectReference", "direct .nupkg", "source checkout reference", "template", "draft", "dry-run", "dashboard", "candidate", "build-only", "dependency-probe-only", "blocked-by-cuda-driver")) {
  $items.Add((New-ValidationItem -Id "raw-marker-$($marker.Replace(' ', '-').Replace('.', 'dot'))" -Passed (Test-RawContains -Raw $raw -Needle $marker) -Severity "blocker" -Detail "Contract raw JSON must contain marker $marker.")) | Out-Null
}

foreach ($contractItem in $contractItems) {
  $id = [string](Get-PropertyOrDefault -Object $contractItem -Name "id" -DefaultValue "")
  $packageIdentity = Get-PropertyOrDefault -Object $contractItem -Name "packageIdentity" -DefaultValue $null
  $sourceProofLinkage = Get-PropertyOrDefault -Object $contractItem -Name "sourceProofLinkage" -DefaultValue $null
  $hostIdentity = Get-PropertyOrDefault -Object $contractItem -Name "hostIdentity" -DefaultValue $null
  $forbiddenMarkers = Convert-ToStringArray (Get-PropertyOrDefault -Object $contractItem -Name "forbiddenSubstituteMarkers" -DefaultValue @())
  $boundary = [string](Get-PropertyOrDefault -Object $contractItem -Name "boundary" -DefaultValue "")

  $items.Add((New-ValidationItem -Id "$id-state" -Passed ([string](Get-PropertyOrDefault -Object $contractItem -Name "proofState" -DefaultValue "") -eq "blocked-post-publish-clean-consumer-proof-required") -Severity "blocker" -Detail "$id must remain blocked.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-package-identity" -Passed ($null -ne $packageIdentity -and $packageIdentity.PSObject.Properties.Name -contains "publicPackageUrl" -and $packageIdentity.PSObject.Properties.Name -contains "packageSource" -and $packageIdentity.PSObject.Properties.Name -contains "nupkgSha256" -and $packageIdentity.PSObject.Properties.Name -contains "managedPackageDownloadUrl" -and $packageIdentity.PSObject.Properties.Name -contains "runtimePackageDownloadUrl") -Severity "blocker" -Detail "$id must expose package identity and public download fields.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-source-proof-linkage" -Passed ($null -ne $sourceProofLinkage -and $sourceProofLinkage.PSObject.Properties.Name -contains "githubActionsRunEvidenceReady" -and $sourceProofLinkage.PSObject.Properties.Name -contains "ownerPublicPublishResultReady" -and $sourceProofLinkage.PSObject.Properties.Name -contains "publicPackageDownloadProofReady" -and $sourceProofLinkage.PSObject.Properties.Name -contains "sourceOwnerPublicPackageSha256" -and $sourceProofLinkage.PSObject.Properties.Name -contains "githubReleaseAssetSha256") -Severity "blocker" -Detail "$id must expose upstream GitHub Actions, Owner public publish, and public download proof linkage fields.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-host-identity" -Passed ($null -ne $hostIdentity -and $hostIdentity.PSObject.Properties.Name -contains "machineName" -and $hostIdentity.PSObject.Properties.Name -contains "cudaVersion" -and $hostIdentity.PSObject.Properties.Name -contains "tensorrtVersion" -and $hostIdentity.PSObject.Properties.Name -contains "driverVersion") -Severity "blocker" -Detail "$id must expose host identity fields.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-forbidden-markers" -Passed (($forbiddenMarkers -join "`n").IndexOf("local feed", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and ($forbiddenMarkers -join "`n").IndexOf("ProjectReference", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and ($forbiddenMarkers -join "`n").IndexOf("direct .nupkg", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and ($forbiddenMarkers -join "`n").IndexOf("source checkout reference", [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "$id must preserve forbidden substitute markers.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-not-ready" -Passed (-not [bool](Get-PropertyOrDefault -Object $contractItem -Name "readyForPreflight" -DefaultValue $true)) -Severity "blocker" -Detail "$id must not be ready by default.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-boundary" -Passed ($boundary.IndexOf("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("not package push", [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "$id must state non-proof boundary.")) | Out-Null
}

$failedItems = @($items | Where-Object { -not $_.passed })
$failedBlockers = @($failedItems | Where-Object { $_.severity -eq "blocker" }).Count
$failedActionRequired = $contractItems.Count
$validationState = if ($failedBlockers -eq 0) { "blocked-post-publish-clean-consumer-proof-required" } else { "invalid-final-post-publish-clean-consumer-proof-record-contract" }

$validation = [ordered]@{
  recordKind = "final-post-publish-clean-consumer-proof-record-contract-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  contractState = [string](Get-PropertyOrDefault -Object $record -Name "contractState" -DefaultValue "")
  contractItemCount = $contractItems.Count
  blockedContractItemCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedContractItemCount" -DefaultValue 0)
  readyForPreflightCount = [int](Get-PropertyOrDefault -Object $record -Name "readyForPreflightCount" -DefaultValue 0)
  requiredEvidenceFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "requiredEvidenceFieldCount" -DefaultValue 0)
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

$jsonPath = Join-Path $OutputRoot "final-post-publish-clean-consumer-proof-record-contract-validation.json"
$markdownPath = Join-Path $OutputRoot "final-post-publish-clean-consumer-proof-record-contract-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 32)
$markdown = @(
  "# Final Post-Publish Clean Consumer Proof Record Contract Validation",
  "",
  "- validationState: ``$validationState``",
  "- failedBlockerCount: ``$failedBlockers``",
  "- failedActionRequiredCount: ``$failedActionRequired``",
  "- contractItemCount: ``$($contractItems.Count)``",
  "- requiredEvidenceFieldCount: ``$($validation.requiredEvidenceFieldCount)``"
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"

if ($Strict -and $failedBlockers -gt 0) {
  throw "Final post-publish clean consumer proof record contract validation failed with $failedBlockers blocker(s)."
}
