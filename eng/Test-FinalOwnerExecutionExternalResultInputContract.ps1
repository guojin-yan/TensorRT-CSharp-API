[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-external-result-input-contract.json",
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
  throw "Final Owner external result input contract not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$contractItems = @((Get-PropertyOrDefault -Object $record -Name "contractItems" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-execution-external-result-input-contract") -Severity "blocker" -Detail "recordKind must be final-owner-execution-external-result-input-contract.")) | Out-Null
$items.Add((New-ValidationItem -Id "contract-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "contractState" -DefaultValue "") -eq "blocked-owner-real-external-result-required") -Severity "blocker" -Detail "Contract must remain blocked until real Owner external result input is filled.")) | Out-Null
$items.Add((New-ValidationItem -Id "eight-contract-items" -Passed ($contractItems.Count -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "contractItemCount" -DefaultValue 0) -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedContractItemCount" -DefaultValue 0) -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "readyForImportCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Contract must mirror all eight owner draft items and expose zero ready lanes by default.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "Contract must stay non-proof and non-publish.")) | Out-Null
$items.Add((New-ValidationItem -Id "placeholder-field-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "placeholderFieldCount" -DefaultValue 0) -ge 180) -Severity "blocker" -Detail "Contract must expose a broad set of Owner-fill placeholders.")) | Out-Null

foreach ($marker in @("realExecutionRoot", "stdoutPath", "stderrPath", "mergedTranscriptPath", "validatorOutputPath", "stdoutSha256", "stderrSha256", "mergedTranscriptSha256", "validatorOutputSha256", "exitCode", "executedCommand", "executedAtUtc", "hostIdentity", "packageIdentity", "ownerReviewer", "ownerReviewTimestampUtc", "local feed", "ProjectReference", "direct .nupkg", "template", "draft", "dry-run", "dashboard", "candidate", "build-only", "dependency-probe-only", "blocked-by-cuda-driver")) {
  $items.Add((New-ValidationItem -Id "raw-marker-$($marker.Replace(' ', '-').Replace('.', 'dot'))" -Passed (Test-RawContains -Raw $raw -Needle $marker) -Severity "blocker" -Detail "Contract raw JSON must contain marker $marker.")) | Out-Null
}

foreach ($item in $contractItems) {
  $id = [string](Get-PropertyOrDefault -Object $item -Name "sourceDraftItemId" -DefaultValue "")
  $hostIdentity = Get-PropertyOrDefault -Object $item -Name "hostIdentity" -DefaultValue $null
  $packageIdentity = Get-PropertyOrDefault -Object $item -Name "packageIdentity" -DefaultValue $null
  $confirmations = @((Get-PropertyOrDefault -Object $item -Name "ownerProvidedNonSubstituteConfirmations" -DefaultValue @()))
  $forbidden = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "forbiddenSubstituteMarkers" -DefaultValue @())
  $boundary = [string](Get-PropertyOrDefault -Object $item -Name "boundary" -DefaultValue "")

  $items.Add((New-ValidationItem -Id "$id-state" -Passed ([string](Get-PropertyOrDefault -Object $item -Name "ownerInputState" -DefaultValue "") -eq "blocked-owner-real-external-result-required") -Severity "blocker" -Detail "$id must require real external result input.")) | Out-Null
  foreach ($field in @("realExecutionRoot", "stdoutPath", "stderrPath", "mergedTranscriptPath", "validatorOutputPath", "stdoutSha256", "stderrSha256", "mergedTranscriptSha256", "validatorOutputSha256", "exitCode", "executedCommand", "executedAtUtc", "ownerReviewer", "ownerReviewTimestampUtc")) {
    $value = [string](Get-PropertyOrDefault -Object $item -Name $field -DefaultValue "")
    $items.Add((New-ValidationItem -Id "$id-$field-placeholder" -Passed ($value.StartsWith("<owner-fill-real-", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "$id must expose placeholder for $field.")) | Out-Null
  }

  $items.Add((New-ValidationItem -Id "$id-host-identity" -Passed ($null -ne $hostIdentity -and $hostIdentity.PSObject.Properties.Name -contains "machineName" -and $hostIdentity.PSObject.Properties.Name -contains "cudaVersion" -and $hostIdentity.PSObject.Properties.Name -contains "tensorrtVersion" -and $hostIdentity.PSObject.Properties.Name -contains "driverVersion") -Severity "blocker" -Detail "$id must expose host identity fields.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-package-identity" -Passed ($null -ne $packageIdentity -and $packageIdentity.PSObject.Properties.Name -contains "packageId" -and $packageIdentity.PSObject.Properties.Name -contains "packageVersion" -and $packageIdentity.PSObject.Properties.Name -contains "packageSource" -and $packageIdentity.PSObject.Properties.Name -contains "nupkgSha256" -and $packageIdentity.PSObject.Properties.Name -contains "publishedPackageUrl") -Severity "blocker" -Detail "$id must expose package identity fields.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-confirmations" -Passed ($confirmations.Count -ge 8) -Severity "blocker" -Detail "$id must preserve non-substitute confirmations.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-forbidden-markers" -Passed (($forbidden -join "`n").IndexOf("local feed", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and ($forbidden -join "`n").IndexOf("ProjectReference", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and ($forbidden -join "`n").IndexOf("direct .nupkg", [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "$id must preserve forbidden substitute markers.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-not-ready" -Passed (-not [bool](Get-PropertyOrDefault -Object $item -Name "readyForImport" -DefaultValue $true)) -Severity "blocker" -Detail "$id must not be ready by default.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $item -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "$id must remain non-proof guidance.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-boundary" -Passed ($boundary.IndexOf("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("not package push", [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "$id must state non-proof boundary.")) | Out-Null
}

$failedItems = @($items | Where-Object { -not $_.passed })
$failedBlockers = @($failedItems | Where-Object { $_.severity -eq "blocker" }).Count
$failedActionRequired = $contractItems.Count
$validationState = if ($failedBlockers -eq 0) { "blocked-owner-real-external-result-required" } else { "invalid-final-owner-execution-external-result-input-contract" }

$validation = [ordered]@{
  recordKind = "final-owner-execution-external-result-input-contract-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  contractState = [string](Get-PropertyOrDefault -Object $record -Name "contractState" -DefaultValue "")
  contractItemCount = $contractItems.Count
  blockedContractItemCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedContractItemCount" -DefaultValue 0)
  readyForImportCount = [int](Get-PropertyOrDefault -Object $record -Name "readyForImportCount" -DefaultValue 0)
  placeholderFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "placeholderFieldCount" -DefaultValue 0)
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

$jsonPath = Join-Path $OutputRoot "final-owner-execution-external-result-input-contract-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-external-result-input-contract-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 32)
$markdown = @(
  "# Final Owner Execution External Result Input Contract Validation",
  "",
  "- validationState: ``$validationState``",
  "- failedBlockerCount: ``$failedBlockers``",
  "- failedActionRequiredCount: ``$failedActionRequired``",
  "- contractItemCount: ``$($contractItems.Count)``",
  "- readyForImportCount: ``$($validation.readyForImportCount)``",
  "- placeholderFieldCount: ``$($validation.placeholderFieldCount)``"
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"

if ($Strict -and $failedBlockers -gt 0) {
  throw "Final Owner external result input contract validation failed with $failedBlockers blocker(s)."
}
