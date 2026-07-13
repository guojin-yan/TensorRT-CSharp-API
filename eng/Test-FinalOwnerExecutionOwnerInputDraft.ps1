[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-owner-input-draft.json",
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

function Test-ArrayTextContains {
  param([AllowNull()][object]$Values, [string]$Needle)
  return ((Convert-ToStringArray $Values) -join "`n").IndexOf($Needle, [StringComparison]::OrdinalIgnoreCase) -ge 0
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final owner execution owner input draft not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$draftItems = @((Get-PropertyOrDefault -Object $record -Name "draftItems" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-execution-owner-input-draft") -Severity "blocker" -Detail "recordKind must be final-owner-execution-owner-input-draft.")) | Out-Null
$items.Add((New-ValidationItem -Id "draft-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "draftState" -DefaultValue "") -eq "blocked-owner-fill-required") -Severity "blocker" -Detail "Draft must remain blocked until Owner fills real evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "eight-draft-items" -Passed ($draftItems.Count -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "draftItemCount" -DefaultValue 0) -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedDraftItemCount" -DefaultValue 0) -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "readyDraftItemCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Draft must mirror all eight skeleton items.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "Draft must stay non-proof and non-publish.")) | Out-Null

foreach ($summaryField in @("missingFileInputCount", "missingSha256InputCount", "missingIdentityInputCount", "missingConfirmationCount", "totalMissingInputCount")) {
  $items.Add((New-ValidationItem -Id "$summaryField-positive" -Passed ([int](Get-PropertyOrDefault -Object $record -Name $summaryField -DefaultValue 0) -gt 0) -Severity "blocker" -Detail "$summaryField must be positive.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "none-ready-for-import" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "readyForImportCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Draft must not mark any item ready for import before Owner fills real evidence.")) | Out-Null

$sourceArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())
foreach ($artifact in @(
  "artifacts/final-release/final-owner-execution-repair-input-skeleton.json",
  "artifacts/final-release/final-owner-execution-repair-input-skeleton-validation.json",
  "artifacts/final-release/final-owner-execution-repair-checklist.json",
  "artifacts/final-release/final-owner-execution-repair-checklist-validation.json"
)) {
  $items.Add((New-ValidationItem -Id "source-$($artifact.Replace('/', '-').Replace('.', '-'))" -Passed ($sourceArtifacts -contains $artifact) -Severity "blocker" -Detail "Source artifact $artifact must be listed.")) | Out-Null
}

foreach ($marker in @("sourceSkeletonItemId", "ownerProvidedFiles", "ownerProvidedSha256", "ownerProvidedIdentity", "ownerProvidedNonSubstituteConfirmations", "missingFileInputCount", "missingSha256InputCount", "missingIdentityInputCount", "missingConfirmationCount", "readyForImport", "local feed", "ProjectReference", "direct .nupkg", "template", "draft", "dry-run", "dashboard", "candidate", "build-only", "dependency-probe-only", "blocked-by-cuda-driver")) {
  $items.Add((New-ValidationItem -Id "raw-marker-$($marker.Replace(' ', '-').Replace('.', 'dot'))" -Passed ($raw.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Owner input draft raw JSON must contain marker $marker.")) | Out-Null
}

foreach ($item in $draftItems) {
  $id = [string](Get-PropertyOrDefault -Object $item -Name "sourceSkeletonItemId" -DefaultValue "")
  $files = Get-PropertyOrDefault -Object $item -Name "ownerProvidedFiles" -DefaultValue $null
  $sha = Get-PropertyOrDefault -Object $item -Name "ownerProvidedSha256" -DefaultValue $null
  $identity = Get-PropertyOrDefault -Object $item -Name "ownerProvidedIdentity" -DefaultValue $null
  $confirmations = @((Get-PropertyOrDefault -Object $item -Name "ownerProvidedNonSubstituteConfirmations" -DefaultValue @()))
  $validatorCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "expectedValidatorCommands" -DefaultValue @())
  $expectedArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "expectedResultArtifacts" -DefaultValue @())
  $cannotUseMarkers = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "cannotUseMarkers" -DefaultValue @())
  $boundary = [string](Get-PropertyOrDefault -Object $item -Name "boundary" -DefaultValue "")

  $items.Add((New-ValidationItem -Id "$id-state" -Passed ([string](Get-PropertyOrDefault -Object $item -Name "ownerInputState" -DefaultValue "") -eq "owner-fill-required") -Severity "blocker" -Detail "$id must require Owner fill.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-files-present" -Passed ($null -ne $files -and [int](Get-PropertyOrDefault -Object $item -Name "missingFileInputCount" -DefaultValue 0) -gt 0) -Severity "blocker" -Detail "$id must expose ownerProvidedFiles and missing file count.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-sha-present" -Passed ($null -ne $sha -and [int](Get-PropertyOrDefault -Object $item -Name "missingSha256InputCount" -DefaultValue 0) -gt 0) -Severity "blocker" -Detail "$id must expose ownerProvidedSha256 and missing hash count.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-identity-present" -Passed ($null -ne $identity -and [int](Get-PropertyOrDefault -Object $item -Name "missingIdentityInputCount" -DefaultValue 0) -gt 0) -Severity "blocker" -Detail "$id must expose ownerProvidedIdentity and missing identity count.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-confirmations-present" -Passed ($confirmations.Count -ge 8 -and [int](Get-PropertyOrDefault -Object $item -Name "missingConfirmationCount" -DefaultValue 0) -ge 8) -Severity "blocker" -Detail "$id must expose owner confirmations and missing confirmation count.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-not-ready" -Passed (-not [bool](Get-PropertyOrDefault -Object $item -Name "readyForImport" -DefaultValue $true)) -Severity "blocker" -Detail "$id must not be ready for import before Owner fills real values.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-validator-commands" -Passed ($validatorCommands.Count -ge 1) -Severity "blocker" -Detail "$id must preserve validator commands.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-result-artifacts" -Passed ($expectedArtifacts.Count -ge 1) -Severity "blocker" -Detail "$id must preserve expected result artifacts.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-cannot-use" -Passed ((Test-ArrayTextContains -Values $cannotUseMarkers -Needle "local feed") -and (Test-ArrayTextContains -Values $cannotUseMarkers -Needle "ProjectReference") -and (Test-ArrayTextContains -Values $cannotUseMarkers -Needle "direct .nupkg") -and (Test-ArrayTextContains -Values $cannotUseMarkers -Needle "blocked-by-cuda-driver")) -Severity "blocker" -Detail "$id must preserve cannot-use markers.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $item -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "$id must remain non-proof guidance.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-boundary" -Passed ($boundary.Contains("Owner input draft only") -and $boundary.Contains("not runtime proof") -and $boundary.Contains("not package push")) -Severity "blocker" -Detail "$id must state owner-input-draft non-proof boundary.")) | Out-Null
}

$failedItems = @($items | Where-Object { -not $_.passed })
$failedBlockers = @($failedItems | Where-Object { $_.severity -eq "blocker" }).Count
$failedActionRequired = @($draftItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "ownerInputState" -DefaultValue "") -eq "owner-fill-required" }).Count
$validationState = if ($failedBlockers -eq 0) { "blocked-owner-fill-required" } else { "invalid-final-owner-execution-owner-input-draft" }

$validation = [ordered]@{
  recordKind = "final-owner-execution-owner-input-draft-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  draftState = [string](Get-PropertyOrDefault -Object $record -Name "draftState" -DefaultValue "")
  draftItemCount = $draftItems.Count
  blockedDraftItemCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedDraftItemCount" -DefaultValue 0)
  readyDraftItemCount = [int](Get-PropertyOrDefault -Object $record -Name "readyDraftItemCount" -DefaultValue 0)
  missingFileInputCount = [int](Get-PropertyOrDefault -Object $record -Name "missingFileInputCount" -DefaultValue 0)
  missingSha256InputCount = [int](Get-PropertyOrDefault -Object $record -Name "missingSha256InputCount" -DefaultValue 0)
  missingIdentityInputCount = [int](Get-PropertyOrDefault -Object $record -Name "missingIdentityInputCount" -DefaultValue 0)
  missingConfirmationCount = [int](Get-PropertyOrDefault -Object $record -Name "missingConfirmationCount" -DefaultValue 0)
  totalMissingInputCount = [int](Get-PropertyOrDefault -Object $record -Name "totalMissingInputCount" -DefaultValue 0)
  readyForImportCount = [int](Get-PropertyOrDefault -Object $record -Name "readyForImportCount" -DefaultValue 0)
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

$jsonPath = Join-Path $OutputRoot "final-owner-execution-owner-input-draft-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-owner-input-draft-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 24)
$markdown = @(
  "# Final Owner Execution Owner Input Draft Validation",
  "",
  "- validationState: ``$validationState``",
  "- failedBlockerCount: ``$failedBlockers``",
  "- failedActionRequiredCount: ``$failedActionRequired``",
  "- draftItemCount: ``$($draftItems.Count)``",
  "- missingFileInputCount: ``$($validation.missingFileInputCount)``",
  "- missingSha256InputCount: ``$($validation.missingSha256InputCount)``",
  "- missingIdentityInputCount: ``$($validation.missingIdentityInputCount)``",
  "- missingConfirmationCount: ``$($validation.missingConfirmationCount)``"
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"

if ($Strict -and $failedBlockers -gt 0) {
  throw "Final owner execution owner input draft validation failed with $failedBlockers blocker(s)."
}
