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

function New-Blocker {
  param([string]$Id, [string]$Category, [string]$FieldPath, [string]$OwnerAction, [string]$Source)
  [pscustomobject]@{
    id = $Id
    category = $Category
    fieldPath = $FieldPath
    ownerAction = $OwnerAction
    source = $Source
    resolved = $false
    readyForImport = $false
    ownerActionRequired = $true
  }
}

$skeleton = Read-JsonOrNull "artifacts\final-release\final-owner-execution-input-skeleton.json"
$skeletonValidation = Read-JsonOrNull "artifacts\final-release\final-owner-execution-input-skeleton-validation.json"
$groups = @(Convert-ToArray (Get-PropertyOrDefault -Object $skeleton -Name "fieldGroups" -DefaultValue @()))
$fields = @($groups | ForEach-Object { Convert-ToArray (Get-PropertyOrDefault -Object $_ -Name "fields" -DefaultValue @()) })

$blockers = New-Object System.Collections.Generic.List[object]
foreach ($field in $fields) {
  $id = [string](Get-PropertyOrDefault -Object $field -Name "id" -DefaultValue "")
  $fieldPath = [string](Get-PropertyOrDefault -Object $field -Name "fieldPath" -DefaultValue "")
  $kind = [string](Get-PropertyOrDefault -Object $field -Name "kind" -DefaultValue "")
  $status = [string](Get-PropertyOrDefault -Object $field -Name "status" -DefaultValue "")
  if ($status -eq "missing owner input") {
    $blockers.Add((New-Blocker -Id "$id-missing" -Category "missing owner input" -FieldPath $fieldPath -OwnerAction "Owner must replace placeholder with real evidence value." -Source $id)) | Out-Null
  }
  if ([bool](Get-PropertyOrDefault -Object $field -Name "placeholder" -DefaultValue $false)) {
    $blockers.Add((New-Blocker -Id "$id-placeholder" -Category "placeholder" -FieldPath $fieldPath -OwnerAction "Placeholder value must be replaced before import." -Source $id)) | Out-Null
  }
  if ($kind -eq "path") {
    $blockers.Add((New-Blocker -Id "$id-path-missing" -Category "path missing" -FieldPath $fieldPath -OwnerAction "Owner must provide an existing evidence file path." -Source $id)) | Out-Null
  }
  if ($kind -eq "sha256") {
    $blockers.Add((New-Blocker -Id "$id-sha256-invalid" -Category "SHA256 invalid" -FieldPath $fieldPath -OwnerAction "Owner must provide a 64-character SHA256 and optionally match it to the evidence file." -Source $id)) | Out-Null
  }
}

$blockers.Add((New-Blocker -Id "strict-validator-not-run" -Category "strict validator not run" -FieldPath "strictValidators.outputPath" -OwnerAction "Run strict validators after real Owner input is filled." -Source "strict-validator-chain")) | Out-Null
$blockers.Add((New-Blocker -Id "strict-validator-failed" -Category "strict validator failed" -FieldPath "strictValidators.chainState" -OwnerAction "Strict validator chain must pass before import can promote any proof candidate." -Source "strict-validator-chain")) | Out-Null

$categories = @(
  "missing owner input",
  "placeholder",
  "path missing",
  "SHA256 invalid",
  "strict validator not run",
  "strict validator failed",
  "ready for import candidate"
)

$categoryRows = foreach ($category in $categories) {
  $items = @($blockers | Where-Object { $_.category -eq $category })
  [pscustomobject]@{
    category = $category
    blockerCount = $items.Count
    resolvedCount = @($items | Where-Object { [bool]$_.resolved }).Count
    remainingCount = @($items | Where-Object { -not [bool]$_.resolved }).Count
  }
}

$remainingBlockers = @($blockers | Where-Object { -not [bool]$_.resolved })

$record = [pscustomobject]@{
  recordKind = "final-owner-execution-blocker-ledger"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  ledgerState = "blocked-final-owner-real-input-required"
  sourceSkeletonState = [string](Get-PropertyOrDefault -Object $skeleton -Name "skeletonState" -DefaultValue "missing-final-owner-execution-input-skeleton")
  sourceSkeletonValidationState = [string](Get-PropertyOrDefault -Object $skeletonValidation -Name "validationState" -DefaultValue "missing-final-owner-execution-input-skeleton-validation")
  categoryCount = $categories.Count
  blockerCount = $blockers.Count
  remainingBlockerCount = $remainingBlockers.Count
  readyForImportCandidateCount = @($blockers | Where-Object { $_.category -eq "ready for import candidate" }).Count
  categories = @($categoryRows)
  blockers = @($blockers.ToArray())
  remainingBlockers = @($remainingBlockers)
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  nonSubstituteProofKinds = @("final owner execution blocker ledger", "owner blocker ledger", "blocked owner input ledger")
  sourceArtifacts = @(
    "artifacts/final-release/final-owner-execution-input-skeleton.json",
    "artifacts/final-release/final-owner-execution-input-skeleton-validation.json"
  )
  boundary = "Final Owner execution blocker ledger is a blocked owner-action ledger only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. remainingBlockerCount must be zero and strict validators must pass before any later import can be considered."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-blocker-ledger.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-blocker-ledger.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 20)
$categoryMarkdownRows = foreach ($row in $categoryRows) {
  "| ``$(ConvertTo-MarkdownCell $row.category)`` | ``$($row.blockerCount)`` | ``$($row.remainingCount)`` |"
}
$blockerRows = foreach ($blocker in $blockers) {
  "| ``$(ConvertTo-MarkdownCell $blocker.id)`` | ``$(ConvertTo-MarkdownCell $blocker.category)`` | ``$(ConvertTo-MarkdownCell $blocker.fieldPath)`` | $(ConvertTo-MarkdownCell $blocker.ownerAction) |"
}

$markdown = @"
# Final Owner Execution Blocker Ledger

| Field | Value |
|---|---|
| ledgerState | ``$($record.ledgerState)`` |
| blockerCount | ``$($record.blockerCount)`` |
| remainingBlockerCount | ``$($record.remainingBlockerCount)`` |
| readyForImportCandidateCount | ``$($record.readyForImportCandidateCount)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Categories

| Category | Blockers | Remaining |
|---|---:|---:|
$($categoryMarkdownRows -join "`r`n")

## Blockers

| ID | Category | Field | Owner Action |
|---|---|---|---|
$($blockerRows -join "`r`n")

## Boundary

$($record.boundary)
"@
Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown

Write-Host "Final owner execution blocker ledger written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "LedgerState=$($record.ledgerState) Remaining=$($record.remainingBlockerCount)"
