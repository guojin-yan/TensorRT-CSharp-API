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
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-OwnerProvidedMap {
  param([AllowNull()][object[]]$Inputs, [string]$PlaceholderPrefix)

  $map = [ordered]@{}
  foreach ($input in @($Inputs)) {
    $name = [string](Get-PropertyOrDefault -Object $input -Name "name" -DefaultValue "")
    if ([string]::IsNullOrWhiteSpace($name)) {
      continue
    }

    $map[$name] = "<owner-fill-$PlaceholderPrefix-$name>"
  }

  return [pscustomobject]$map
}

function New-OwnerInputDraftItem {
  param([object]$SkeletonItem)

  $sourceRepairItemId = [string](Get-PropertyOrDefault -Object $SkeletonItem -Name "sourceRepairItemId" -DefaultValue "")
  $sourceExecutionStepId = [string](Get-PropertyOrDefault -Object $SkeletonItem -Name "sourceExecutionStepId" -DefaultValue "")
  $fileInputs = @((Get-PropertyOrDefault -Object $SkeletonItem -Name "fileInputs" -DefaultValue @()))
  $shaInputs = @((Get-PropertyOrDefault -Object $SkeletonItem -Name "sha256Inputs" -DefaultValue @()))
  $identityInputs = @((Get-PropertyOrDefault -Object $SkeletonItem -Name "identityInputs" -DefaultValue @()))
  $confirmations = Convert-ToStringArray (Get-PropertyOrDefault -Object $SkeletonItem -Name "nonSubstituteConfirmations" -DefaultValue @())
  $validatorCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $SkeletonItem -Name "expectedValidatorCommands" -DefaultValue @())
  $expectedArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $SkeletonItem -Name "expectedResultArtifacts" -DefaultValue @())
  $cannotUseMarkers = Convert-ToStringArray (Get-PropertyOrDefault -Object $SkeletonItem -Name "cannotUseMarkers" -DefaultValue @())

  [pscustomobject]@{
    sourceSkeletonItemId = "skeleton-$sourceRepairItemId"
    sourceRepairItemId = $sourceRepairItemId
    sourceExecutionStepId = $sourceExecutionStepId
    laneId = [string](Get-PropertyOrDefault -Object $SkeletonItem -Name "laneId" -DefaultValue "")
    actionRequiredId = [string](Get-PropertyOrDefault -Object $SkeletonItem -Name "actionRequiredId" -DefaultValue "")
    ownerInputState = "owner-fill-required"
    ownerProvidedFiles = New-OwnerProvidedMap -Inputs $fileInputs -PlaceholderPrefix "file"
    ownerProvidedSha256 = New-OwnerProvidedMap -Inputs $shaInputs -PlaceholderPrefix "sha256"
    ownerProvidedIdentity = New-OwnerProvidedMap -Inputs $identityInputs -PlaceholderPrefix "identity"
    ownerProvidedNonSubstituteConfirmations = @($confirmations | ForEach-Object { [pscustomobject]@{ confirmation = $_; ownerConfirmed = $false; placeholder = "<owner-confirm-required>" } })
    missingFileInputCount = $fileInputs.Count
    missingSha256InputCount = $shaInputs.Count
    missingIdentityInputCount = $identityInputs.Count
    missingConfirmationCount = $confirmations.Count
    totalMissingInputCount = ($fileInputs.Count + $shaInputs.Count + $identityInputs.Count + $confirmations.Count)
    expectedValidatorCommands = @($validatorCommands)
    expectedResultArtifacts = @($expectedArtifacts)
    cannotUseMarkers = @($cannotUseMarkers)
    importPreflight = Get-PropertyOrDefault -Object $SkeletonItem -Name "importPreflight" -DefaultValue $null
    evidenceRootPolicy = Get-PropertyOrDefault -Object $SkeletonItem -Name "evidenceRootPolicy" -DefaultValue $null
    readyForImport = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner input draft only. It is a placeholder contract for Owner-filled real evidence; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$skeleton = Read-JsonOrNull "artifacts\final-release\final-owner-execution-repair-input-skeleton.json"
$skeletonValidation = Read-JsonOrNull "artifacts\final-release\final-owner-execution-repair-input-skeleton-validation.json"

$skeletonItems = @()
if ($null -ne $skeleton) {
  $skeletonItems = @((Get-PropertyOrDefault -Object $skeleton -Name "skeletonItems" -DefaultValue @()))
}

$draftItems = @($skeletonItems | ForEach-Object { New-OwnerInputDraftItem -SkeletonItem $_ } | Sort-Object sourceExecutionStepId)
$missingFileInputCount = ($draftItems | ForEach-Object { [int]$_.missingFileInputCount } | Measure-Object -Sum).Sum
if ($null -eq $missingFileInputCount) { $missingFileInputCount = 0 }
$missingSha256InputCount = ($draftItems | ForEach-Object { [int]$_.missingSha256InputCount } | Measure-Object -Sum).Sum
if ($null -eq $missingSha256InputCount) { $missingSha256InputCount = 0 }
$missingIdentityInputCount = ($draftItems | ForEach-Object { [int]$_.missingIdentityInputCount } | Measure-Object -Sum).Sum
if ($null -eq $missingIdentityInputCount) { $missingIdentityInputCount = 0 }
$missingConfirmationCount = ($draftItems | ForEach-Object { [int]$_.missingConfirmationCount } | Measure-Object -Sum).Sum
if ($null -eq $missingConfirmationCount) { $missingConfirmationCount = 0 }
$totalMissingInputCount = ($draftItems | ForEach-Object { [int]$_.totalMissingInputCount } | Measure-Object -Sum).Sum
if ($null -eq $totalMissingInputCount) { $totalMissingInputCount = 0 }

$record = [ordered]@{
  recordKind = "final-owner-execution-owner-input-draft"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  draftState = "blocked-owner-fill-required"
  sourceSkeletonState = [string](Get-PropertyOrDefault -Object $skeleton -Name "skeletonState" -DefaultValue "missing-final-owner-execution-repair-input-skeleton")
  sourceSkeletonValidationState = [string](Get-PropertyOrDefault -Object $skeletonValidation -Name "validationState" -DefaultValue "missing-final-owner-execution-repair-input-skeleton-validation")
  draftItemCount = $draftItems.Count
  blockedDraftItemCount = $draftItems.Count
  readyDraftItemCount = 0
  missingFileInputCount = [int]$missingFileInputCount
  missingSha256InputCount = [int]$missingSha256InputCount
  missingIdentityInputCount = [int]$missingIdentityInputCount
  missingConfirmationCount = [int]$missingConfirmationCount
  totalMissingInputCount = [int]$totalMissingInputCount
  readyForImportCount = 0
  draftItems = @($draftItems)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    "artifacts/final-release/final-owner-execution-repair-input-skeleton.json",
    "artifacts/final-release/final-owner-execution-repair-input-skeleton.md",
    "artifacts/final-release/final-owner-execution-repair-input-skeleton-validation.json",
    "artifacts/final-release/final-owner-execution-repair-input-skeleton-validation.md",
    "artifacts/final-release/final-owner-execution-repair-checklist.json",
    "artifacts/final-release/final-owner-execution-repair-checklist-validation.json"
  )
  boundary = "Final owner execution owner input draft is owner input draft only. It records missing real evidence fields and placeholders; it does not publish, does not close the release, is not runtime proof, is not post-publish proof, is not publish approval, is not release close approval, and is not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-owner-input-draft.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-owner-input-draft.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 24)
$rows = foreach ($item in $draftItems) {
  "| ``$(ConvertTo-MarkdownCell $item.sourceExecutionStepId)`` | ``$(ConvertTo-MarkdownCell $item.laneId)`` | ``$($item.missingFileInputCount)`` | ``$($item.missingSha256InputCount)`` | ``$($item.missingIdentityInputCount)`` | ``$($item.missingConfirmationCount)`` | ``$($item.readyForImport)`` |"
}

$markdown = @(
  "# Final Owner Execution Owner Input Draft",
  "",
  "- draftState: ``$($record.draftState)``",
  "- draftItemCount: ``$($record.draftItemCount)``",
  "- blockedDraftItemCount: ``$($record.blockedDraftItemCount)``",
  "- readyDraftItemCount: ``0``",
  "- missingFileInputCount: ``$missingFileInputCount``",
  "- missingSha256InputCount: ``$missingSha256InputCount``",
  "- missingIdentityInputCount: ``$missingIdentityInputCount``",
  "- missingConfirmationCount: ``$missingConfirmationCount``",
  "- totalMissingInputCount: ``$totalMissingInputCount``",
  "- boundary: $($record.boundary)",
  "",
  "> This is an Owner input draft only. Replace placeholders with real external evidence before running strict import validators.",
  "",
  "| Execution Step | Lane | Files Missing | SHA256 Missing | Identity Missing | Confirmations Missing | Ready For Import |",
  "|---|---|---:|---:|---:|---:|---|",
  @($rows)
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
