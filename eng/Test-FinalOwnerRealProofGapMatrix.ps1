[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-real-proof-gap-matrix.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

foreach ($pathName in @("InputPath", "OutputRoot")) {
  if (-not [System.IO.Path]::IsPathRooted((Get-Variable $pathName).Value)) {
    Set-Variable -Name $pathName -Value (Join-Path $RepositoryRoot (Get-Variable $pathName).Value)
  }
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

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerRealProofGapMatrix.ps1") -RepositoryRoot $RepositoryRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$gaps = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "gaps" -DefaultValue @()))
$categories = @($gaps | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "category" -DefaultValue "") } | Sort-Object -Unique)
$gapTypes = @($gaps | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "gapType" -DefaultValue "") } | Sort-Object -Unique)

$recordKindOk = [string](Get-PropertyOrDefault $record "recordKind" "") -eq "final-owner-real-proof-gap-matrix"
$blockedOk = [string](Get-PropertyOrDefault $record "matrixState" "") -eq "blocked-final-owner-real-proof-gaps-remain" -and [bool](Get-PropertyOrDefault $record "ownerActionRequired" $false) -and -not [bool](Get-PropertyOrDefault $record "passed" $true)
$categoriesOk = @("external-clean-consumer-runtime", "post-publish-clean-consumer-proof", "rollback-review", "final-close-decision", "release-evidence-refresh" | Where-Object { $categories -notcontains $_ }).Count -eq 0
$gapTypesOk = @("missing-field", "missing-file", "missing-sha256", "missing-host-metadata", "missing-owner-confirmation" | Where-Object { $gapTypes -notcontains $_ }).Count -eq 0
$nonProofFlagsOk = -not [bool](Get-PropertyOrDefault $record "performsPublish" $true) -and -not [bool](Get-PropertyOrDefault $record "performsRuntimeExecution" $true) -and -not [bool](Get-PropertyOrDefault $record "canPromoteRuntimeProof" $true) -and -not [bool](Get-PropertyOrDefault $record "canCloseReleaseIssue" $true) -and -not [bool](Get-PropertyOrDefault $record "isRuntimeExecutionProof" $true) -and -not [bool](Get-PropertyOrDefault $record "isPostPublishProof" $true)
$gapCountOk = $gaps.Count -ge 10
$boundary = [string](Get-PropertyOrDefault $record "boundary" "")
$boundaryOk = $boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem "record-kind" $recordKindOk "blocker" "recordKind must match final-owner-real-proof-gap-matrix.")) | Out-Null
$items.Add((New-ValidationItem "blocked-default" $blockedOk "blocker" "Gap matrix must remain blocked and owner-action-required.")) | Out-Null
$items.Add((New-ValidationItem "required-categories" $categoriesOk "blocker" "Gap matrix must include all required categories.")) | Out-Null
$items.Add((New-ValidationItem "required-gap-types" $gapTypesOk "blocker" "Gap matrix must include field/file/SHA256/host/confirmation gap types.")) | Out-Null
$items.Add((New-ValidationItem "non-proof-flags" $nonProofFlagsOk "blocker" "Gap matrix cannot claim proof, publish, or close readiness.")) | Out-Null
$items.Add((New-ValidationItem "gap-count" $gapCountOk "blocker" "Gap matrix should enumerate real missing Owner evidence.")) | Out-Null
$items.Add((New-ValidationItem "boundary" $boundaryOk "blocker" "Boundary must preserve all non-proof classifications.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "final-owner-real-proof-gap-matrix-ready-non-proof" } else { "blocked-final-owner-real-proof-gap-matrix-invalid" }

$validation = [pscustomobject]@{
  recordKind = "final-owner-real-proof-gap-matrix-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  validationItemCount = $validationItems.Count
  failedBlockerCount = $failedBlockers.Count
  gapCount = $gaps.Count
  categories = @($categories)
  gapTypes = @($gapTypes)
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
  validationItems = @($validationItems)
  boundary = "Validation checks the final Owner real proof gap matrix only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-real-proof-gap-matrix-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-real-proof-gap-matrix-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validationItems) {
  "| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

Write-Utf8File -LiteralPath $markdownPath -InputObject @(
  "# Final Owner Real Proof Gap Matrix Validation",
  "",
  "- validationState: ``$validationState``",
  "- failedBlockerCount: ``$($failedBlockers.Count)``",
  "- gapCount: ``$($gaps.Count)``",
  "- passed: ``False``",
  "",
  "| ID | Passed | Severity | Detail |",
  "|---|---:|---|---|",
  @($rows),
  "",
  "## Boundary",
  "",
  $validation.boundary
)

Write-Host "FinalOwnerRealProofGapMatrixValidationState=$validationState FailedBlockers=$($failedBlockers.Count) GapCount=$($gaps.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final Owner real proof gap matrix validation failed."
}
