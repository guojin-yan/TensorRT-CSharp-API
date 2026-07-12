[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-script-reference-index.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

if (-not [System.IO.Path]::IsPathRooted($InputPath)) {
  $InputPath = Join-Path $RepositoryRoot $InputPath
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-ReleaseScriptReferenceIndex.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$text = $record | ConvertTo-Json -Depth 12
$findings = New-Object System.Collections.Generic.List[object]

function Add-Finding {
  param([string]$Id, [string]$Message)
  $findings.Add([pscustomobject]@{
      id = $Id
      severity = "blocker"
      message = $Message
    })
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

if ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -ne "release-script-reference-index") {
  Add-Finding "record-kind" "recordKind must be release-script-reference-index."
}

if ([string](Get-PropertyOrDefault -Object $record -Name "indexState" -DefaultValue "") -ne "release-script-reference-index-ready-non-proof") {
  Add-Finding "index-state" "indexState must remain release-script-reference-index-ready-non-proof."
}

if ([int](Get-PropertyOrDefault -Object $record -Name "uniqueScriptReferenceCount" -DefaultValue 0) -lt 100) {
  Add-Finding "unique-script-reference-count" "Index must scan tracked test script references."
}

if ([int](Get-PropertyOrDefault -Object $record -Name "trackedScriptReferenceCount" -DefaultValue 0) -le 0) {
  Add-Finding "tracked-script-reference-count" "Index must report tracked script references."
}

if ([int](Get-PropertyOrDefault -Object $record -Name "missingScriptFileCount" -DefaultValue -1) -ne 0) {
  Add-Finding "missing-script-file-count" "Every referenced script file should exist locally before deciding tracking status."
}

if ([int](Get-PropertyOrDefault -Object $record -Name "untrackedScriptReferenceCount" -DefaultValue -1) -lt 0) {
  Add-Finding "untracked-script-reference-count" "Index must report remaining untracked script references."
}

foreach ($flag in @("performsPublish", "canPublishPublicly", "canCloseReleaseIssue", "isRuntimeExecutionProof", "isPostPublishProof", "isReleaseCloseProof", "isPackageConsumerRuntimeProof", "canPromoteRuntimeProof")) {
  if ([bool](Get-PropertyOrDefault -Object $record -Name $flag -DefaultValue $true)) {
    Add-Finding "flag-$flag" "$flag must remain false."
  }
}

foreach ($marker in @("not runtime proof", "not package-consumer proof", "not publish approval", "not release close approval", "not package push")) {
  if ($text.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
    Add-Finding "missing-boundary-$marker" "Boundary is missing marker: $marker"
  }
}

$validationState = if ($findings.Count -eq 0) { "release-script-reference-index-validation-ready-non-proof" } else { "release-script-reference-index-validation-failed" }
$validation = [pscustomobject]@{
  recordKind = "release-script-reference-index-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  findingCount = [int]$findings.Count
  failedBlockerCount = [int]$findings.Count
  uniqueScriptReferenceCount = [int](Get-PropertyOrDefault -Object $record -Name "uniqueScriptReferenceCount" -DefaultValue 0)
  trackedScriptReferenceCount = [int](Get-PropertyOrDefault -Object $record -Name "trackedScriptReferenceCount" -DefaultValue 0)
  untrackedScriptReferenceCount = [int](Get-PropertyOrDefault -Object $record -Name "untrackedScriptReferenceCount" -DefaultValue 0)
  missingScriptFileCount = [int](Get-PropertyOrDefault -Object $record -Name "missingScriptFileCount" -DefaultValue 0)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isPackageConsumerRuntimeProof = $false
  canPromoteRuntimeProof = $false
  findings = @($findings.ToArray())
  boundary = "Release script reference index validation only: not runtime proof, not package-consumer proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-script-reference-index-validation.json"
$markdownPath = Join-Path $OutputRoot "release-script-reference-index-validation.md"
$validation | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$md = @(
  "# Release Script Reference Index Validation",
  "",
  "- validationState: ``$validationState``",
  "- findingCount: ``$($validation.findingCount)``",
  "- uniqueScriptReferenceCount: ``$($validation.uniqueScriptReferenceCount)``",
  "- trackedScriptReferenceCount: ``$($validation.trackedScriptReferenceCount)``",
  "- untrackedScriptReferenceCount: ``$($validation.untrackedScriptReferenceCount)``",
  "- missingScriptFileCount: ``$($validation.missingScriptFileCount)``",
  "",
  $validation.boundary
)
[System.IO.File]::WriteAllText($markdownPath, (($md -join [Environment]::NewLine) + [Environment]::NewLine), $utf8)

Write-Host "ReleaseScriptReferenceIndexValidationState=$validationState Findings=$($findings.Count) Unique=$($validation.uniqueScriptReferenceCount) Tracked=$($validation.trackedScriptReferenceCount) Untracked=$($validation.untrackedScriptReferenceCount) Missing=$($validation.missingScriptFileCount)"
if ($Strict.IsPresent -and $findings.Count -gt 0) {
  throw "Release script reference index validation failed."
}
