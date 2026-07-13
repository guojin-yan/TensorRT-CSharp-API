[CmdletBinding()]
param(
  [string]$ImportPath = "artifacts\final-release\final-owner-execution-real-input-import.json",
  [string]$CandidatePath = "artifacts\final-release\final-owner-execution-real-input-candidate.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

foreach ($pathName in @("ImportPath", "CandidatePath", "OutputRoot")) {
  if (-not [System.IO.Path]::IsPathRooted((Get-Variable $pathName).Value)) {
    Set-Variable -Name $pathName -Value (Join-Path $RepositoryRoot (Get-Variable $pathName).Value)
  }
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

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

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

if (-not (Test-Path -LiteralPath $ImportPath -PathType Leaf) -or -not (Test-Path -LiteralPath $CandidatePath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Import-FinalOwnerExecutionRealInput.ps1") -RepositoryRoot $RepositoryRoot
}

$import = Get-Content -LiteralPath $ImportPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $CandidatePath -Raw -Encoding utf8
$candidate = Get-Content -LiteralPath $CandidatePath -Raw -Encoding utf8 | ConvertFrom-Json
$fieldResults = @(Convert-ToArray (Get-PropertyOrDefault -Object $candidate -Name "fieldResults" -DefaultValue @()))
$boundary = [string](Get-PropertyOrDefault -Object $candidate -Name "boundary" -DefaultValue "")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kinds" -Passed ([string]$import.recordKind -eq "final-owner-execution-real-input-import" -and [string]$candidate.recordKind -eq "final-owner-execution-real-input-candidate") -Severity "blocker" -Detail "Import and candidate recordKind values must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-default" -Passed ([string]$import.importState -eq "blocked-final-owner-real-input-required" -and [string]$candidate.candidateState -eq "blocked-final-owner-real-input-required") -Severity "blocker" -Detail "Default template import must remain blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "field-results" -Passed ($fieldResults.Count -ge 47 -and [int]$candidate.fieldResultCount -eq $fieldResults.Count) -Severity "blocker" -Detail "Candidate must carry per-field overlay results, including dual-package route proof fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "dual-package-route-fields" -Passed ($raw.Contains("dualPackageRoutes.nugetSmallBridgeCore.ownerAuthorizationUrl", [StringComparison]::OrdinalIgnoreCase) -and $raw.Contains("dualPackageRoutes.githubPackagesFullRuntime.runtimeDllResolutionReportPath", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Candidate import must preserve both NuGet and GitHub Packages dual-package route proof fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "placeholder-tracking" -Passed ([int]$candidate.placeholderFieldCount -gt 0 -and [int]$candidate.readyFieldCount -lt [int]$candidate.fieldResultCount) -Severity "blocker" -Detail "Template import must report placeholders and not be ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed ((-not [bool]$import.performsPublish) -and (-not [bool]$import.performsRuntimeExecution) -and (-not [bool]$import.canPromoteRuntimeProof) -and (-not [bool]$import.canPublishPublicly) -and (-not [bool]$import.canCloseReleaseIssue) -and (-not [bool]$candidate.isRuntimeExecutionProof) -and (-not [bool]$candidate.isPostPublishProof) -and (-not [bool]$candidate.isReleaseCloseProof)) -Severity "blocker" -Detail "Import and candidate must remain non-proof and non-publish.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed ($boundary.Contains("not runtime proof") -and $boundary.Contains("not post-publish proof") -and $boundary.Contains("not publish approval") -and $boundary.Contains("not release close approval") -and $boundary.Contains("not package push")) -Severity "blocker" -Detail "Boundary must exclude proof, publish, close, and package push.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "blocked-final-owner-real-input-required" } else { "invalid-final-owner-execution-real-input-import" }

$validation = [pscustomobject]@{
  recordKind = "final-owner-execution-real-input-import-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  importPath = $ImportPath
  candidatePath = $CandidatePath
  fieldResultCount = $fieldResults.Count
  readyFieldCount = [int](Get-PropertyOrDefault -Object $candidate -Name "readyFieldCount" -DefaultValue 0)
  placeholderFieldCount = [int](Get-PropertyOrDefault -Object $candidate -Name "placeholderFieldCount" -DefaultValue 0)
  invalidSha256FieldCount = [int](Get-PropertyOrDefault -Object $candidate -Name "invalidSha256FieldCount" -DefaultValue 0)
  missingPathFieldCount = [int](Get-PropertyOrDefault -Object $candidate -Name "missingPathFieldCount" -DefaultValue 0)
  forbiddenSubstituteFieldCount = [int](Get-PropertyOrDefault -Object $candidate -Name "forbiddenSubstituteFieldCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = $validationItems
  boundary = "Final Owner execution real input import validation checks candidate shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-real-input-import-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-real-input-import-validation.md"
$validation | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validationItems) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Final Owner Execution Real Input Import Validation

| Field | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| fieldResultCount | ``$($validation.fieldResultCount)`` |
| readyFieldCount | ``$($validation.readyFieldCount)`` |
| placeholderFieldCount | ``$($validation.placeholderFieldCount)`` |
| invalidSha256FieldCount | ``$($validation.invalidSha256FieldCount)`` |
| missingPathFieldCount | ``$($validation.missingPathFieldCount)`` |
| forbiddenSubstituteFieldCount | ``$($validation.forbiddenSubstituteFieldCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@
Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown

Write-Host "Final owner execution real input import validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final owner execution real input import validation failed with $($failedBlockers.Count) blocker(s)."
}
