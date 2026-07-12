[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/final-release/owner-real-evidence-input-import.json",
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

. (Join-Path $PSScriptRoot "OwnerRealEvidenceInput.Common.ps1")

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$inputFullPath = Resolve-OwnerRepositoryPath -RepositoryRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $inputFullPath -PathType Leaf)) {
  throw "Missing owner real evidence input import report: $InputPath"
}

$import = Get-Content -LiteralPath $inputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$schemaFullPath = Resolve-OwnerRepositoryPath -RepositoryRoot $RepositoryRoot -Path "artifacts/final-release/owner-input/owner-real-evidence-input.schema.json"
$templateFullPath = Resolve-OwnerRepositoryPath -RepositoryRoot $RepositoryRoot -Path "artifacts/final-release/owner-input/owner-real-evidence-input.template.json"
$schemaText = if (Test-Path -LiteralPath $schemaFullPath -PathType Leaf) { Get-Content -LiteralPath $schemaFullPath -Raw -Encoding utf8 } else { "" }
$templateText = if (Test-Path -LiteralPath $templateFullPath -PathType Leaf) { Get-Content -LiteralPath $templateFullPath -Raw -Encoding utf8 } else { "" }
$importText = $import | ConvertTo-Json -Depth 18

$items = @(
  New-OwnerValidationItem -Id "record-kind" -Passed ([string]$import.recordKind -eq "owner-real-evidence-input-import") -Severity "blocker" -Detail "Import report must use owner-real-evidence-input-import recordKind."
  New-OwnerValidationItem -Id "missing-input-blocked-not-passed" -Passed (([bool]$import.inputExists -and [string]$import.importState -ne "blocked-owner-input-file-required") -or (-not [bool]$import.inputExists -and [string]$import.importState -eq "blocked-owner-input-file-required")) -Severity "blocker" -Detail "Missing owner input must produce blocked-owner-input-file-required instead of a fake pass."
  New-OwnerValidationItem -Id "schema-and-template-exist" -Passed ((Test-Path -LiteralPath $schemaFullPath -PathType Leaf) -and (Test-Path -LiteralPath $templateFullPath -PathType Leaf)) -Severity "blocker" -Detail "Schema and template must exist under artifacts/final-release/owner-input."
  New-OwnerValidationItem -Id "schema-covers-six-lanes" -Passed ((@($script:OwnerRealEvidenceRequiredLaneIds | Where-Object { $schemaText -notmatch [regex]::Escape($_) -or $templateText -notmatch [regex]::Escape($_) })).Count -eq 0) -Severity "blocker" -Detail "Schema/template must cover all six final action-required lanes."
  New-OwnerValidationItem -Id "required-fields-covered" -Passed ($schemaText.Contains("artifactPath") -and $schemaText.Contains("artifactSha256") -and $schemaText.Contains("hostMetadata") -and $schemaText.Contains("commandLine") -and $schemaText.Contains("exitCode") -and $schemaText.Contains("ownerReview") -and $schemaText.Contains("publicPackageIdentity") -and $schemaText.Contains("rollbackPlan")) -Severity "blocker" -Detail "Schema must cover path/hash/log/host/command/exit/owner/public package/rollback fields."
  New-OwnerValidationItem -Id "forbidden-substitutes-covered" -Passed ((@($script:OwnerRealEvidenceForbiddenSubstitutes | Where-Object { $schemaText -notmatch [regex]::Escape($_) -or $templateText -notmatch [regex]::Escape($_) -or $importText -notmatch [regex]::Escape($_) })).Count -eq 0) -Severity "blocker" -Detail "Schema/template/import must keep forbidden substitutes visible."
  New-OwnerValidationItem -Id "flags-remain-false" -Passed (-not [bool]$import.performsPublish -and -not [bool]$import.canPublishPublicly -and -not [bool]$import.canCloseReleaseIssue -and -not [bool]$import.canPromotePackageConsumerRuntime -and -not [bool]$import.canPromoteRuntimeProof -and -not [bool]$import.executesDotnetNugetPush -and -not [bool]$import.uploadsGitHubReleaseAssets) -Severity "blocker" -Detail "Import validation must not publish, upload assets, close, or promote proof."
)

$failedBlockerCount = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" }).Count
$failedActionRequiredCount = if ($failedBlockerCount -eq 0) { [int]$import.failedActionRequiredCount } else { 0 }
$validationState = if ($failedBlockerCount -eq 0) {
  if ([string]$import.importState -eq "accepted-real-owner-evidence-input") { "accepted-real-owner-evidence-input-valid" } else { "blocked-owner-real-evidence-input-required-validation-valid" }
}
else {
  "failed-owner-real-evidence-input-validation"
}

$report = [pscustomobject]@{
  recordKind = "owner-real-evidence-input-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  sourceImport = $InputPath
  importState = [string]$import.importState
  inputExists = [bool]$import.inputExists
  laneCount = [int]$import.laneCount
  acceptedLaneCount = [int]$import.acceptedLaneCount
  blockedLaneCount = [int]$import.blockedLaneCount
  failedBlockerCount = [int]$failedBlockerCount
  failedActionRequiredCount = [int]$failedActionRequiredCount
  validationItems = @($items)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  executesDotnetNugetPush = $false
  uploadsGitHubReleaseAssets = $false
}

$jsonPath = Join-Path $OutputRoot "owner-real-evidence-input-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-real-evidence-input-validation.md"
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $items) {
  "| ``$(ConvertTo-OwnerMarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-OwnerMarkdownCell $item.severity)`` | $(ConvertTo-OwnerMarkdownCell $item.detail) |"
}

$markdown = @"
# Owner Real Evidence Input Validation

Generated at: ``$($report.generatedAtUtc)``

## Summary

- validationState: ``$($report.validationState)``
- importState: ``$($report.importState)``
- inputExists: ``$($report.inputExists)``
- laneCount: ``$($report.laneCount)``
- acceptedLaneCount: ``$($report.acceptedLaneCount)``
- blockedLaneCount: ``$($report.blockedLaneCount)``
- failedBlockerCount: ``$($report.failedBlockerCount)``
- failedActionRequiredCount: ``$($report.failedActionRequiredCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Validation Items

| Item | Passed | Severity | Detail |
| --- | --- | --- | --- |
$($rows -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

if ($Strict -and $failedBlockerCount -gt 0) {
  throw "Owner real evidence input validation failed with $failedBlockerCount blocker(s)."
}

Write-Output "Owner real evidence input validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$validationState FailedBlockerCount=$failedBlockerCount FailedActionRequiredCount=$failedActionRequiredCount"
