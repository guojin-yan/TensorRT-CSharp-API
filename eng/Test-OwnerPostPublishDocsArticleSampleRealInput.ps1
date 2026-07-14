[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-post-publish-docs-article-sample-real-input-import.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$resolvedInputPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Import-OwnerPostPublishDocsArticleSampleRealInput.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$analysis = Get-PropertyOrDefault -Object $record -Name "analysis" -DefaultValue $null
$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string]$record.recordKind -eq "owner-post-publish-docs-article-sample-real-input-import") "blocker" "Import recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-default" ([string]$record.importState -eq "blocked-owner-post-publish-docs-article-sample-real-input-required" -and -not [bool]$record.candidateReady) "blocker" "Default import must remain blocked without real Owner input.")) | Out-Null
$items.Add((New-OwnerValidationItem "field-results" ([int]$record.fieldResultCount -ge 37 -and [int]$record.blockedFieldCount -gt 0) "blocker" "Import must carry per-field results and blocked fields.")) | Out-Null
$items.Add((New-OwnerValidationItem "placeholder-tracking" ([int]$record.placeholderFieldCount -gt 0) "blocker" "Template fallback must report placeholders.")) | Out-Null
$items.Add((New-OwnerValidationItem "template-rejected" ((-not [bool]$record.realOwnerInputPresent) -or [bool]$record.inputPathForbidden -or [int]$record.placeholderFieldCount -gt 0) "blocker" "Template/missing Owner input must be rejected as proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "non-proof-flags" (Assert-OwnerPostPublishFalseFlags -Record $record) "blocker" "Import must not publish, close, or promote proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "analysis-present" ($null -ne $analysis -and [int](Get-PropertyOrDefault -Object $analysis -Name "fieldResultCount" -DefaultValue 0) -eq [int]$record.fieldResultCount) "blocker" "Import must include reusable candidate analysis.")) | Out-Null

$failedBlockers = @($items.ToArray() | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "owner-post-publish-docs-article-sample-real-input-import-ready-non-proof" } else { "invalid-owner-post-publish-docs-article-sample-real-input-import" }

$validation = [pscustomobject]@{
  recordKind = "owner-post-publish-docs-article-sample-real-input-import-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  importState = [string]$record.importState
  realOwnerInputPresent = [bool]$record.realOwnerInputPresent
  inputPathForbidden = [bool]$record.inputPathForbidden
  fieldResultCount = [int]$record.fieldResultCount
  readyFieldCount = [int]$record.readyFieldCount
  blockedFieldCount = [int]$record.blockedFieldCount
  placeholderFieldCount = [int]$record.placeholderFieldCount
  invalidFormatFieldCount = [int]$record.invalidFormatFieldCount
  forbiddenSubstituteFieldCount = [int]$record.forbiddenSubstituteFieldCount
  failedBlockerCount = $failedBlockers.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner post-publish docs/article/sample real input import validation is non-proof; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-post-publish-docs-article-sample-real-input-import-validation.json"
$mdPath = Join-Path $OutputRoot "owner-post-publish-docs-article-sample-real-input-import-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Post-Publish Docs Article Sample Real Input Import Validation",
  "",
  "- validationState: ``$state``",
  "- importState: ``$($validation.importState)``",
  "- realOwnerInputPresent: ``$($validation.realOwnerInputPresent)``",
  "- fieldResultCount: ``$($validation.fieldResultCount)``",
  "- blockedFieldCount: ``$($validation.blockedFieldCount)``",
  "- placeholderFieldCount: ``$($validation.placeholderFieldCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- canCloseReleaseIssue: ``False``",
  "",
  $validation.boundary
)

Write-Host "OwnerPostPublishDocsArticleSampleRealInputValidationState=$state FailedBlockers=$($failedBlockers.Count) BlockedFields=$($validation.blockedFieldCount)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Owner post-publish real input import validation failed." }
