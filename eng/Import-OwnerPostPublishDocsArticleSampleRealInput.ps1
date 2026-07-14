[CmdletBinding()]
param(
  [string]$OwnerInputPath = "artifacts\final-release\owner-post-publish-docs-article-sample-real-input.json",
  [string]$TemplatePath = "artifacts\final-release\owner-post-publish-docs-article-sample-real-input.template.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$resolvedOwnerInputPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $OwnerInputPath
$resolvedTemplatePath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $TemplatePath

if (-not (Test-Path -LiteralPath $resolvedTemplatePath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerPostPublishDocsArticleSampleRealInputTemplate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$realOwnerInputPresent = Test-Path -LiteralPath $resolvedOwnerInputPath -PathType Leaf
$effectiveInputPath = if ($realOwnerInputPresent) { $resolvedOwnerInputPath } else { $resolvedTemplatePath }
$inputRecord = Get-Content -LiteralPath $effectiveInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$analysis = Get-OwnerPostPublishImportAnalysis -InputRecord $inputRecord -InputPath $effectiveInputPath -RepositoryRoot $RepositoryRoot
$inputPathForbidden = [bool](Get-PropertyOrDefault -Object $analysis -Name "inputPathForbidden" -DefaultValue $true)
$candidateReady = [bool](Get-PropertyOrDefault -Object $analysis -Name "allFieldsReady" -DefaultValue $false) -and $realOwnerInputPresent -and (-not $inputPathForbidden)
$importState = if ($candidateReady) { "owner-post-publish-docs-article-sample-real-input-candidate-ready" } else { "blocked-owner-post-publish-docs-article-sample-real-input-required" }

$record = [pscustomobject]@{
  recordKind = "owner-post-publish-docs-article-sample-real-input-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = $importState
  requestedOwnerInputPath = $resolvedOwnerInputPath
  effectiveInputPath = $effectiveInputPath
  realOwnerInputPresent = $realOwnerInputPresent
  inputPathForbidden = $inputPathForbidden
  fieldResultCount = [int]$analysis.fieldResultCount
  readyFieldCount = [int]$analysis.readyFieldCount
  blockedFieldCount = [int]$analysis.blockedFieldCount
  placeholderFieldCount = [int]$analysis.placeholderFieldCount
  invalidFormatFieldCount = [int]$analysis.invalidFormatFieldCount
  forbiddenSubstituteFieldCount = [int]$analysis.forbiddenSubstituteFieldCount
  candidateReady = $candidateReady
  analysis = $analysis
  ownerActionRequired = -not $candidateReady
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner post-publish docs/article/sample real input import checks owner supplied fields only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-post-publish-docs-article-sample-real-input-import.json"
$mdPath = Join-Path $OutputRoot "owner-post-publish-docs-article-sample-real-input-import.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 18)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Post-Publish Docs Article Sample Real Input Import",
  "",
  "- importState: ``$($record.importState)``",
  "- realOwnerInputPresent: ``$($record.realOwnerInputPresent)``",
  "- inputPathForbidden: ``$($record.inputPathForbidden)``",
  "- fieldResultCount: ``$($record.fieldResultCount)``",
  "- readyFieldCount: ``$($record.readyFieldCount)``",
  "- blockedFieldCount: ``$($record.blockedFieldCount)``",
  "- placeholderFieldCount: ``$($record.placeholderFieldCount)``",
  "- candidateReady: ``$($record.candidateReady)``",
  "- canCloseReleaseIssue: ``False``",
  "",
  $record.boundary
)

Write-Host "OwnerPostPublishDocsArticleSampleRealInputImportState=$($record.importState) RealInput=$realOwnerInputPresent ReadyFields=$($record.readyFieldCount)/$($record.fieldResultCount)"
