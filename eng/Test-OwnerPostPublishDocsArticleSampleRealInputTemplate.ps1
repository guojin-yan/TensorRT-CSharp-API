[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-post-publish-docs-article-sample-real-input.template.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-OwnerPostPublishDocsArticleSampleRealInputTemplate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @()))
$fields = @($lanes | ForEach-Object { Convert-ToArray (Get-PropertyOrDefault -Object $_ -Name "fields" -DefaultValue @()) })
$laneIds = @($lanes | ForEach-Object { [string]$_.id })
$requiredLaneIds = @("public-package-urls-and-hashes", "external-clean-consumer-logs", "yolovision-real-model-assets", "article-publication-urls", "release-issue-close-material")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string]$record.recordKind -eq "owner-post-publish-docs-article-sample-real-input-template") "blocker" "Template recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "template-state" ([string]$record.templateState -eq "blocked-owner-post-publish-real-input-template") "blocker" "Template must stay blocked.")) | Out-Null
$items.Add((New-OwnerValidationItem "lane-coverage" (@($requiredLaneIds | Where-Object { $laneIds -notcontains $_ }).Count -eq 0 -and [int]$record.laneCount -eq 5) "blocker" "Template must include all five Owner lanes.")) | Out-Null
$items.Add((New-OwnerValidationItem "field-coverage" ([int]$record.requiredFieldCount -ge 37 -and $fields.Count -eq [int]$record.requiredFieldCount) "blocker" "Template must include broad required fields.")) | Out-Null
$items.Add((New-OwnerValidationItem "placeholder-only" ([int]$record.placeholderFieldCount -eq [int]$record.requiredFieldCount -and @($fields | Where-Object { -not (Test-OwnerPlaceholder -Value $_.value) }).Count -eq 0) "blocker" "Every template value must stay placeholder-only.")) | Out-Null
$items.Add((New-OwnerValidationItem "forbidden-substitute-breadth" ([int]$record.forbiddenSubstituteCount -ge 12) "blocker" "Template must list rejected substitutes.")) | Out-Null
$items.Add((New-OwnerValidationItem "non-proof-flags" (Assert-OwnerPostPublishFalseFlags -Record $record) "blocker" "Template must not publish, close, or promote proof.")) | Out-Null

$failedBlockers = @($items.ToArray() | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "owner-post-publish-docs-article-sample-real-input-template-ready-non-proof" } else { "invalid-owner-post-publish-docs-article-sample-real-input-template" }

$validation = [pscustomobject]@{
  recordKind = "owner-post-publish-docs-article-sample-real-input-template-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  templateState = [string]$record.templateState
  laneCount = [int]$record.laneCount
  requiredFieldCount = [int]$record.requiredFieldCount
  placeholderFieldCount = [int]$record.placeholderFieldCount
  forbiddenSubstituteCount = [int]$record.forbiddenSubstituteCount
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
  boundary = "Owner post-publish docs/article/sample real input template validation is schema-only and non-proof; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-post-publish-docs-article-sample-real-input-template-validation.json"
$mdPath = Join-Path $OutputRoot "owner-post-publish-docs-article-sample-real-input-template-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Post-Publish Docs Article Sample Real Input Template Validation",
  "",
  "- validationState: ``$state``",
  "- laneCount: ``$($validation.laneCount)``",
  "- requiredFieldCount: ``$($validation.requiredFieldCount)``",
  "- placeholderFieldCount: ``$($validation.placeholderFieldCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- canCloseReleaseIssue: ``False``",
  "",
  $validation.boundary
)

Write-Host "OwnerPostPublishDocsArticleSampleRealInputTemplateValidationState=$state FailedBlockers=$($failedBlockers.Count) Fields=$($validation.requiredFieldCount)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Owner post-publish real input template validation failed." }
