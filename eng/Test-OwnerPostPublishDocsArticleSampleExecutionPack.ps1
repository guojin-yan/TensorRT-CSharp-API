[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-post-publish-docs-article-sample-execution-pack.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

$resolvedInputPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerPostPublishDocsArticleSampleExecutionPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @()))
$laneIds = @($lanes | ForEach-Object { [string]$_.id })
$requiredLaneIds = @("public-package-urls-and-hashes", "external-clean-consumer-logs", "yolovision-real-model-assets", "article-publication-urls", "release-issue-close-material")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string]$record.recordKind -eq "owner-post-publish-docs-article-sample-execution-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "state-blocked" ([string]$record.executionPackState -eq "blocked-owner-post-publish-docs-article-sample-real-input-required") "blocker" "Execution pack must stay blocked until Owner real inputs exist.")) | Out-Null
$items.Add((New-OwnerValidationItem "required-lanes" (@($requiredLaneIds | Where-Object { $laneIds -notcontains $_ }).Count -eq 0 -and [int]$record.laneCount -ge 5) "blocker" "Execution pack must include public package, clean consumer, YoloVision, article URL, and close material lanes.")) | Out-Null
$items.Add((New-OwnerValidationItem "field-command-breadth" ([int]$record.requiredFieldCount -ge 35 -and [int]$record.copyableCommandCount -ge 9 -and [int]$record.forbiddenSubstituteCount -ge 50) "blocker" "Execution pack must contain broad Owner fields, copyable commands, and forbidden substitute rules.")) | Out-Null
$items.Add((New-OwnerValidationItem "all-lanes-blocked" ([int]$record.blockedLaneCount -eq $lanes.Count -and [int]$record.proofReadyLaneCount -eq 0 -and @($lanes | Where-Object { [bool]$_.proofReady }).Count -eq 0) "blocker" "All lanes must remain blocked and proofReady=false by default.")) | Out-Null
$items.Add((New-OwnerValidationItem "non-proof-flags" ((-not [bool]$record.performsPublish) -and (-not [bool]$record.usesPublishToken) -and (-not [bool]$record.canPublishPublicly) -and (-not [bool]$record.canCloseReleaseIssue) -and (-not [bool]$record.canPromoteRuntimeProof) -and (-not [bool]$record.isRuntimeExecutionProof) -and (-not [bool]$record.isPostPublishProof) -and (-not [bool]$record.isReleaseCloseProof)) "blocker" "Execution pack must not publish, use tokens, close, or promote proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "lane-non-proof-flags" (@($lanes | Where-Object { [bool]$_.performsPublish -or [bool]$_.usesPublishToken -or [bool]$_.canPublishPublicly -or [bool]$_.canCloseReleaseIssue -or [bool]$_.isRuntimeExecutionProof -or [bool]$_.isPostPublishProof -or [bool]$_.isReleaseCloseProof }).Count -eq 0) "blocker" "Every lane must keep non-proof flags false.")) | Out-Null

$failedBlockers = @($items.ToArray() | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "owner-post-publish-docs-article-sample-execution-pack-ready-non-proof" } else { "invalid-owner-post-publish-docs-article-sample-execution-pack" }

$validation = [pscustomobject]@{
  recordKind = "owner-post-publish-docs-article-sample-execution-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  executionPackState = [string]$record.executionPackState
  laneCount = [int]$record.laneCount
  blockedLaneCount = [int]$record.blockedLaneCount
  proofReadyLaneCount = [int]$record.proofReadyLaneCount
  requiredFieldCount = [int]$record.requiredFieldCount
  copyableCommandCount = [int]$record.copyableCommandCount
  forbiddenSubstituteCount = [int]$record.forbiddenSubstituteCount
  failedBlockerCount = $failedBlockers.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner post-publish docs/article/sample execution pack validation is non-proof validation only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-post-publish-docs-article-sample-execution-pack-validation.json"
$mdPath = Join-Path $OutputRoot "owner-post-publish-docs-article-sample-execution-pack-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Post-Publish Docs Article Sample Execution Pack Validation",
  "",
  "- validationState: ``$state``",
  "- laneCount: ``$($validation.laneCount)``",
  "- blockedLaneCount: ``$($validation.blockedLaneCount)``",
  "- proofReadyLaneCount: ``0``",
  "- requiredFieldCount: ``$($validation.requiredFieldCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- canCloseReleaseIssue: ``False``",
  "",
  $validation.boundary
)

Write-Host "OwnerPostPublishDocsArticleSampleExecutionPackValidationState=$state FailedBlockers=$($failedBlockers.Count) Lanes=$($validation.laneCount)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Owner post-publish docs/article/sample execution pack validation failed." }
