[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-final-authorization-request-summary.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerFinalAuthorizationRequestSummary.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$text = $record | ConvertTo-Json -Depth 12
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-final-authorization-request-summary") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-and-readonly-commands" ([int](Get-PropertyOrDefault -Object $record -Name "blockedCommandCount" -DefaultValue 0) -ge 5 -and [int](Get-PropertyOrDefault -Object $record -Name "allowedReadonlyCommandCount" -DefaultValue 0) -ge 3) "blocker" "Summary must include blocked publish/release/article commands and allowed readonly commands.")) | Out-Null
$items.Add((New-OwnerValidationItem "authorization-decision" ($text.Contains("Authorize real public publish") -and $text.Contains("keep all publish/article/release-close commands blocked")) "blocker" "Summary must request an explicit Owner decision.")) | Out-Null
$items.Add((New-OwnerValidationItem "real-proof-gaps" ($text.Contains("owner-public-publish-evidence") -and $text.Contains("post-publish-owner-input") -and $text.Contains("article-source-patch-readiness")) "blocker" "Summary must include real proof gaps.")) | Out-Null
$items.Add((New-OwnerValidationItem "forbidden-substitutes" ($text.Contains("local feed") -and $text.Contains("ProjectReference") -and $text.Contains("direct nupkg") -and $text.Contains("TensorRtExec report")) "blocker" "Summary must preserve forbidden substitute warnings.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Summary must not publish packages/articles or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "owner-final-authorization-request-summary-validation-ready-non-proof" } else { "blocked-owner-final-authorization-request-summary-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "owner-final-authorization-request-summary-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  blockedCommandCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedCommandCount" -DefaultValue 0)
  allowedReadonlyCommandCount = [int](Get-PropertyOrDefault -Object $record -Name "allowedReadonlyCommandCount" -DefaultValue 0)
  articleProposalCount = [int](Get-PropertyOrDefault -Object $record -Name "articleProposalCount" -DefaultValue 0)
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner final authorization request summary validation only; not package publication, workflow dispatch, article publication, destructive package action, proof promotion, or release close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-final-authorization-request-summary-validation.json"
$mdPath = Join-Path $OutputRoot "owner-final-authorization-request-summary-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Final Authorization Request Summary Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- blockedCommandCount: ``$($validation.blockedCommandCount)``",
  "- allowedReadonlyCommandCount: ``$($validation.allowedReadonlyCommandCount)``",
  "- articleProposalCount: ``$($validation.articleProposalCount)``",
  "",
  $validation.boundary
)
Write-Host "OwnerFinalAuthorizationRequestSummaryValidationState=$state FailedBlockers=$failedBlockerCount BlockedCommands=$($validation.blockedCommandCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Owner final authorization request summary validation failed." }
