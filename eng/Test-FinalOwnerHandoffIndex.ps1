[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-handoff-index.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerHandoffIndex.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$sections = @(Get-PropertyOrDefault -Object $record -Name "sections" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-handoff-index") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "section-count" ($sections.Count -ge 9 -and [int](Get-PropertyOrDefault -Object $record -Name "sectionCount" -DefaultValue 0) -eq $sections.Count) "blocker" "Handoff index must include the final Owner handoff sections.")) | Out-Null
$items.Add((New-OwnerValidationItem "section-scope" ($text.Contains("owner-final-authorization-request-summary") -and $text.Contains("owner-authorization-command-guard") -and $text.Contains("article-source-patch-readiness") -and $text.Contains("github-actions-runner-non-proof-guard") -and $text.Contains("release-close-strict-closure")) "blocker" "Handoff index must cover authorization, command guard, article readiness, runner guard, and release close.")) | Out-Null
$items.Add((New-OwnerValidationItem "validator-bindings" ($text.Contains("Test-OwnerFinalAuthorizationRequestSummary.ps1") -and $text.Contains("Test-GitHubActionsRunnerNonProofGuardPack.ps1") -and $text.Contains("Test-ReleaseCloseStrictEvidenceClosure.ps1")) "blocker" "Handoff index must include concrete validator commands.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Handoff index must not publish packages/articles or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "final-owner-handoff-index-validation-ready-non-proof" } else { "blocked-final-owner-handoff-index-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "final-owner-handoff-index-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  sectionCount = [int](Get-PropertyOrDefault -Object $record -Name "sectionCount" -DefaultValue 0)
  blockedSectionCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedSectionCount" -DefaultValue 0)
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner handoff index validation only; not workflow dispatch, not package publication, not article publication, not proof promotion, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "final-owner-handoff-index-validation.json"
$mdPath = Join-Path $OutputRoot "final-owner-handoff-index-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Final Owner Handoff Index Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- sectionCount: ``$($validation.sectionCount)``",
  "- blockedSectionCount: ``$($validation.blockedSectionCount)``",
  "",
  $validation.boundary
)
Write-Host "FinalOwnerHandoffIndexValidationState=$state FailedBlockers=$failedBlockerCount Sections=$($validation.sectionCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Final Owner handoff index validation failed." }
