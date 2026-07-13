[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-next-decision-gate.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerNextDecisionGate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$text = $record | ConvertTo-Json -Depth 12
$options = @(Get-PropertyOrDefault -Object $record -Name "decisionOptions" -DefaultValue @())
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-next-decision-gate") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-state" ([string](Get-PropertyOrDefault -Object $record -Name "gateState" -DefaultValue "") -eq "blocked-final-owner-next-decision-required" -and [string](Get-PropertyOrDefault -Object $record -Name "recommendedDefault" -DefaultValue "") -eq "keep-blocked-wait-for-owner") "blocker" "Gate must remain blocked by default while Owner authorization is missing.")) | Out-Null
$items.Add((New-OwnerValidationItem "decision-options" ($options.Count -eq 3 -and $text.Contains("authorize-real-public-publish") -and $text.Contains("authorize-article-source-patch-only") -and $text.Contains("keep-blocked-wait-for-owner")) "blocker" "Gate must expose the three allowed Owner decision paths.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-commands" ([int](Get-PropertyOrDefault -Object $record -Name "blockedCommandCount" -DefaultValue 0) -ge 5 -and $text.Contains("dotnet-nuget-push") -and $text.Contains("github-packages-push") -and $text.Contains("workflow-dispatch-publish") -and $text.Contains("release-close")) "blocker" "Gate must preserve publish/workflow/release blockers.")) | Out-Null
$items.Add((New-OwnerValidationItem "article-patch-boundary" ($text.Contains("apply-public-article-source-patch-proposals-after-owner-approval") -and $text.Contains("no public publish claim without real proof")) "blocker" "Gate must separate article source patch approval from article publication/proof promotion.")) | Out-Null
$items.Add((New-OwnerValidationItem "forbidden-substitutes" ($text.Contains("local feed") -and $text.Contains("ProjectReference") -and $text.Contains("direct .nupkg") -and $text.Contains("queued workflow") -and $text.Contains("TensorRtExec report")) "blocker" "Gate must preserve forbidden proof substitute warnings.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Gate must not publish packages/articles or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "final-owner-next-decision-gate-validation-ready-non-proof" } else { "blocked-final-owner-next-decision-gate-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "final-owner-next-decision-gate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  decisionOptionCount = [int](Get-PropertyOrDefault -Object $record -Name "decisionOptionCount" -DefaultValue 0)
  blockedCommandCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedCommandCount" -DefaultValue 0)
  blockedHandoffSectionCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedHandoffSectionCount" -DefaultValue 0)
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner next decision gate validation only; not package publication, workflow dispatch, article patch application, article publication, proof promotion, or release close approval."
}

$jsonPath = Join-Path $OutputRoot "final-owner-next-decision-gate-validation.json"
$mdPath = Join-Path $OutputRoot "final-owner-next-decision-gate-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Final Owner Next Decision Gate Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- decisionOptionCount: ``$($validation.decisionOptionCount)``",
  "- blockedCommandCount: ``$($validation.blockedCommandCount)``",
  "- blockedHandoffSectionCount: ``$($validation.blockedHandoffSectionCount)``",
  "",
  $validation.boundary
)
Write-Host "FinalOwnerNextDecisionGateValidationState=$state FailedBlockers=$failedBlockerCount Options=$($validation.decisionOptionCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Final Owner next decision gate validation failed." }
