[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-strict-evidence-closure.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-ReleaseCloseStrictEvidenceClosure.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @(Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @())
$crossChecks = @(Get-PropertyOrDefault -Object $record -Name "crossChecks" -DefaultValue @())
$fakeReadyCases = @(Get-PropertyOrDefault -Object $record -Name "fakeReadySubstituteCases" -DefaultValue @())
$forbidden = @(Get-PropertyOrDefault -Object $record -Name "forbiddenNonProofSubstitutes" -DefaultValue @())
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$requiredLaneIds = @(
  "owner-public-publish-result",
  "post-publish-owner-input",
  "post-publish-record",
  "rollback-review",
  "close-decision",
  "release-evidence-bundle",
  "classification-audit",
  "final-public-publish-acceptance-gate"
)

$requiredSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "dashboard",
  "dry-run",
  "manual approval",
  "queued GitHub Actions run",
  "missing self-hosted runner",
  "sidecar-only",
  "TensorRtExec report"
)

$validationItems = New-Object System.Collections.Generic.List[object]
$recordKindOk = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-close-strict-evidence-closure"
$closureState = [string](Get-PropertyOrDefault -Object $record -Name "closureState" -DefaultValue "")
$defaultBlockedOk = $closureState.IndexOf("blocked", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and -not [bool](Get-PropertyOrDefault -Object $record -Name "strictEvidenceClosureReady" -DefaultValue $true)
$sideEffectFreeOk = -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "approvesPublicRelease" -DefaultValue $true)
$failedBlockerCountNotProofOk = [bool](Get-PropertyOrDefault -Object $record -Name "failedBlockerCountIsNotProof" -DefaultValue $false)
$nonProofFlagsOk = -not [bool](Get-PropertyOrDefault -Object $record -Name "dashboardIsProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "dryRunIsProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "manualApprovalIsProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "queuedWorkflowIsProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "missingRunnerIsProof" -DefaultValue $true)
$laneCountOk = $lanes.Count -eq $requiredLaneIds.Count

$validationItems.Add((New-OwnerValidationItem "record-kind" $recordKindOk "blocker" "recordKind must match.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "default-blocked" $defaultBlockedOk "blocker" "Strict closure must remain blocked until real owner publish, PostPublish, rollback, close decision, evidence bundle, classification audit, and final gate all align.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "side-effect-free" $sideEffectFreeOk "blocker" "Strict closure must not publish or approve public release.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "failed-blocker-count-not-proof" $failedBlockerCountNotProofOk "blocker" "failedBlockerCount=0 must not be treated as proof.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "dashboard-dryrun-non-proof" $nonProofFlagsOk "blocker" "Dashboard, dry-run, manual approval, queued workflow, and missing runner must stay non-proof.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "lane-count" $laneCountOk "blocker" "Strict closure must aggregate every required lane.")) | Out-Null

foreach ($id in $requiredLaneIds) {
  $validationItems.Add((New-OwnerValidationItem "lane-$id" (@($lanes | Where-Object { [string]$_.id -eq $id }).Count -eq 1) "blocker" "Missing required strict closure lane: $id")) | Out-Null
}

foreach ($substitute in $requiredSubstitutes) {
  $hasForbidden = @($forbidden | Where-Object { [string]$_ -eq $substitute }).Count -eq 1
  $hasBlockedCase = @($fakeReadyCases | Where-Object { [string]$_.substitute -eq $substitute -and [bool]$_.fakeReadyShapeBlocked -and -not [bool]$_.canCloseReleaseIssue -and -not [bool]$_.isReleaseCloseProof }).Count -eq 1
  $validationItems.Add((New-OwnerValidationItem "fake-ready-$substitute" ($hasForbidden -and $hasBlockedCase) "blocker" "Fake-ready substitute must be blocked and non-proof: $substitute")) | Out-Null
}

$crossChecksPresentOk = $crossChecks.Count -ge 6 -and [int](Get-PropertyOrDefault -Object $record -Name "failedCrossCheckCount" -DefaultValue -1) -ge 1
$boundaryTextOk = $boundary.IndexOf("never executes dotnet nuget push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("delete", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("TensorRtExec", [StringComparison]::OrdinalIgnoreCase) -ge 0
$validationItems.Add((New-OwnerValidationItem "cross-checks-present" $crossChecksPresentOk "blocker" "Strict closure must include failed cross-checks while real evidence is missing.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "boundary-text" $boundaryTextOk "blocker" "Boundary must explicitly forbid publish/delete-like actions and TensorRtExec substitution.")) | Out-Null

$failed = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failed.Count -eq 0) { "release-close-strict-evidence-closure-validation-ready-non-proof" } else { "blocked-release-close-strict-evidence-closure-validation-invalid" }
$failedBlockerCount = [int]$failed.Count
$validationItemCount = [int]$validationItems.Count
$laneCount = [int]$lanes.Count
$blockedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue -1)
$crossCheckCount = [int]$crossChecks.Count
$failedCrossCheckCount = [int](Get-PropertyOrDefault -Object $record -Name "failedCrossCheckCount" -DefaultValue -1)
$fakeReadySubstituteCaseCount = [int]$fakeReadyCases.Count
$blockedFakeReadySubstituteCaseCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedFakeReadySubstituteCaseCount" -DefaultValue -1)
$validationItemArray = @($validationItems.ToArray())
$validation = [pscustomobject]@{
  recordKind = "release-close-strict-evidence-closure-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = $validationItemCount
  laneCount = $laneCount
  blockedLaneCount = $blockedLaneCount
  crossCheckCount = $crossCheckCount
  failedCrossCheckCount = $failedCrossCheckCount
  fakeReadySubstituteCaseCount = $fakeReadySubstituteCaseCount
  blockedFakeReadySubstituteCaseCount = $blockedFakeReadySubstituteCaseCount
  validationItems = $validationItemArray
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Validation of strict closure only; non-proof while real public publish, PostPublish, rollback, close decision, and public acceptance evidence are incomplete."
}

$jsonPath = Join-Path $OutputRoot "release-close-strict-evidence-closure-validation.json"
$mdPath = Join-Path $OutputRoot "release-close-strict-evidence-closure-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# ReleaseClose Strict Evidence Closure Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$($failed.Count)``",
  "- blockedLaneCount: ``$($validation.blockedLaneCount)``",
  "- failedCrossCheckCount: ``$($validation.failedCrossCheckCount)``",
  "- blockedFakeReadySubstituteCaseCount: ``$($validation.blockedFakeReadySubstituteCaseCount)``",
  "",
  "## Boundary",
  "",
  $validation.boundary
)
Write-Host "ReleaseCloseStrictEvidenceClosureValidationState=$state FailedBlockers=$failedBlockerCount BlockedLanes=$blockedLaneCount FailedCrossChecks=$failedCrossCheckCount"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "ReleaseClose strict evidence closure validation failed." }
