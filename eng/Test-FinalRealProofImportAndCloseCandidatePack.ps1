[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-real-proof-import-and-close-candidate-pack.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-FinalRealProofImportAndCloseCandidatePack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$phases = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "phases" -DefaultValue @()))
$acceptedProofPhases = @($phases | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "acceptedProof" -DefaultValue $false) })
$missingProofPhases = @($phases | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "missingProof" -DefaultValue $false) })
$invalidProofPhases = @($phases | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "invalidProof" -DefaultValue $false) })
$closeCandidateReady = [bool](Get-PropertyOrDefault -Object $record -Name "closeCandidateReady" -DefaultValue $false)
$forbiddenKinds = @(Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteKinds" -DefaultValue @())
$requiredForbiddenKinds = @("local feed", "ProjectReference", "direct .nupkg", "queued workflow", "missing runner", "dry-run", "template", "dashboard", "audit", "bundle")
$missingForbiddenKinds = @($requiredForbiddenKinds | Where-Object { $forbiddenKinds -notcontains $_ })

$closeDecisionPhase = $null
foreach ($phase in $phases) {
  if ([string](Get-PropertyOrDefault -Object $phase -Name "id" -DefaultValue "") -eq "release-issue-close-owner-decision") {
    $closeDecisionPhase = $phase
  }
}
$closeDecisionBlockedReasons = @(Get-PropertyOrDefault -Object $closeDecisionPhase -Name "blockedReasons" -DefaultValue @())

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-real-proof-import-and-close-candidate-pack") "blocker" "recordKind must match final real proof import and close candidate pack.")) | Out-Null
$items.Add((New-OwnerValidationItem "phase-count" ($phases.Count -ge 12 -and [int](Get-PropertyOrDefault -Object $record -Name "phaseCount" -DefaultValue 0) -eq $phases.Count) "blocker" "Close candidate pack must cover all final proof phases.")) | Out-Null
$items.Add((New-OwnerValidationItem "accepted-count" ([int](Get-PropertyOrDefault -Object $record -Name "acceptedProofPhaseCount" -DefaultValue -1) -eq $acceptedProofPhases.Count) "blocker" "acceptedProofPhaseCount must match acceptedProof flags.")) | Out-Null
$items.Add((New-OwnerValidationItem "missing-count" ([int](Get-PropertyOrDefault -Object $record -Name "missingProofPhaseCount" -DefaultValue -1) -eq $missingProofPhases.Count) "blocker" "missingProofPhaseCount must match missingProof flags.")) | Out-Null
$items.Add((New-OwnerValidationItem "invalid-count" ([int](Get-PropertyOrDefault -Object $record -Name "invalidProofPhaseCount" -DefaultValue -1) -eq $invalidProofPhases.Count) "blocker" "invalidProofPhaseCount must match invalidProof flags.")) | Out-Null
$items.Add((New-OwnerValidationItem "close-candidate-consistency" ((-not $closeCandidateReady) -or ($acceptedProofPhases.Count -eq $phases.Count -and $missingProofPhases.Count -eq 0 -and $invalidProofPhases.Count -eq 0)) "blocker" "closeCandidateReady requires every real proof phase to be accepted.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-state-without-all-proof" (($acceptedProofPhases.Count -eq $phases.Count) -or ([string](Get-PropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "") -eq "blocked-final-real-proof-import-and-close-candidate-owner-evidence-required")) "blocker" "Candidate pack must remain blocked until all real proof phases are accepted.")) | Out-Null
$items.Add((New-OwnerValidationItem "approved-close-rejected-while-final-bridge-blocked" (([bool](Get-PropertyOrDefault -Object $record -Name "closeDecisionBlockedByFinalBridge" -DefaultValue $false) -and ($closeDecisionBlockedReasons -contains "approved-close-decision-rejected-while-final-bridge-blocked")) -or $closeCandidateReady) "blocker" "Approved close decision must stay rejected while final bridge is blocked.")) | Out-Null
$items.Add((New-OwnerValidationItem "forbidden-substitutes-rejected" ([int](Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteRejectionCount" -DefaultValue 0) -gt 0 -and $missingForbiddenKinds.Count -eq 0) "blocker" ("Forbidden substitute list must include local feed, ProjectReference, direct nupkg, queued workflow, missing runner, dry-run, template, dashboard, audit and bundle. Missing: " + ($missingForbiddenKinds -join ", ")))) | Out-Null
$items.Add((New-OwnerValidationItem "non-proof-flags" ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseRecordProof" -DefaultValue $true))) "blocker" "Close candidate pack must not publish, use tokens, promote proof, or close release issue.")) | Out-Null

foreach ($phase in $phases) {
  $id = [string](Get-PropertyOrDefault -Object $phase -Name "id" -DefaultValue "")
  $boundary = [string](Get-PropertyOrDefault -Object $phase -Name "boundary" -DefaultValue "")
  $items.Add((New-OwnerValidationItem "phase-$id-boundary" ($boundary.Contains("local feeds", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("direct .nupkg", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("close the release issue", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Each phase must state forbidden substitute and no-close boundaries.")) | Out-Null
}

$failedBlockers = @($items.ToArray() | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "final-real-proof-import-and-close-candidate-pack-ready-non-proof" } else { "invalid-final-real-proof-import-and-close-candidate-pack" }

$validation = [pscustomobject]@{
  recordKind = "final-real-proof-import-and-close-candidate-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $state
  candidateState = [string](Get-PropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "")
  phaseCount = $phases.Count
  acceptedProofPhaseCount = $acceptedProofPhases.Count
  missingProofPhaseCount = $missingProofPhases.Count
  invalidProofPhaseCount = $invalidProofPhases.Count
  closeCandidateReady = $closeCandidateReady
  closeDecisionBlockedByFinalBridge = [bool](Get-PropertyOrDefault -Object $record -Name "closeDecisionBlockedByFinalBridge" -DefaultValue $false)
  forbiddenSubstituteRejectionCount = [int](Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteRejectionCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isReleaseCloseRecordProof = $false
  boundary = "Final real proof import and close candidate validation is non-proof structure validation only; it cannot publish, promote proof, or close release issue."
}

$jsonPath = Join-Path $OutputRoot "final-real-proof-import-and-close-candidate-pack-validation.json"
$mdPath = Join-Path $OutputRoot "final-real-proof-import-and-close-candidate-pack-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 14)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Final Real Proof Import And Close Candidate Pack Validation",
  "",
  "- validationState: ``$state``",
  "- candidateState: ``$($validation.candidateState)``",
  "- phaseCount: ``$($validation.phaseCount)``",
  "- acceptedProofPhaseCount: ``$($validation.acceptedProofPhaseCount)``",
  "- missingProofPhaseCount: ``$($validation.missingProofPhaseCount)``",
  "- invalidProofPhaseCount: ``$($validation.invalidProofPhaseCount)``",
  "- closeCandidateReady: ``$($validation.closeCandidateReady)``",
  "- closeDecisionBlockedByFinalBridge: ``$($validation.closeDecisionBlockedByFinalBridge)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "## Boundary",
  "",
  $validation.boundary
)

Write-Host "FinalRealProofImportAndCloseCandidatePackValidationState=$state FailedBlockers=$($failedBlockers.Count) Phases=$($validation.phaseCount)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final real proof import and close candidate pack validation failed."
}
