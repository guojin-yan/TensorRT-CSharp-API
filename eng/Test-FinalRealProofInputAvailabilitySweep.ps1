[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-real-proof-input-availability-sweep.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-FinalRealProofInputAvailabilitySweep.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$phases = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "phases" -DefaultValue @()))
$phaseIds = @($phases | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$requiredPhaseIds = @(
  "claim-boundary-preflight",
  "owner-public-publish-execution",
  "github-actions-run-proof",
  "public-managed-package-download-proof",
  "public-runtime-package-download-proof",
  "repository-external-clean-consumer",
  "post-publish-clean-consumer-proof",
  "post-publish-user-verification",
  "strict-close-convergence",
  "release-issue-close-owner-decision",
  "release-issue-close-record",
  "final-bundle-classification-lock"
)

$missingPhaseIds = @($requiredPhaseIds | Where-Object { $phaseIds -notcontains $_ })
$proofReadyPhases = @($phases | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "proofReady" -DefaultValue $false) })
$invalidInputPhases = @($phases | Where-Object { [int](Get-PropertyOrDefault -Object $_ -Name "invalidRealInputFileCount" -DefaultValue 0) -gt 0 })
$unsafePaths = New-Object System.Collections.Generic.List[object]
foreach ($phase in $phases) {
  foreach ($path in @(Get-PropertyOrDefault -Object $phase -Name "acceptedOwnerInputPaths" -DefaultValue @())) {
    $text = [string]$path
    if ($text.EndsWith(".template.json", [System.StringComparison]::OrdinalIgnoreCase) -or
      $text.EndsWith(".example.json", [System.StringComparison]::OrdinalIgnoreCase) -or
      $text.EndsWith(".draft.json", [System.StringComparison]::OrdinalIgnoreCase) -or
      $text.EndsWith(".misuse.json", [System.StringComparison]::OrdinalIgnoreCase) -or
      $text.EndsWith(".ready.json", [System.StringComparison]::OrdinalIgnoreCase) -or
      $text.EndsWith(".debug-ready.json", [System.StringComparison]::OrdinalIgnoreCase) -or
      $text.EndsWith("-validation.json", [System.StringComparison]::OrdinalIgnoreCase)) {
      $unsafePaths.Add([pscustomobject]@{ phaseId = [string]$phase.id; path = $text }) | Out-Null
    }
  }
}

$forbiddenKinds = @(Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteKinds" -DefaultValue @())
$requiredForbiddenKinds = @("local feed", "ProjectReference", "direct .nupkg", "queued workflow", "missing runner", "dry-run", "template", "dashboard", "audit", "bundle")
$missingForbiddenKinds = @($requiredForbiddenKinds | Where-Object { $forbiddenKinds -notcontains $_ })

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-real-proof-input-availability-sweep") "blocker" "recordKind must match final real proof input availability sweep.")) | Out-Null
$items.Add((New-OwnerValidationItem "sweep-state" ([string](Get-PropertyOrDefault -Object $record -Name "sweepState" -DefaultValue "") -in @("blocked-final-real-proof-inputs-required", "final-real-proof-inputs-available-and-validator-accepted")) "blocker" "Sweep state must be blocked or all-real-proof-ready.")) | Out-Null
$items.Add((New-OwnerValidationItem "required-phases" ($missingPhaseIds.Count -eq 0 -and $phases.Count -ge 12) "blocker" ("Missing phases: " + ($missingPhaseIds -join ", ")))) | Out-Null
$items.Add((New-OwnerValidationItem "phase-count" ([int](Get-PropertyOrDefault -Object $record -Name "phaseCount" -DefaultValue 0) -eq $phases.Count) "blocker" "phaseCount must match phases array.")) | Out-Null
$items.Add((New-OwnerValidationItem "invalid-real-inputs-zero" ($invalidInputPhases.Count -eq 0) "blocker" "Malformed .owner.json/.real.json Owner inputs must block strict validation.")) | Out-Null
$items.Add((New-OwnerValidationItem "accepted-paths-safe" ($unsafePaths.Count -eq 0) "blocker" "Accepted real input paths must not include template/example/draft/misuse/ready/validation files.")) | Out-Null
$items.Add((New-OwnerValidationItem "forbidden-substitutes-rejected" ([int](Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteCount" -DefaultValue 0) -gt 0 -and $missingForbiddenKinds.Count -eq 0) "blocker" ("Forbidden substitute scan must reject local feed, ProjectReference, direct nupkg, queued workflow, missing runner, dry-run, template, dashboard, audit and bundle. Missing: " + ($missingForbiddenKinds -join ", ")))) | Out-Null
$items.Add((New-OwnerValidationItem "proof-ready-consistency" ([int](Get-PropertyOrDefault -Object $record -Name "proofReadyPhaseCount" -DefaultValue -1) -eq $proofReadyPhases.Count) "blocker" "proofReadyPhaseCount must match phase proofReady flags.")) | Out-Null
$items.Add((New-OwnerValidationItem "missing-available-consistency" (([int](Get-PropertyOrDefault -Object $record -Name "missingRealInputPhaseCount" -DefaultValue -1) + [int](Get-PropertyOrDefault -Object $record -Name "availableRealInputPhaseCount" -DefaultValue -1)) -eq $phases.Count) "blocker" "Missing plus available phase counts must equal phase count.")) | Out-Null
$items.Add((New-OwnerValidationItem "close-candidate-false" (-not [bool](Get-PropertyOrDefault -Object $record -Name "closeCandidateReady" -DefaultValue $true)) "blocker" "Availability sweep cannot declare a close candidate.")) | Out-Null
$items.Add((New-OwnerValidationItem "non-proof-flags" ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) "blocker" "Sweep must not publish, use tokens, promote runtime proof, or close release issue.")) | Out-Null

$failedBlockers = @($items.ToArray() | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "final-real-proof-input-availability-sweep-ready-non-proof" } else { "invalid-final-real-proof-input-availability-sweep" }

$validation = [pscustomobject]@{
  recordKind = "final-real-proof-input-availability-sweep-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $state
  sweepState = [string](Get-PropertyOrDefault -Object $record -Name "sweepState" -DefaultValue "")
  phaseCount = $phases.Count
  availableRealInputPhaseCount = [int](Get-PropertyOrDefault -Object $record -Name "availableRealInputPhaseCount" -DefaultValue 0)
  acceptedRealInputPhaseCount = [int](Get-PropertyOrDefault -Object $record -Name "acceptedRealInputPhaseCount" -DefaultValue 0)
  missingRealInputPhaseCount = [int](Get-PropertyOrDefault -Object $record -Name "missingRealInputPhaseCount" -DefaultValue 0)
  invalidRealInputPhaseCount = [int](Get-PropertyOrDefault -Object $record -Name "invalidRealInputPhaseCount" -DefaultValue 0)
  proofReadyPhaseCount = [int](Get-PropertyOrDefault -Object $record -Name "proofReadyPhaseCount" -DefaultValue 0)
  forbiddenSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteCount" -DefaultValue 0)
  forbiddenSubstituteKinds = @($forbiddenKinds)
  failedBlockerCount = $failedBlockers.Count
  validationItems = @($items.ToArray())
  unsafeAcceptedPaths = @($unsafePaths.ToArray())
  ownerActionRequired = $true
  closeCandidateReady = $false
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final real proof input availability sweep validation is non-proof structure validation only; it rejects forbidden substitutes and cannot publish, promote proof, or close release issue."
}

$jsonPath = Join-Path $OutputRoot "final-real-proof-input-availability-sweep-validation.json"
$mdPath = Join-Path $OutputRoot "final-real-proof-input-availability-sweep-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 14)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Final Real Proof Input Availability Sweep Validation",
  "",
  "- validationState: ``$state``",
  "- sweepState: ``$($validation.sweepState)``",
  "- phaseCount: ``$($validation.phaseCount)``",
  "- availableRealInputPhaseCount: ``$($validation.availableRealInputPhaseCount)``",
  "- missingRealInputPhaseCount: ``$($validation.missingRealInputPhaseCount)``",
  "- invalidRealInputPhaseCount: ``$($validation.invalidRealInputPhaseCount)``",
  "- proofReadyPhaseCount: ``$($validation.proofReadyPhaseCount)``",
  "- forbiddenSubstituteCount: ``$($validation.forbiddenSubstituteCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "## Boundary",
  "",
  $validation.boundary
)

Write-Host "FinalRealProofInputAvailabilitySweepValidationState=$state FailedBlockers=$($failedBlockers.Count) Phases=$($validation.phaseCount)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final real proof input availability sweep validation failed."
}
