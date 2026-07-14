[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-public-publish-execution-final-intake-pack.json",
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
  throw "Owner public publish execution final intake pack not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$phases = @((Get-PropertyOrDefault -Object $record -Name "phases" -DefaultValue @()))
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
$blockedPhases = @($phases | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "blocked" -DefaultValue $false) })
$proofReadyPhases = @($phases | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "proofReady" -DefaultValue $false) })
$unsafePhases = @($phases | Where-Object {
  [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "usesPublishToken" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "isRuntimeExecutionProof" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "isPostPublishProof" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "isReleaseCloseProof" -DefaultValue $true)
})

$unsafeFields = New-Object System.Collections.Generic.List[object]
$boundaryFailures = New-Object System.Collections.Generic.List[object]
foreach ($phase in $phases) {
  $boundary = [string](Get-PropertyOrDefault -Object $phase -Name "boundary" -DefaultValue "")
  foreach ($marker in @("not runtime proof", "not post-publish proof", "not publish approval", "not release close approval", "not package push")) {
    if (-not $boundary.Contains($marker, [StringComparison]::OrdinalIgnoreCase)) {
      $boundaryFailures.Add([pscustomobject]@{ id = [string](Get-PropertyOrDefault -Object $phase -Name "id" -DefaultValue "missing-id"); missingMarker = $marker }) | Out-Null
    }
  }

  foreach ($field in @((Get-PropertyOrDefault -Object $phase -Name "requiredFields" -DefaultValue @()))) {
    if ([bool](Get-PropertyOrDefault -Object $field -Name "acceptsTemplate" -DefaultValue $true) -or
      [bool](Get-PropertyOrDefault -Object $field -Name "acceptsDraft" -DefaultValue $true) -or
      [bool](Get-PropertyOrDefault -Object $field -Name "acceptsLocalFeed" -DefaultValue $true) -or
      [bool](Get-PropertyOrDefault -Object $field -Name "acceptsProjectReference" -DefaultValue $true) -or
      [bool](Get-PropertyOrDefault -Object $field -Name "acceptsDirectNupkg" -DefaultValue $true) -or
      [bool](Get-PropertyOrDefault -Object $field -Name "acceptsDryRun" -DefaultValue $true) -or
      [bool](Get-PropertyOrDefault -Object $field -Name "acceptsDashboard" -DefaultValue $true) -or
      [bool](Get-PropertyOrDefault -Object $field -Name "acceptsQueuedWorkflow" -DefaultValue $true)) {
      $unsafeFields.Add($field) | Out-Null
    }
  }
}

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-public-publish-execution-final-intake-pack") "blocker" "recordKind must be owner-public-publish-execution-final-intake-pack.")) | Out-Null
$items.Add((New-OwnerValidationItem "intake-state" ([string](Get-PropertyOrDefault -Object $record -Name "intakeState" -DefaultValue "") -eq "blocked-owner-public-publish-execution-final-intake-real-evidence-required") "blocker" "Final intake pack must remain blocked until real Owner evidence exists.")) | Out-Null
$items.Add((New-OwnerValidationItem "required-phases-present" ($missingPhaseIds.Count -eq 0) "blocker" ("Missing phases: " + ($missingPhaseIds -join ", ")))) | Out-Null
$items.Add((New-OwnerValidationItem "phase-count" ($phases.Count -ge 12 -and [int](Get-PropertyOrDefault -Object $record -Name "phaseCount" -DefaultValue 0) -eq $phases.Count) "blocker" "Final intake pack must cover at least 12 final evidence phases.")) | Out-Null
$items.Add((New-OwnerValidationItem "all-phases-blocked" ($blockedPhases.Count -eq $phases.Count -and [int](Get-PropertyOrDefault -Object $record -Name "blockedPhaseCount" -DefaultValue 0) -eq $phases.Count) "blocker" "All phases must remain blocked by default.")) | Out-Null
$items.Add((New-OwnerValidationItem "proof-ready-zero" ($proofReadyPhases.Count -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "proofReadyPhaseCount" -DefaultValue -1) -eq 0) "blocker" "No phase can be proof-ready without Owner real evidence.")) | Out-Null
$items.Add((New-OwnerValidationItem "field-count" ([int](Get-PropertyOrDefault -Object $record -Name "requiredEvidenceFieldCount" -DefaultValue 0) -ge 110) "blocker" "Final intake pack should expose a broad real evidence field map.")) | Out-Null
$items.Add((New-OwnerValidationItem "validator-count" ([int](Get-PropertyOrDefault -Object $record -Name "validatorScriptCount" -DefaultValue 0) -ge 25) "blocker" "Final intake pack should map to the strict validator chain.")) | Out-Null
$items.Add((New-OwnerValidationItem "owner-action-count" ([int](Get-PropertyOrDefault -Object $record -Name "ownerActionCount" -DefaultValue 0) -ge 24) "blocker" "Each final phase should list concrete Owner actions.")) | Out-Null
$items.Add((New-OwnerValidationItem "forbidden-substitute-count" ([int](Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteCount" -DefaultValue 0) -ge 120) "blocker" "Every phase must carry forbidden substitute markers.")) | Out-Null
$items.Add((New-OwnerValidationItem "no-unsafe-fields" ($unsafeFields.Count -eq 0) "blocker" "Fields must reject templates, drafts, local feeds, direct nupkg, queued workflows, dashboards, and dry-runs.")) | Out-Null
$items.Add((New-OwnerValidationItem "no-unsafe-phases" ($unsafePhases.Count -eq 0) "blocker" "Phases must not publish, use tokens, promote proof, or close release issue.")) | Out-Null
$items.Add((New-OwnerValidationItem "boundaries-complete" ($boundaryFailures.Count -eq 0) "blocker" "Every phase must state non-proof and non-publish boundaries.")) | Out-Null
$items.Add((New-OwnerValidationItem "pack-no-side-effects" ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) "blocker" "Pack must not publish, use tokens, promote proof, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "owner-public-publish-execution-final-intake-pack-ready-non-proof" } else { "invalid-owner-public-publish-execution-final-intake-pack" }

$validation = [pscustomobject]@{
  recordKind = "owner-public-publish-execution-final-intake-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $state
  intakeState = [string](Get-PropertyOrDefault -Object $record -Name "intakeState" -DefaultValue "")
  phaseCount = $phases.Count
  blockedPhaseCount = $blockedPhases.Count
  proofReadyPhaseCount = $proofReadyPhases.Count
  requiredEvidenceFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "requiredEvidenceFieldCount" -DefaultValue 0)
  validatorScriptCount = [int](Get-PropertyOrDefault -Object $record -Name "validatorScriptCount" -DefaultValue 0)
  ownerActionCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerActionCount" -DefaultValue 0)
  forbiddenSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  finalCloseCandidateReady = $false
  ownerActionRequired = $true
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundaryFailures = @($boundaryFailures.ToArray())
  safetyBoundary = "Owner public publish execution final intake pack validation is non-proof boundary validation only; it does not execute publish commands, does not download packages, does not run clean consumer smoke, and cannot close release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-public-publish-execution-final-intake-pack-validation.json"
$mdPath = Join-Path $OutputRoot "owner-public-publish-execution-final-intake-pack-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 14)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Public Publish Execution Final Intake Pack Validation",
  "",
  "- validationState: ``$state``",
  "- phaseCount: ``$($validation.phaseCount)``",
  "- blockedPhaseCount: ``$($validation.blockedPhaseCount)``",
  "- proofReadyPhaseCount: ``$($validation.proofReadyPhaseCount)``",
  "- requiredEvidenceFieldCount: ``$($validation.requiredEvidenceFieldCount)``",
  "- validatorScriptCount: ``$($validation.validatorScriptCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- canPublishPublicly: ``False``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "## Boundary",
  "",
  $validation.safetyBoundary
)

Write-Host "OwnerPublicPublishExecutionFinalIntakePackValidationState=$state FailedBlockers=$($failedBlockers.Count) Phases=$($validation.phaseCount)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Owner public publish execution final intake pack validation failed."
}
