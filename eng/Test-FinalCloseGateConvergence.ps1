[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-close-gate-convergence.json",
  [string]$OutputRoot = "artifacts\final-release",
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function ConvertTo-Array {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function Get-LaneById {
  param([AllowNull()][object[]]$Lanes, [string]$Id)
  return @($Lanes | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq $Id } | Select-Object -First 1)[0]
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  throw "Final close gate convergence not found: $InputPath"
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @((Get-PropertyOrDefault -Object $record -Name "gateLanes" -DefaultValue @()))
$blockedLanes = @($lanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$forbiddenSubstituteMarkers = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteMarkers" -DefaultValue @())) | ForEach-Object { [string]$_ })
$rejectedCloseSubstitutes = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "rejectedCloseSubstitutes" -DefaultValue @())) | ForEach-Object { [string]$_ })
$strictValidatorSourceArtifacts = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "strictValidatorSourceArtifacts" -DefaultValue @())) | ForEach-Object { [string]$_ })
$acceptedProofSources = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "finalCloseAcceptedProofSources" -DefaultValue @())) | ForEach-Object { [string]$_ })
$finalCloseProofAdmissionRequiredFields = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "finalCloseProofAdmissionRequiredFields" -DefaultValue @())) | ForEach-Object { [string]$_ })
$rejectedNonProofStates = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "rejectedNonProofStates" -DefaultValue @())) | ForEach-Object { [string]$_ })
$acceptedProofAdmissionContract = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "acceptedProofAdmissionContract" -DefaultValue @())))
$acceptedProofAdmissionContractLaneIds = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "acceptedProofAdmissionContractLaneIds" -DefaultValue @())) | ForEach-Object { [string]$_ })
$summary = Get-PropertyOrDefault -Object $record -Name "summary" -DefaultValue $null
$remoteProofIds = @("github-actions-run-proof", "owner-public-publish-result", "public-package-download-proof", "post-publish-clean-consumer-proof")
$remoteProofLanes = @($remoteProofIds | ForEach-Object { Get-LaneById -Lanes $lanes -Id $_ })
$remoteProofLanesPresent = @($remoteProofLanes | Where-Object { $null -ne $_ }).Count -eq $remoteProofIds.Count
$remoteProofLanesBlockClose = $remoteProofLanesPresent -and @($remoteProofLanes | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $true) }).Count -eq 0
$dualPackageLaneIds = @("dual-package-nuget-small-bridge-core", "dual-package-github-packages-bridge")
$dualPackageLanes = @($dualPackageLaneIds | ForEach-Object { Get-LaneById -Lanes $lanes -Id $_ })
$dualPackageLanesPresent = @($dualPackageLanes | Where-Object { $null -ne $_ }).Count -eq $dualPackageLaneIds.Count
$dualPackageLanesBlockClose = $dualPackageLanesPresent -and @($dualPackageLanes | Where-Object {
  [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $true) -or
  -not [bool](Get-PropertyOrDefault -Object $_ -Name "ownerActionRequired" -DefaultValue $false) -or
  -not [bool](Get-PropertyOrDefault -Object $_ -Name "externalProofRequired" -DefaultValue $false) -or
  -not [bool](Get-PropertyOrDefault -Object $_ -Name "postPublishProofRequired" -DefaultValue $false) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "acceptsSubstituteProof" -DefaultValue $true) -or
  [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "nextOwnerAction" -DefaultValue "")) -or
  [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "externalProofMissingReason" -DefaultValue "")) -or
  [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "postPublishProofMissingReason" -DefaultValue ""))
}).Count -eq 0
$postPublishRemoteLane = Get-LaneById -Lanes $lanes -Id "post-publish-clean-consumer-proof"
$postPublishRemoteLaneRequiresCandidate = $null -ne $postPublishRemoteLane -and
  [bool](Get-PropertyOrDefault -Object $postPublishRemoteLane -Name "requireProofReady" -DefaultValue $false) -and
  [string](Get-PropertyOrDefault -Object $postPublishRemoteLane -Name "proofReadyProperty" -DefaultValue "") -eq "proofCandidateReady" -and
  -not [bool](Get-PropertyOrDefault -Object $postPublishRemoteLane -Name "proofReady" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $postPublishRemoteLane -Name "ready" -DefaultValue $true)
$requiredAdmissionLaneIds = @("github-actions-run-proof", "owner-public-publish-result", "public-package-download-proof", "post-publish-clean-consumer-proof", "release-issue-close-record-strict-validation")
$requiredAdmissionFields = @("publicPackageSourceUrl", "publicPackageDownloadUrl", "managedNupkgSha256", "runtimeNupkgSha256", "externalCleanConsumerProjectIdentity", "smokeCommandRuntimePackageKey", "hostCudaVersion", "hostTensorRtVersion", "hostCudnnVersion", "stdoutSha256", "stderrSha256", "mergedTranscriptSha256", "githubRunId", "githubHeadSha", "githubLogSha256", "githubArtifactSha256", "ownerReviewer", "ownerAuthorizationLink", "rollbackReview", "finalCloseDecision")
$requiredRejectedStates = @("template-only", "candidate-only", "draft-rich-but-not-proof", "draft-blocked-by-cuda-driver", "not-requested", "validation-ready-without-proof-candidate", "dashboard-only", "runbook-only", "local-feed-only", "project-reference-only")
$missingAdmissionLaneIds = @($requiredAdmissionLaneIds | Where-Object { $acceptedProofAdmissionContractLaneIds -notcontains $_ })
$missingAdmissionFields = @($requiredAdmissionFields | Where-Object { $finalCloseProofAdmissionRequiredFields -notcontains $_ })
$missingRejectedStates = @($requiredRejectedStates | Where-Object { $rejectedNonProofStates -notcontains $_ })
$invalidAdmissionContracts = @($acceptedProofAdmissionContract | Where-Object {
  $laneId = [string](Get-PropertyOrDefault -Object $_ -Name "laneId" -DefaultValue "")
  $acceptedState = [string](Get-PropertyOrDefault -Object $_ -Name "requiredAcceptedState" -DefaultValue "")
  $strictValidator = [string](Get-PropertyOrDefault -Object $_ -Name "strictValidator" -DefaultValue "")
  $fields = @((ConvertTo-Array (Get-PropertyOrDefault -Object $_ -Name "requiredEvidenceFields" -DefaultValue @())) | ForEach-Object { [string]$_ })
  $states = @((ConvertTo-Array (Get-PropertyOrDefault -Object $_ -Name "rejectsNonProofStates" -DefaultValue @())) | ForEach-Object { [string]$_ })
  $requiredAdmissionLaneIds -notcontains $laneId -or
    [string]::IsNullOrWhiteSpace($acceptedState) -or
    $acceptedState -notlike "accepted-*" -or
    [string]::IsNullOrWhiteSpace($strictValidator) -or
    -not [bool](Get-PropertyOrDefault -Object $_ -Name "acceptedOnlyAfterStrictValidator" -DefaultValue $false) -or
    [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $true) -or
    $fields.Count -lt 5 -or
    @($requiredRejectedStates | Where-Object { $states -notcontains $_ }).Count -gt 0
})

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-close-gate-convergence") -Severity "blocker" -Detail "recordKind must be final-close-gate-convergence.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "convergenceState" -DefaultValue "") -eq "blocked-final-close-gate-owner-proof-required") -Severity "blocker" -Detail "Convergence must stay blocked until all real close proof lanes pass.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-shape" -Passed ($lanes.Count -ge 16) -Severity "blocker" -Detail "Convergence must expose all final close gate lanes, including remote proof, dual-package route, owner import, and strict validator bridge lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-lanes-required" -Passed ($blockedLanes.Count -eq 0) -Severity "action-required" -Detail "Owner must complete all final close gate lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Convergence must not publish, approve, promote proof, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "remote-proof-lanes-present" -Passed $remoteProofLanesPresent -Severity "blocker" -Detail "Final close convergence must include GitHub Actions, owner public publish, public package download, and post-publish clean consumer proof dependency lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "remote-proof-lanes-block-close" -Passed $remoteProofLanesBlockClose -Severity "blocker" -Detail "Remote/public proof dependency lanes must block final close until real proof is imported.")) | Out-Null
$items.Add((New-ValidationItem -Id "dual-package-lanes-present" -Passed ($dualPackageLanesPresent -and [int](Get-PropertyOrDefault -Object $record -Name "dualPackageRouteCount" -DefaultValue 0) -eq 2) -Severity "blocker" -Detail "Final close convergence must include NuGet managed and GitHub Packages bridge-only lanes, with NVIDIA dependencies remaining host-installed.")) | Out-Null
$items.Add((New-ValidationItem -Id "dual-package-lanes-block-close" -Passed $dualPackageLanesBlockClose -Severity "blocker" -Detail "Dual-package route lanes must block final close until Owner authorization, external proof, and post-publish proof are imported; substitute proof is not accepted.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-proof-lane-requires-proof-candidate-ready" -Passed $postPublishRemoteLaneRequiresCandidate -Severity "blocker" -Detail "Final close post-publish lane must require proofCandidateReady=true, not validation-ready alone.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-validator-sources" -Passed ($strictValidatorSourceArtifacts -contains "artifacts/final-release/real-external-proof-record-import-validator-validation.json" -and $strictValidatorSourceArtifacts -contains "artifacts/final-release/release-close-real-proof-import-bridge-validation.json" -and $acceptedProofSources -contains "strict-validator-accepted-real-external-proof-record") -Severity "blocker" -Detail "Final close must name strict validator accepted real proof as the only promotable source.")) | Out-Null
$items.Add((New-ValidationItem -Id "accepted-proof-admission-contract" -Passed ($missingAdmissionLaneIds.Count -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "acceptedProofAdmissionContractCount" -DefaultValue 0) -ge 5 -and $acceptedProofAdmissionContract.Count -ge 5 -and $invalidAdmissionContracts.Count -eq 0) -Severity "blocker" -Detail "Final close must expose accepted-proof admission lanes with accepted states, strict validators, required fields, non-proof rejection lists, and no direct runtime promotion.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-proof-field-contract" -Passed ($missingAdmissionFields.Count -eq 0) -Severity "blocker" -Detail "Final close admission must require public URLs, package hashes, clean consumer identity, runtime key, host CUDA/TensorRT/cuDNN metadata, stdout/stderr/transcript hashes, GitHub run/log/artifact hashes, owner authorization, rollback review, and final close decision.")) | Out-Null
$items.Add((New-ValidationItem -Id "rejected-non-proof-states" -Passed ($missingRejectedStates.Count -eq 0) -Severity "blocker" -Detail "Final close must explicitly reject template, candidate, draft-rich, driver-blocked, not-requested, dashboard, runbook, local-feed, and ProjectReference states.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-bridge-input-only" -Passed ([bool](Get-PropertyOrDefault -Object $summary -Name "candidateInputOnly" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $summary -Name "bridgeInputOnly" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $summary -Name "strictValidatorRequired" -DefaultValue $false)) -Severity "blocker" -Detail "Candidate and bridge records must remain strict-validator input only.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitute-markers" -Passed ((@("candidate","draft","dashboard","dry-run","local feed","ProjectReference","direct .nupkg","template","build-only","blocked-by-cuda-driver") | Where-Object { $forbiddenSubstituteMarkers -notcontains $_ -or $rejectedCloseSubstitutes -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Final close must reject candidate, draft, dashboard, dry-run, local feed, ProjectReference, direct nupkg, template, build-only, and blocked-by-driver substitutes.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-boundary-fields" -Passed (@($lanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "strictValidatorRequired" -DefaultValue $false) -or -not [bool](Get-PropertyOrDefault -Object $_ -Name "strictValidatorInputOnly" -DefaultValue $false) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "blockedReason" -DefaultValue "")) }).Count -eq 0) -Severity "blocker" -Detail "Every lane must carry strict-validator and blocked reason boundary fields.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-final-close-gate-convergence" } else { "blocked-final-close-gate-owner-proof-required" }

$validation = [pscustomobject]@{
  recordKind = "final-close-gate-convergence-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $InputPath
  validationState = $validationState
  laneCount = $lanes.Count
  blockedLaneCount = $blockedLanes.Count
  dualPackageRouteCount = [int](Get-PropertyOrDefault -Object $record -Name "dualPackageRouteCount" -DefaultValue 0)
  dualPackageBlockedLaneCount = @($dualPackageLanes | Where-Object { $null -ne $_ -and -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) }).Count
  dualPackageAcceptsSubstituteProof = [bool](Get-PropertyOrDefault -Object $record -Name "dualPackageAcceptsSubstituteProof" -DefaultValue $true)
  acceptedProofAdmissionContractCount = [int](Get-PropertyOrDefault -Object $record -Name "acceptedProofAdmissionContractCount" -DefaultValue 0)
  finalCloseProofAdmissionRequiredFieldCount = $finalCloseProofAdmissionRequiredFields.Count
  rejectedNonProofStateCount = $rejectedNonProofStates.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  validationItems = @($items.ToArray())
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  boundary = "Validation checks final close convergence shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-close-gate-convergence-validation.json"
$markdownPath = Join-Path $OutputRoot "final-close-gate-convergence-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Final Close Gate Convergence Validation",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| validationState | ``$($validation.validationState)`` |",
  "| laneCount | ``$($validation.laneCount)`` |",
  "| blockedLaneCount | ``$($validation.blockedLaneCount)`` |",
  "| failedBlockerCount | ``$($validation.failedBlockerCount)`` |",
  "| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |",
  "| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |",
  "",
  "## Boundary",
  "",
  $validation.boundary
)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final close gate convergence validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) Lanes=$($validation.laneCount) Blocked=$($validation.blockedLaneCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Final close gate convergence validation failed with $($failedBlockers.Count) blocker(s)."
}
