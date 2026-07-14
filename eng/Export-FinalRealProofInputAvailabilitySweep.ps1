[CmdletBinding()]
param(
  [string]$OwnerInputRoot = "artifacts\final-release",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$OwnerInputRoot = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $OwnerInputRoot

function New-PhaseSpec {
  param(
    [string]$Id,
    [string]$Title,
    [string[]]$AcceptedOwnerInputPaths,
    [string[]]$RequiredFields,
    [string[]]$ValidationPaths,
    [string[]]$OwnerActions
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    acceptedOwnerInputPaths = @($AcceptedOwnerInputPaths)
    requiredFields = @($RequiredFields)
    validationPaths = @($ValidationPaths)
    ownerActions = @($OwnerActions)
  }
}

function Get-ValidationState {
  param([string]$RelativePath)
  $record = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $RelativePath
  return [string](Get-PropertyOrDefault -Object $record -Name "validationState" -DefaultValue "missing-$([System.IO.Path]::GetFileNameWithoutExtension($RelativePath))-validation")
}

function Test-AcceptedValidatorState {
  param([AllowNull()][string]$State)
  if ([string]::IsNullOrWhiteSpace($State)) { return $false }

  foreach ($blocked in @("missing", "blocked", "invalid", "failed", "required", "template", "draft", "dry-run", "dryrun", "non-proof", "no-proof", "owner-action")) {
    if ($State.IndexOf($blocked, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
      return $false
    }
  }

  foreach ($accepted in @("accepted", "ready", "passed", "proof-ready", "real-proof")) {
    if ($State.IndexOf($accepted, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
      return $true
    }
  }

  return $false
}

function Test-ForbiddenInputFileName {
  param([string]$Path)
  $name = [System.IO.Path]::GetFileName($Path)
  return $name.EndsWith(".template.json", [System.StringComparison]::OrdinalIgnoreCase) -or
    $name.EndsWith(".example.json", [System.StringComparison]::OrdinalIgnoreCase) -or
    $name.EndsWith(".draft.json", [System.StringComparison]::OrdinalIgnoreCase) -or
    $name.EndsWith(".misuse.json", [System.StringComparison]::OrdinalIgnoreCase) -or
    $name.EndsWith(".ready.json", [System.StringComparison]::OrdinalIgnoreCase) -or
    $name.EndsWith(".debug-ready.json", [System.StringComparison]::OrdinalIgnoreCase) -or
    $name.EndsWith("-validation.json", [System.StringComparison]::OrdinalIgnoreCase) -or
    $name.IndexOf("dashboard", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $name.IndexOf("audit", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $name.IndexOf("bundle", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $name.IndexOf("runbook", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $name.IndexOf("checklist", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $name.IndexOf("dry-run", [System.StringComparison]::OrdinalIgnoreCase) -ge 0
}

function Get-ForbiddenKindsFromText {
  param([AllowNull()][string]$Text)
  $kinds = New-Object System.Collections.Generic.List[string]
  $checks = [ordered]@{
    "local feed" = @("local feed", "RestoreSources local", "--source ./", "artifacts/package-managed-dry-run")
    "ProjectReference" = @("ProjectReference", "../src", "..\src")
    "direct .nupkg" = @("direct .nupkg", ".nupkg")
    "queued workflow" = @("queued workflow", "queued")
    "missing runner" = @("missing runner", "runner missing")
    "dry-run" = @("dry-run", "dry run", "package-managed-dry-run")
    "template" = @("<owner-fill", "<external-", ".template", "template only")
    "dashboard" = @("dashboard")
    "audit" = @("audit")
    "bundle" = @("bundle")
  }

  foreach ($entry in $checks.GetEnumerator()) {
    foreach ($marker in @($entry.Value)) {
      if (-not [string]::IsNullOrWhiteSpace($Text) -and $Text.IndexOf([string]$marker, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
        if (-not $kinds.Contains([string]$entry.Key)) {
          $kinds.Add([string]$entry.Key) | Out-Null
        }
      }
    }
  }

  return @($kinds.ToArray())
}

function Test-OwnerInputJson {
  param(
    [string]$RelativePath,
    [string]$PhaseId,
    [string[]]$RequiredFields
  )

  $path = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  $issues = New-Object System.Collections.Generic.List[string]
  $jsonText = [System.IO.File]::ReadAllText($path, (Get-OwnerUtf8Encoding))
  $record = $null
  try {
    $record = $jsonText | ConvertFrom-Json
  }
  catch {
    $issues.Add("invalid-json") | Out-Null
  }

  if (Test-ForbiddenInputFileName -Path $path) {
    $issues.Add("forbidden-file-name") | Out-Null
  }

  $forbiddenKinds = @(Get-ForbiddenKindsFromText -Text $jsonText)
  if ($forbiddenKinds.Count -gt 0) {
    $issues.Add(("forbidden-substitute-text:" + ($forbiddenKinds -join ","))) | Out-Null
  }

  if ($null -ne $record) {
    $ownerEvidenceKind = [string](Get-PropertyOrDefault -Object $record -Name "ownerEvidenceKind" -DefaultValue "")
    $isRealOwnerProof = [bool](Get-PropertyOrDefault -Object $record -Name "isRealOwnerProof" -DefaultValue $false)
    $noLocalSubstitute = [bool](Get-PropertyOrDefault -Object $record -Name "noLocalSubstituteConfirmation" -DefaultValue (Get-PropertyOrDefault -Object $record -Name "noLocalSubstitutes" -DefaultValue $false))
    $phase = [string](Get-PropertyOrDefault -Object $record -Name "phaseId" -DefaultValue $PhaseId)
    $isTemplate = [bool](Get-PropertyOrDefault -Object $record -Name "isTemplate" -DefaultValue $false)

    if ($ownerEvidenceKind -ne "real-owner-public-release-proof" -and -not $isRealOwnerProof) {
      $issues.Add("missing-real-owner-proof-kind") | Out-Null
    }

    if (-not $noLocalSubstitute) {
      $issues.Add("missing-no-local-substitute-confirmation") | Out-Null
    }

    if ($phase -ne $PhaseId) {
      $issues.Add("phase-id-mismatch") | Out-Null
    }

    if ($isTemplate) {
      $issues.Add("is-template-true") | Out-Null
    }

    foreach ($field in @($RequiredFields)) {
      $value = Get-PropertyOrDefault -Object $record -Name $field -DefaultValue $null
      if (Test-OwnerPlaceholder -Value $value) {
        $issues.Add("missing-or-placeholder-field:$field") | Out-Null
      }
    }
  }

  $accepted = $issues.Count -eq 0
  return [pscustomobject]@{
    path = $RelativePath
    exists = $true
    accepted = $accepted
    invalid = -not $accepted
    issues = @($issues.ToArray())
    forbiddenSubstituteKinds = @($forbiddenKinds)
  }
}

$phaseSpecs = @(
  New-PhaseSpec -Id "claim-boundary-preflight" -Title "Claim boundary and documentation preflight" -AcceptedOwnerInputPaths @("artifacts/final-release/claim-boundary-preflight.owner.json", "artifacts/final-release/claim-boundary-preflight.real.json") -RequiredFields @("releaseVersion", "packageIds", "runtimePackageKeys", "knownLimitationsUrl", "nonProofBoundaryAcknowledgement", "docsHash", "readmeHash", "nugetMetadataReview", "ownerReviewer") -ValidationPaths @("artifacts/final-release/public-publish-final-owner-execution-pack-validation.json", "artifacts/final-release/release-docs-and-nuget-metadata-audit-validation.json") -OwnerActions @("Owner reviews public claims and metadata.", "Owner confirms no local-only artifact is presented as public proof.")
  New-PhaseSpec -Id "owner-public-publish-execution" -Title "Owner public publish execution result" -AcceptedOwnerInputPaths @("artifacts/final-release/owner-public-publish-execution-result.owner.json", "artifacts/final-release/public-publish-result-owner-input.real.json") -RequiredFields @("selectedChannel", "managedPackageId", "managedPackageVersion", "runtimePackageId", "runtimePackageVersion", "runtimePackageKey", "publicPackagePageUrl", "publicPackageDownloadUrl", "publishStartedAtUtc", "publishCompletedAtUtc", "publishTranscriptSha256", "ownerReviewer") -ValidationPaths @("artifacts/final-release/public-publish-result-import-validation.json", "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json") -OwnerActions @("Owner executes public publish outside automation.", "Owner imports the real public publish result.")
  New-PhaseSpec -Id "github-actions-run-proof" -Title "GitHub Actions and CI proof import" -AcceptedOwnerInputPaths @("artifacts/final-release/github-actions-run-evidence.owner.json", "artifacts/final-release/github-actions-run-proof.real.json") -RequiredFields @("workflowName", "runId", "runUrl", "headSha", "branch", "conclusion", "createdAtUtc", "completedAtUtc", "logSha256", "artifactManifestSha256", "releaseQualityGateSummarySha256") -ValidationPaths @("artifacts/final-release/github-publish-and-ci-status-snapshot-validation.json", "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.json") -OwnerActions @("Record completed GitHub Actions run without dispatching publish.", "Cross-check run SHA and logs.")
  New-PhaseSpec -Id "public-managed-package-download-proof" -Title "Public managed package download proof" -AcceptedOwnerInputPaths @("artifacts/final-release/public-managed-package-download-proof.owner.json", "artifacts/final-release/public-package-download-proof-input.real.json") -RequiredFields @("managedPublicPackageUrl", "managedPublicDownloadUrl", "managedPackageId", "managedPackageVersion", "managedNupkgPath", "managedNupkgSha256", "managedNupkgSizeBytes", "downloadedAtUtc", "downloadTranscriptSha256") -ValidationPaths @("artifacts/final-release/public-package-download-proof-input-validation.json", "artifacts/final-release/public-package-download-proof-candidate-validation.json") -OwnerActions @("Download managed package from public channel.", "Record public URL, path, size and SHA256.")
  New-PhaseSpec -Id "public-runtime-package-download-proof" -Title "Public runtime package download proof" -AcceptedOwnerInputPaths @("artifacts/final-release/public-runtime-package-download-proof.owner.json", "artifacts/final-release/public-runtime-package-download-proof.real.json") -RequiredFields @("runtimePublicPackageUrl", "runtimePublicDownloadUrl", "runtimePackageId", "runtimePackageVersion", "runtimePackageKey", "runtimeNupkgPath", "runtimeNupkgSha256", "runtimeNupkgSizeBytes", "downloadedAtUtc", "downloadTranscriptSha256") -ValidationPaths @("artifacts/final-release/public-package-download-proof-input-validation.json", "artifacts/final-release/public-package-download-proof-candidate-validation.json") -OwnerActions @("Download runtime package or release asset from public channel.", "Record runtime package key, SHA256 and public source.")
  New-PhaseSpec -Id "repository-external-clean-consumer" -Title "Repository-external clean consumer proof" -AcceptedOwnerInputPaths @("artifacts/final-release/repository-external-clean-consumer-proof.owner.json", "artifacts/final-release/external-clean-consumer-execution-result.real.json") -RequiredFields @("externalWorkspaceRoot", "consumerProjectPath", "restoreSource", "restoreCommand", "buildCommand", "smokeCommand", "restoreLogSha256", "buildLogSha256", "smokeLogSha256", "hostMetadataSha256", "noLocalSubstituteConfirmation") -ValidationPaths @("artifacts/final-release/external-clean-consumer-execution-result-validation.json", "artifacts/final-release/clean-consumer-runtime-proof-cross-check-gate-validation.json") -OwnerActions @("Run clean consumer outside this repository.", "Use public package sources only.")
  New-PhaseSpec -Id "post-publish-clean-consumer-proof" -Title "Post-publish clean consumer proof" -AcceptedOwnerInputPaths @("artifacts/final-release/post-publish-clean-consumer-proof.owner.json", "artifacts/final-release/post-publish-clean-consumer-proof-result.real.json") -RequiredFields @("postPublishPublicPackageUrl", "postPublishPackageSha256", "postPublishConsumerRoot", "postPublishRestoreLogSha256", "postPublishBuildLogSha256", "postPublishSmokeLogSha256", "stdoutSummary", "stderrSummary", "hostMetadata", "ownerReviewedAtUtc") -ValidationPaths @("artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json", "artifacts/final-release/post-publish-clean-consumer-real-proof-from-owner-result-validation.json") -OwnerActions @("Run post-publication clean consumer validation.", "Link public package hashes to clean consumer logs.")
  New-PhaseSpec -Id "post-publish-user-verification" -Title "Post-publish user verification" -AcceptedOwnerInputPaths @("artifacts/final-release/post-publish-user-verification.owner.json", "artifacts/final-release/post-publish-verification-owner-input.real.json") -RequiredFields @("managedPackageId", "managedPackageVersion", "runtimePackageId", "runtimePackageVersion", "consumerProjectIdentity", "smokeCommand", "stdoutSummary", "stderrSummary", "allLogSha256Matches", "ownerReviewer") -ValidationPaths @("artifacts/final-release/post-publish-user-verification-pack-validation.json", "artifacts/final-release/post-publish-verification-record-validation.json") -OwnerActions @("Record end-user style verification after public publication.", "Confirm stdout/stderr summaries and log hashes.")
  New-PhaseSpec -Id "strict-close-convergence" -Title "Strict close convergence" -AcceptedOwnerInputPaths @("artifacts/final-release/strict-close-convergence.owner.json", "artifacts/final-release/strict-close-convergence.real.json") -RequiredFields @("strictCloseReadyState", "finalClosureBridgeState", "acceptedProofLaneCount", "blockedProofLaneCount", "classificationAuditState", "releaseQualityGateState", "evidenceBundleSha256", "classificationAuditSha256") -ValidationPaths @("artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json", "artifacts/final-release/final-public-release-closure-bridge-validation.json", "artifacts/final-release/release-close-final-candidate-audit-pack-validation.json") -OwnerActions @("Refresh strict close dashboards after real proof inputs are accepted.", "Keep close candidate blocked until every lane is accepted.")
  New-PhaseSpec -Id "release-issue-close-owner-decision" -Title "Release Issue close Owner decision" -AcceptedOwnerInputPaths @("artifacts/final-release/release-issue-close-owner-decision.owner.json", "artifacts/final-release/release-issue-close-owner-decision-input.real.json") -RequiredFields @("releaseIssueUrl", "releaseIssueNumber", "ownerCloseDecision", "ownerName", "ownerEmail", "ownerDecisionTimestampUtc", "approvedPublicPackageProofHash", "approvedPostPublishProofHash", "rollbackPlan", "knownLimitationsAcknowledgement") -ValidationPaths @("artifacts/final-release/release-issue-close-owner-decision-input-validation.json", "artifacts/final-release/final-release-close-approval-real-input-from-owner-result-validation.json") -OwnerActions @("Owner records close decision only after real post-publish evidence is accepted.", "Map decision to bundle hash, proof hashes and rollback plan.")
  New-PhaseSpec -Id "release-issue-close-record" -Title "Release Issue close record" -AcceptedOwnerInputPaths @("artifacts/final-release/release-issue-close-record.owner.json", "artifacts/final-release/release-issue-close-record.real.json") -RequiredFields @("closeRecordId", "releaseIssueUrl", "ownerCloseDecision", "releaseEvidenceBundleSha256", "classificationAuditSha256", "postPublishProofSha256", "rollbackDecision", "staleClaimAuditState") -ValidationPaths @("artifacts/final-release/release-issue-close-record-validation.json", "artifacts/final-release/release-issue-close-record-real-input-map-validation.json") -OwnerActions @("Import final close record after Owner decision.", "Verify close record hashes against current bundle and audit.")
  New-PhaseSpec -Id "final-bundle-classification-lock" -Title "Final bundle and classification audit lock" -AcceptedOwnerInputPaths @("artifacts/final-release/final-bundle-classification-lock.owner.json", "artifacts/final-release/final-bundle-classification-lock.real.json") -RequiredFields @("releaseEvidenceBundleSha256", "classificationAuditSha256", "requiredNonProofItemCount", "nonSubstituteProofKindCount", "auditedNonProofItemCount", "findingCount", "bundleState", "ownerActionStatus", "canCloseReleaseIssue") -ValidationPaths @("artifacts/final-release/release-evidence-bundle.json", "artifacts/final-release/release-evidence-classification-audit.json") -OwnerActions @("Refresh evidence bundle after every real proof import.", "Confirm classification audit still blocks templates and local substitutes.")
)

$allFiles = @()
if (Test-Path -LiteralPath $OwnerInputRoot -PathType Container) {
  $allFiles = @(Get-ChildItem -LiteralPath $OwnerInputRoot -File -Filter "*.json")
}

$forbiddenFiles = New-Object System.Collections.Generic.List[object]
$forbiddenKinds = New-Object System.Collections.Generic.List[string]
foreach ($file in $allFiles) {
  $text = ""
  try { $text = [System.IO.File]::ReadAllText($file.FullName, (Get-OwnerUtf8Encoding)) } catch { $text = "" }
  $fileKinds = @(Get-ForbiddenKindsFromText -Text ($file.Name + " " + $text))
  if ((Test-ForbiddenInputFileName -Path $file.FullName) -or $fileKinds.Count -gt 0) {
    foreach ($kind in $fileKinds) {
      if (-not $forbiddenKinds.Contains($kind)) {
        $forbiddenKinds.Add($kind) | Out-Null
      }
    }

    if ($forbiddenFiles.Count -lt 120) {
      $forbiddenFiles.Add([pscustomobject]@{
          path = ("artifacts/final-release/" + $file.Name)
          rejected = $true
          substituteKinds = @($fileKinds)
        }) | Out-Null
    }
  }
}

foreach ($requiredKind in @("local feed", "ProjectReference", "direct .nupkg", "queued workflow", "missing runner", "dry-run", "template", "dashboard", "audit", "bundle")) {
  if (-not $forbiddenKinds.Contains($requiredKind)) {
    $forbiddenKinds.Add($requiredKind) | Out-Null
  }
}

foreach ($canonicalMisuseFile in @(
    "public-package-download-proof-input.misuse.json",
    "release-issue-close-owner-decision-input.misuse.json",
    "post-publish-proof-input.misuse.json",
    "clean-external-consumer-smoke-input.misuse.json",
    "release-candidate-public-proof-final-audit.misuse.json"
  )) {
  $alreadyListed = $false
  foreach ($entry in @($forbiddenFiles.ToArray())) {
    if ([string](Get-PropertyOrDefault -Object $entry -Name "path" -DefaultValue "") -eq "artifacts/final-release/$canonicalMisuseFile") {
      $alreadyListed = $true
      break
    }
  }

  $canonicalPath = Join-Path $OwnerInputRoot $canonicalMisuseFile
  if (-not $alreadyListed -and (Test-Path -LiteralPath $canonicalPath -PathType Leaf)) {
    $canonicalText = [System.IO.File]::ReadAllText($canonicalPath, (Get-OwnerUtf8Encoding))
    $forbiddenFiles.Add([pscustomobject]@{
        path = "artifacts/final-release/$canonicalMisuseFile"
        rejected = $true
        substituteKinds = @(Get-ForbiddenKindsFromText -Text ($canonicalMisuseFile + " " + $canonicalText))
        canonicalMisuseFixture = $true
      }) | Out-Null
  }
}

$phaseRecords = New-Object System.Collections.Generic.List[object]
foreach ($spec in $phaseSpecs) {
  $candidateRecords = New-Object System.Collections.Generic.List[object]
  foreach ($path in @($spec.acceptedOwnerInputPaths)) {
    $candidate = Test-OwnerInputJson -RelativePath $path -PhaseId $spec.id -RequiredFields $spec.requiredFields
    if ($null -ne $candidate) {
      $candidateRecords.Add($candidate) | Out-Null
    }
  }

  $validationStates = @(
    foreach ($path in @($spec.validationPaths)) {
      [pscustomobject]@{
        path = $path
        state = Get-ValidationState -RelativePath $path
      }
    }
  )

  $validatorsAccepted = $true
  foreach ($state in $validationStates) {
    if (-not (Test-AcceptedValidatorState -State $state.state)) {
      $validatorsAccepted = $false
    }
  }

  $availableCount = $candidateRecords.Count
  $acceptedCount = @($candidateRecords | Where-Object { [bool]$_.accepted }).Count
  $invalidCount = @($candidateRecords | Where-Object { [bool]$_.invalid }).Count
  $proofReady = $acceptedCount -gt 0 -and $invalidCount -eq 0 -and $validatorsAccepted
  $phaseState = if ($proofReady) {
    "proof-ready-real-owner-input-and-validators-accepted"
  }
  elseif ($invalidCount -gt 0) {
    "blocked-invalid-real-owner-input"
  }
  elseif ($availableCount -gt 0) {
    "blocked-real-owner-input-validator-chain-not-accepted"
  }
  else {
    "missing-real-owner-input"
  }

  $phaseRecords.Add([pscustomobject]@{
      id = $spec.id
      title = $spec.title
      phaseState = $phaseState
      acceptedOwnerInputPaths = @($spec.acceptedOwnerInputPaths)
      availableRealInputFileCount = $availableCount
      acceptedRealInputFileCount = $acceptedCount
      invalidRealInputFileCount = $invalidCount
      missingRealInput = $availableCount -eq 0
      proofReady = $proofReady
      requiredFields = @($spec.requiredFields)
      requiredFieldCount = @($spec.requiredFields).Count
      validationStates = @($validationStates)
      validatorsAccepted = $validatorsAccepted
      ownerActions = @($spec.ownerActions)
      ownerActionRequired = -not $proofReady
      blocked = -not $proofReady
      ownerInputs = @($candidateRecords.ToArray())
      forbiddenSubstitutes = @("template", "example", "draft", "misuse", "ready fixture", "debug-ready fixture", "validation output", "dashboard", "audit", "bundle", "local feed", "ProjectReference", "direct .nupkg", "queued workflow", "missing runner", "dry-run")
      boundary = "Final real proof input availability phase only; accepts explicit .owner.json/.real.json Owner inputs after strict validators and rejects templates, dashboards, local feeds, ProjectReference, direct .nupkg, queued workflows, missing runners, dry-runs, ready fixtures, and validation outputs."
    }) | Out-Null
}

$phaseArray = @($phaseRecords.ToArray())
$availableRealInputPhaseCount = @($phaseArray | Where-Object { [int]$_.availableRealInputFileCount -gt 0 }).Count
$acceptedRealInputPhaseCount = @($phaseArray | Where-Object { [int]$_.acceptedRealInputFileCount -gt 0 -and [int]$_.invalidRealInputFileCount -eq 0 }).Count
$invalidRealInputPhaseCount = @($phaseArray | Where-Object { [int]$_.invalidRealInputFileCount -gt 0 }).Count
$missingRealInputPhaseCount = @($phaseArray | Where-Object { [bool]$_.missingRealInput }).Count
$proofReadyPhaseCount = @($phaseArray | Where-Object { [bool]$_.proofReady }).Count
$blockedPhaseCount = @($phaseArray | Where-Object { [bool]$_.blocked }).Count
$requiredEvidenceFieldCount = 0
$ownerActionCount = 0
foreach ($phase in $phaseArray) {
  $requiredEvidenceFieldCount += [int]$phase.requiredFieldCount
  $ownerActionCount += @($phase.ownerActions).Count
}

$record = [pscustomobject]@{
  recordKind = "final-real-proof-input-availability-sweep"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  sweepState = if ($proofReadyPhaseCount -eq $phaseArray.Count) { "final-real-proof-inputs-available-and-validator-accepted" } else { "blocked-final-real-proof-inputs-required" }
  ownerInputRoot = $OwnerInputRoot
  phaseCount = $phaseArray.Count
  availableRealInputPhaseCount = $availableRealInputPhaseCount
  acceptedRealInputPhaseCount = $acceptedRealInputPhaseCount
  missingRealInputPhaseCount = $missingRealInputPhaseCount
  invalidRealInputPhaseCount = $invalidRealInputPhaseCount
  proofReadyPhaseCount = $proofReadyPhaseCount
  blockedPhaseCount = $blockedPhaseCount
  requiredEvidenceFieldCount = $requiredEvidenceFieldCount
  ownerActionCount = $ownerActionCount
  forbiddenSubstituteCount = $forbiddenFiles.Count
  forbiddenSubstituteKinds = @($forbiddenKinds.ToArray())
  forbiddenSubstituteFiles = @($forbiddenFiles.ToArray())
  phases = @($phaseArray)
  ownerActionRequired = $proofReadyPhaseCount -lt $phaseArray.Count
  closeCandidateReady = $false
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
  boundary = "This sweep only checks availability and safety of explicit Owner real proof input files. It does not publish packages, does not download packages, does not run clean consumers, does not execute runtime proof, does not approve public release, and cannot close the release issue."
}

$jsonPath = Join-Path $OutputRoot "final-real-proof-input-availability-sweep.json"
$mdPath = Join-Path $OutputRoot "final-real-proof-input-availability-sweep.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 20)

$rows = foreach ($phase in $phaseArray) {
  "| ``$($phase.id)`` | ``$($phase.phaseState)`` | ``$($phase.availableRealInputFileCount)`` | ``$($phase.acceptedRealInputFileCount)`` | ``$($phase.proofReady)`` |"
}

Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Final Real Proof Input Availability Sweep",
  "",
  "- sweepState: ``$($record.sweepState)``",
  "- phaseCount: ``$($record.phaseCount)``",
  "- availableRealInputPhaseCount: ``$($record.availableRealInputPhaseCount)``",
  "- acceptedRealInputPhaseCount: ``$($record.acceptedRealInputPhaseCount)``",
  "- missingRealInputPhaseCount: ``$($record.missingRealInputPhaseCount)``",
  "- invalidRealInputPhaseCount: ``$($record.invalidRealInputPhaseCount)``",
  "- proofReadyPhaseCount: ``$($record.proofReadyPhaseCount)``",
  "- forbiddenSubstituteCount: ``$($record.forbiddenSubstituteCount)``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "| Phase | State | Real Inputs | Accepted Inputs | Proof Ready |",
  "| --- | --- | ---: | ---: | ---: |",
  @($rows),
  "",
  "## Boundary",
  "",
  $record.boundary
)

Write-Host "FinalRealProofInputAvailabilitySweepState=$($record.sweepState) Phases=$($record.phaseCount) Available=$availableRealInputPhaseCount Missing=$missingRealInputPhaseCount ProofReady=$proofReadyPhaseCount"
