[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
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

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function ConvertTo-StringArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value | ForEach-Object { [string]$_ }) }
  return @([string]$Value)
}

function New-ReadinessCheck {
  param(
    [string]$Id,
    [AllowNull()][object]$Record,
    [string]$StateProperty,
    [string]$DefaultState,
    [string]$OwnerAction,
    [string]$Validator
  )

  $blockedReasons = ConvertTo-StringArray (Get-PropertyOrDefault -Object $Record -Name "blockedReasons" -DefaultValue @())
  $blockedReason = [string](Get-PropertyOrDefault -Object $Record -Name "blockedReason" -DefaultValue "")
  if (-not [string]::IsNullOrWhiteSpace($blockedReason) -and $blockedReasons -notcontains $blockedReason) {
    $blockedReasons = @($blockedReasons + $blockedReason)
  }

  [pscustomobject]@{
    id = $Id
    state = [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue $DefaultState)
    ready = $false
    strictValidatorRequired = $true
    strictValidatorInputOnly = $true
    bridgeInputOnly = [bool](Get-PropertyOrDefault -Object $Record -Name "bridgeInputOnly" -DefaultValue $true)
    blockedReason = if ($blockedReasons.Count -gt 0) { [string]$blockedReasons[0] } else { "owner-real-proof-required" }
    blockedReasons = @($blockedReasons)
    sourceOwnerResultRows = @(ConvertTo-StringArray (Get-PropertyOrDefault -Object $Record -Name "sourceOwnerResultRow" -DefaultValue @()))
    resultArtifactPaths = @(ConvertTo-StringArray (Get-PropertyOrDefault -Object $Record -Name "resultArtifactPaths" -DefaultValue @()))
    hashProof = Get-PropertyOrDefault -Object $Record -Name "hashProof" -DefaultValue $null
    ownerAction = $OwnerAction
    validator = $Validator
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
    boundary = "Final owner close readiness check only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$publicPublishDraftValidation = Read-JsonOrNull "artifacts\final-release\public-publish-real-result-record-draft-validation.json"
$cleanConsumerDraftValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-record-draft-validation.json"
$forbiddenScanValidation = Read-JsonOrNull "artifacts\final-release\public-publish-forbidden-substitute-scan-validation.json"
$cleanExternalRunbookValidation = Read-JsonOrNull "artifacts\final-release\clean-external-package-consumer-owner-runbook-validation.json"
$postPublishRunbookValidation = Read-JsonOrNull "artifacts\final-release\post-publish-owner-verification-runbook-validation.json"
$bridgeValidation = Read-JsonOrNull "artifacts\final-release\release-close-real-proof-import-bridge-validation.json"
$ownerResultImportValidation = Read-JsonOrNull "artifacts\final-release\owner-external-proof-execution-result-import-validation.json"
$realExternalImportValidatorValidation = Read-JsonOrNull "artifacts\final-release\real-external-proof-record-import-validator-validation.json"
$candidateFromOwnerResultValidation = Read-JsonOrNull "artifacts\final-release\real-proof-record-candidate-from-owner-result-import-validation.json"
$strictDecisionImportValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-strict-owner-decision-import-validation.json"
$finalCloseGateValidation = Read-JsonOrNull "artifacts\final-release\final-close-gate-convergence-validation.json"
$releaseIssueCloseRecordValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"

$checks = @(
  New-ReadinessCheck -Id "real-public-package-result-ready" -Record $publicPublishDraftValidation -StateProperty "validationState" -DefaultState "missing-public-publish-real-result-record-draft-validation" -OwnerAction "Fill real public package URL/hash/timestamp/transcript/reviewer fields." -Validator "eng\Test-PublicPublishRealResultRecordDraft.ps1 -Strict"
  New-ReadinessCheck -Id "clean-consumer-smoke-ready" -Record $cleanConsumerDraftValidation -StateProperty "validationState" -DefaultState "missing-post-publish-clean-consumer-proof-record-draft-validation" -OwnerAction "Fill repository-external restore/build/smoke logs, hashes, host metadata, and no-substitute confirmations." -Validator "eng\Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict"
  New-ReadinessCheck -Id "clean-external-package-consumer-owner-runbook-ready" -Record $cleanExternalRunbookValidation -StateProperty "validationState" -DefaultState "missing-clean-external-package-consumer-owner-runbook-validation" -OwnerAction "Execute the clean external package consumer runbook outside the repository and import real clean consumer logs, hashes, host metadata, owner review, and non-substitute confirmations." -Validator "eng\Test-CleanExternalPackageConsumerOwnerRunbook.ps1 -Strict"
  New-ReadinessCheck -Id "post-publish-owner-verification-runbook-ready" -Record $postPublishRunbookValidation -StateProperty "validationState" -DefaultState "missing-post-publish-owner-verification-runbook-validation" -OwnerAction "After owner manual publish, execute post-publish verification from the selected public package source and import public package URLs, downloaded nupkg SHA256, logs, and host metadata." -Validator "eng\Test-PostPublishOwnerVerificationRunbook.ps1 -Strict"
  New-ReadinessCheck -Id "forbidden-substitute-scan-ready" -Record $forbiddenScanValidation -StateProperty "validationState" -DefaultState "missing-public-publish-forbidden-substitute-scan-validation" -OwnerAction "Prove no local feed, ProjectReference, direct nupkg, template, dry-run, dashboard, audit pack, or local-only artifact scan was substituted." -Validator "eng\Test-PublicPublishForbiddenSubstituteScan.ps1 -Strict"
  New-ReadinessCheck -Id "owner-external-result-import-ready" -Record $ownerResultImportValidation -StateProperty "validationState" -DefaultState "missing-owner-external-proof-execution-result-import-validation" -OwnerAction "Import owner execution result rows only when passed=true and stdout/stderr/merged transcript/result artifact paths/SHA256 are present." -Validator "eng\Test-OwnerExternalProofExecutionResultImport.ps1 -Strict"
  New-ReadinessCheck -Id "real-external-import-validator-ready" -Record $realExternalImportValidatorValidation -StateProperty "validationState" -DefaultState "missing-real-external-proof-record-import-validator-validation" -OwnerAction "Run strict real external proof import validator; candidate and dashboard artifacts cannot bypass it." -Validator "eng\Test-RealExternalProofRecordImportValidator.ps1 -Strict"
  New-ReadinessCheck -Id "owner-result-candidate-bridge-ready" -Record $candidateFromOwnerResultValidation -StateProperty "validationState" -DefaultState "missing-real-proof-record-candidate-from-owner-result-import-validation" -OwnerAction "Keep owner result candidates as strict-validator input only and propagate sourceOwnerResultRow, resultArtifactPaths, and hashProof." -Validator "eng\Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict"
  New-ReadinessCheck -Id "real-proof-import-bridge-ready" -Record $bridgeValidation -StateProperty "validationState" -DefaultState "missing-release-close-real-proof-import-bridge-validation" -OwnerAction "Re-run real proof import bridge after public result and clean consumer records are filled." -Validator "eng\Test-ReleaseCloseRealProofImportBridge.ps1 -Strict"
  New-ReadinessCheck -Id "strict-owner-decision-ready" -Record $strictDecisionImportValidation -StateProperty "validationState" -DefaultState "missing-release-issue-close-strict-owner-decision-import-validation" -OwnerAction "Import final owner close decision and rollback review after real records pass." -Validator "eng\Test-ReleaseIssueCloseStrictOwnerDecisionImport.ps1 -Strict"
  New-ReadinessCheck -Id "final-close-gate-convergence-ready" -Record $finalCloseGateValidation -StateProperty "validationState" -DefaultState "missing-final-close-gate-convergence-validation" -OwnerAction "Refresh final close gate convergence after all real proof lanes pass." -Validator "eng\Test-FinalCloseGateConvergence.ps1 -Strict"
  New-ReadinessCheck -Id "release-issue-close-record-ready" -Record $releaseIssueCloseRecordValidation -StateProperty "validationState" -DefaultState "missing-release-issue-close-record-validation" -OwnerAction "Run strict release issue close validator only after all real proof and owner decision records pass." -Validator "eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
)

$blockedChecks = @($checks | Where-Object { -not [bool]$_.ready })
$forbiddenSubstituteMarkers = @(
  "candidate",
  "draft",
  "dashboard",
  "dry-run",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "template",
  "build-only",
  "blocked-by-cuda-driver"
)

$record = [pscustomobject]@{
  recordKind = "final-owner-close-readiness-checkpoint"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  checkpointState = "blocked-final-owner-close-readiness-owner-proof-required"
  readinessCheckCount = $checks.Count
  blockedReadinessCheckCount = $blockedChecks.Count
  readyReadinessCheckCount = 0
  readinessChecks = @($checks)
  forbiddenSubstituteMarkers = $forbiddenSubstituteMarkers
  finalCloseAcceptedProofSources = @(
    "strict-validator-accepted-real-external-proof-record",
    "strict-release-issue-close-record-validation"
  )
  summary = [pscustomobject]@{
    strictValidatorRequired = $true
    candidateInputOnly = $true
    bridgeInputOnly = $true
    dashboardsCloseNothing = $true
    publicDocsAndArticlesAreNotProof = $true
  }
  sourceArtifacts = @(
    "artifacts/final-release/public-publish-real-result-record-draft-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.json",
    "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json",
    "artifacts/final-release/post-publish-owner-verification-runbook-validation.json",
    "artifacts/final-release/public-publish-forbidden-substitute-scan-validation.json",
    "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
    "artifacts/final-release/real-external-proof-record-import-validator-validation.json",
    "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json",
    "artifacts/final-release/release-close-real-proof-import-bridge-validation.json",
    "artifacts/final-release/release-issue-close-strict-owner-decision-import-validation.json",
    "artifacts/final-release/final-close-gate-convergence-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json"
  )
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
  boundary = "This checkpoint summarizes final owner close readiness blockers only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-close-readiness-checkpoint.json"
$markdownPath = Join-Path $OutputRoot "final-owner-close-readiness-checkpoint.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Final Owner Close Readiness Checkpoint",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| checkpointState | ``$($record.checkpointState)`` |",
  "| readinessCheckCount | ``$($record.readinessCheckCount)`` |",
  "| blockedReadinessCheckCount | ``$($record.blockedReadinessCheckCount)`` |",
  "| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |",
  "",
  "## Readiness Checks",
  "",
  "| Check | State | Owner Action | Validator |",
  "| --- | --- | --- | --- |"
)

foreach ($check in $checks) {
  $markdown += "| $($check.id) | ``$($check.state)`` | $($check.ownerAction) | ``$($check.validator)`` |"
}

$markdown += @("", "## Boundary", "", $record.boundary)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final owner close readiness checkpoint written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "CheckpointState=$($record.checkpointState) Checks=$($record.readinessCheckCount) Blocked=$($record.blockedReadinessCheckCount)"
