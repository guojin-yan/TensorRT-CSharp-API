[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Detail,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    detail = $Detail
    boundary = $Boundary
  }
}

$summary = Read-JsonOrNull "artifacts\release\release-candidate-freeze-summary.json"
$checklist = Read-JsonOrNull "artifacts\release\release-candidate-freeze-checklist.json"

if ($null -eq $summary) {
  throw "Missing artifacts/release/release-candidate-freeze-summary.json. Run Export-ReleaseCandidateFreezeSummary.ps1 first."
}

$recordKind = [string](Get-PropertyOrDefault -Object $summary -Name "recordKind" -DefaultValue "")
$summaryRuntimePackageKey = [string](Get-PropertyOrDefault -Object $summary -Name "runtimePackageKey" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $summary -Name "performsPublish" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $summary -Name "canCloseReleaseIssue" -DefaultValue $true)
$closeReadinessConsistent = [bool](Get-PropertyOrDefault -Object $summary -Name "closeReadinessConsistent" -DefaultValue $false)
$realExternalRuntimeProofReady = [bool](Get-PropertyOrDefault -Object $summary -Name "realExternalRuntimeProofReady" -DefaultValue $false)
$realPostPublishVerificationReady = [bool](Get-PropertyOrDefault -Object $summary -Name "realPostPublishVerificationReady" -DefaultValue $false)
$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $summary -Name "externalRuntimeProofState" -DefaultValue "missing")
$postPublishVerificationState = [string](Get-PropertyOrDefault -Object $summary -Name "postPublishVerificationState" -DefaultValue "missing")
$externalRuntimeProofBackfillPlanState = [string](Get-PropertyOrDefault -Object $summary -Name "externalRuntimeProofBackfillPlanState" -DefaultValue "missing-external-runtime-proof-backfill-plan")
$externalRuntimeProofBackfillStepCount = [int](Get-PropertyOrDefault -Object $summary -Name "externalRuntimeProofBackfillStepCount" -DefaultValue 0)
$externalRuntimeProofBackfillCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $summary -Name "externalRuntimeProofBackfillCanPromoteRuntimeProof" -DefaultValue $true)
$postPublishVerificationBackfillPlanState = [string](Get-PropertyOrDefault -Object $summary -Name "postPublishVerificationBackfillPlanState" -DefaultValue "missing-post-publish-verification-backfill-plan")
$postPublishVerificationBackfillStepCount = [int](Get-PropertyOrDefault -Object $summary -Name "postPublishVerificationBackfillStepCount" -DefaultValue 0)
$postPublishVerificationBackfillCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $summary -Name "postPublishVerificationBackfillCanCloseReleaseIssue" -DefaultValue $true)
$runtimeProofStatus = [string](Get-PropertyOrDefault -Object $summary -Name "runtimeProofStatus" -DefaultValue "missing")
$canPublish = [bool](Get-PropertyOrDefault -Object $summary -Name "canPublish" -DefaultValue $true)
$canPromote = [bool](Get-PropertyOrDefault -Object $summary -Name "canPromote" -DefaultValue $true)
$blockingItems = @(Get-PropertyOrDefault -Object $summary -Name "blockingItems" -DefaultValue @())

$validationItems = @(
  New-ValidationItem -Id "record-kind" -Passed ([string]::Equals($recordKind, "release-candidate-freeze-summary", [System.StringComparison]::Ordinal)) -Detail "recordKind=$recordKind" -Boundary "Freeze summary must be a dedicated record."
  New-ValidationItem -Id "runtime-package-key" -Passed ([string]::Equals($summaryRuntimePackageKey, $RuntimePackageKey, [System.StringComparison]::Ordinal)) -Detail "runtimePackageKey=$summaryRuntimePackageKey" -Boundary "Freeze summary must target the requested runtime package key."
  New-ValidationItem -Id "no-publish-side-effect" -Passed (-not $performsPublish) -Detail "performsPublish=$performsPublish" -Boundary "Freeze summary must not run public publish, upload, delete, delist, or withdraw."
  New-ValidationItem -Id "close-readiness-consistency" -Passed $closeReadinessConsistent -Detail "closeReadinessConsistent=$closeReadinessConsistent" -Boundary "Release evidence, final dry run, publish checklist, promotion issue, and post-publish validation must agree on close readiness."
  New-ValidationItem -Id "no-close-without-post-publish-proof" -Passed (-not $canCloseReleaseIssue -or $realPostPublishVerificationReady) -Detail "canCloseReleaseIssue=$canCloseReleaseIssue; realPostPublishVerificationReady=$realPostPublishVerificationReady" -Boundary "Release issue cannot close without a real post-publish verification record."
  New-ValidationItem -Id "no-publish-without-external-runtime-proof" -Passed (-not $canPublish -or $realExternalRuntimeProofReady) -Detail "canPublish=$canPublish; realExternalRuntimeProofReady=$realExternalRuntimeProofReady" -Boundary "Public publish readiness cannot ignore compatible-host package-consumer runtime proof."
  New-ValidationItem -Id "no-promote-without-external-runtime-proof" -Passed (-not $canPromote -or $realExternalRuntimeProofReady) -Detail "canPromote=$canPromote; realExternalRuntimeProofReady=$realExternalRuntimeProofReady" -Boundary "Promotion readiness cannot be based on templates, drafts, runbooks, collection bundles, or dependency probes."
  New-ValidationItem -Id "blocked-cuda-driver-visible" -Passed (-not ([string]::Equals($runtimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase) -and $realExternalRuntimeProofReady)) -Detail "runtimeProofStatus=$runtimeProofStatus; realExternalRuntimeProofReady=$realExternalRuntimeProofReady" -Boundary "blocked-by-cuda-driver is not smoke passed."
  New-ValidationItem -Id "external-proof-blocking-item-visible" -Passed (@($blockingItems | Where-Object { [string]$_.id -eq "real-external-runtime-proof" }).Count -gt 0) -Detail "externalRuntimeProofState=$externalRuntimeProofState" -Boundary "Missing real external runtime proof must remain visible to the owner."
  New-ValidationItem -Id "post-publish-blocking-item-visible" -Passed (@($blockingItems | Where-Object { [string]$_.id -eq "post-publish-verification" }).Count -gt 0) -Detail "postPublishVerificationState=$postPublishVerificationState" -Boundary "Missing real post-publish proof must remain visible to the owner."
  New-ValidationItem -Id "external-backfill-plan-visible-and-blocked" -Passed ([string]::Equals($externalRuntimeProofBackfillPlanState, "blocked-compatible-host-proof-required", [System.StringComparison]::Ordinal) -and $externalRuntimeProofBackfillStepCount -ge 7 -and -not $externalRuntimeProofBackfillCanPromoteRuntimeProof -and (@($blockingItems | Where-Object { [string]$_.id -eq "external-runtime-proof-backfill-required" }).Count -gt 0)) -Detail "externalRuntimeProofBackfillPlanState=$externalRuntimeProofBackfillPlanState; stepCount=$externalRuntimeProofBackfillStepCount; canPromoteRuntimeProof=$externalRuntimeProofBackfillCanPromoteRuntimeProof" -Boundary "External runtime proof backfill plan must stay visible as blocked guidance only."
  New-ValidationItem -Id "post-publish-backfill-plan-visible-and-blocked" -Passed ([string]::Equals($postPublishVerificationBackfillPlanState, "blocked-real-post-publish-proof-required", [System.StringComparison]::Ordinal) -and $postPublishVerificationBackfillStepCount -ge 9 -and -not $postPublishVerificationBackfillCanCloseReleaseIssue -and (@($blockingItems | Where-Object { [string]$_.id -eq "post-publish-verification-backfill-required" }).Count -gt 0)) -Detail "postPublishVerificationBackfillPlanState=$postPublishVerificationBackfillPlanState; stepCount=$postPublishVerificationBackfillStepCount; canCloseReleaseIssue=$postPublishVerificationBackfillCanCloseReleaseIssue" -Boundary "Post-publish verification backfill plan must stay visible as blocked guidance only."
)

if ($checklist) {
  $checklistKind = [string](Get-PropertyOrDefault -Object $checklist -Name "recordKind" -DefaultValue "")
  $checklistPerformsPublish = [bool](Get-PropertyOrDefault -Object $checklist -Name "performsPublish" -DefaultValue $true)
  $checklistCanClose = [bool](Get-PropertyOrDefault -Object $checklist -Name "canCloseReleaseIssue" -DefaultValue $true)
  $validationItems += New-ValidationItem -Id "checklist-kind" -Passed ([string]::Equals($checklistKind, "release-candidate-freeze-checklist", [System.StringComparison]::Ordinal)) -Detail "checklistKind=$checklistKind" -Boundary "Freeze checklist must be a dedicated owner-facing record."
  $validationItems += New-ValidationItem -Id "checklist-no-publish-side-effect" -Passed (-not $checklistPerformsPublish) -Detail "checklistPerformsPublish=$checklistPerformsPublish" -Boundary "Freeze checklist can contain placeholders but must not perform publish."
  $validationItems += New-ValidationItem -Id "checklist-close-consistency" -Passed ($checklistCanClose -eq $canCloseReleaseIssue) -Detail "checklistCanClose=$checklistCanClose; summaryCanClose=$canCloseReleaseIssue" -Boundary "Freeze checklist and summary must agree on release issue close readiness."
}

$failedItems = @($validationItems | Where-Object { -not $_.passed })
$validationState = if ($failedItems.Count -eq 0) {
  if ([string]::Equals([string](Get-PropertyOrDefault -Object $summary -Name "freezeState" -DefaultValue ""), "freeze-ready-for-owner-publish-authorization", [System.StringComparison]::Ordinal)) {
    "freeze-ready-for-owner-publish-authorization"
  }
  else {
    "blocked-freeze-owner-action-required"
  }
}
else {
  "invalid-freeze-summary"
}

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationKind = "release-candidate-freeze-summary-validation"
  runtimePackageKey = $RuntimePackageKey
  validationState = $validationState
  failedValidationItemCount = $failedItems.Count
  canCloseReleaseIssue = $canCloseReleaseIssue
  canPublish = $canPublish
  canPromote = $canPromote
  realExternalRuntimeProofReady = $realExternalRuntimeProofReady
  realPostPublishVerificationReady = $realPostPublishVerificationReady
  externalRuntimeProofBackfillPlanState = $externalRuntimeProofBackfillPlanState
  externalRuntimeProofBackfillStepCount = $externalRuntimeProofBackfillStepCount
  postPublishVerificationBackfillPlanState = $postPublishVerificationBackfillPlanState
  postPublishVerificationBackfillStepCount = $postPublishVerificationBackfillStepCount
  validationItems = $validationItems
  safetyNotes = @(
    "blocked-by-cuda-driver is not smoke passed.",
    "Template, draft, example, runbook, collection bundle, and dependency-probe-only records are not proof.",
    "Backfill plans are guidance only; they are not runtime proof, post-publish proof, publication approval, release close approval, or package push.",
    "No public publish, upload, delete, delist, or withdraw is performed by this validation.",
    "Release issue close readiness requires real post-publish verification proof."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "release-candidate-freeze-validation.json"
$markdownPath = Join-Path $outputRoot "release-candidate-freeze-validation.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Candidate Freeze Validation")
$lines.Add("")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- failed validation items: $($failedItems.Count)")
$lines.Add("- can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("- real external runtime proof ready: ``$realExternalRuntimeProofReady``")
$lines.Add("- real post-publish verification ready: ``$realPostPublishVerificationReady``")
$lines.Add("- external runtime proof backfill plan: ``$externalRuntimeProofBackfillPlanState``")
$lines.Add("- external runtime proof backfill step count: ``$externalRuntimeProofBackfillStepCount``")
$lines.Add("- post-publish verification backfill plan: ``$postPublishVerificationBackfillPlanState``")
$lines.Add("- post-publish verification backfill step count: ``$postPublishVerificationBackfillStepCount``")
$lines.Add("")
$lines.Add("## Validation Items")
$lines.Add("")
$lines.Add("| ID | Passed | Detail |")
$lines.Add("| --- | --- | --- |")
foreach ($item in $validationItems) {
  $lines.Add("| ``$($item.id)`` | ``$($item.passed)`` | $($item.detail.Replace("|", "\|")) |")
}
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate freeze validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedValidationItemCount=$($failedItems.Count) CanCloseReleaseIssue=$canCloseReleaseIssue"

if ($failedItems.Count -gt 0) {
  throw "Release candidate freeze summary validation failed with $($failedItems.Count) item(s)."
}
