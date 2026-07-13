[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

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
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-VerificationLane {
  param(
    [string]$Id,
    [string]$Title,
    [string]$State,
    [string]$RequiredState,
    [bool]$Ready,
    [string]$SourceArtifact,
    [string]$OwnerAction,
    [string]$Boundary
  )

  [pscustomobject]@{
    laneId = $Id
    title = $Title
    state = $State
    requiredState = $RequiredState
    laneState = if ($Ready) { "ready-non-proof" } else { "blocked-post-publish-user-verification-required" }
    ready = $Ready
    sourceArtifact = $SourceArtifact
    ownerAction = $OwnerAction
    boundary = $Boundary
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    canPromotePublicProof = $false
    canPromotePostPublishProof = $false
    isRuntimeExecutionProof = $false
    isPackageConsumerRuntimeProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

$docsAuditValidation = Read-JsonOrNull "artifacts\final-release\release-docs-and-nuget-metadata-audit-validation.json"
$publicDocsGate = Read-JsonOrNull "artifacts\final-release\public-docs-package-metadata-gate.json"
$preReleaseMatrix = Read-JsonOrNull "artifacts\final-release\pre-release-package-proof-readiness-matrix.json"
$ownerPublishAuthorization = Read-JsonOrNull "artifacts\final-release\owner-publish-authorization-input-validation.json"
$ownerPublishExecution = Read-JsonOrNull "artifacts\final-release\owner-publish-execution-result-input-validation.json"
$publicDownloadCandidate = Read-JsonOrNull "artifacts\final-release\public-package-download-proof-candidate-validation.json"
$cleanConsumer = Read-JsonOrNull "artifacts\final-release\clean-external-consumer-smoke-input-validation.json"
$postPublishCleanConsumer = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-validation.json"
$finalPostPublishAuditPack = Read-JsonOrNull "artifacts\final-release\final-post-publish-audit-pack-validation.json"

$preReleaseBlockedLaneCount = [int](Get-PropertyOrDefault -Object $preReleaseMatrix -Name "blockedLaneCount" -DefaultValue -1)
$ownerAuthorizationReady = [bool](Get-PropertyOrDefault -Object $ownerPublishAuthorization -Name "ownerPublishAuthorizationReady" -DefaultValue $false)
$ownerExecutionReady = [bool](Get-PropertyOrDefault -Object $ownerPublishExecution -Name "ownerExecutionResultReady" -DefaultValue $false)
$publicDownloadReady = [bool](Get-PropertyOrDefault -Object $publicDownloadCandidate -Name "publicPackageDownloadProofCandidateReady" -DefaultValue $false)
$cleanConsumerReady = [bool](Get-PropertyOrDefault -Object $cleanConsumer -Name "cleanExternalConsumerSmokeReady" -DefaultValue $false)
$postPublishProofReady = [bool](Get-PropertyOrDefault -Object $postPublishCleanConsumer -Name "proofCandidateReady" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $postPublishCleanConsumer -Name "sourceProofLinkageReady" -DefaultValue $false)
$finalPostPublishReady = [string](Get-PropertyOrDefault -Object $finalPostPublishAuditPack -Name "validationState" -DefaultValue "") -eq "final-post-publish-audit-ready"

$lanes = @(
  New-VerificationLane `
    -Id "release-docs-and-nuget-metadata-audit" `
    -Title "Release docs and NuGet metadata audit" `
    -State ([string](Get-PropertyOrDefault -Object $docsAuditValidation -Name "validationState" -DefaultValue "missing-release-docs-and-nuget-metadata-audit-validation")) `
    -RequiredState "release-docs-and-nuget-metadata-audit-ready-non-proof" `
    -Ready ([string](Get-PropertyOrDefault -Object $docsAuditValidation -Name "validationState" -DefaultValue "") -eq "release-docs-and-nuget-metadata-audit-ready-non-proof") `
    -SourceArtifact "artifacts/final-release/release-docs-and-nuget-metadata-audit-validation.json" `
    -OwnerAction "Keep public docs and NuGet metadata claim-safe before publication; this lane is not publish proof." `
    -Boundary "Documentation and metadata audit only; not public package download proof, runtime proof, post-publish proof, or release close approval."

  New-VerificationLane `
    -Id "public-docs-package-metadata-gate" `
    -Title "Public docs/package metadata negative claim gate" `
    -State ([string](Get-PropertyOrDefault -Object $publicDocsGate -Name "gateState" -DefaultValue "missing-public-docs-package-metadata-gate")) `
    -RequiredState "blocked-owner-public-postpublish-proof-required with failedBlockerCount=0" `
    -Ready ($null -ne $publicDocsGate -and [int](Get-PropertyOrDefault -Object $publicDocsGate -Name "failedBlockerCount" -DefaultValue -1) -eq 0) `
    -SourceArtifact "artifacts/final-release/public-docs-package-metadata-gate.json" `
    -OwnerAction "Keep stale/overclaim scanner clean before public handoff." `
    -Boundary "Negative claim-safety gate only; its safe result still says owner public/post-publish proof is required."

  New-VerificationLane `
    -Id "pre-release-package-proof-readiness-matrix" `
    -Title "Pre-release package proof readiness matrix" `
    -State ([string](Get-PropertyOrDefault -Object $preReleaseMatrix -Name "matrixState" -DefaultValue "missing-pre-release-package-proof-readiness-matrix")) `
    -RequiredState "all readiness lanes unblocked" `
    -Ready ($preReleaseBlockedLaneCount -eq 0) `
    -SourceArtifact "artifacts/final-release/pre-release-package-proof-readiness-matrix.json" `
    -OwnerAction "Clear package dry-run, public download, package-consumer, and post-publish readiness blockers with real owner evidence." `
    -Boundary "Readiness classifier only; source-quality, dispatch packs, templates, local artifacts, and dry-runs remain non-proof."

  New-VerificationLane `
    -Id "owner-publish-authorization" `
    -Title "Owner publish authorization input" `
    -State ([string](Get-PropertyOrDefault -Object $ownerPublishAuthorization -Name "validationState" -DefaultValue "missing-owner-publish-authorization-input-validation")) `
    -RequiredState "owner-publish-authorization-input-ready" `
    -Ready $ownerAuthorizationReady `
    -SourceArtifact "artifacts/final-release/owner-publish-authorization-input-validation.json" `
    -OwnerAction "Owner must explicitly approve a manual publish run and review the readiness matrix; automation must not publish." `
    -Boundary "Owner authorization validation only; no token use, package push, runtime proof, or release close approval."

  New-VerificationLane `
    -Id "owner-publish-execution-result" `
    -Title "Owner publish execution result" `
    -State ([string](Get-PropertyOrDefault -Object $ownerPublishExecution -Name "validationState" -DefaultValue "missing-owner-publish-execution-result-input-validation")) `
    -RequiredState "owner-publish-execution-result-input-ready" `
    -Ready $ownerExecutionReady `
    -SourceArtifact "artifacts/final-release/owner-publish-execution-result-input-validation.json" `
    -OwnerAction "Owner must provide real publish transcript, public package URLs, downloaded hashes, and no-token review fields." `
    -Boundary "Execution result shape validation only; this script does not execute dotnet nuget push and cannot close release issues."

  New-VerificationLane `
    -Id "public-package-download-proof" `
    -Title "Public package download proof" `
    -State ([string](Get-PropertyOrDefault -Object $publicDownloadCandidate -Name "validationState" -DefaultValue "missing-public-package-download-proof-candidate-validation")) `
    -RequiredState "public-package-download-proof-candidate-ready" `
    -Ready $publicDownloadReady `
    -SourceArtifact "artifacts/final-release/public-package-download-proof-candidate-validation.json" `
    -OwnerAction "After real publication, download managed/runtime packages from public channels and record URLs, sizes, hashes, and source linkage." `
    -Boundary "Public download proof candidate only; local feed, direct .nupkg, dry-run, ProjectReference, queued workflow, or missing runner cannot substitute."

  New-VerificationLane `
    -Id "clean-external-consumer-smoke" `
    -Title "Clean external consumer smoke" `
    -State ([string](Get-PropertyOrDefault -Object $cleanConsumer -Name "validationState" -DefaultValue "missing-clean-external-consumer-smoke-input-validation")) `
    -RequiredState "clean-external-consumer-smoke-ready" `
    -Ready $cleanConsumerReady `
    -SourceArtifact "artifacts/final-release/clean-external-consumer-smoke-input-validation.json" `
    -OwnerAction "Run a repository-external consumer using public package sources, no ProjectReference, no source path, no local restore source, and real host logs/hashes." `
    -Boundary "Clean consumer input validation only; in-repo samples, local feeds, direct .nupkg, source path leakage, build-only, or dependency-probe-only cannot promote runtime proof."

  New-VerificationLane `
    -Id "post-publish-clean-consumer-proof" `
    -Title "Post-publish clean consumer proof" `
    -State ([string](Get-PropertyOrDefault -Object $postPublishCleanConsumer -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-result-validation")) `
    -RequiredState "post-publish clean consumer proof candidate and source linkage ready" `
    -Ready $postPublishProofReady `
    -SourceArtifact "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json" `
    -OwnerAction "After publication, link real public package proof to a clean external restore/build/smoke result with logs and hashes." `
    -Boundary "Post-publish import validation only; not runtime proof, not package push, not release close approval, and not a substitute for public download proof."

  New-VerificationLane `
    -Id "final-post-publish-audit-pack" `
    -Title "Final post-publish audit pack" `
    -State ([string](Get-PropertyOrDefault -Object $finalPostPublishAuditPack -Name "validationState" -DefaultValue "missing-final-post-publish-audit-pack-validation")) `
    -RequiredState "final-post-publish-audit-ready" `
    -Ready $finalPostPublishReady `
    -SourceArtifact "artifacts/final-release/final-post-publish-audit-pack-validation.json" `
    -OwnerAction "Complete all final public/post-publish/close lanes before any release issue close decision." `
    -Boundary "Final post-publish audit aggregation only; it cannot publish, promote proof, or close a release issue."
)

$blockedLanes = @($lanes | Where-Object { -not [bool]$_.ready })
$readyLanes = @($lanes | Where-Object { [bool]$_.ready })
$packState = if ($blockedLanes.Count -eq 0) {
  "post-publish-user-verification-ready-for-owner-close-review"
}
else {
  "blocked-post-publish-user-verification-required"
}

$record = [ordered]@{
  recordKind = "post-publish-user-verification-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  packState = $packState
  verificationLaneCount = $lanes.Count
  readyVerificationLaneCount = $readyLanes.Count
  blockedVerificationLaneCount = $blockedLanes.Count
  lanes = $lanes
  requiredOwnerActions = @($blockedLanes | ForEach-Object { [pscustomobject]@{ laneId = $_.laneId; ownerAction = $_.ownerAction; sourceArtifact = $_.sourceArtifact } })
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  canPromotePublicProof = $false
  canPromotePostPublishProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    "artifacts/final-release/release-docs-and-nuget-metadata-audit-validation.json",
    "artifacts/final-release/public-docs-package-metadata-gate.json",
    "artifacts/final-release/pre-release-package-proof-readiness-matrix.json",
    "artifacts/final-release/owner-publish-authorization-input-validation.json",
    "artifacts/final-release/owner-publish-execution-result-input-validation.json",
    "artifacts/final-release/public-package-download-proof-candidate-validation.json",
    "artifacts/final-release/clean-external-consumer-smoke-input-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json",
    "artifacts/final-release/final-post-publish-audit-pack-validation.json"
  )
  safetyBoundary = "Post-publish user verification pack is an owner action aggregator only. It does not publish packages, use tokens, download public packages, run runtime smoke, promote runtime/public/post-publish proof, or close release issues. Public availability, clean external package-consumer-runtime, and post-publish proof require real owner evidence."
}

$jsonPath = Join-Path $OutputRoot "post-publish-user-verification-pack.json"
$markdownPath = Join-Path $OutputRoot "post-publish-user-verification-pack.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$laneRows = foreach ($lane in $lanes) {
  "| ``$(ConvertTo-MarkdownCell $lane.laneId)`` | ``$(ConvertTo-MarkdownCell $lane.state)`` | ``$(ConvertTo-MarkdownCell $lane.requiredState)`` | ``$($lane.ready)`` | $(ConvertTo-MarkdownCell $lane.ownerAction) |"
}

$markdown = @"
# Post-Publish User Verification Pack

Generated at: ``$($record.generatedAtUtc)``

## Summary

- recordKind: ``$($record.recordKind)``
- packState: ``$($record.packState)``
- verificationLaneCount: ``$($record.verificationLaneCount)``
- readyVerificationLaneCount: ``$($record.readyVerificationLaneCount)``
- blockedVerificationLaneCount: ``$($record.blockedVerificationLaneCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromoteRuntimeProof: ``False``

## Verification Lanes

| Lane | State | Required State | Ready | Owner Action |
| --- | --- | --- | ---: | --- |
$($laneRows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Post-publish user verification pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackState=$($record.packState) Lanes=$($record.verificationLaneCount) Blocked=$($record.blockedVerificationLaneCount) Ready=$($record.readyVerificationLaneCount)"
