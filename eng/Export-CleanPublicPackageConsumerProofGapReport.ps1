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

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Write-Utf8File {
  param(
    [string]$LiteralPath,
    [AllowNull()][object]$InputObject
  )

  $content = @($InputObject) -join [Environment]::NewLine
  [System.IO.File]::WriteAllText($LiteralPath, $content + [Environment]::NewLine, $script:utf8)
}

function New-Gap {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string]$CurrentSignal,
    [string[]]$RequiredEvidence,
    [string[]]$Accepts,
    [string[]]$Rejects,
    [string]$OwnerNextAction
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    currentSignal = $CurrentSignal
    ownerActionRequired = $true
    readyForPromotion = $false
    requiredEvidence = @($RequiredEvidence)
    accepts = @($Accepts)
    rejects = @($Rejects)
    ownerNextAction = $OwnerNextAction
  }
}

$ownerInputValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input-validation.json"
$recordValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-record-validation.json"
$executionBundle = Read-JsonOrNull "artifacts\final-release\clean-consumer-proof-execution-bundle.json"
$closurePackValidation = Read-JsonOrNull "artifacts\final-release\clean-consumer-external-proof-closure-pack-validation.json"
$postPublishPreflight = Read-JsonOrNull "artifacts\final-release\final-post-publish-clean-consumer-proof-preflight.json"

$ownerInputValidationState = [string](Get-PropertyOrDefault -Object $ownerInputValidation -Name "validationState" -DefaultValue "missing-package-consumer-runtime-proof-owner-input-validation")
$ownerInputFailedActionRequiredCount = [int](Get-PropertyOrDefault -Object $ownerInputValidation -Name "failedActionRequiredCount" -DefaultValue -1)
$ownerInputBlockedReason = [string](Get-PropertyOrDefault -Object $ownerInputValidation -Name "ownerInputBlockedReason" -DefaultValue "")
$ownerInputCleanReady = [bool](Get-PropertyOrDefault -Object $ownerInputValidation -Name "cleanOwnerInputReady" -DefaultValue $false)
$ownerInputForbiddenSubstituteFree = [bool](Get-PropertyOrDefault -Object $ownerInputValidation -Name "ownerInputForbiddenSubstituteFree" -DefaultValue $false)
$ownerInputHashFieldsReady = [bool](Get-PropertyOrDefault -Object $ownerInputValidation -Name "ownerInputHashFieldsReady" -DefaultValue $false)
$ownerInputSmokeLogReady = [bool](Get-PropertyOrDefault -Object $ownerInputValidation -Name "ownerInputSmokeLogReady" -DefaultValue $false)
$ownerInputHostMetadataReady = [bool](Get-PropertyOrDefault -Object $ownerInputValidation -Name "ownerInputHostMetadataReady" -DefaultValue $false)
$ownerInputCommandEvidenceReady = [bool](Get-PropertyOrDefault -Object $ownerInputValidation -Name "ownerInputCommandEvidenceReady" -DefaultValue $false)

$recordValidationState = [string](Get-PropertyOrDefault -Object $recordValidation -Name "validationState" -DefaultValue "missing-package-consumer-runtime-proof-record-validation")
$recordFailedProofItemCount = [int](Get-PropertyOrDefault -Object $recordValidation -Name "failedProofItemCount" -DefaultValue -1)
$recordFailedActionRequiredCount = [int](Get-PropertyOrDefault -Object $recordValidation -Name "failedActionRequiredCount" -DefaultValue -1)
$recordBlockedReason = [string](Get-PropertyOrDefault -Object $recordValidation -Name "ownerInputBlockedReason" -DefaultValue "")

$executionBundleState = [string](Get-PropertyOrDefault -Object $executionBundle -Name "bundleState" -DefaultValue "missing-clean-consumer-proof-execution-bundle")
$closurePackValidationState = [string](Get-PropertyOrDefault -Object $closurePackValidation -Name "validationState" -DefaultValue "missing-clean-consumer-external-proof-closure-pack-validation")
$postPublishPreflightState = [string](Get-PropertyOrDefault -Object $postPublishPreflight -Name "preflightState" -DefaultValue "missing-final-post-publish-clean-consumer-proof-preflight")
$postPublishBlockedProofCandidateCount = [int](Get-PropertyOrDefault -Object $postPublishPreflight -Name "blockedProofCandidateCount" -DefaultValue -1)
$postPublishFailedActionRequiredCount = [int](Get-PropertyOrDefault -Object $postPublishPreflight -Name "failedActionRequiredCount" -DefaultValue -1)

$forbiddenSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "build-only",
  "DependencyProbe-only",
  "bridge-only compatible-host smoke",
  "pre-publish smoke reused as post-publish proof",
  "template-only owner input",
  "dashboard-only evidence",
  "hash-only package listing"
)

$gaps = @(
  New-Gap -Order 1 -Id "missing-public-package-source" -Title "Missing public package source" -CurrentSignal $ownerInputValidationState -RequiredEvidence @(
    "publicPackageSource URI for the real package channel",
    "managed package id/version from the public source",
    "runtime package id/version from the public source",
    "owner confirmation that local feeds are not used"
  ) -Accepts @(
    "nuget.org or another explicitly approved public package source",
    "package source visible in restore logs",
    "owner-filled package identity fields"
  ) -Rejects @("local feed", "direct nupkg", "ProjectReference") -OwnerNextAction "Run a repository-external clean consumer restore from the public package source and capture the exact source URL plus package ids/versions."
  New-Gap -Order 2 -Id "missing-public-package-hashes" -Title "Missing public package hashes" -CurrentSignal "hashFieldsReady=$ownerInputHashFieldsReady" -RequiredEvidence @(
    "managed nupkg SHA256",
    "runtime nupkg SHA256",
    "hashes computed from packages downloaded by the clean consumer"
  ) -Accepts @("SHA256 values matching downloaded public packages") -Rejects @("local artifact hash only", "hash without package source") -OwnerNextAction "Compute hashes from the public-source clean consumer package cache and enter them into the owner proof input."
  New-Gap -Order 3 -Id "missing-clean-external-consumer-path" -Title "Missing clean external consumer path" -CurrentSignal "cleanOwnerInputReady=$ownerInputCleanReady" -RequiredEvidence @(
    "cleanExternalConsumerRoot outside repository",
    "consumerProjectPath outside repository",
    "project scan showing no ProjectReference, no local feed, no direct nupkg"
  ) -Accepts @("external consumer project with PackageReference only") -Rejects @("ProjectReference", "local feed", "direct nupkg") -OwnerNextAction "Create or rerun a clean consumer project outside the repository and preserve its project path plus scan result."
  New-Gap -Order 4 -Id "missing-restore-build-smoke-logs" -Title "Missing restore/build/smoke logs" -CurrentSignal "smokeLogReady=$ownerInputSmokeLogReady; commandEvidenceReady=$ownerInputCommandEvidenceReady" -RequiredEvidence @(
    "restore command and log",
    "build command and log",
    "runtime smoke command including explicit runtime package key",
    "stdout/stderr summaries"
  ) -Accepts @("existing logs captured from real clean consumer commands") -Rejects @("build-only", "DependencyProbe-only", "bridge-only compatible-host smoke") -OwnerNextAction "Capture restore/build/smoke command lines, stdout/stderr, exit code, and timestamps from the clean external consumer."
  New-Gap -Order 5 -Id "missing-log-sha256" -Title "Missing stdout/stderr SHA256 evidence" -CurrentSignal $ownerInputBlockedReason -RequiredEvidence @(
    "restore log SHA256",
    "build log SHA256",
    "runtime smoke stdout SHA256",
    "runtime smoke stderr SHA256"
  ) -Accepts @("SHA256 values matching existing owner-supplied logs") -Rejects @("summaries without log hashes", "mismatched hashes") -OwnerNextAction "Hash every submitted log and ensure validator fields match the files on disk."
  New-Gap -Order 6 -Id "missing-host-runtime-metadata" -Title "Missing host and runtime metadata" -CurrentSignal "hostMetadataReady=$ownerInputHostMetadataReady" -RequiredEvidence @(
    "ownerName",
    "machineName",
    "OS and host architecture",
    "GPU name and driver version",
    "CUDA driver/runtime",
    "TensorRT runtime version",
    "cuDNN version",
    "tensorRtLine"
  ) -Accepts @("real compatible-host metadata captured with the proof run") -Rejects @("placeholder host metadata", "host metadata without smoke logs") -OwnerNextAction "Fill owner and host metadata from the machine that executed the clean consumer smoke."
)

$requiredEvidence = @($gaps | ForEach-Object { $_.requiredEvidence } | Select-Object -Unique)

$record = [pscustomobject]@{
  schemaVersion = 1
  recordKind = "clean-public-package-consumer-proof-gap-report"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  reportState = "blocked-clean-public-package-consumer-proof-owner-action-required"
  ownerInputValidationState = $ownerInputValidationState
  ownerInputFailedActionRequiredCount = $ownerInputFailedActionRequiredCount
  ownerInputBlockedReason = $ownerInputBlockedReason
  ownerInputCleanReady = $ownerInputCleanReady
  ownerInputForbiddenSubstituteFree = $ownerInputForbiddenSubstituteFree
  ownerInputHashFieldsReady = $ownerInputHashFieldsReady
  ownerInputSmokeLogReady = $ownerInputSmokeLogReady
  ownerInputHostMetadataReady = $ownerInputHostMetadataReady
  ownerInputCommandEvidenceReady = $ownerInputCommandEvidenceReady
  recordValidationState = $recordValidationState
  recordFailedProofItemCount = $recordFailedProofItemCount
  recordFailedActionRequiredCount = $recordFailedActionRequiredCount
  recordBlockedReason = $recordBlockedReason
  executionBundleState = $executionBundleState
  closurePackValidationState = $closurePackValidationState
  postPublishPreflightState = $postPublishPreflightState
  postPublishBlockedProofCandidateCount = $postPublishBlockedProofCandidateCount
  postPublishFailedActionRequiredCount = $postPublishFailedActionRequiredCount
  gapCount = $gaps.Count
  ownerActionRequiredCount = @($gaps | Where-Object { $_.ownerActionRequired }).Count
  readyForPromotionCount = @($gaps | Where-Object { $_.readyForPromotion }).Count
  requiredEvidence = $requiredEvidence
  forbiddenSubstitutes = $forbiddenSubstitutes
  gaps = $gaps
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  sourceArtifacts = @(
    "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
    "artifacts/final-release/package-consumer-runtime-proof-record-validation.json",
    "artifacts/final-release/clean-consumer-proof-execution-bundle.json",
    "artifacts/final-release/clean-consumer-external-proof-closure-pack-validation.json",
    "artifacts/final-release/final-post-publish-clean-consumer-proof-preflight.json",
    "eng/Test-PackageConsumerRuntimeProofOwnerInput.ps1",
    "eng/Test-PackageConsumerRuntimeProofRecord.ps1"
  )
  boundary = "This gap report is owner-action planning evidence only. It does not publish packages, run a clean consumer, run runtime smoke, approve release close, or promote package-consumer-runtime proof."
}

$jsonPath = Join-Path $OutputRoot "clean-public-package-consumer-proof-gap-report.json"
$markdownPath = Join-Path $OutputRoot "clean-public-package-consumer-proof-gap-report.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 16)

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Clean Public Package Consumer Proof Gap Report")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| reportState | ``$($record.reportState)`` |")
$lines.Add("| ownerInputValidationState | ``$ownerInputValidationState`` |")
$lines.Add("| ownerInputFailedActionRequiredCount | ``$ownerInputFailedActionRequiredCount`` |")
$lines.Add("| recordValidationState | ``$recordValidationState`` |")
$lines.Add("| recordFailedProofItemCount | ``$recordFailedProofItemCount`` |")
$lines.Add("| gapCount | ``$($record.gapCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |")
$lines.Add("")
$lines.Add("## Gap Lanes")
$lines.Add("")
$lines.Add("| Order | ID | Current Signal | Owner Next Action | Rejects |")
$lines.Add("| ---: | --- | --- | --- | --- |")
foreach ($gap in $gaps) {
  $lines.Add("| $($gap.order) | ``$($gap.id)`` | $(ConvertTo-MarkdownCell $gap.currentSignal) | $(ConvertTo-MarkdownCell $gap.ownerNextAction) | $(ConvertTo-MarkdownCell ($gap.rejects -join ", ")) |")
}
$lines.Add("")
$lines.Add("## Forbidden Substitutes")
$lines.Add("")
foreach ($item in $forbiddenSubstitutes) {
  $lines.Add("- ``$item``")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)

Write-Utf8File -LiteralPath $markdownPath -InputObject $lines

Write-Host "Clean public package consumer proof gap report written to $jsonPath"
Write-Host "Clean public package consumer proof gap report written to $markdownPath"
