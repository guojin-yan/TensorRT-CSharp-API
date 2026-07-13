[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-proof-execution-record-projection.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

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

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-ProofLane {
  param([string]$TrackId)

  switch ($TrackId) {
    "package-consumer-runtime-proof-execution" { return "package-consumer-runtime" }
    "post-publish-verification-execution" { return "post-publish-verification" }
    "linux-runner-proof-execution" { return "linux-runner-proof" }
    "real-model-runtime-proof-execution" { return "real-model-runtime" }
    "release-close-owner-input-execution" { return "release-close-owner-input" }
    "strict-close-validation-execution" { return "strict-close-validation" }
    default { return "unknown-proof-lane" }
  }
}

function New-OwnerInput {
  param([string]$Name, [string]$Source, [AllowNull()][object]$CurrentValue)

  [pscustomobject]@{
    name = $Name
    source = $Source
    currentValue = $CurrentValue
    required = $true
    ownerStatus = "owner-action-required"
    ready = $false
  }
}

function New-EvidenceFile {
  param([string]$Name, [string]$Path, [string]$Sha256)

  [pscustomobject]@{
    name = $Name
    path = $Path
    sha256 = $Sha256
    required = $true
    exists = $false
    hashReady = $false
    ownerStatus = "owner-action-required"
  }
}

function New-RequiredHash {
  param([string]$Name, [string]$Path, [string]$Sha256, [string]$ComputedSha256, [bool]$Matches)

  [pscustomobject]@{
    name = $Name
    path = $Path
    sha256 = $Sha256
    computedSha256 = $ComputedSha256
    matches = $Matches
    required = $true
    ownerStatus = "owner-action-required"
  }
}

function New-RequiredCommand {
  param([object]$Command)

  [pscustomobject]@{
    command = [string](Get-PropertyOrDefault -Object $Command -Name "command" -DefaultValue "<owner-fill-command>")
    exitCode = Get-PropertyOrDefault -Object $Command -Name "exitCode" -DefaultValue "<owner-fill-exit-code>"
    startedAtUtc = [string](Get-PropertyOrDefault -Object $Command -Name "startedAtUtc" -DefaultValue "<owner-fill-started-at-utc>")
    completedAtUtc = [string](Get-PropertyOrDefault -Object $Command -Name "completedAtUtc" -DefaultValue "<owner-fill-completed-at-utc>")
    workingDirectory = [string](Get-PropertyOrDefault -Object $Command -Name "workingDirectory" -DefaultValue "<owner-fill-working-directory>")
    logPath = [string](Get-PropertyOrDefault -Object $Command -Name "logPath" -DefaultValue "<owner-fill-command-log-path>")
    logSha256 = [string](Get-PropertyOrDefault -Object $Command -Name "logSha256" -DefaultValue "<owner-fill-command-log-sha256>")
    required = $true
    captured = $false
    ownerStatus = "owner-action-required"
  }
}

function New-RequiredValidator {
  param([object]$Validator)

  [pscustomobject]@{
    command = [string](Get-PropertyOrDefault -Object $Validator -Name "command" -DefaultValue "<owner-fill-validator-command>")
    exitCode = Get-PropertyOrDefault -Object $Validator -Name "exitCode" -DefaultValue "<owner-fill-validator-exit-code>"
    logPath = [string](Get-PropertyOrDefault -Object $Validator -Name "logPath" -DefaultValue "<owner-fill-validator-log-path>")
    logSha256 = [string](Get-PropertyOrDefault -Object $Validator -Name "logSha256" -DefaultValue "<owner-fill-validator-log-sha256>")
    passed = [bool](Get-PropertyOrDefault -Object $Validator -Name "passed" -DefaultValue $false)
    required = $true
    ownerStatus = "owner-action-required"
  }
}

function New-ForbiddenChecklistItem {
  param([object]$Check)

  [pscustomobject]@{
    name = [string](Get-PropertyOrDefault -Object $Check -Name "name" -DefaultValue "unknown-forbidden-substitute")
    checked = [bool](Get-PropertyOrDefault -Object $Check -Name "checked" -DefaultValue $false)
    present = [bool](Get-PropertyOrDefault -Object $Check -Name "present" -DefaultValue $true)
    ownerEvidence = [string](Get-PropertyOrDefault -Object $Check -Name "ownerEvidence" -DefaultValue "<owner-fill-forbidden-substitute-check-evidence>")
    passed = [bool](Get-PropertyOrDefault -Object $Check -Name "passed" -DefaultValue $false)
    requiredAbsent = $true
    ownerStatus = "owner-action-required"
  }
}

function New-ReviewChecklistItem {
  param([string]$Id, [string]$Description)

  [pscustomobject]@{
    id = $Id
    description = $Description
    required = $true
    passed = $false
    ownerStatus = "owner-action-required"
  }
}

function New-ProofReportItem {
  param([object]$Record)

  $recordId = [string](Get-PropertyOrDefault -Object $Record -Name "recordId" -DefaultValue "unknown-record")
  $sourceTrackId = [string](Get-PropertyOrDefault -Object $Record -Name "sourceTrackId" -DefaultValue "unknown-track")
  $hostMetadata = Get-PropertyOrDefault -Object $Record -Name "hostMetadata" -DefaultValue $null
  $execution = Get-PropertyOrDefault -Object $Record -Name "execution" -DefaultValue $null
  $commands = @(Get-PropertyOrDefault -Object $execution -Name "commands" -DefaultValue @())
  $logs = @(Get-PropertyOrDefault -Object $Record -Name "logs" -DefaultValue @())
  $hashes = @(Get-PropertyOrDefault -Object $Record -Name "hashes" -DefaultValue @())
  $validators = @(Get-PropertyOrDefault -Object $Record -Name "validatorOutputs" -DefaultValue @())
  $forbiddenChecks = @(Get-PropertyOrDefault -Object $Record -Name "forbiddenSubstituteChecks" -DefaultValue @())

  $ownerInputs = @(
    New-OwnerInput -Name "hostOs" -Source "hostMetadata" -CurrentValue (Get-PropertyOrDefault -Object $hostMetadata -Name "hostOs" -DefaultValue "<owner-fill-host-os>")
    New-OwnerInput -Name "hostArchitecture" -Source "hostMetadata" -CurrentValue (Get-PropertyOrDefault -Object $hostMetadata -Name "hostArchitecture" -DefaultValue "<owner-fill-host-architecture>")
    New-OwnerInput -Name "cudaDriverVersion" -Source "hostMetadata" -CurrentValue (Get-PropertyOrDefault -Object $hostMetadata -Name "cudaDriverVersion" -DefaultValue "<owner-fill-cuda-driver-version>")
    New-OwnerInput -Name "cudaRuntimeVersion" -Source "hostMetadata" -CurrentValue (Get-PropertyOrDefault -Object $hostMetadata -Name "cudaRuntimeVersion" -DefaultValue "<owner-fill-cuda-runtime-version>")
    New-OwnerInput -Name "tensorRtVersion" -Source "hostMetadata" -CurrentValue (Get-PropertyOrDefault -Object $hostMetadata -Name "tensorRtVersion" -DefaultValue "<owner-fill-tensorrt-version>")
    New-OwnerInput -Name "runtimePackageKey" -Source "hostMetadata" -CurrentValue (Get-PropertyOrDefault -Object $hostMetadata -Name "runtimePackageKey" -DefaultValue "<owner-fill-runtime-package-key>")
  )

  $evidenceFiles = @()
  foreach ($command in $commands) {
    $evidenceFiles += New-EvidenceFile -Name "command-log" -Path ([string](Get-PropertyOrDefault -Object $command -Name "logPath" -DefaultValue "<owner-fill-command-log-path>")) -Sha256 ([string](Get-PropertyOrDefault -Object $command -Name "logSha256" -DefaultValue "<owner-fill-command-log-sha256>"))
  }
  foreach ($log in $logs) {
    $evidenceFiles += New-EvidenceFile -Name ([string](Get-PropertyOrDefault -Object $log -Name "name" -DefaultValue "log")) -Path ([string](Get-PropertyOrDefault -Object $log -Name "path" -DefaultValue "<owner-fill-log-path>")) -Sha256 ([string](Get-PropertyOrDefault -Object $log -Name "sha256" -DefaultValue "<owner-fill-log-sha256>"))
  }
  foreach ($validator in $validators) {
    $evidenceFiles += New-EvidenceFile -Name "validator-log" -Path ([string](Get-PropertyOrDefault -Object $validator -Name "logPath" -DefaultValue "<owner-fill-validator-log-path>")) -Sha256 ([string](Get-PropertyOrDefault -Object $validator -Name "logSha256" -DefaultValue "<owner-fill-validator-log-sha256>"))
  }

  $requiredHashes = @()
  foreach ($hash in $hashes) {
    $requiredHashes += New-RequiredHash -Name ([string](Get-PropertyOrDefault -Object $hash -Name "name" -DefaultValue "hash")) -Path ([string](Get-PropertyOrDefault -Object $hash -Name "path" -DefaultValue "<owner-fill-hash-path>")) -Sha256 ([string](Get-PropertyOrDefault -Object $hash -Name "sha256" -DefaultValue "<owner-fill-hash-sha256>")) -ComputedSha256 ([string](Get-PropertyOrDefault -Object $hash -Name "computedSha256" -DefaultValue "<validator-computed-sha256>")) -Matches ([bool](Get-PropertyOrDefault -Object $hash -Name "matches" -DefaultValue $false))
  }

  [pscustomobject]@{
    reportItemId = "$sourceTrackId-report"
    sourceRecordId = $recordId
    sourceTrackId = $sourceTrackId
    proofKind = [string](Get-PropertyOrDefault -Object $Record -Name "proofKind" -DefaultValue "unknown-proof-kind")
    reportState = "blocked-owner-real-proof-report-input-required"
    ownerStatus = "owner-action-required"
    proofLane = Get-ProofLane -TrackId $sourceTrackId
    requiredOwnerInputs = @($ownerInputs)
    requiredEvidenceFiles = @($evidenceFiles)
    requiredHashes = @($requiredHashes)
    requiredCommands = @($commands | ForEach-Object { New-RequiredCommand -Command $_ })
    requiredValidators = @($validators | ForEach-Object { New-RequiredValidator -Validator $_ })
    forbiddenSubstituteChecklist = @($forbiddenChecks | ForEach-Object { New-ForbiddenChecklistItem -Check $_ })
    reviewChecklist = @(
      New-ReviewChecklistItem -Id "owner-reviewed-host-metadata" -Description "Owner reviewed host OS, architecture, CUDA, TensorRT, and runtime package key."
      New-ReviewChecklistItem -Id "owner-reviewed-command-logs" -Description "Owner reviewed captured command logs and exit codes."
      New-ReviewChecklistItem -Id "owner-reviewed-hashes" -Description "Owner reviewed expected and computed SHA256 values."
      New-ReviewChecklistItem -Id "owner-reviewed-forbidden-substitutes" -Description "Owner confirmed forbidden substitute evidence is absent."
      New-ReviewChecklistItem -Id "owner-reviewed-validator-output" -Description "Owner reviewed validator output and promotion boundary."
    )
    readyForOwnerReview = $false
    readyForPromotion = $false
    promotionFlags = [pscustomobject]@{
      canPromoteRuntimeProof = $false
      canPublishPublicly = $false
      canCloseReleaseIssue = $false
      isRuntimeExecutionProof = $false
      isReleaseCloseProof = $false
    }
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    boundary = "This report item is an owner-fill proof report package item. It is not proof, not publish approval, not post-publish verification, and not release-close approval."
  }
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Real proof execution record projection not found: $resolvedInputPath"
}

$projection = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$records = @(Get-PropertyOrDefault -Object $projection -Name "proofExecutionRecords" -DefaultValue @())
$reportItems = @($records | ForEach-Object { New-ProofReportItem -Record $_ })

$recordOut = [pscustomobject]@{
  recordKind = "owner-real-proof-report-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  sourceProjectionPath = $resolvedInputPath
  sourceProjectionState = [string](Get-PropertyOrDefault -Object $projection -Name "projectionState" -DefaultValue "missing-real-proof-execution-record-projection-state")
  packState = "blocked-owner-real-proof-report-input-required"
  reportItemCount = $reportItems.Count
  blockedReportItemCount = $reportItems.Count
  readyForOwnerReviewCount = 0
  readyForPromotionCount = 0
  proofReportItems = @($reportItems)
  requiredProofLanes = @(
    "package-consumer-runtime",
    "post-publish-verification",
    "linux-runner-proof",
    "real-model-runtime",
    "release-close-owner-input",
    "strict-close-validation"
  )
  forbiddenSubstitutes = @(Get-PropertyOrDefault -Object $projection -Name "forbiddenSubstitutes" -DefaultValue @())
  sourceArtifacts = @(
    "artifacts/final-release/real-proof-execution-record-projection.json",
    "artifacts/final-release/real-proof-execution-record-projection-validation.json",
    "artifacts/final-release/real-proof-runner-input-backfill.template.json",
    "artifacts/final-release/real-external-proof-backfill-execution-bundle.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This owner report pack organizes fields required for future real proof submission. It is not proof, not publish approval, not post-publish verification, and not release-close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-real-proof-report-pack.json"
$markdownPath = Join-Path $OutputRoot "owner-real-proof-report-pack.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($recordOut | ConvertTo-Json -Depth 18)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner Real Proof Report Pack")
$lines.Add("")
$lines.Add("`owner-real-proof-report-pack` 将 projection records 收敛为 Owner 可填写的真实 proof 填报包。它不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| packState | ``$(ConvertTo-MarkdownCell $recordOut.packState)`` |")
$lines.Add("| reportItemCount | ``$($recordOut.reportItemCount)`` |")
$lines.Add("| blockedReportItemCount | ``$($recordOut.blockedReportItemCount)`` |")
$lines.Add("| readyForOwnerReviewCount | ``$($recordOut.readyForOwnerReviewCount)`` |")
$lines.Add("| readyForPromotionCount | ``$($recordOut.readyForPromotionCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($recordOut.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($recordOut.canCloseReleaseIssue)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($recordOut.isRuntimeExecutionProof)`` |")
$lines.Add("| isReleaseCloseProof | ``$($recordOut.isReleaseCloseProof)`` |")
$lines.Add("")
$lines.Add("## Report Items")
$lines.Add("")
$lines.Add("| Report Item | Proof Lane | Source Track | Owner Inputs | Evidence Files | Hashes | Validators |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- |")
foreach ($item in $reportItems) {
  $lines.Add("| $(ConvertTo-MarkdownCell $item.reportItemId) | $(ConvertTo-MarkdownCell $item.proofLane) | $(ConvertTo-MarkdownCell $item.sourceTrackId) | ``$(@($item.requiredOwnerInputs).Count)`` | ``$(@($item.requiredEvidenceFiles).Count)`` | ``$(@($item.requiredHashes).Count)`` | ``$(@($item.requiredValidators).Count)`` |")
}
$lines.Add("")
$lines.Add("## Required Proof Lanes")
$lines.Add("")
foreach ($lane in $recordOut.requiredProofLanes) {
  $lines.Add("- ``$lane``")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($recordOut.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner real proof report pack written to $jsonPath"
Write-Host "Owner real proof report pack markdown written to $markdownPath"
Write-Host "PackState=$($recordOut.packState) ReportItems=$($recordOut.reportItemCount) Blocked=$($recordOut.blockedReportItemCount) ReadyForOwnerReview=$($recordOut.readyForOwnerReviewCount) ReadyForPromotion=$($recordOut.readyForPromotionCount)"
