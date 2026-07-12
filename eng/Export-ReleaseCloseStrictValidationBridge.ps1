[CmdletBinding()]
param(
  [string]$RuntimeProofInputValidationPath = "artifacts\final-release\runtime-proof-execution-input-record-validation.json",
  [string]$OwnerRuntimeProofRunbookValidationPath = "artifacts\final-release\owner-runtime-proof-execution-runbook-validation.json",
  [string]$PostPublishVerificationValidationPath = "artifacts\final-release\post-publish-verification-validation.json",
  [string]$ReleaseIssueCloseRecordValidationPath = "artifacts\final-release\release-issue-close-record-validation.json",
  [string]$ReleaseCloseFinalOwnerRunbookValidationPath = "artifacts\final-release\release-close-final-owner-runbook-validation.json",
  [string]$ReleaseEvidenceBundlePath = "artifacts\final-release\release-evidence-bundle.json",
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

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)

  $resolved = Resolve-InputPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
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

function Test-ReadyState {
  param(
    [AllowNull()][object]$Object,
    [string]$StateName,
    [string[]]$ReadyValues,
    [string]$BooleanName
  )

  if ($null -eq $Object) { return $false }
  if (-not [string]::IsNullOrWhiteSpace($BooleanName) -and [bool](Get-PropertyOrDefault -Object $Object -Name $BooleanName -DefaultValue $false)) { return $true }
  $state = [string](Get-PropertyOrDefault -Object $Object -Name $StateName -DefaultValue "")
  return @($ReadyValues) -contains $state
}

$runtimeProofInputValidation = Read-JsonOrNull $RuntimeProofInputValidationPath
$ownerRuntimeProofRunbookValidation = Read-JsonOrNull $OwnerRuntimeProofRunbookValidationPath
$postPublishVerificationValidation = Read-JsonOrNull $PostPublishVerificationValidationPath
$releaseIssueCloseRecordValidation = Read-JsonOrNull $ReleaseIssueCloseRecordValidationPath
$releaseCloseFinalOwnerRunbookValidation = Read-JsonOrNull $ReleaseCloseFinalOwnerRunbookValidationPath
$releaseEvidenceBundle = Read-JsonOrNull $ReleaseEvidenceBundlePath

$runtimeProofInputReady = Test-ReadyState -Object $runtimeProofInputValidation -StateName "validationState" -ReadyValues @("runtime-proof-execution-input-ready") -BooleanName "canPromoteRuntimeProof"
$ownerRuntimeRunbookReady = Test-ReadyState -Object $ownerRuntimeProofRunbookValidation -StateName "validationState" -ReadyValues @("owner-runtime-proof-execution-ready") -BooleanName "canPromoteRuntimeProof"
$postPublishReady = Test-ReadyState -Object $postPublishVerificationValidation -StateName "validationState" -ReadyValues @("post-publish-verification-complete", "complete-post-publish-verification") -BooleanName "canCloseReleaseIssue"
$releaseCloseRecordReady = Test-ReadyState -Object $releaseIssueCloseRecordValidation -StateName "validationState" -ReadyValues @("release-issue-close-ready", "valid-release-issue-close-record") -BooleanName "canCloseReleaseIssue"
$finalOwnerRunbookReady = Test-ReadyState -Object $releaseCloseFinalOwnerRunbookValidation -StateName "validationState" -ReadyValues @("release-close-final-owner-runbook-ready") -BooleanName "canCloseReleaseIssue"
$releaseEvidenceBundleReady = Test-ReadyState -Object $releaseEvidenceBundle -StateName "bundleState" -ReadyValues @("release-evidence-complete") -BooleanName "canCloseReleaseIssue"

$lanes = @(
  "package-consumer-runtime",
  "post-publish-verification",
  "linux-runner-proof",
  "real-model-runtime",
  "release-close-owner-input",
  "strict-close-validation"
)

$bridgeItems = foreach ($lane in $lanes) {
  $blockedReasons = New-Object System.Collections.Generic.List[string]
  if (-not $runtimeProofInputReady) { $blockedReasons.Add("runtime proof execution input validation is not ready") | Out-Null }
  if (-not $ownerRuntimeRunbookReady) { $blockedReasons.Add("owner runtime proof execution runbook is not executed") | Out-Null }
  if (-not $postPublishReady) { $blockedReasons.Add("post-publish verification is not ready") | Out-Null }
  if (-not $releaseCloseRecordReady) { $blockedReasons.Add("release issue close record validation is not ready") | Out-Null }
  if (-not $finalOwnerRunbookReady) { $blockedReasons.Add("release close final owner runbook validation is not ready") | Out-Null }
  if (-not $releaseEvidenceBundleReady) { $blockedReasons.Add("release evidence bundle is not complete") | Out-Null }

  [pscustomobject]@{
    bridgeItemId = "$lane-release-close-strict-validation-bridge"
    proofLane = $lane
    sourceExecutionInputId = "$lane-runtime-proof-execution-input"
    runtimeProofInputReady = $runtimeProofInputReady
    ownerRuntimeRunbookReady = $ownerRuntimeRunbookReady
    postPublishReady = $postPublishReady
    releaseCloseRecordReady = $releaseCloseRecordReady
    finalOwnerRunbookReady = $finalOwnerRunbookReady
    releaseEvidenceBundleReady = $releaseEvidenceBundleReady
    strictCloseCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
    blockedReasons = @($blockedReasons.ToArray())
    bridgeState = "blocked-release-close-strict-validation-required"
    notProofBoundary = "This bridge only aggregates strict-close prerequisites. It cannot substitute real runtime execution logs, post-publish proof, owner rollback approval, final close decision, or release issue close validation."
    readyForStrictClose = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
  }
}

$blockedBridgeItemCount = @($bridgeItems | Where-Object { [string]$_.bridgeState -eq "blocked-release-close-strict-validation-required" }).Count
$readyBridgeItemCount = @($bridgeItems | Where-Object { [bool]$_.readyForStrictClose }).Count

$bridge = [pscustomobject]@{
  recordKind = "release-close-strict-validation-bridge"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  bridgeState = "blocked-release-close-strict-validation-required"
  runtimeProofInputValidationState = [string](Get-PropertyOrDefault -Object $runtimeProofInputValidation -Name "validationState" -DefaultValue "missing-runtime-proof-execution-input-record-validation")
  ownerRuntimeProofRunbookValidationState = [string](Get-PropertyOrDefault -Object $ownerRuntimeProofRunbookValidation -Name "validationState" -DefaultValue "missing-owner-runtime-proof-execution-runbook-validation")
  postPublishValidationState = [string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
  releaseIssueCloseRecordValidationState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseRecordValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")
  releaseCloseFinalOwnerRunbookValidationState = [string](Get-PropertyOrDefault -Object $releaseCloseFinalOwnerRunbookValidation -Name "validationState" -DefaultValue "missing-release-close-final-owner-runbook-validation")
  releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  runtimeProofInputReady = $runtimeProofInputReady
  ownerRuntimeRunbookReady = $ownerRuntimeRunbookReady
  postPublishReady = $postPublishReady
  releaseCloseRecordReady = $releaseCloseRecordReady
  finalOwnerRunbookReady = $finalOwnerRunbookReady
  releaseEvidenceBundleReady = $releaseEvidenceBundleReady
  bridgeItemCount = @($bridgeItems).Count
  blockedBridgeItemCount = $blockedBridgeItemCount
  readyBridgeItemCount = $readyBridgeItemCount
  bridgeItems = @($bridgeItems)
  sourceArtifacts = @(
    "artifacts/final-release/runtime-proof-execution-input-record-validation.json",
    "artifacts/final-release/owner-runtime-proof-execution-runbook-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json",
    "artifacts/final-release/release-close-final-owner-runbook-validation.json",
    "artifacts/final-release/release-evidence-bundle.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This bridge is a blocked strict-close prerequisite aggregator. It cannot run proof, publish packages, approve rollback, verify post-publish state, or close the release issue."
}

$jsonPath = Join-Path $OutputRoot "release-close-strict-validation-bridge.json"
$markdownPath = Join-Path $OutputRoot "release-close-strict-validation-bridge.md"
$bridge | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Close Strict Validation Bridge")
$lines.Add("")
$lines.Add("`release-close-strict-validation-bridge` 串联 runtime proof input validation、Owner runtime runbook、post-publish verification、release issue close record、final owner runbook 和 release evidence bundle。默认 blocked，不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| bridgeState | ``$(ConvertTo-MarkdownCell $bridge.bridgeState)`` |")
$lines.Add("| bridgeItemCount | ``$($bridge.bridgeItemCount)`` |")
$lines.Add("| blockedBridgeItemCount | ``$($bridge.blockedBridgeItemCount)`` |")
$lines.Add("| readyBridgeItemCount | ``$($bridge.readyBridgeItemCount)`` |")
$lines.Add("| runtimeProofInputValidationState | ``$(ConvertTo-MarkdownCell $bridge.runtimeProofInputValidationState)`` |")
$lines.Add("| ownerRuntimeProofRunbookValidationState | ``$(ConvertTo-MarkdownCell $bridge.ownerRuntimeProofRunbookValidationState)`` |")
$lines.Add("| postPublishValidationState | ``$(ConvertTo-MarkdownCell $bridge.postPublishValidationState)`` |")
$lines.Add("| releaseIssueCloseRecordValidationState | ``$(ConvertTo-MarkdownCell $bridge.releaseIssueCloseRecordValidationState)`` |")
$lines.Add("| releaseCloseFinalOwnerRunbookValidationState | ``$(ConvertTo-MarkdownCell $bridge.releaseCloseFinalOwnerRunbookValidationState)`` |")
$lines.Add("| releaseEvidenceBundleState | ``$(ConvertTo-MarkdownCell $bridge.releaseEvidenceBundleState)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($bridge.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Bridge Items")
$lines.Add("")
$lines.Add("| Lane | State | Blocked Reasons | Strict Close Command |")
$lines.Add("| --- | --- | ---: | --- |")
foreach ($item in $bridgeItems) {
  $lines.Add("| $(ConvertTo-MarkdownCell $item.proofLane) | $(ConvertTo-MarkdownCell $item.bridgeState) | ``$(@($item.blockedReasons).Count)`` | $(ConvertTo-MarkdownCell $item.strictCloseCommand) |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($bridge.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close strict validation bridge written to $jsonPath"
Write-Host "Release close strict validation bridge markdown written to $markdownPath"
Write-Host "BridgeState=$($bridge.bridgeState) Items=$($bridge.bridgeItemCount) Blocked=$($bridge.blockedBridgeItemCount) Ready=$($bridge.readyBridgeItemCount)"
