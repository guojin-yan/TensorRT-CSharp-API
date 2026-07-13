[CmdletBinding()]
param(
  [string]$ValidatorPath = "artifacts\final-release\real-proof-record-validator.json",
  [string]$PromotionGuardPath = "artifacts\final-release\real-proof-candidate-promotion-guard.json",
  [string]$DeltaPackPath = "artifacts\final-release\owner-real-proof-field-delta-pack.json",
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

function Get-FirstCommandForLane {
  param([string]$ProofLane)

  switch -Regex ($ProofLane) {
    "package-consumer|external-runtime" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\New-PackageConsumerExternalSmokeScaffold.ps1; run restore/build/smoke from a clean directory and capture logs." }
    "linux" { return "Run the Linux runner evidence commands on a Linux x64 CUDA/TensorRT host and capture runner logs." }
    "real-model|sample|yolo|classification" { return "Run the real model sample with owner-supplied ONNX/assets, capture stdout/stderr, and hash all logs." }
    "post-publish" { return "Install from the public package channel in a clean consumer and run post-publish verification smoke." }
    "release-issue|close" { return "Refresh release issue close record after all real proof validators pass, then run Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady." }
    default { return "Collect the real runtime proof evidence for this proof lane, then run the strict validator command with hashed logs." }
  }
}

function New-ClosureItem {
  param(
    [object]$Contract,
    [object[]]$OwnerDeltas,
    [object]$GuardItem
  )

  $candidateId = [string](Get-PropertyOrDefault -Object $Contract -Name "candidateId" -DefaultValue "unknown-candidate")
  $proofLane = [string](Get-PropertyOrDefault -Object $Contract -Name "proofLane" -DefaultValue "unknown-proof-lane")
  $ownerDeltaIds = @($OwnerDeltas | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "fieldDeltaId" -DefaultValue "unknown-delta") })
  $validatorCommand = [string](Get-PropertyOrDefault -Object $Contract -Name "validatorCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordValidator.ps1 -Strict")
  $blockedReasons = @(Get-PropertyOrDefault -Object $Contract -Name "blockedReasons" -DefaultValue @())
  $blockedRequirementCount = [int](Get-PropertyOrDefault -Object $GuardItem -Name "blockedRequirementCount" -DefaultValue 0)

  [pscustomobject]@{
    closureItemId = "$candidateId-execution-closure"
    candidateId = $candidateId
    proofLane = $proofLane
    ownerDeltaIds = $ownerDeltaIds
    firstCommand = Get-FirstCommandForLane -ProofLane $proofLane
    expectedArtifacts = @(
      "real proof input record for $proofLane",
      "stdout/stderr logs captured outside template-only flow",
      "SHA256 files for each captured log",
      "strict validator JSON and markdown output",
      "updated promotion guard / close record only after validator pass"
    )
    requiredLogs = @(
      "command stdout log",
      "command stderr log",
      "merged proof transcript",
      "validator output log"
    )
    requiredSha256 = @(
      "nupkg or package source SHA256 where applicable",
      "stdout log SHA256",
      "stderr log SHA256",
      "merged proof transcript SHA256",
      "validator output SHA256"
    )
    validatorCommands = @(
      $validatorCommand,
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofCandidatePromotionGuard.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict"
    )
    promotionGuardRequirements = @(Get-PropertyOrDefault -Object $GuardItem -Name "requirements" -DefaultValue @())
    blockedRequirementCount = $blockedRequirementCount
    releaseCloseFollowUp = "After this closure item has real logs, hashes, owner review, and passing validator output, refresh release issue close inputs and run Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady."
    notProofBoundary = "This closure item is an owner execution plan only. It is not runtime proof, not package publish approval, not post-publish verification, and not release-close approval."
    blockedReasons = $blockedReasons
    readyForExecutionClosure = $false
    closureState = "blocked-owner-real-proof-execution-closure-required"
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
  }
}

$resolvedValidatorPath = Resolve-InputPath -Path $ValidatorPath
$resolvedPromotionGuardPath = Resolve-InputPath -Path $PromotionGuardPath
$resolvedDeltaPackPath = Resolve-InputPath -Path $DeltaPackPath
foreach ($path in @($resolvedValidatorPath, $resolvedPromotionGuardPath, $resolvedDeltaPackPath)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required closure input not found: $path" }
}

$validator = Get-Content -LiteralPath $resolvedValidatorPath -Raw -Encoding utf8 | ConvertFrom-Json
$promotionGuard = Get-Content -LiteralPath $resolvedPromotionGuardPath -Raw -Encoding utf8 | ConvertFrom-Json
$deltaPack = Get-Content -LiteralPath $resolvedDeltaPackPath -Raw -Encoding utf8 | ConvertFrom-Json

$contracts = @(Get-PropertyOrDefault -Object $validator -Name "validatorContracts" -DefaultValue @())
$guardItems = @(Get-PropertyOrDefault -Object $promotionGuard -Name "guardItems" -DefaultValue @())
$deltas = @(Get-PropertyOrDefault -Object $deltaPack -Name "fieldDeltas" -DefaultValue @())

$closureItems = @()
foreach ($contract in $contracts) {
  $candidateId = [string](Get-PropertyOrDefault -Object $contract -Name "candidateId" -DefaultValue "unknown-candidate")
  $contractDeltas = @($deltas | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "candidateId" -DefaultValue "") -eq $candidateId })
  $guardItem = @($guardItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "candidateId" -DefaultValue "") -eq $candidateId } | Select-Object -First 1)[0]
  $closureItems += New-ClosureItem -Contract $contract -OwnerDeltas $contractDeltas -GuardItem $guardItem
}

$readyClosureItemCount = @($closureItems | Where-Object { [bool]$_.readyForExecutionClosure }).Count
$blockedClosureItemCount = @($closureItems | Where-Object { [string]$_.closureState -eq "blocked-owner-real-proof-execution-closure-required" }).Count

$recordOut = [pscustomobject]@{
  recordKind = "owner-real-proof-execution-closure-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validatorPath = $resolvedValidatorPath
  promotionGuardPath = $resolvedPromotionGuardPath
  deltaPackPath = $resolvedDeltaPackPath
  closureState = "blocked-owner-real-proof-execution-closure-required"
  closureItemCount = $closureItems.Count
  readyClosureItemCount = $readyClosureItemCount
  blockedClosureItemCount = $blockedClosureItemCount
  closureItems = @($closureItems)
  sourceArtifacts = @(
    "artifacts/final-release/real-proof-record-validator.json",
    "artifacts/final-release/real-proof-record-validator-validation.json",
    "artifacts/final-release/real-proof-candidate-promotion-guard.json",
    "artifacts/final-release/real-proof-candidate-promotion-guard-validation.json",
    "artifacts/final-release/owner-real-proof-field-delta-pack.json",
    "artifacts/final-release/owner-real-proof-field-delta-pack-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This closure pack is an owner execution checklist for real proof collection. It does not run runtime proof, publish packages, verify post-publish channels, or close the release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-real-proof-execution-closure-pack.json"
$markdownPath = Join-Path $OutputRoot "owner-real-proof-execution-closure-pack.md"
$recordOut | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner Real Proof Execution Closure Pack")
$lines.Add("")
$lines.Add("`owner-real-proof-execution-closure-pack` 把 validator contract 转成 Owner 下一步真实执行闭环。它不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| closureState | ``$(ConvertTo-MarkdownCell $recordOut.closureState)`` |")
$lines.Add("| closureItemCount | ``$($recordOut.closureItemCount)`` |")
$lines.Add("| readyClosureItemCount | ``$($recordOut.readyClosureItemCount)`` |")
$lines.Add("| blockedClosureItemCount | ``$($recordOut.blockedClosureItemCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($recordOut.canPromoteRuntimeProof)`` |")
$lines.Add("| canPublishPublicly | ``$($recordOut.canPublishPublicly)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($recordOut.canCloseReleaseIssue)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($recordOut.isRuntimeExecutionProof)`` |")
$lines.Add("| isReleaseCloseProof | ``$($recordOut.isReleaseCloseProof)`` |")
$lines.Add("")
$lines.Add("## Closure Items")
$lines.Add("")
$lines.Add("| Candidate | Lane | State | Owner Deltas | First Command |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($item in $closureItems) {
  $lines.Add("| $(ConvertTo-MarkdownCell $item.candidateId) | $(ConvertTo-MarkdownCell $item.proofLane) | $(ConvertTo-MarkdownCell $item.closureState) | ``$(@($item.ownerDeltaIds).Count)`` | $(ConvertTo-MarkdownCell $item.firstCommand) |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($recordOut.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner real proof execution closure pack written to $jsonPath"
Write-Host "Owner real proof execution closure pack markdown written to $markdownPath"
Write-Host "ClosureState=$($recordOut.closureState) Items=$($recordOut.closureItemCount) Blocked=$($recordOut.blockedClosureItemCount) Ready=$($recordOut.readyClosureItemCount)"
