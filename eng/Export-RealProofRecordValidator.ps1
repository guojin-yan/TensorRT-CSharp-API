[CmdletBinding()]
param(
  [string]$StrictRecordPath = "artifacts\final-release\real-proof-input-candidate-strict-record.json",
  [string]$DeltaPackPath = "artifacts\final-release\owner-real-proof-field-delta-pack.json",
  [string]$PromotionGuardPath = "artifacts\final-release\real-proof-candidate-promotion-guard.json",
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

function New-ValidatorContract {
  param(
    [object]$Candidate,
    [object]$GuardItem,
    [object[]]$CandidateDeltas
  )

  $candidateId = [string](Get-PropertyOrDefault -Object $Candidate -Name "candidateId" -DefaultValue "unknown-candidate")
  $proofLane = [string](Get-PropertyOrDefault -Object $Candidate -Name "proofLane" -DefaultValue "unknown-proof-lane")
  $promotionGuardState = [string](Get-PropertyOrDefault -Object $GuardItem -Name "guardState" -DefaultValue "missing-promotion-guard")
  $blockedFieldCount = [int](Get-PropertyOrDefault -Object $Candidate -Name "blockedFieldCount" -DefaultValue 0)
  $blockedDeltaCount = @($CandidateDeltas | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) }).Count
  $blockedRequirementCount = [int](Get-PropertyOrDefault -Object $GuardItem -Name "blockedRequirementCount" -DefaultValue 0)
  $promotionAllowed = [bool](Get-PropertyOrDefault -Object $GuardItem -Name "promotionAllowed" -DefaultValue $false)

  $requiredRuntimeEvidence = @(
    "real runtime execution log captured on compatible CUDA/TensorRT host",
    "stdout/stderr summary with command exit code",
    "runtime package key and TensorRT/CUDA/cuDNN version metadata",
    "evidence file SHA256 values computed from actual logs"
  )
  $requiredHostMetadata = @(
    "operating system and architecture",
    "GPU name and driver version",
    "CUDA runtime/toolkit version",
    "TensorRT runtime version and package identity",
    ".NET SDK/runtime version"
  )
  $requiredPackageIdentity = @(
    "package id and version under test",
    "nupkg path or published package URL",
    "nupkg SHA256",
    "clean consumer restore source and lock file context"
  )
  $requiredCommandCapture = @(
    "exact command line",
    "working directory",
    "environment variables relevant to native library loading",
    "exit code",
    "stdout log path",
    "stderr log path"
  )
  $requiredLogHash = @(
    "stdout SHA256",
    "stderr SHA256",
    "merged proof log SHA256",
    "validator output SHA256"
  )
  $requiredValidatorOutput = @(
    "strict validator JSON output",
    "strict validator markdown output",
    "failedBlockerCount=0",
    "no forbidden substitute present",
    "promotion guard requirements satisfied before promotion review"
  )
  $forbiddenSubstitutes = @(
    "template-only record",
    "candidate-only record",
    "field delta pack",
    "promotion guard",
    "hash-only evidence",
    "local-feed-only package check",
    "ProjectReference consumer",
    "direct nupkg smoke without package identity and logs",
    "DependencyProbe-only output",
    "sidecar-only or parse-only evidence"
  )

  $blockedReasons = @()
  if ($blockedFieldCount -gt 0) { $blockedReasons += "strict candidate still has blocked field contracts: $blockedFieldCount" }
  if ($blockedDeltaCount -gt 0) { $blockedReasons += "Owner field deltas still blocked: $blockedDeltaCount" }
  if ($blockedRequirementCount -gt 0) { $blockedReasons += "promotion guard requirements still blocked: $blockedRequirementCount" }
  if (-not $promotionAllowed) { $blockedReasons += "promotion guard has not allowed candidate review" }

  $validatorReady = $blockedReasons.Count -eq 0

  [pscustomobject]@{
    validatorContractId = "$candidateId-validator-contract"
    candidateId = $candidateId
    proofLane = $proofLane
    sourceReportItemId = [string](Get-PropertyOrDefault -Object $Candidate -Name "sourceReportItemId" -DefaultValue "unknown-report-item")
    sourceRecordId = [string](Get-PropertyOrDefault -Object $Candidate -Name "sourceRecordId" -DefaultValue "unknown-source-record")
    sourceTrackId = [string](Get-PropertyOrDefault -Object $Candidate -Name "sourceTrackId" -DefaultValue "unknown-source-track")
    requiredRuntimeEvidence = $requiredRuntimeEvidence
    requiredHostMetadata = $requiredHostMetadata
    requiredPackageIdentity = $requiredPackageIdentity
    requiredCommandCapture = $requiredCommandCapture
    requiredLogHash = $requiredLogHash
    requiredValidatorOutput = $requiredValidatorOutput
    forbiddenSubstitutes = $forbiddenSubstitutes
    ownerReviewRequired = $true
    promotionGuardState = $promotionGuardState
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordValidator.ps1 -Strict"
    proofBoundary = "This validator contract defines the minimum fields for future real proof records. It is not runtime execution proof, package publish approval, post-publish verification, or release-close approval."
    blockedReasons = @($blockedReasons)
    validatorReady = $validatorReady
    contractState = if ($validatorReady) { "ready-real-proof-record-validator-contract" } else { "blocked-real-proof-record-validation-input-required" }
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
  }
}

$resolvedStrictRecordPath = Resolve-InputPath -Path $StrictRecordPath
$resolvedDeltaPackPath = Resolve-InputPath -Path $DeltaPackPath
$resolvedPromotionGuardPath = Resolve-InputPath -Path $PromotionGuardPath
foreach ($path in @($resolvedStrictRecordPath, $resolvedDeltaPackPath, $resolvedPromotionGuardPath)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required real proof input not found: $path" }
}

$strictRecord = Get-Content -LiteralPath $resolvedStrictRecordPath -Raw -Encoding utf8 | ConvertFrom-Json
$deltaPack = Get-Content -LiteralPath $resolvedDeltaPackPath -Raw -Encoding utf8 | ConvertFrom-Json
$promotionGuard = Get-Content -LiteralPath $resolvedPromotionGuardPath -Raw -Encoding utf8 | ConvertFrom-Json

$candidates = @(Get-PropertyOrDefault -Object $strictRecord -Name "strictCandidateRecords" -DefaultValue @())
$deltas = @(Get-PropertyOrDefault -Object $deltaPack -Name "fieldDeltas" -DefaultValue @())
$guardItems = @(Get-PropertyOrDefault -Object $promotionGuard -Name "guardItems" -DefaultValue @())

$contracts = @()
foreach ($candidate in $candidates) {
  $candidateId = [string](Get-PropertyOrDefault -Object $candidate -Name "candidateId" -DefaultValue "unknown-candidate")
  $guardItem = @($guardItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "candidateId" -DefaultValue "") -eq $candidateId } | Select-Object -First 1)[0]
  $candidateDeltas = @($deltas | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "candidateId" -DefaultValue "") -eq $candidateId })
  $contracts += New-ValidatorContract -Candidate $candidate -GuardItem $guardItem -CandidateDeltas $candidateDeltas
}

$blockedValidatorContractCount = @($contracts | Where-Object { -not [bool]$_.validatorReady }).Count
$readyValidatorContractCount = @($contracts | Where-Object { [bool]$_.validatorReady }).Count

$recordOut = [pscustomobject]@{
  recordKind = "real-proof-record-validator"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  strictRecordPath = $resolvedStrictRecordPath
  deltaPackPath = $resolvedDeltaPackPath
  promotionGuardPath = $resolvedPromotionGuardPath
  validatorState = "blocked-real-proof-record-validation-input-required"
  candidateCount = $contracts.Count
  blockedValidatorContractCount = $blockedValidatorContractCount
  readyValidatorContractCount = $readyValidatorContractCount
  validatorContracts = @($contracts)
  sourceArtifacts = @(
    "artifacts/final-release/real-proof-input-candidate-strict-record.json",
    "artifacts/final-release/real-proof-input-candidate-strict-record-validation.json",
    "artifacts/final-release/owner-real-proof-field-delta-pack.json",
    "artifacts/final-release/owner-real-proof-field-delta-pack-validation.json",
    "artifacts/final-release/real-proof-candidate-promotion-guard.json",
    "artifacts/final-release/real-proof-candidate-promotion-guard-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This record defines strict validator contracts for future real proof input. It does not execute runtime smoke tests, publish packages, verify post-publish channels, or close the release issue."
}

$jsonPath = Join-Path $OutputRoot "real-proof-record-validator.json"
$markdownPath = Join-Path $OutputRoot "real-proof-record-validator.md"
$recordOut | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Real Proof Record Validator")
$lines.Add("")
$lines.Add("`real-proof-record-validator` 将 strict candidate、Owner delta 与 promotion guard 收敛为未来真实 proof record 的严格 validator contract。它不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| validatorState | ``$(ConvertTo-MarkdownCell $recordOut.validatorState)`` |")
$lines.Add("| candidateCount | ``$($recordOut.candidateCount)`` |")
$lines.Add("| blockedValidatorContractCount | ``$($recordOut.blockedValidatorContractCount)`` |")
$lines.Add("| readyValidatorContractCount | ``$($recordOut.readyValidatorContractCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($recordOut.canPromoteRuntimeProof)`` |")
$lines.Add("| canPublishPublicly | ``$($recordOut.canPublishPublicly)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($recordOut.canCloseReleaseIssue)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($recordOut.isRuntimeExecutionProof)`` |")
$lines.Add("| isReleaseCloseProof | ``$($recordOut.isReleaseCloseProof)`` |")
$lines.Add("")
$lines.Add("## Validator Contracts")
$lines.Add("")
$lines.Add("| Candidate | Lane | State | Blocked Reasons | Validator Command |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($contract in $contracts) {
  $lines.Add("| $(ConvertTo-MarkdownCell $contract.candidateId) | $(ConvertTo-MarkdownCell $contract.proofLane) | $(ConvertTo-MarkdownCell $contract.contractState) | ``$(@($contract.blockedReasons).Count)`` | ``$(ConvertTo-MarkdownCell $contract.validatorCommand)`` |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($recordOut.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real proof record validator written to $jsonPath"
Write-Host "Real proof record validator markdown written to $markdownPath"
Write-Host "ValidatorState=$($recordOut.validatorState) Candidates=$($recordOut.candidateCount) Blocked=$($recordOut.blockedValidatorContractCount) Ready=$($recordOut.readyValidatorContractCount)"
