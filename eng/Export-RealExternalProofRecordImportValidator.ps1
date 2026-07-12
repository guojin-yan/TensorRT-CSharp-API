[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-external-proof-execution-result-import.json",
  [string]$ResultImportValidationPath = "artifacts\final-release\owner-external-proof-execution-result-import-validation.json",
  [string]$ExecutionBundlePath = "artifacts\final-release\owner-external-proof-execution-bundle.json",
  [string]$OutputRoot,
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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Resolve-RepositoryPath -Path $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-CandidateContract {
  param([object]$ImportItem)

  $itemId = [string](Get-PropertyOrDefault -Object $ImportItem -Name "resultImportItemId" -DefaultValue "unknown-import-item")
  $ready = [bool](Get-PropertyOrDefault -Object $ImportItem -Name "readyForRealProofRecordImport" -DefaultValue $false)
  $missingCount = [int](Get-PropertyOrDefault -Object $ImportItem -Name "missingResultFieldCount" -DefaultValue 0)
  $failedFileCount = [int](Get-PropertyOrDefault -Object $ImportItem -Name "failedFileEvidenceCheckCount" -DefaultValue 0)
  $fileMissingCount = [int](Get-PropertyOrDefault -Object $ImportItem -Name "fileMissingCount" -DefaultValue 0)
  $invalidSha256Count = [int](Get-PropertyOrDefault -Object $ImportItem -Name "invalidSha256Count" -DefaultValue 0)
  $hashMismatchCount = [int](Get-PropertyOrDefault -Object $ImportItem -Name "hashMismatchCount" -DefaultValue 0)
  $outsideAllowedEvidenceRootCount = [int](Get-PropertyOrDefault -Object $ImportItem -Name "outsideAllowedEvidenceRootCount" -DefaultValue 0)
  $forbiddenSubstituteFindingCount = [int](Get-PropertyOrDefault -Object $ImportItem -Name "forbiddenSubstituteFindingCount" -DefaultValue 0)
  $passedTrue = [bool](Get-PropertyOrDefault -Object $ImportItem -Name "passedTrue" -DefaultValue $false)
  $runtimeExecutionFieldsReady = [bool](Get-PropertyOrDefault -Object $ImportItem -Name "runtimeExecutionFieldsReady" -DefaultValue $false)
  $blockedReasons = New-Object System.Collections.Generic.List[string]
  if (-not $ready) { $blockedReasons.Add("owner external proof result import is not ready") | Out-Null }
  if ($missingCount -gt 0) { $blockedReasons.Add("missing real result fields: $missingCount") | Out-Null }
  if ($failedFileCount -gt 0) { $blockedReasons.Add("missing or mismatched file/hash evidence: $failedFileCount") | Out-Null }
  if ($fileMissingCount -gt 0) { $blockedReasons.Add("real evidence files missing: $fileMissingCount") | Out-Null }
  if ($invalidSha256Count -gt 0) { $blockedReasons.Add("missing or invalid SHA256 values: $invalidSha256Count") | Out-Null }
  if ($hashMismatchCount -gt 0) { $blockedReasons.Add("SHA256 mismatches: $hashMismatchCount") | Out-Null }
  if ($outsideAllowedEvidenceRootCount -gt 0) { $blockedReasons.Add("evidence paths outside allowed roots: $outsideAllowedEvidenceRootCount") | Out-Null }
  if ($forbiddenSubstituteFindingCount -gt 0) { $blockedReasons.Add("forbidden substitute findings: $forbiddenSubstituteFindingCount") | Out-Null }
  if (-not $passedTrue) { $blockedReasons.Add("passed flag is missing or not true") | Out-Null }
  if (-not $runtimeExecutionFieldsReady) { $blockedReasons.Add("stdout, stderr, and merged transcript paths are required") | Out-Null }

  [pscustomobject]@{
    candidateContractId = "$itemId-real-external-proof-record-import-contract"
    resultImportItemId = $itemId
    resultInputId = [string](Get-PropertyOrDefault -Object $ImportItem -Name "resultInputId" -DefaultValue "")
    executionInputId = [string](Get-PropertyOrDefault -Object $ImportItem -Name "executionInputId" -DefaultValue "")
    candidateId = [string](Get-PropertyOrDefault -Object $ImportItem -Name "candidateId" -DefaultValue "")
    proofLane = [string](Get-PropertyOrDefault -Object $ImportItem -Name "proofLane" -DefaultValue "")
    runtimePackageKey = [string](Get-PropertyOrDefault -Object $ImportItem -Name "runtimePackageKey" -DefaultValue "")
    contractState = if ($ready) { "real-external-proof-record-import-ready" } else { "blocked-real-external-proof-record-import-required" }
    ownerResultProvided = [bool](Get-PropertyOrDefault -Object $ImportItem -Name "ownerResultProvided" -DefaultValue $false)
    missingResultFieldCount = $missingCount
    failedFileEvidenceCheckCount = $failedFileCount
    fileMissingCount = $fileMissingCount
    invalidSha256Count = $invalidSha256Count
    hashMismatchCount = $hashMismatchCount
    outsideAllowedEvidenceRootCount = $outsideAllowedEvidenceRootCount
    forbiddenSubstituteFindingCount = $forbiddenSubstituteFindingCount
    passedTrue = $passedTrue
    runtimeExecutionFieldsReady = $runtimeExecutionFieldsReady
    blockedReasons = @($blockedReasons.ToArray())
    forbiddenSubstitutes = @(
      "local feed substituted for public package proof",
      "ProjectReference substituted for package-consumer proof",
      "direct nupkg substituted for post-publish proof",
      "DependencyProbe-only output",
      "build-only output",
      "precheck-only output",
      "sidecar-only output",
      "skipped run",
      "mismatched SHA256"
    )
    strictValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealExternalProofRecordImportValidator.ps1 -Strict"
    downstreamStrictValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordValidator.ps1 -Strict"
    importTarget = "artifacts/final-release/real-external-proof-record-import-candidate.json"
    readyForPromotionGuard = $ready
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    nonProofBoundary = "This contract can only become promotable after owner result import has real files, matching SHA256, host metadata, package identity, validator output, reviewer data, and non-substitute checks."
  }
}

$resultImport = Read-JsonOrNull $InputPath
$resultImportValidation = Read-JsonOrNull $ResultImportValidationPath
$executionBundle = Read-JsonOrNull $ExecutionBundlePath

if ($null -eq $resultImport) {
  throw "Missing owner external proof execution result import: $(Resolve-RepositoryPath -Path $InputPath). Run Import-OwnerExternalProofExecutionResult.ps1 first."
}

$importItems = @(Get-PropertyOrDefault -Object $resultImport -Name "resultImportItems" -DefaultValue @())
$contracts = @($importItems | ForEach-Object { New-CandidateContract -ImportItem $_ })
$blockedContracts = @($contracts | Where-Object { [string]$_.contractState -eq "blocked-real-external-proof-record-import-required" })
$readyContracts = @($contracts | Where-Object { [bool]$_.readyForPromotionGuard })
$blockedReasonCount = ($contracts | ForEach-Object { @($_.blockedReasons).Count } | Measure-Object -Sum).Sum
if ($null -eq $blockedReasonCount) { $blockedReasonCount = 0 }
$fileMissingCount = ($contracts | ForEach-Object { [int]$_.fileMissingCount } | Measure-Object -Sum).Sum
if ($null -eq $fileMissingCount) { $fileMissingCount = 0 }
$invalidSha256Count = ($contracts | ForEach-Object { [int]$_.invalidSha256Count } | Measure-Object -Sum).Sum
if ($null -eq $invalidSha256Count) { $invalidSha256Count = 0 }
$hashMismatchCount = ($contracts | ForEach-Object { [int]$_.hashMismatchCount } | Measure-Object -Sum).Sum
if ($null -eq $hashMismatchCount) { $hashMismatchCount = 0 }
$outsideAllowedEvidenceRootCount = ($contracts | ForEach-Object { [int]$_.outsideAllowedEvidenceRootCount } | Measure-Object -Sum).Sum
if ($null -eq $outsideAllowedEvidenceRootCount) { $outsideAllowedEvidenceRootCount = 0 }
$forbiddenSubstituteFindingCount = ($contracts | ForEach-Object { [int]$_.forbiddenSubstituteFindingCount } | Measure-Object -Sum).Sum
if ($null -eq $forbiddenSubstituteFindingCount) { $forbiddenSubstituteFindingCount = 0 }
$laneReadinessSummary = @($contracts | ForEach-Object {
    [pscustomobject]@{
      proofLane = [string]$_.proofLane
      contractState = [string]$_.contractState
      readyForStrictValidator = [bool]$_.readyForPromotionGuard
      blockedReasons = @($_.blockedReasons)
      strictValidatorInputOnly = $true
      canPromoteRuntimeProof = $false
      canCloseReleaseIssue = $false
    }
  })
$summary = [pscustomobject]@{
  readyForStrictValidatorContractCount = $readyContracts.Count
  blockedContractCount = $blockedContracts.Count
  readyForStrictValidatorLanes = @($laneReadinessSummary | Where-Object { [bool]$_.readyForStrictValidator } | ForEach-Object { [string]$_.proofLane })
  blockedLanes = @($laneReadinessSummary | Where-Object { -not [bool]$_.readyForStrictValidator } | ForEach-Object { [string]$_.proofLane })
  strictValidatorInputOnly = $true
  proofPromotionAllowed = $false
  releaseCloseAllowed = $false
}

$record = [pscustomobject]@{
  recordKind = "real-external-proof-record-import-validator"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validatorState = if ($readyContracts.Count -eq $contracts.Count -and $contracts.Count -gt 0) { "real-external-proof-record-import-ready" } else { "blocked-real-external-proof-record-import-required" }
  ownerExternalProofExecutionResultImportState = [string](Get-PropertyOrDefault -Object $resultImport -Name "importState" -DefaultValue "missing-owner-external-proof-execution-result-import")
  ownerExternalProofExecutionResultImportValidationState = [string](Get-PropertyOrDefault -Object $resultImportValidation -Name "validationState" -DefaultValue "missing-owner-external-proof-execution-result-import-validation")
  ownerExternalProofExecutionBundleState = [string](Get-PropertyOrDefault -Object $executionBundle -Name "bundleState" -DefaultValue "missing-owner-external-proof-execution-bundle")
  candidateContractCount = $contracts.Count
  blockedCandidateContractCount = $blockedContracts.Count
  readyCandidateContractCount = $readyContracts.Count
  blockedReasonCount = [int]$blockedReasonCount
  fileMissingCount = [int]$fileMissingCount
  invalidSha256Count = [int]$invalidSha256Count
  hashMismatchCount = [int]$hashMismatchCount
  outsideAllowedEvidenceRootCount = [int]$outsideAllowedEvidenceRootCount
  forbiddenSubstituteFindingCount = [int]$forbiddenSubstituteFindingCount
  summary = $summary
  laneReadinessSummary = $laneReadinessSummary
  candidateContracts = $contracts
  sourceArtifacts = @(
    $InputPath,
    $ResultImportValidationPath,
    $ExecutionBundlePath
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  safetyBoundary = "Real external proof record import validator is a blocked validator contract until owner-provided external proof records are complete and non-substitute checks pass. It is not proof by itself."
}

$artifactRoot = if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  Join-Path $RepositoryRoot "artifacts\final-release"
}
else {
  Resolve-RepositoryPath -Path $OutputRoot
}
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "real-external-proof-record-import-validator.json"
$markdownPath = Join-Path $artifactRoot "real-external-proof-record-import-validator.md"
$record | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Real External Proof Record Import Validator")
$lines.Add("")
$lines.Add("| 项目 | 当前值 |")
$lines.Add("|---|---|")
$lines.Add("| validatorState | ``$(ConvertTo-MarkdownCell $record.validatorState)`` |")
$lines.Add("| candidateContractCount | ``$($record.candidateContractCount)`` |")
$lines.Add("| blockedCandidateContractCount | ``$($record.blockedCandidateContractCount)`` |")
$lines.Add("| readyCandidateContractCount | ``$($record.readyCandidateContractCount)`` |")
$lines.Add("| blockedReasonCount | ``$($record.blockedReasonCount)`` |")
$lines.Add("| fileMissingCount | ``$($record.fileMissingCount)`` |")
$lines.Add("| invalidSha256Count | ``$($record.invalidSha256Count)`` |")
$lines.Add("| hashMismatchCount | ``$($record.hashMismatchCount)`` |")
$lines.Add("| outsideAllowedEvidenceRootCount | ``$($record.outsideAllowedEvidenceRootCount)`` |")
$lines.Add("| forbiddenSubstituteFindingCount | ``$($record.forbiddenSubstituteFindingCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Candidate Contracts")
$lines.Add("")
$lines.Add("| Contract | Lane | State | Blockers |")
$lines.Add("|---|---|---|---:|")
foreach ($contract in $contracts) {
  $lines.Add("| $(ConvertTo-MarkdownCell $contract.candidateContractId) | $(ConvertTo-MarkdownCell $contract.proofLane) | $(ConvertTo-MarkdownCell $contract.contractState) | ``$(@($contract.blockedReasons).Count)`` |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.safetyBoundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real external proof record import validator written to $jsonPath"
Write-Host "Real external proof record import validator markdown written to $markdownPath"
Write-Host "ValidatorState=$($record.validatorState) Contracts=$($record.candidateContractCount) Blocked=$($record.blockedCandidateContractCount) BlockedReasons=$($record.blockedReasonCount)"
