[CmdletBinding()]
param(
  [string]$ImportValidatorPath = "artifacts\final-release\real-external-proof-record-import-validator.json",
  [string]$OwnerResultImportPath = "artifacts\final-release\owner-external-proof-execution-result-import.json",
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

function Get-ImportItemById {
  param([object[]]$ImportItems, [string]$ResultImportItemId)

  return @($ImportItems | Where-Object {
      [string](Get-PropertyOrDefault -Object $_ -Name "resultImportItemId" -DefaultValue "") -eq $ResultImportItemId
    } | Select-Object -First 1)[0]
}

function New-ProofCandidate {
  param(
    [object]$Contract,
    [AllowNull()][object]$ImportItem
  )

  $contractId = [string](Get-PropertyOrDefault -Object $Contract -Name "candidateContractId" -DefaultValue "unknown-contract")
  $resultImportItemId = [string](Get-PropertyOrDefault -Object $Contract -Name "resultImportItemId" -DefaultValue "")
  $proofLane = [string](Get-PropertyOrDefault -Object $Contract -Name "proofLane" -DefaultValue "")
  $candidateId = [string](Get-PropertyOrDefault -Object $Contract -Name "candidateId" -DefaultValue "")
  $runtimePackageKey = [string](Get-PropertyOrDefault -Object $Contract -Name "runtimePackageKey" -DefaultValue "")
  $fileEvidenceChecks = @(Get-PropertyOrDefault -Object $ImportItem -Name "fileEvidenceChecks" -DefaultValue @())
  $forbiddenSubstituteFindings = @(Get-PropertyOrDefault -Object $ImportItem -Name "forbiddenSubstituteFindings" -DefaultValue @())
  $blockedReasons = New-Object System.Collections.Generic.List[string]

  foreach ($reason in @(Get-PropertyOrDefault -Object $Contract -Name "blockedReasons" -DefaultValue @())) {
    if (-not [string]::IsNullOrWhiteSpace([string]$reason)) {
      $blockedReasons.Add([string]$reason) | Out-Null
    }
  }

  if ($null -eq $ImportItem) {
    $blockedReasons.Add("owner result import item missing for ready contract") | Out-Null
  }

  $strictValidatorInputsReady = $null -ne $ImportItem -and
    [bool](Get-PropertyOrDefault -Object $Contract -Name "readyForPromotionGuard" -DefaultValue $false) -and
    [bool](Get-PropertyOrDefault -Object $ImportItem -Name "readyForRealProofRecordImport" -DefaultValue $false) -and
    -not [bool](Get-PropertyOrDefault -Object $ImportItem -Name "canPromoteLaneResult" -DefaultValue $true)
  $resultArtifactPaths = @($fileEvidenceChecks | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "path" -DefaultValue "") } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  $hashProof = @($fileEvidenceChecks | ForEach-Object {
      [pscustomobject]@{
        id = [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")
        path = [string](Get-PropertyOrDefault -Object $_ -Name "path" -DefaultValue "")
        sha256 = [string](Get-PropertyOrDefault -Object $_ -Name "sha256" -DefaultValue "")
        actualSha256 = [string](Get-PropertyOrDefault -Object $_ -Name "actualSha256" -DefaultValue "")
        hashMatches = [bool](Get-PropertyOrDefault -Object $_ -Name "hashMatches" -DefaultValue $false)
        existsAndHashMatches = [bool](Get-PropertyOrDefault -Object $_ -Name "existsAndHashMatches" -DefaultValue $false)
      }
    })
  $blockedReasonText = if ($blockedReasons.Count -eq 0) { "" } else { @($blockedReasons.ToArray()) -join "; " }

  [pscustomobject]@{
    recordKind = "real-proof-record-candidate-from-owner-result-import-item"
    proofRecordCandidateId = "$candidateId-owner-result-import-candidate"
    sourceCandidateContractId = $contractId
    resultImportItemId = $resultImportItemId
    resultInputId = [string](Get-PropertyOrDefault -Object $Contract -Name "resultInputId" -DefaultValue "")
    executionInputId = [string](Get-PropertyOrDefault -Object $Contract -Name "executionInputId" -DefaultValue "")
    candidateId = $candidateId
    proofLane = $proofLane
    runtimePackageKey = $runtimePackageKey
    candidateState = if ($strictValidatorInputsReady) { "ready-for-strict-validator-input" } else { "blocked-owner-result-import-candidate-required" }
    strictValidatorInputReady = $strictValidatorInputsReady
    sourceContractState = [string](Get-PropertyOrDefault -Object $Contract -Name "contractState" -DefaultValue "")
    readyForPromotionGuard = [bool](Get-PropertyOrDefault -Object $Contract -Name "readyForPromotionGuard" -DefaultValue $false)
    ownerResultProvided = [bool](Get-PropertyOrDefault -Object $Contract -Name "ownerResultProvided" -DefaultValue $false)
    fileEvidenceChecks = @($fileEvidenceChecks)
    fileEvidenceCheckCount = $fileEvidenceChecks.Count
    sourceOwnerResultRow = [pscustomobject]@{
      resultImportItemId = $resultImportItemId
      resultInputId = [string](Get-PropertyOrDefault -Object $Contract -Name "resultInputId" -DefaultValue "")
      executionInputId = [string](Get-PropertyOrDefault -Object $Contract -Name "executionInputId" -DefaultValue "")
      proofLane = $proofLane
      runtimePackageKey = $runtimePackageKey
      ownerReviewer = [string](Get-PropertyOrDefault -Object $ImportItem -Name "ownerReviewer" -DefaultValue "")
      ownerReviewTimestampUtc = [string](Get-PropertyOrDefault -Object $ImportItem -Name "ownerReviewTimestampUtc" -DefaultValue "")
    }
    resultArtifactPaths = @($resultArtifactPaths)
    resultArtifactPath = if ($resultArtifactPaths.Count -gt 0) { [string]$resultArtifactPaths[0] } else { "" }
    hashProof = @($hashProof)
    blockedReason = $blockedReasonText
    missingResultFieldCount = [int](Get-PropertyOrDefault -Object $Contract -Name "missingResultFieldCount" -DefaultValue 0)
    failedFileEvidenceCheckCount = [int](Get-PropertyOrDefault -Object $Contract -Name "failedFileEvidenceCheckCount" -DefaultValue 0)
    fileMissingCount = [int](Get-PropertyOrDefault -Object $Contract -Name "fileMissingCount" -DefaultValue 0)
    invalidSha256Count = [int](Get-PropertyOrDefault -Object $Contract -Name "invalidSha256Count" -DefaultValue 0)
    hashMismatchCount = [int](Get-PropertyOrDefault -Object $Contract -Name "hashMismatchCount" -DefaultValue 0)
    outsideAllowedEvidenceRootCount = [int](Get-PropertyOrDefault -Object $Contract -Name "outsideAllowedEvidenceRootCount" -DefaultValue 0)
    forbiddenSubstituteFindingCount = [int](Get-PropertyOrDefault -Object $Contract -Name "forbiddenSubstituteFindingCount" -DefaultValue 0)
    forbiddenSubstituteFindings = @($forbiddenSubstituteFindings)
    ownerReviewer = [string](Get-PropertyOrDefault -Object $ImportItem -Name "ownerReviewer" -DefaultValue "")
    ownerReviewTimestampUtc = [string](Get-PropertyOrDefault -Object $ImportItem -Name "ownerReviewTimestampUtc" -DefaultValue "")
    ownerReviewReady = [bool](Get-PropertyOrDefault -Object $ImportItem -Name "ownerReviewReady" -DefaultValue $false)
    nonSubstituteConfirmationCount = [int](Get-PropertyOrDefault -Object $ImportItem -Name "nonSubstituteConfirmationCount" -DefaultValue 0)
    nonSubstituteConfirmationsReady = [bool](Get-PropertyOrDefault -Object $ImportItem -Name "nonSubstituteConfirmationsReady" -DefaultValue $false)
    blockedReasons = @($blockedReasons.ToArray())
    strictValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict"
    nextValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordValidator.ps1 -Strict"
    promotionGuardCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofCandidatePromotionGuard.ps1 -Strict"
    postPublishBoundary = if ($proofLane -eq "post-publish-verification") { "Post-publish candidate still requires a real public/private channel publish and clean consumer verification; package-consumer-runtime cannot substitute it." } else { "This candidate is not post-publish proof and cannot satisfy post-publish verification." }
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    proofBoundary = "Ready owner result import contracts may become strict validator input candidates only. They are not runtime proof, post-publish proof, publication approval, or release-close approval."
  }
}

$resolvedImportValidatorPath = Resolve-InputPath -Path $ImportValidatorPath
$resolvedOwnerResultImportPath = Resolve-InputPath -Path $OwnerResultImportPath
if (-not (Test-Path -LiteralPath $resolvedImportValidatorPath -PathType Leaf)) {
  throw "Real external proof record import validator not found: $resolvedImportValidatorPath"
}
if (-not (Test-Path -LiteralPath $resolvedOwnerResultImportPath -PathType Leaf)) {
  throw "Owner external proof execution result import not found: $resolvedOwnerResultImportPath"
}

$validator = Get-Content -LiteralPath $resolvedImportValidatorPath -Raw -Encoding utf8 | ConvertFrom-Json
$ownerImport = Get-Content -LiteralPath $resolvedOwnerResultImportPath -Raw -Encoding utf8 | ConvertFrom-Json
$contracts = @(Get-PropertyOrDefault -Object $validator -Name "candidateContracts" -DefaultValue @())
$readyContracts = @($contracts | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "readyForPromotionGuard" -DefaultValue $false) })
$importItems = @(Get-PropertyOrDefault -Object $ownerImport -Name "resultImportItems" -DefaultValue @())
$candidateItems = @($readyContracts | ForEach-Object {
    $importItem = Get-ImportItemById -ImportItems $importItems -ResultImportItemId ([string](Get-PropertyOrDefault -Object $_ -Name "resultImportItemId" -DefaultValue ""))
    New-ProofCandidate -Contract $_ -ImportItem $importItem
  })

$packageConsumerCandidateCount = @($candidateItems | Where-Object { [string]$_.proofLane -eq "package-consumer-runtime" }).Count
$postPublishCandidateCount = @($candidateItems | Where-Object { [string]$_.proofLane -eq "post-publish-verification" }).Count
$strictValidatorReadyCandidateCount = @($candidateItems | Where-Object { [bool]$_.strictValidatorInputReady }).Count
$laneGroups = @($candidateItems | Group-Object -Property proofLane | ForEach-Object {
    [pscustomobject]@{
      proofLane = [string]$_.Name
      candidateCount = @($_.Group).Count
      strictValidatorReadyCandidateCount = @($_.Group | Where-Object { [bool]$_.strictValidatorInputReady }).Count
      canPromoteRuntimeProof = $false
      isRuntimeExecutionProof = $false
      isPostPublishProof = $false
      canCloseReleaseIssue = $false
    }
  })
$summary = [pscustomobject]@{
  strictValidatorInputOnly = $true
  candidateLanes = @($laneGroups | ForEach-Object { [string]$_.proofLane })
  laneGroups = $laneGroups
  candidateCount = $candidateItems.Count
  strictValidatorReadyCandidateCount = $strictValidatorReadyCandidateCount
  proofPromotionAllowed = $false
  releaseCloseAllowed = $false
}

$recordOut = [pscustomobject]@{
  recordKind = "real-proof-record-candidate-from-owner-result-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  candidateState = if ($candidateItems.Count -gt 0) { "owner-result-import-candidate-ready-for-strict-validator" } else { "blocked-owner-result-import-candidate-required" }
  importValidatorPath = $resolvedImportValidatorPath
  ownerResultImportPath = $resolvedOwnerResultImportPath
  sourceValidatorState = [string](Get-PropertyOrDefault -Object $validator -Name "validatorState" -DefaultValue "")
  sourceCandidateContractCount = $contracts.Count
  sourceReadyCandidateContractCount = $readyContracts.Count
  candidateCount = $candidateItems.Count
  strictValidatorReadyCandidateCount = $strictValidatorReadyCandidateCount
  packageConsumerRuntimeCandidateCount = $packageConsumerCandidateCount
  postPublishVerificationCandidateCount = $postPublishCandidateCount
  summary = $summary
  laneGroups = $laneGroups
  candidateItems = @($candidateItems)
  sourceArtifacts = @(
    $ImportValidatorPath,
    $OwnerResultImportPath
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This record projects ready owner result import contracts into strict-validator input candidates only. It is not a real proof record, not post-publish proof, not publish approval, and not release-close approval."
}

$jsonPath = Join-Path $OutputRoot "real-proof-record-candidate-from-owner-result-import.json"
$markdownPath = Join-Path $OutputRoot "real-proof-record-candidate-from-owner-result-import.md"
$recordOut | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Real Proof Record Candidate From Owner Result Import")
$lines.Add("")
$lines.Add("该产物只把 ready owner result import contract 投影为 strict validator 的输入候选，不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| candidateState | ``$(ConvertTo-MarkdownCell $recordOut.candidateState)`` |")
$lines.Add("| sourceCandidateContractCount | ``$($recordOut.sourceCandidateContractCount)`` |")
$lines.Add("| sourceReadyCandidateContractCount | ``$($recordOut.sourceReadyCandidateContractCount)`` |")
$lines.Add("| candidateCount | ``$($recordOut.candidateCount)`` |")
$lines.Add("| strictValidatorReadyCandidateCount | ``$($recordOut.strictValidatorReadyCandidateCount)`` |")
$lines.Add("| packageConsumerRuntimeCandidateCount | ``$($recordOut.packageConsumerRuntimeCandidateCount)`` |")
$lines.Add("| postPublishVerificationCandidateCount | ``$($recordOut.postPublishVerificationCandidateCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($recordOut.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($recordOut.canCloseReleaseIssue)`` |")
$lines.Add("| isPostPublishProof | ``$($recordOut.isPostPublishProof)`` |")
$lines.Add("")
$lines.Add("## Candidates")
$lines.Add("")
$lines.Add("| Candidate | Lane | State | Files | Owner | Non-substitute |")
$lines.Add("| --- | --- | --- | ---: | --- | --- |")
foreach ($candidate in $candidateItems) {
  $lines.Add("| $(ConvertTo-MarkdownCell $candidate.proofRecordCandidateId) | $(ConvertTo-MarkdownCell $candidate.proofLane) | $(ConvertTo-MarkdownCell $candidate.candidateState) | ``$($candidate.fileEvidenceCheckCount)`` | $(ConvertTo-MarkdownCell $candidate.ownerReviewer) | ``$($candidate.nonSubstituteConfirmationsReady)`` |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($recordOut.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real proof record candidate from owner result import written to $jsonPath"
Write-Host "Real proof record candidate from owner result import markdown written to $markdownPath"
Write-Host "CandidateState=$($recordOut.candidateState) Candidates=$($recordOut.candidateCount) StrictValidatorReady=$($recordOut.strictValidatorReadyCandidateCount)"
