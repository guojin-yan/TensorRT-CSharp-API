[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-real-proof-report-pack.json",
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

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-ArrayReady {
  param([AllowNull()][object[]]$Items, [scriptblock]$Predicate)

  $values = @($Items)
  if ($values.Count -eq 0) { return $false }

  foreach ($item in $values) {
    if (-not (& $Predicate $item)) { return $false }
  }

  return $true
}

function New-CandidateFieldContract {
  param(
    [string]$Name,
    [string]$Source,
    [string]$Rule,
    [bool]$Ready
  )

  [pscustomobject]@{
    name = $Name
    source = $Source
    required = $true
    rule = $Rule
    ready = $Ready
    candidateStatus = if ($Ready) { "candidate-field-ready" } else { "blocked-candidate-field-input-required" }
  }
}

function New-CandidateStrictRecord {
  param([object]$ReportItem)

  $reportItemId = [string](Get-PropertyOrDefault -Object $ReportItem -Name "reportItemId" -DefaultValue "unknown-report-item")
  $proofLane = [string](Get-PropertyOrDefault -Object $ReportItem -Name "proofLane" -DefaultValue "unknown-proof-lane")
  $ownerInputs = @(Get-PropertyOrDefault -Object $ReportItem -Name "requiredOwnerInputs" -DefaultValue @())
  $evidenceFiles = @(Get-PropertyOrDefault -Object $ReportItem -Name "requiredEvidenceFiles" -DefaultValue @())
  $hashes = @(Get-PropertyOrDefault -Object $ReportItem -Name "requiredHashes" -DefaultValue @())
  $commands = @(Get-PropertyOrDefault -Object $ReportItem -Name "requiredCommands" -DefaultValue @())
  $validators = @(Get-PropertyOrDefault -Object $ReportItem -Name "requiredValidators" -DefaultValue @())
  $forbiddenChecklist = @(Get-PropertyOrDefault -Object $ReportItem -Name "forbiddenSubstituteChecklist" -DefaultValue @())
  $reviewChecklist = @(Get-PropertyOrDefault -Object $ReportItem -Name "reviewChecklist" -DefaultValue @())

  $ownerInputsReady = Test-ArrayReady -Items $ownerInputs -Predicate {
    param($item)
    (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $item -Name "currentValue" -DefaultValue ""))) -and
      [bool](Get-PropertyOrDefault -Object $item -Name "ready" -DefaultValue $false)
  }
  $evidenceFilesReady = Test-ArrayReady -Items $evidenceFiles -Predicate {
    param($item)
    (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $item -Name "path" -DefaultValue ""))) -and
      (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $item -Name "sha256" -DefaultValue ""))) -and
      [bool](Get-PropertyOrDefault -Object $item -Name "exists" -DefaultValue $false) -and
      [bool](Get-PropertyOrDefault -Object $item -Name "hashReady" -DefaultValue $false)
  }
  $hashesReady = Test-ArrayReady -Items $hashes -Predicate {
    param($item)
    (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $item -Name "path" -DefaultValue ""))) -and
      (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $item -Name "sha256" -DefaultValue ""))) -and
      [bool](Get-PropertyOrDefault -Object $item -Name "matches" -DefaultValue $false)
  }
  $commandsReady = Test-ArrayReady -Items $commands -Predicate {
    param($item)
    (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $item -Name "command" -DefaultValue ""))) -and
      (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $item -Name "logPath" -DefaultValue ""))) -and
      (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $item -Name "logSha256" -DefaultValue ""))) -and
      [bool](Get-PropertyOrDefault -Object $item -Name "captured" -DefaultValue $false)
  }
  $validatorsReady = Test-ArrayReady -Items $validators -Predicate {
    param($item)
    (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $item -Name "command" -DefaultValue ""))) -and
      (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $item -Name "logPath" -DefaultValue ""))) -and
      (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $item -Name "logSha256" -DefaultValue ""))) -and
      [bool](Get-PropertyOrDefault -Object $item -Name "passed" -DefaultValue $false)
  }
  $forbiddenReady = Test-ArrayReady -Items $forbiddenChecklist -Predicate {
    param($item)
    [bool](Get-PropertyOrDefault -Object $item -Name "checked" -DefaultValue $false) -and
      (-not [bool](Get-PropertyOrDefault -Object $item -Name "present" -DefaultValue $true)) -and
      [bool](Get-PropertyOrDefault -Object $item -Name "passed" -DefaultValue $false)
  }
  $reviewReady = Test-ArrayReady -Items $reviewChecklist -Predicate {
    param($item)
    [bool](Get-PropertyOrDefault -Object $item -Name "passed" -DefaultValue $false)
  }

  $fieldContracts = @(
    New-CandidateFieldContract -Name "ownerInputs" -Source "requiredOwnerInputs" -Rule "all owner input values must be non-placeholder and ready=true" -Ready $ownerInputsReady
    New-CandidateFieldContract -Name "evidenceFiles" -Source "requiredEvidenceFiles" -Rule "all evidence paths and sha256 values must be non-placeholder; exists/hashReady must be true" -Ready $evidenceFilesReady
    New-CandidateFieldContract -Name "hashes" -Source "requiredHashes" -Rule "all expected/computed hashes must be non-placeholder and matches=true" -Ready $hashesReady
    New-CandidateFieldContract -Name "commands" -Source "requiredCommands" -Rule "all commands must include captured logs and log hashes" -Ready $commandsReady
    New-CandidateFieldContract -Name "validators" -Source "requiredValidators" -Rule "all validators must include command, log path, log hash, and passed=true" -Ready $validatorsReady
    New-CandidateFieldContract -Name "forbiddenSubstitutes" -Source "forbiddenSubstituteChecklist" -Rule "all forbidden substitute checks must be checked, absent, and passed" -Ready $forbiddenReady
    New-CandidateFieldContract -Name "ownerReview" -Source "reviewChecklist" -Rule "all owner review checklist items must pass" -Ready $reviewReady
  )

  $blockedFieldCount = @($fieldContracts | Where-Object { -not [bool]$_.ready }).Count
  $ready = $blockedFieldCount -eq 0

  [pscustomobject]@{
    candidateId = "$reportItemId-strict-candidate"
    sourceReportItemId = $reportItemId
    sourceRecordId = [string](Get-PropertyOrDefault -Object $ReportItem -Name "sourceRecordId" -DefaultValue "unknown-source-record")
    sourceTrackId = [string](Get-PropertyOrDefault -Object $ReportItem -Name "sourceTrackId" -DefaultValue "unknown-source-track")
    proofKind = [string](Get-PropertyOrDefault -Object $ReportItem -Name "proofKind" -DefaultValue "unknown-proof-kind")
    proofLane = $proofLane
    candidateState = if ($ready) { "strict-candidate-ready-for-owner-review" } else { "blocked-real-proof-input-candidate-required" }
    fieldContracts = $fieldContracts
    blockedFieldCount = $blockedFieldCount
    readyForOwnerReview = $false
    readyForPromotion = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    boundary = "This strict candidate record checks whether owner-filled report fields are complete enough to become a candidate. It is not runtime proof, publish approval, post-publish verification, or release-close approval."
  }
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner real proof report pack not found: $resolvedInputPath"
}

$pack = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$reportItems = @(Get-PropertyOrDefault -Object $pack -Name "proofReportItems" -DefaultValue @())
$candidates = @($reportItems | ForEach-Object { New-CandidateStrictRecord -ReportItem $_ })
$readyCandidateCount = @($candidates | Where-Object { [string]$_.candidateState -eq "strict-candidate-ready-for-owner-review" }).Count
$blockedCandidateCount = @($candidates | Where-Object { ([string]$_.candidateState).StartsWith("blocked", [StringComparison]::Ordinal) }).Count

$recordOut = [pscustomobject]@{
  recordKind = "real-proof-input-candidate-strict-record"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  sourceReportPackPath = $resolvedInputPath
  sourceReportPackState = [string](Get-PropertyOrDefault -Object $pack -Name "packState" -DefaultValue "missing-owner-real-proof-report-pack-state")
  candidateState = "blocked-real-proof-input-candidate-required"
  candidateCount = $candidates.Count
  blockedCandidateCount = $blockedCandidateCount
  readyCandidateCount = $readyCandidateCount
  strictCandidateRecords = $candidates
  requiredProofLanes = @(Get-PropertyOrDefault -Object $pack -Name "requiredProofLanes" -DefaultValue @())
  forbiddenSubstitutes = @(Get-PropertyOrDefault -Object $pack -Name "forbiddenSubstitutes" -DefaultValue @())
  sourceArtifacts = @(
    "artifacts/final-release/owner-real-proof-report-pack.json",
    "artifacts/final-release/owner-real-proof-report-pack-validation.json",
    "artifacts/final-release/real-proof-execution-record-projection.json",
    "artifacts/final-release/real-proof-execution-record-projection-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This strict candidate record converts owner report pack items into candidate contracts only. It is blocked until real owner evidence, logs, hashes, validators, forbidden-substitute checks, and owner review are complete; it is not proof or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "real-proof-input-candidate-strict-record.json"
$markdownPath = Join-Path $OutputRoot "real-proof-input-candidate-strict-record.md"
$recordOut | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Real Proof Input Candidate Strict Record")
$lines.Add("")
$lines.Add("`real-proof-input-candidate-strict-record` 将 Owner real proof report pack 收敛为更严格的 candidate contract。它不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| candidateState | ``$(ConvertTo-MarkdownCell $recordOut.candidateState)`` |")
$lines.Add("| candidateCount | ``$($recordOut.candidateCount)`` |")
$lines.Add("| blockedCandidateCount | ``$($recordOut.blockedCandidateCount)`` |")
$lines.Add("| readyCandidateCount | ``$($recordOut.readyCandidateCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($recordOut.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($recordOut.canCloseReleaseIssue)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($recordOut.isRuntimeExecutionProof)`` |")
$lines.Add("| isReleaseCloseProof | ``$($recordOut.isReleaseCloseProof)`` |")
$lines.Add("")
$lines.Add("## Strict Candidates")
$lines.Add("")
$lines.Add("| Candidate | Proof Lane | State | Blocked Fields |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($candidate in $candidates) {
  $lines.Add("| $(ConvertTo-MarkdownCell $candidate.candidateId) | $(ConvertTo-MarkdownCell $candidate.proofLane) | $(ConvertTo-MarkdownCell $candidate.candidateState) | ``$($candidate.blockedFieldCount)`` |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($recordOut.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real proof input candidate strict record written to $jsonPath"
Write-Host "Real proof input candidate strict record markdown written to $markdownPath"
Write-Host "CandidateState=$($recordOut.candidateState) Candidates=$($recordOut.candidateCount) Blocked=$($recordOut.blockedCandidateCount) Ready=$($recordOut.readyCandidateCount)"
