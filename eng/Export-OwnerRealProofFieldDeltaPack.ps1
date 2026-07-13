[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-proof-input-candidate-strict-record.json",
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

function Get-OwnerAction {
  param([string]$FieldName)

  switch ($FieldName) {
    "ownerInputs" { return "Fill host/runtime/package metadata and mark each owner input ready." }
    "evidenceFiles" { return "Provide real evidence file paths, SHA256 values, exists=true, and hashReady=true." }
    "hashes" { return "Provide expected/computed hash evidence and make every hash match." }
    "commands" { return "Capture real command execution logs with log SHA256 values." }
    "validators" { return "Run strict validators and provide passing validator logs with hashes." }
    "forbiddenSubstitutes" { return "Review forbidden substitutes and prove they are absent." }
    "ownerReview" { return "Complete Owner review checklist after real evidence and validators are present." }
    default { return "Complete the blocked strict candidate field contract." }
  }
}

function New-FieldDelta {
  param(
    [object]$Candidate,
    [object]$FieldContract
  )

  $candidateId = [string](Get-PropertyOrDefault -Object $Candidate -Name "candidateId" -DefaultValue "unknown-candidate")
  $fieldName = [string](Get-PropertyOrDefault -Object $FieldContract -Name "name" -DefaultValue "unknown-field")

  [pscustomobject]@{
    fieldDeltaId = "$candidateId-$fieldName-delta"
    candidateId = $candidateId
    sourceReportItemId = [string](Get-PropertyOrDefault -Object $Candidate -Name "sourceReportItemId" -DefaultValue "unknown-report-item")
    sourceRecordId = [string](Get-PropertyOrDefault -Object $Candidate -Name "sourceRecordId" -DefaultValue "unknown-source-record")
    sourceTrackId = [string](Get-PropertyOrDefault -Object $Candidate -Name "sourceTrackId" -DefaultValue "unknown-source-track")
    proofLane = [string](Get-PropertyOrDefault -Object $Candidate -Name "proofLane" -DefaultValue "unknown-proof-lane")
    fieldName = $fieldName
    source = [string](Get-PropertyOrDefault -Object $FieldContract -Name "source" -DefaultValue "unknown-source")
    rule = [string](Get-PropertyOrDefault -Object $FieldContract -Name "rule" -DefaultValue "missing-rule")
    currentStatus = [string](Get-PropertyOrDefault -Object $FieldContract -Name "candidateStatus" -DefaultValue "blocked-candidate-field-input-required")
    ownerAction = Get-OwnerAction -FieldName $fieldName
    targetReportPackPath = "artifacts/final-release/owner-real-proof-report-pack.json"
    targetStrictRecordPath = "artifacts/final-release/real-proof-input-candidate-strict-record.json"
    targetValidationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofInputCandidateStrictRecord.ps1 -Strict"
    required = $true
    ready = [bool](Get-PropertyOrDefault -Object $FieldContract -Name "ready" -DefaultValue $false)
    deltaState = if ([bool](Get-PropertyOrDefault -Object $FieldContract -Name "ready" -DefaultValue $false)) { "ready-field-contract" } else { "blocked-owner-real-proof-field-delta-required" }
    nonSubstituteBoundary = "Template, candidate, report-pack, hash-only, local-feed, ProjectReference, direct nupkg, DependencyProbe, and checklist-only evidence cannot satisfy this field delta."
  }
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Real proof input candidate strict record not found: $resolvedInputPath"
}

$strictRecord = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$candidates = @(Get-PropertyOrDefault -Object $strictRecord -Name "strictCandidateRecords" -DefaultValue @())
$fieldDeltas = @()

foreach ($candidate in $candidates) {
  foreach ($contract in @(Get-PropertyOrDefault -Object $candidate -Name "fieldContracts" -DefaultValue @())) {
    if (-not [bool](Get-PropertyOrDefault -Object $contract -Name "ready" -DefaultValue $false)) {
      $fieldDeltas += New-FieldDelta -Candidate $candidate -FieldContract $contract
    }
  }
}

$blockedFieldContractCount = @($fieldDeltas | Where-Object { [string]$_.deltaState -eq "blocked-owner-real-proof-field-delta-required" }).Count
$readyFieldContractCount = @($fieldDeltas | Where-Object { [bool]$_.ready }).Count

$recordOut = [pscustomobject]@{
  recordKind = "owner-real-proof-field-delta-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  sourceStrictRecordPath = $resolvedInputPath
  sourceStrictRecordState = [string](Get-PropertyOrDefault -Object $strictRecord -Name "candidateState" -DefaultValue "missing-real-proof-input-candidate-strict-record-state")
  deltaState = "blocked-owner-real-proof-field-delta-required"
  candidateCount = $candidates.Count
  fieldDeltaCount = $fieldDeltas.Count
  blockedFieldContractCount = $blockedFieldContractCount
  readyFieldContractCount = $readyFieldContractCount
  fieldDeltas = @($fieldDeltas)
  sourceArtifacts = @(
    "artifacts/final-release/real-proof-input-candidate-strict-record.json",
    "artifacts/final-release/real-proof-input-candidate-strict-record-validation.json",
    "artifacts/final-release/owner-real-proof-report-pack.json",
    "artifacts/final-release/owner-real-proof-report-pack-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This delta pack converts blocked strict candidate field contracts into Owner actions only. It is not proof, not publish approval, not post-publish verification, and not release-close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-real-proof-field-delta-pack.json"
$markdownPath = Join-Path $OutputRoot "owner-real-proof-field-delta-pack.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($recordOut | ConvertTo-Json -Depth 14)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner Real Proof Field Delta Pack")
$lines.Add("")
$lines.Add("`owner-real-proof-field-delta-pack` 将 strict candidate record 的 blocked field contracts 转为 Owner 可执行字段填报 delta。它不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| deltaState | ``$(ConvertTo-MarkdownCell $recordOut.deltaState)`` |")
$lines.Add("| candidateCount | ``$($recordOut.candidateCount)`` |")
$lines.Add("| fieldDeltaCount | ``$($recordOut.fieldDeltaCount)`` |")
$lines.Add("| blockedFieldContractCount | ``$($recordOut.blockedFieldContractCount)`` |")
$lines.Add("| readyFieldContractCount | ``$($recordOut.readyFieldContractCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($recordOut.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($recordOut.canCloseReleaseIssue)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($recordOut.isRuntimeExecutionProof)`` |")
$lines.Add("| isReleaseCloseProof | ``$($recordOut.isReleaseCloseProof)`` |")
$lines.Add("")
$lines.Add("## Field Deltas")
$lines.Add("")
$lines.Add("| Candidate | Lane | Field | State | Owner Action |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($delta in $fieldDeltas) {
  $lines.Add("| $(ConvertTo-MarkdownCell $delta.candidateId) | $(ConvertTo-MarkdownCell $delta.proofLane) | $(ConvertTo-MarkdownCell $delta.fieldName) | $(ConvertTo-MarkdownCell $delta.deltaState) | $(ConvertTo-MarkdownCell $delta.ownerAction) |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($recordOut.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner real proof field delta pack written to $jsonPath"
Write-Host "Owner real proof field delta pack markdown written to $markdownPath"
Write-Host "DeltaState=$($recordOut.deltaState) Candidates=$($recordOut.candidateCount) FieldDeltas=$($recordOut.fieldDeltaCount) BlockedFields=$($recordOut.blockedFieldContractCount) ReadyFields=$($recordOut.readyFieldContractCount)"
