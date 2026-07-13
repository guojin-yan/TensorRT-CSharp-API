[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function Get-RepairCategory {
  param([string]$Field)

  $value = $Field.ToLowerInvariant()
  if ($value -match "sha256|hash|digest") {
    return "fieldsRequiringSha256"
  }

  if ($value -match "path|log|artifact|file|nupkg|package|project|model|asset|record|manifest|bundle|evidence") {
    return "fieldsRequiringExistingFiles"
  }

  if ($value -match "consumer|clean|projectreference|local feed|package source|smoke|runtimepackagekey|runtime package key") {
    return "fieldsRequiringCleanConsumerEvidence"
  }

  if ($value -match "owner|decision|approval|authorization|publish|channel|close") {
    return "fieldsRequiringOwnerDecision"
  }

  if ($value -match "rollback|yank|deprecat|recover") {
    return "fieldsRequiringRollbackPlan"
  }

  return "placeholderFieldsToReplace"
}

function New-InputDraftPath {
  param([string]$ProofLineId)

  switch ($ProofLineId) {
    "owner-authorization" { return "artifacts/final-release/release-owner-proof-input-record.input-draft.json" }
    "package-consumer-runtime" { return "artifacts/final-release/external-runtime-proof-record.input-draft.json" }
    "linux-runner-proof" { return "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record.input-draft.json" }
    "real-model-runtime" { return "artifacts/user-acceptance/sample-run-evidence-record.input-draft.json" }
    "post-publish-verification" { return "artifacts/final-release/post-publish-verification-record.input-draft.json" }
    "release-issue-close-record" { return "artifacts/final-release/release-issue-close-record.input-draft.json" }
    default { return "artifacts/final-release/$ProofLineId.input-draft.json" }
  }
}

function New-FirstRepairCommand {
  param(
    [string]$ProofLineId,
    [string]$FirstCommand
  )

  if (-not [string]::IsNullOrWhiteSpace($FirstCommand)) {
    return $FirstCommand
  }

  switch ($ProofLineId) {
    "owner-authorization" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerProofInputRecordTemplate.ps1" }
    "package-consumer-runtime" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordTemplate.ps1 -RuntimePackageKey $RuntimePackageKey" }
    "linux-runner-proof" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-LinuxRunnerEvidenceRecordTemplate.ps1 -RuntimePackageKey $LinuxRuntimePackageKey" }
    "real-model-runtime" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleRunEvidenceRecordTemplate.ps1" }
    "post-publish-verification" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordTemplate.ps1 -RuntimePackageKey $RuntimePackageKey" }
    "release-issue-close-record" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordTemplate.ps1" }
    default { return "Review the expected artifact template and replace every placeholder with real owner input." }
  }
}

function New-RepairItem {
  param([object]$PreflightLine)

  $id = [string](Get-PropertyOrDefault -Object $PreflightLine -Name "id" -DefaultValue "")
  $requiredRealInputs = Convert-ToStringArray (Get-PropertyOrDefault -Object $PreflightLine -Name "requiredRealInputs" -DefaultValue @())
  $expectedArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $PreflightLine -Name "expectedArtifacts" -DefaultValue @())
  $sourceArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $PreflightLine -Name "sourceArtifacts" -DefaultValue @())
  $cannotUse = Convert-ToStringArray (Get-PropertyOrDefault -Object $PreflightLine -Name "cannotUse" -DefaultValue $script:NonSubstituteProofKinds)
  $riskMarkers = Convert-ToStringArray (Get-PropertyOrDefault -Object $PreflightLine -Name "riskMarkers" -DefaultValue @())

  $placeholderFieldsToReplace = New-Object System.Collections.Generic.List[string]
  $fieldsRequiringExistingFiles = New-Object System.Collections.Generic.List[string]
  $fieldsRequiringSha256 = New-Object System.Collections.Generic.List[string]
  $fieldsRequiringCleanConsumerEvidence = New-Object System.Collections.Generic.List[string]
  $fieldsRequiringOwnerDecision = New-Object System.Collections.Generic.List[string]
  $fieldsRequiringRollbackPlan = New-Object System.Collections.Generic.List[string]

  foreach ($field in $requiredRealInputs) {
    switch (Get-RepairCategory -Field $field) {
      "fieldsRequiringExistingFiles" { $fieldsRequiringExistingFiles.Add($field) }
      "fieldsRequiringSha256" { $fieldsRequiringSha256.Add($field) }
      "fieldsRequiringCleanConsumerEvidence" { $fieldsRequiringCleanConsumerEvidence.Add($field) }
      "fieldsRequiringOwnerDecision" { $fieldsRequiringOwnerDecision.Add($field) }
      "fieldsRequiringRollbackPlan" { $fieldsRequiringRollbackPlan.Add($field) }
      default { $placeholderFieldsToReplace.Add($field) }
    }
  }

  foreach ($artifact in $expectedArtifacts) {
    if (-not $fieldsRequiringExistingFiles.Contains($artifact)) {
      $fieldsRequiringExistingFiles.Add($artifact)
    }
  }

  $validatorCommand = [string](Get-PropertyOrDefault -Object $PreflightLine -Name "validatorCommand" -DefaultValue "")
  $firstCommand = New-FirstRepairCommand -ProofLineId $id -FirstCommand ([string](Get-PropertyOrDefault -Object $PreflightLine -Name "firstCommand" -DefaultValue ""))
  $candidateClassification = [string](Get-PropertyOrDefault -Object $PreflightLine -Name "candidateClassification" -DefaultValue "missing")
  $missingRealInputCount = [int](Get-PropertyOrDefault -Object $PreflightLine -Name "missingRealInputCount" -DefaultValue $requiredRealInputs.Count)

  [pscustomobject]@{
    id = $id
    proofClass = [string](Get-PropertyOrDefault -Object $PreflightLine -Name "proofClass" -DefaultValue "")
    currentCandidateClassification = $candidateClassification
    currentState = [string](Get-PropertyOrDefault -Object $PreflightLine -Name "currentState" -DefaultValue "")
    canPromoteProof = $false
    repairState = if ($missingRealInputCount -eq 0 -and $candidateClassification -eq "candidate-needs-owner-review") { "candidate-review-required" } else { "blocked-real-input-repair-required" }
    requiredRealInputs = $requiredRealInputs
    requiredRealInputCount = $requiredRealInputs.Count
    missingRealInputCount = $missingRealInputCount
    placeholderFieldsToReplace = @($placeholderFieldsToReplace)
    fieldsRequiringExistingFiles = @($fieldsRequiringExistingFiles)
    fieldsRequiringSha256 = @($fieldsRequiringSha256)
    fieldsRequiringCleanConsumerEvidence = @($fieldsRequiringCleanConsumerEvidence)
    fieldsRequiringOwnerDecision = @($fieldsRequiringOwnerDecision)
    fieldsRequiringRollbackPlan = @($fieldsRequiringRollbackPlan)
    expectedArtifacts = $expectedArtifacts
    sourceArtifacts = $sourceArtifacts
    inputDraftPath = New-InputDraftPath -ProofLineId $id
    inputDraftIsProof = $false
    repairMarkdownPath = "artifacts/final-release/$id.repair.md"
    firstRepairCommand = $firstCommand
    validatorCommand = $validatorCommand
    ownerNextAction = [string](Get-PropertyOrDefault -Object $PreflightLine -Name "ownerNextAction" -DefaultValue "")
    cannotUseMarkers = @($cannotUse + $riskMarkers | Select-Object -Unique)
    blockedReason = [string](Get-PropertyOrDefault -Object $PreflightLine -Name "blockedReason" -DefaultValue "blocked-real-proof-required")
    realProofPromotionRequirement = "Fill the real owner input record, attach existing files and SHA256 values, run the validator command in strict proof mode, then refresh release evidence. This repair item and any input draft remain non-proof."
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$preflight = Read-JsonOrNull "artifacts\final-release\owner-external-proof-input-preflight.json"
if ($null -eq $preflight) {
  throw "owner-external-proof-input-preflight.json is missing. Run Export-OwnerExternalProofInputPreflight.ps1 first."
}

$preflightLines = @(Get-PropertyOrDefault -Object $preflight -Name "proofLines" -DefaultValue @())
if ($preflightLines.Count -eq 0) {
  throw "owner-external-proof-input-preflight.json has no proofLines."
}

$script:NonSubstituteProofKinds = Convert-ToStringArray (Get-PropertyOrDefault -Object $preflight -Name "nonSubstituteProofKinds" -DefaultValue @(
  "template",
  "draft",
  "runbook",
  "collection package",
  "handoff",
  "local feed",
  "ProjectReference",
  "direct .nupkg reference",
  "precheck-only",
  "dry-run-only",
  "schema-only",
  "readiness snapshot",
  "helper scan",
  "dependency-probe-only",
  "blocked-by-cuda-driver",
  "Windows handoff for Linux proof",
  "release-issue-close-record-template.json"
))

$repairItems = @($preflightLines | ForEach-Object { New-RepairItem -PreflightLine $_ })
$readyRepairItemCount = @($repairItems | Where-Object { $_.repairState -eq "candidate-review-required" }).Count
$blockedRepairItemCount = $repairItems.Count - $readyRepairItemCount
$templateOnlyRepairItemCount = @($repairItems | Where-Object { $_.currentCandidateClassification -eq "template-only" }).Count
$requiredRealInputCount = ($repairItems | ForEach-Object { $_.requiredRealInputCount } | Measure-Object -Sum).Sum
$missingRealInputCount = ($repairItems | ForEach-Object { $_.missingRealInputCount } | Measure-Object -Sum).Sum

$record = [pscustomobject]@{
  recordKind = "owner-proof-input-repair-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  repairPackState = if ($blockedRepairItemCount -eq 0) { "ready-for-owner-candidate-review" } else { "blocked-real-input-repair-required" }
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  repairItemCount = $repairItems.Count
  readyRepairItemCount = $readyRepairItemCount
  blockedRepairItemCount = $blockedRepairItemCount
  templateOnlyRepairItemCount = $templateOnlyRepairItemCount
  requiredRealInputCount = [int]$requiredRealInputCount
  missingRealInputCount = [int]$missingRealInputCount
  repairItems = $repairItems
  sourceArtifacts = @(
    "artifacts/final-release/owner-external-proof-input-preflight.json",
    "artifacts/final-release/owner-proof-execution-handoff.json",
    "artifacts/final-release/owner-proof-backfill-execution-pack.json",
    "artifacts/final-release/release-evidence-bundle.json"
  )
  nonSubstituteProofKinds = $script:NonSubstituteProofKinds
  safetyBoundary = "owner-proof-input-repair-pack is owner repair guidance only. It writes no proof, performs no publish, approves no public release, closes no issue, and any input draft remains non-proof until strict validators promote real records."
  nextOwnerSequence = @(
    "Choose one blocked repair item.",
    "Run firstRepairCommand to refresh the template or draft surface.",
    "Replace placeholders with real owner input, existing file paths, SHA256 values, clean consumer evidence, owner decisions, and rollback plans.",
    "Run validatorCommand in the compatible environment required by that proof line.",
    "Refresh owner-external-proof-input-preflight and release-evidence-bundle after validators pass."
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-proof-input-repair-pack.json"
$markdownPath = Join-Path $artifactRoot "owner-proof-input-repair-pack.md"

$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $repairItems | ForEach-Object {
  $ownerAction = ([string]$_.ownerNextAction).Replace("|", "\|")
  $validator = ([string]$_.validatorCommand).Replace("|", "\|")
  "| ``$($_.id)`` | ``$($_.currentCandidateClassification)`` | ``$($_.repairState)`` | ``$($_.missingRealInputCount)`` | ``$($_.fieldsRequiringExistingFiles.Count)`` | ``$($_.fieldsRequiringSha256.Count)`` | ``$($_.fieldsRequiringCleanConsumerEvidence.Count)`` | ``$($_.fieldsRequiringOwnerDecision.Count)`` | ``$($_.fieldsRequiringRollbackPlan.Count)`` | $ownerAction | ``$validator`` |"
}

$itemDetails = $repairItems | ForEach-Object {
  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("### ``$($_.id)``")
  $lines.Add("")
  $lines.Add("- currentCandidateClassification: ``$($_.currentCandidateClassification)``")
  $lines.Add("- repairState: ``$($_.repairState)``")
  $lines.Add("- inputDraftPath: ``$($_.inputDraftPath)``")
  $lines.Add("- inputDraftIsProof: ``False``")
  $lines.Add("- firstRepairCommand: ``$($_.firstRepairCommand)``")
  $lines.Add("- validatorCommand: ``$($_.validatorCommand)``")
  $lines.Add("- blockedReason: ``$($_.blockedReason)``")
  $requiredRealInputText = $_.requiredRealInputs -join "``; ``"
  $cannotUseMarkerText = $_.cannotUseMarkers -join "``; ``"
  $lines.Add("- requiredRealInputs: ``$requiredRealInputText``")
  $lines.Add("- cannotUseMarkers: ``$cannotUseMarkerText``")
  $lines.Add("")
  $lines -join "`r`n"
}

$sourceLines = $record.sourceArtifacts | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $script:NonSubstituteProofKinds | ForEach-Object { "- ``$_``" }
$nextOwnerLines = $record.nextOwnerSequence | ForEach-Object { "- $_" }

$markdown = @"
# Owner Proof Input Repair Pack

生成时间：$($record.generatedAtUtc)

## 总结

``owner-proof-input-repair-pack`` 基于 ``owner-external-proof-input-preflight`` 的 6 条 proof line 生成真实输入修复清单。它把每条 template-only/candidate proof line 拆成可替换 placeholder、必须存在的文件、SHA256、clean consumer evidence、owner decision 和 rollback plan。它仍然只是 owner guidance：``performsPublish=false``，``canPublishPublicly=false``，``canCloseReleaseIssue=false``。

## 当前 Gate

| 项目 | 当前值 |
|---|---|
| repairPackState | ``$($record.repairPackState)`` |
| repairItemCount | ``$($record.repairItemCount)`` |
| readyRepairItemCount | ``$($record.readyRepairItemCount)`` |
| blockedRepairItemCount | ``$($record.blockedRepairItemCount)`` |
| templateOnlyRepairItemCount | ``$($record.templateOnlyRepairItemCount)`` |
| requiredRealInputCount | ``$($record.requiredRealInputCount)`` |
| missingRealInputCount | ``$($record.missingRealInputCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Repair Items

| ID | Candidate classification | Repair state | Missing real inputs | Existing/file fields | SHA256 fields | Clean consumer fields | Owner decision fields | Rollback fields | Owner next action | Validator |
|---|---|---|---|---|---|---|---|---|---|---|
$($rows -join "`r`n")

## 下一步 Owner 顺序

$($nextOwnerLines -join "`r`n")

## 明细

$($itemDetails -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Release Gate Boundary

repair pack、repair markdown、input draft、template、runbook、collection package、handoff、preflight 和 readiness snapshot 都不是 proof。只有真实记录带 existing files、匹配 SHA256、clean consumer evidence、owner decision、rollback plan，并通过对应 validator 后，才能由下一次 preflight/evidence bundle 反映为可审阅候选或真实 proof。

## Source Artifacts

$($sourceLines -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Owner proof input repair pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "RepairPackState=$($record.repairPackState)"
Write-Output "RepairItemCount=$($record.repairItemCount)"
Write-Output "BlockedRepairItemCount=$($record.blockedRepairItemCount)"
Write-Output "TemplateOnlyRepairItemCount=$($record.templateOnlyRepairItemCount)"
Write-Output "PerformsPublish=False"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
