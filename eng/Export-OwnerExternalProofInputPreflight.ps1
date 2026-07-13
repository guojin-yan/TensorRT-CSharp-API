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

function Test-ArtifactExists {
  param([string]$RelativePath)

  if ([string]::IsNullOrWhiteSpace($RelativePath)) {
    return $false
  }

  $normalized = $RelativePath -replace "/", "\"
  $path = Join-Path $RepositoryRoot $normalized
  return Test-Path -LiteralPath $path -PathType Leaf
}

function Get-RecordClassification {
  param(
    [AllowNull()][object]$ValidationRecord,
    [bool]$CanPromoteProof,
    [string]$DefaultWhenMissing = "missing"
  )

  if ($CanPromoteProof) {
    return "validator-passed-real-proof"
  }

  if ($null -eq $ValidationRecord) {
    return $DefaultWhenMissing
  }

  $classification = [string](Get-PropertyOrDefault -Object $ValidationRecord -Name "proofClassification" -DefaultValue "")
  if (-not [string]::IsNullOrWhiteSpace($classification)) {
    if ($classification -match "template") {
      return "template-only"
    }

    if ($classification -match "schema") {
      return "schema-only"
    }

    if ($classification -match "preflight") {
      return "preflight-only"
    }

    if ($classification -match "guidance|handoff|runbook|collection") {
      return "guidance-only"
    }

    return "candidate-needs-owner-review"
  }

  $validationState = [string](Get-PropertyOrDefault -Object $ValidationRecord -Name "validationState" -DefaultValue "")
  if ($validationState -match "template") {
    return "template-only"
  }

  if ($validationState -match "schema") {
    return "schema-only"
  }

  if ($validationState -match "preflight") {
    return "preflight-only"
  }

  if ($validationState -match "missing") {
    return "missing"
  }

  return "candidate-needs-owner-review"
}

function New-PreflightLine {
  param(
    [object]$HandoffLine,
    [AllowNull()][object]$ValidationRecord,
    [bool]$CanPromoteProof
  )

  $expectedArtifacts = @(Get-PropertyOrDefault -Object $HandoffLine -Name "expectedArtifacts" -DefaultValue @())
  $sourceArtifacts = @(Get-PropertyOrDefault -Object $HandoffLine -Name "sourceArtifacts" -DefaultValue @())
  $requiredRealInputs = @(Get-PropertyOrDefault -Object $HandoffLine -Name "requiredRealInputs" -DefaultValue @())
  $cannotUse = @(Get-PropertyOrDefault -Object $HandoffLine -Name "cannotUse" -DefaultValue $script:NonSubstituteProofKinds)

  $existingExpectedArtifacts = @($expectedArtifacts | Where-Object { Test-ArtifactExists $_ })
  $missingExpectedArtifacts = @($expectedArtifacts | Where-Object { -not (Test-ArtifactExists $_) })
  $existingSourceArtifacts = @($sourceArtifacts | Where-Object { Test-ArtifactExists $_ })
  $missingSourceArtifacts = @($sourceArtifacts | Where-Object { -not (Test-ArtifactExists $_) })
  $candidateClassification = Get-RecordClassification -ValidationRecord $ValidationRecord -CanPromoteProof $CanPromoteProof

  $riskMarkers = @()
  foreach ($marker in @(
      "local feed",
      "ProjectReference",
      "direct .nupkg reference",
      "template-only record",
      "schema-only record",
      "preflight-only close record",
      "readiness snapshot",
      "dependency-probe-only",
      "release-issue-close-record-template.json",
      "missing log hash",
      "mismatched SHA256",
      "missing owner final close decision",
      "missing rollback plan"
    )) {
    $riskMarkers += $marker
  }

  [pscustomobject]@{
    id = [string](Get-PropertyOrDefault -Object $HandoffLine -Name "id" -DefaultValue "")
    proofClass = [string](Get-PropertyOrDefault -Object $HandoffLine -Name "proofClass" -DefaultValue "")
    currentState = [string](Get-PropertyOrDefault -Object $HandoffLine -Name "currentState" -DefaultValue "missing-state")
    candidateClassification = $candidateClassification
    canPromoteProof = $CanPromoteProof
    ownerNextAction = [string](Get-PropertyOrDefault -Object $HandoffLine -Name "ownerNextAction" -DefaultValue "")
    firstCommand = [string](Get-PropertyOrDefault -Object $HandoffLine -Name "firstCommand" -DefaultValue "")
    validatorCommand = [string](Get-PropertyOrDefault -Object $HandoffLine -Name "validatorCommand" -DefaultValue "")
    requiredRealInputs = $requiredRealInputs
    requiredRealInputCount = $requiredRealInputs.Count
    missingRealInputCount = [int](Get-PropertyOrDefault -Object $HandoffLine -Name "missingRealInputCount" -DefaultValue $requiredRealInputs.Count)
    expectedArtifacts = $expectedArtifacts
    expectedArtifactCount = $expectedArtifacts.Count
    existingExpectedArtifacts = $existingExpectedArtifacts
    existingCandidateArtifactCount = $existingExpectedArtifacts.Count
    missingExpectedArtifacts = $missingExpectedArtifacts
    sourceArtifacts = $sourceArtifacts
    sourceArtifactCount = $sourceArtifacts.Count
    existingSourceArtifacts = $existingSourceArtifacts
    missingSourceArtifacts = $missingSourceArtifacts
    cannotUse = $cannotUse
    riskMarkers = $riskMarkers
    blockedReason = [string](Get-PropertyOrDefault -Object $HandoffLine -Name "blockerReason" -DefaultValue "blocked-real-proof-required")
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$handoff = Read-JsonOrNull "artifacts\final-release\owner-proof-execution-handoff.json"
$backfillPack = Read-JsonOrNull "artifacts\final-release\owner-proof-backfill-execution-pack.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$releaseClosePreflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$releaseIssueCloseValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$ownerProofInputValidation = Read-JsonOrNull "artifacts\final-release\release-owner-proof-input-record-validation.json"
$externalRuntimeValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$linuxRunnerValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"
$sampleRunValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"

$script:NonSubstituteProofKinds = @(Get-PropertyOrDefault -Object $handoff -Name "nonSubstituteProofKinds" -DefaultValue @(Get-PropertyOrDefault -Object $backfillPack -Name "nonSubstituteProofKinds" -DefaultValue @(
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
)))

$handoffLines = @(Get-PropertyOrDefault -Object $handoff -Name "handoffLines" -DefaultValue @())
if ($handoffLines.Count -eq 0) {
  throw "owner-proof-execution-handoff.json is missing or has no handoffLines. Run Export-OwnerProofExecutionHandoff.ps1 first."
}

$ownerProofInputCanPromote = [bool](Get-PropertyOrDefault -Object $ownerProofInputValidation -Name "canPromoteOwnerProofInput" -DefaultValue $false)
$externalRuntimeCanPromote = [bool](Get-PropertyOrDefault -Object $externalRuntimeValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$linuxRunnerProof = [bool](Get-PropertyOrDefault -Object $linuxRunnerValidation -Name "isRealLinuxRunnerProof" -DefaultValue $false)
$sampleRunProof = [bool](Get-PropertyOrDefault -Object $sampleRunValidation -Name "isRealSampleRunProof" -DefaultValue $false)
$postPublishProof = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
$postPublishCanClose = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$closeRecordCanPromote = [bool](Get-PropertyOrDefault -Object $releaseIssueCloseValidation -Name "canPromoteReleaseIssueCloseRecord" -DefaultValue $false)
$closeRecordCanClose = [bool](Get-PropertyOrDefault -Object $releaseIssueCloseValidation -Name "canCloseReleaseIssue" -DefaultValue $false)

$validationMap = @{
  "owner-authorization" = $ownerProofInputValidation
  "package-consumer-runtime" = $externalRuntimeValidation
  "linux-runner-proof" = $linuxRunnerValidation
  "real-model-runtime" = $sampleRunValidation
  "post-publish-verification" = $postPublishValidation
  "release-issue-close-record" = $releaseIssueCloseValidation
}

$promotionMap = @{
  "owner-authorization" = $ownerProofInputCanPromote
  "package-consumer-runtime" = $externalRuntimeCanPromote
  "linux-runner-proof" = $linuxRunnerProof
  "real-model-runtime" = $sampleRunProof
  "post-publish-verification" = ($postPublishProof -and $postPublishCanClose)
  "release-issue-close-record" = ($closeRecordCanPromote -and $closeRecordCanClose)
}

$preflightLines = @($handoffLines | ForEach-Object {
  $id = [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")
  New-PreflightLine -HandoffLine $_ -ValidationRecord $validationMap[$id] -CanPromoteProof ([bool]$promotionMap[$id])
})

$readyLineCount = @($preflightLines | Where-Object { $_.canPromoteProof }).Count
$blockedLineCount = $preflightLines.Count - $readyLineCount
$candidateLineCount = @($preflightLines | Where-Object { $_.candidateClassification -eq "candidate-needs-owner-review" }).Count
$templateOnlyLineCount = @($preflightLines | Where-Object { $_.candidateClassification -eq "template-only" }).Count
$missingLineCount = @($preflightLines | Where-Object { $_.candidateClassification -eq "missing" }).Count
$preflightState = if ($blockedLineCount -eq 0) { "ready-for-owner-close-review" } else { "blocked-real-proof-required" }

$releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$releaseEvidenceBundleCanPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canPublishPublicly" -DefaultValue $false)
$releaseEvidenceBundleCanClose = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canCloseReleaseIssue" -DefaultValue $false)
$releaseClosePreflightState = [string](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")
$releaseClosePreflightCanClose = [bool](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "canCloseReleaseIssue" -DefaultValue $false)
$releaseClosePreflightFailedItemCount = [int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "failedItemCount" -DefaultValue -1)
$releaseIssueCloseValidationState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")
$releaseIssueCloseFailedValidationItemCount = [int](Get-PropertyOrDefault -Object $releaseIssueCloseValidation -Name "failedValidationItemCount" -DefaultValue -1)

$sourceArtifacts = @(
  "artifacts/final-release/owner-proof-execution-handoff.json",
  "artifacts/final-release/owner-proof-backfill-execution-pack.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/release-close-preflight.json",
  "artifacts/final-release/release-issue-close-record-validation.json"
)

$record = [pscustomobject]@{
  recordKind = "owner-external-proof-input-preflight"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  preflightState = $preflightState
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  releaseEvidenceBundleState = $releaseEvidenceBundleState
  releaseEvidenceBundleCanPublishPublicly = $releaseEvidenceBundleCanPublish
  releaseEvidenceBundleCanCloseReleaseIssue = $releaseEvidenceBundleCanClose
  releaseClosePreflightState = $releaseClosePreflightState
  releaseClosePreflightCanCloseReleaseIssue = $releaseClosePreflightCanClose
  releaseClosePreflightFailedItemCount = $releaseClosePreflightFailedItemCount
  releaseIssueCloseRecordValidationState = $releaseIssueCloseValidationState
  releaseIssueCloseRecordFailedValidationItemCount = $releaseIssueCloseFailedValidationItemCount
  proofLineCount = $preflightLines.Count
  readyProofLineCount = $readyLineCount
  blockedProofLineCount = $blockedLineCount
  candidateNeedsOwnerReviewLineCount = $candidateLineCount
  templateOnlyLineCount = $templateOnlyLineCount
  missingLineCount = $missingLineCount
  proofLines = $preflightLines
  nonSubstituteProofKinds = $script:NonSubstituteProofKinds
  sourceArtifacts = $sourceArtifacts
  safetyBoundary = "owner-external-proof-input-preflight is candidate audit guidance only; it never promotes proof, publishes packages, approves public publication, or closes the release issue."
  candidateClassificationLegend = @(
    "missing: no relevant validation record or candidate artifact exists",
    "template-only: only template/sample placeholder records are present",
    "guidance-only: runbook/handoff/collection/readiness guidance exists but no real proof exists",
    "candidate-needs-owner-review: candidate artifacts exist but validators have not promoted them",
    "validator-passed-real-proof: validator explicitly promoted real proof"
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-external-proof-input-preflight.json"
$markdownPath = Join-Path $artifactRoot "owner-external-proof-input-preflight.md"

$record | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lineRows = $preflightLines | ForEach-Object {
  $state = ([string]$_.currentState).Replace("|", "\|")
  $action = ([string]$_.ownerNextAction).Replace("|", "\|")
  $validator = ([string]$_.validatorCommand).Replace("|", "\|")
  "| ``$($_.id)`` | ``$($_.candidateClassification)`` | ``$($_.canPromoteProof)`` | ``$($_.missingRealInputCount)`` | ``$($_.existingCandidateArtifactCount)`` | $state | $action | ``$validator`` |"
}
$nonSubstituteLines = $script:NonSubstituteProofKinds | ForEach-Object { "- ``$_``" }
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }
$legendLines = $record.candidateClassificationLegend | ForEach-Object { "- $_" }

$markdown = @"
# Owner External Proof Input Preflight

生成时间：$($record.generatedAtUtc)

## 总结

``owner-external-proof-input-preflight`` 是真实 owner 外部输入候选预审。它读取 ``owner-proof-execution-handoff``，检查 6 条 proof line 的 expected/source artifact、缺失真实输入、候选分类和不可替代 proof 风险。它不执行发布、不批准公开发布、不关闭 release issue：``performsPublish=false``，``canPublishPublicly=false``，``canCloseReleaseIssue=false``。

## 当前 Gate

| 项目 | 当前值 |
|---|---|
| preflightState | ``$preflightState`` |
| proofLineCount | ``$($preflightLines.Count)`` |
| readyProofLineCount | ``$readyLineCount`` |
| blockedProofLineCount | ``$blockedLineCount`` |
| candidateNeedsOwnerReviewLineCount | ``$candidateLineCount`` |
| templateOnlyLineCount | ``$templateOnlyLineCount`` |
| missingLineCount | ``$missingLineCount`` |
| releaseClosePreflightState | ``$releaseClosePreflightState`` |
| releaseClosePreflightFailedItemCount | ``$releaseClosePreflightFailedItemCount`` |
| releaseIssueCloseRecordValidationState | ``$releaseIssueCloseValidationState`` |
| releaseIssueCloseRecordFailedValidationItemCount | ``$releaseIssueCloseFailedValidationItemCount`` |

## Proof Line Preflight

| ID | Candidate classification | Can promote proof | Missing real inputs | Existing candidate artifacts | Current state | Owner next action | Validator |
|---|---|---|---|---|---|---|---|
$($lineRows -join "`r`n")

## Candidate Classification Legend

$($legendLines -join "`r`n")

## Blocked Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Release Gate Boundary

即使所有 candidate artifacts 存在，只要没有真实 validator-passing proof，仍必须保持 ``canPublishPublicly=false`` 和 ``canCloseReleaseIssue=false``。local feed、ProjectReference、direct ``.nupkg``、template-only record、schema-only record、preflight-only close record、readiness snapshot、dependency-probe-only、``release-issue-close-record-template.json``、missing log hash、mismatched SHA256、missing owner final close decision 和 missing rollback plan 都必须阻断 release close。

## Source Artifacts

$($sourceLines -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Owner external proof input preflight written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PreflightState=$preflightState"
Write-Output "ProofLineCount=$($preflightLines.Count)"
Write-Output "ReadyProofLineCount=$readyLineCount"
Write-Output "PerformsPublish=False"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
