[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\yolovision-real-model-post-publish-candidate.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$resolvedInputPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-YoloVisionRealModelPostPublishCandidate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = @(Test-OwnerPostPublishLaneCandidateArtifact -Record $record -RecordKind "yolovision-real-model-post-publish-candidate" -LaneId "yolovision-real-model-assets" -ExpectedBlockedState "blocked-yolovision-real-model-owner-proof-required" -RequiredBoundaryText "not real-model-runtime proof")
$items += (New-OwnerValidationItem "task-coverage" (@(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "requiredTasks" -DefaultValue @())).Count -ge 6 -and [bool](Get-PropertyOrDefault -Object $record -Name "matrixIsForbiddenSubstitute" -DefaultValue $false)) "blocker" "Candidate must preserve YoloVision task coverage and reject matrix-only proof.")
$items += (New-OwnerValidationItem "real-model-execution-surface" ([bool](Get-PropertyOrDefault -Object $record -Name "rejectsReadinessTutorialMatrixArtifacts" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $record -Name "yoloVisionHashFieldCount" -DefaultValue 0) -ge 9 -and -not [bool](Get-PropertyOrDefault -Object $record -Name "realModelExecutionConfirmationReady" -DefaultValue $true)) "blocker" "Candidate must require real model execution confirmation and broad model/labels/input/output/log/host hashes.")
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "yolovision-real-model-post-publish-candidate-ready-non-proof" } else { "invalid-yolovision-real-model-post-publish-candidate" }
$validation = [pscustomobject]@{
  recordKind = "yolovision-real-model-post-publish-candidate-validation"; generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O"); validationState = $state; candidateState = [string]$record.candidateState; requiredFieldCount = [int]$record.requiredFieldCount; blockedFieldCount = [int]$record.blockedFieldCount; yoloVisionTaskReady = [bool](Get-PropertyOrDefault -Object $record -Name "yoloVisionTaskReady" -DefaultValue $false); yoloVisionHashFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "yoloVisionHashFieldCount" -DefaultValue 0); realModelExecutionConfirmationReady = [bool](Get-PropertyOrDefault -Object $record -Name "realModelExecutionConfirmationReady" -DefaultValue $false); rejectsReadinessTutorialMatrixArtifacts = [bool](Get-PropertyOrDefault -Object $record -Name "rejectsReadinessTutorialMatrixArtifacts" -DefaultValue $false); failedBlockerCount = $failedBlockers.Count; validationItems = @($items); ownerActionRequired = $true; performsPublish = $false; usesPublishToken = $false; canPublishPublicly = $false; canCloseReleaseIssue = $false; canPromoteRuntimeProof = $false; isRuntimeExecutionProof = $false; isPostPublishProof = $false; isReleaseCloseProof = $false; boundary = "YoloVision real model post-publish candidate validation is non-proof; not real-model-runtime proof, not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
$jsonPath = Join-Path $OutputRoot "yolovision-real-model-post-publish-candidate-validation.json"; $mdPath = Join-Path $OutputRoot "yolovision-real-model-post-publish-candidate-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath $mdPath -InputObject @("# YoloVision Real Model Post-Publish Candidate Validation", "", "- validationState: ``$state``", "- requiredFieldCount: ``$($validation.requiredFieldCount)``", "- blockedFieldCount: ``$($validation.blockedFieldCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "YoloVisionRealModelPostPublishCandidateValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "YoloVision real model post-publish candidate validation failed." }
