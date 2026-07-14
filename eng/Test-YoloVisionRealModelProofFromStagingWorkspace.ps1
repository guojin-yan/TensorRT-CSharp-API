[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\yolovision-real-model-proof-from-staging-workspace.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict,
  [switch]$FailOnNotProof
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Import-YoloVisionRealModelProofFromStagingWorkspace.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$shapeValid = [bool](Get-PropertyOrDefault -Object $record -Name "ownerEvidenceShapeValid" -DefaultValue $false)
$items = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "yolovision-real-model-proof-from-staging-workspace") "blocker" "recordKind must match."
  New-OwnerValidationItem "lane-surface" ([int](Get-PropertyOrDefault -Object $record -Name "assetFileCount" -DefaultValue 0) -ge 12 -and [int](Get-PropertyOrDefault -Object $record -Name "sha256RequiredFileCount" -DefaultValue 0) -ge 12) "blocker" "YoloVision staging lane must expose all required real-model files."
  New-OwnerValidationItem "non-proof" (-not [bool](Get-PropertyOrDefault -Object $record -Name "proofCandidateReady" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) "blocker" "YoloVision staging admission must not claim runtime proof."
  New-OwnerValidationItem "shape-validity-consistent" ((-not $shapeValid) -or ([int](Get-PropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue -1) -eq 0 -and [bool](Get-PropertyOrDefault -Object $record -Name "assetManifestLinkageReady" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "modelLicenseReady" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "realModelExecutionConfirmationReady" -DefaultValue $false))) "blocker" "Shape-valid state requires manifest, license, and confirmation readiness."
  New-OwnerValidationItem "boundary" ($boundary.Contains("not runtime proof") -and $boundary.Contains("not post-publish proof") -and $boundary.Contains("not package push")) "blocker" "Boundary must preserve non-proof status."
)
$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failed.Count -eq 0) { "yolovision-real-model-proof-from-staging-workspace-validation-ready" } else { "blocked-yolovision-real-model-proof-from-staging-workspace-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "yolovision-real-model-proof-from-staging-workspace-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  importState = [string](Get-PropertyOrDefault -Object $record -Name "importState" -DefaultValue "")
  ownerEvidenceShapeValid = $shapeValid
  assetFileCount = [int](Get-PropertyOrDefault -Object $record -Name "assetFileCount" -DefaultValue 0)
  existingAssetFileCount = [int](Get-PropertyOrDefault -Object $record -Name "existingAssetFileCount" -DefaultValue 0)
  sha256ValidFileCount = [int](Get-PropertyOrDefault -Object $record -Name "sha256ValidFileCount" -DefaultValue 0)
  assetManifestHashMatchCount = [int](Get-PropertyOrDefault -Object $record -Name "assetManifestHashMatchCount" -DefaultValue 0)
  assetManifestLinkageReady = [bool](Get-PropertyOrDefault -Object $record -Name "assetManifestLinkageReady" -DefaultValue $false)
  modelLicenseReady = [bool](Get-PropertyOrDefault -Object $record -Name "modelLicenseReady" -DefaultValue $false)
  realModelExecutionConfirmationReady = [bool](Get-PropertyOrDefault -Object $record -Name "realModelExecutionConfirmationReady" -DefaultValue $false)
  runtimeTranscriptLinkageReady = [bool](Get-PropertyOrDefault -Object $record -Name "runtimeTranscriptLinkageReady" -DefaultValue $false)
  failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue 0)
  failedBlockerCount = $failed.Count
  validationItems = @($items)
  passed = $false
  performsPublish = $false
  usesPublishToken = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Validation checks YoloVision real model staging evidence shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
$jsonPath = Join-Path $OutputRoot "yolovision-real-model-proof-from-staging-workspace-validation.json"
$mdPath = Join-Path $OutputRoot "yolovision-real-model-proof-from-staging-workspace-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @("# YoloVision Real Model Proof From Staging Workspace Validation", "", "- validationState: ``$state``", "- ownerEvidenceShapeValid: ``$shapeValid``", "- failedBlockerCount: ``$($failed.Count)``", "", $validation.boundary)
Write-Host "YoloVisionRealModelProofFromStagingWorkspaceValidationState=$state ShapeValid=$shapeValid FailedBlockers=$($failed.Count)"
if ($Strict.IsPresent -and $failed.Count -gt 0) { throw "YoloVision real model staging validation failed." }
if ($FailOnNotProof.IsPresent -and -not $shapeValid) { throw "YoloVision real model staging workspace evidence is not shape-valid." }
