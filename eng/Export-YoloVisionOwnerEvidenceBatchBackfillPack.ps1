[CmdletBinding()]
param(
  [string]$RepairPackPath = "artifacts/user-acceptance/yolovision-owner-proof-field-delta-repair-pack.json",
  [string]$ExecutionPackPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-execution-pack.json",
  [string]$IntakeDashboardPath = "artifacts/user-acceptance/yolovision-owner-real-evidence-intake-dashboard.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\user-acceptance"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-Group {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Description,
    [string[]]$CategoryMatches,
    [string[]]$PathMatches,
    [string]$FirstCommand,
    [string]$ValidatorCommand
  )

  $deltas = @($script:FieldDeltas | Where-Object {
      $category = [string]$_.category
      $path = [string]$_.jsonPath
      ($CategoryMatches -contains $category) -or (@($PathMatches | Where-Object { $path -match $_ }).Count -gt 0)
    })

  [pscustomobject]@{
    id = $Id
    title = $Title
    description = $Description
    fieldCount = @($deltas).Count
    cases = @($deltas | ForEach-Object { [string]$_.caseId } | Sort-Object -Unique)
    fields = @($deltas | Sort-Object caseId, jsonPath | ForEach-Object {
        [pscustomobject]@{
          caseId = [string]$_.caseId
          jsonPath = [string]$_.jsonPath
          category = [string]$_.category
          validatorItemId = [string]$_.validatorItemId
          repairOrder = [string]$_.repairOrder
          description = [string]$_.description
        }
      })
    firstCommand = $FirstCommand
    hashCommand = "Get-FileHash -Algorithm SHA256 <owner-real-file>"
    validatorCommand = $ValidatorCommand
    forbiddenSubstitutes = @($script:ForbiddenSubstitutes)
    canAutoFill = $false
    canPromoteProof = $false
  }
}

$repairPack = Read-JsonOrNull $RepairPackPath
$executionPack = Read-JsonOrNull $ExecutionPackPath
$intakeDashboard = Read-JsonOrNull $IntakeDashboardPath

if ($null -eq $repairPack) { throw "Repair pack missing: $RepairPackPath" }
if ($null -eq $executionPack) { throw "Execution pack missing: $ExecutionPackPath" }

$script:FieldDeltas = @($repairPack.fieldDeltas)
$script:ForbiddenSubstitutes = @($executionPack.forbiddenSubstitutes)
$validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-YoloVisionRealAssetOwnerProofInput.ps1"

$groups = @(
  (New-Group -Id "01-model-source-license" -Title "模型来源与许可证" -Description "补齐模型下载来源、许可证、ONNX/PT hash 和导出来源。" -CategoryMatches @("owner-input") -PathMatches @("model\.") -FirstCommand "Prepare model source URL, license, ONNX/PT artifacts, and SHA256." -ValidatorCommand $validatorCommand),
  (New-Group -Id "02-labels-input-preprocessed" -Title "Labels / Input / Preprocessed Tensor" -Description "补齐 labels、输入图片、预处理 tensor、形状和 preprocess contract。" -CategoryMatches @("owner-input") -PathMatches @("labels\.", "input\.") -FirstCommand "Prepare labels, redistributable input image, preprocessed tensor, and metadata." -ValidatorCommand $validatorCommand),
  (New-Group -Id "03-tensorrtexec-build-report-engine-logs" -Title "TensorRtExec build report / engine / logs" -Description "运行 TensorRtExec build/report，记录 engine/report/stdout/stderr 路径和 hash；它仍只是 supporting evidence。" -CategoryMatches @("hash") -PathMatches @("tensorRtExec\.", "engine") -FirstCommand "Run TensorRtExec build/report for every YOLOv8n task; keep report as supporting evidence only." -ValidatorCommand $validatorCommand),
  (New-Group -Id "04-yolovision-run-output-json-logs" -Title "YoloVision run log / output JSON" -Description "运行真实 YoloVision case，记录 run log、stdout/stderr、output JSON 和 expected evidence lines。" -CategoryMatches @("expected-evidence-line", "owner-input") -PathMatches @("yoloVision\.", "outputJson", "stdoutSummary", "stderrSummary") -FirstCommand "Run YoloVision with real model/input and capture run log, stdout/stderr, and output JSON." -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"),
  (New-Group -Id "05-sha256-verification" -Title "SHA256 校验" -Description "对模型、labels、输入、tensor、engine、report、log 和 output JSON 计算 SHA256。" -CategoryMatches @("hash") -PathMatches @("sha256", "Sha256", "SHA256") -FirstCommand "Calculate SHA256 for all owner artifacts with Get-FileHash." -ValidatorCommand $validatorCommand),
  (New-Group -Id "06-host-metadata" -Title "Host metadata" -Description "记录真实 OS/GPU/driver/CUDA/TensorRT/cuDNN 环境。" -CategoryMatches @("global-host-metadata") -PathMatches @("host", "gpu", "cuda", "tensorRt", "cudnn") -FirstCommand "Collect host OS, GPU, driver, CUDA, TensorRT, and cuDNN metadata on the real execution host." -ValidatorCommand $validatorCommand),
  (New-Group -Id "07-package-metadata" -Title "Package metadata" -Description "记录 package source/channel/runtime key/version 和 managed/native/runtime package hash。" -CategoryMatches @("global-host-metadata") -PathMatches @("package", "runtimePackage", "managedPackage", "nativeBridge") -FirstCommand "Collect package source, runtime key/version, managed/native/runtime package SHA256 values." -ValidatorCommand $validatorCommand),
  (New-Group -Id "08-owner-review-acceptance" -Title "Owner review / acceptance decision" -Description "补齐 reviewer、review time、acceptance decision 和 notes；未接受前不能晋级。" -CategoryMatches @("owner-review") -PathMatches @("ownerReview", "owner") -FirstCommand "Owner reviews filled evidence and records acceptance decision and notes." -ValidatorCommand $validatorCommand)
)

$pack = [pscustomobject]@{
  recordKind = "yolovision-owner-evidence-batch-backfill-pack"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  packState = "blocked-owner-evidence-backfill-required"
  sourceRepairPack = $RepairPackPath
  sourceExecutionPack = $ExecutionPackPath
  sourceIntakeDashboard = $IntakeDashboardPath
  taskCount = if ($null -ne $intakeDashboard) { [int]$intakeDashboard.taskCount } else { [int]$repairPack.caseCount }
  totalMissingFieldCount = [int]$repairPack.missingFieldCount
  globalMissingFieldCount = [int]$repairPack.globalMissingFieldCount
  groupCount = @($groups).Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canAutoFillOwnerFields = $false
  canPromoteRealModelRuntime = $false
  canPromotePackageConsumerRuntime = $false
  groups = @($groups)
  ownerCommands = @(
    "Fill artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json with real owner evidence.",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-YoloVisionRealAssetOwnerProofInput.ps1",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
  )
  boundary = "This batch backfill pack is an owner worklist only. It does not auto-fill evidence, does not run publish, and cannot promote runtime or package-consumer proof."
}

$jsonPath = Join-Path $OutputRoot "yolovision-owner-evidence-batch-backfill-pack.json"
$markdownPath = Join-Path $OutputRoot "yolovision-owner-evidence-batch-backfill-pack.md"
$pack | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($group in $groups) {
  "| ``$(ConvertTo-MarkdownCell $group.id)`` | $(ConvertTo-MarkdownCell $group.title) | $($group.fieldCount) | ``$(ConvertTo-MarkdownCell (($group.cases | Select-Object -First 8) -join ', '))`` | ``False`` |"
}
$commandLines = $pack.ownerCommands | ForEach-Object { "- ``$_``" }

$markdown = @"
# YoloVision Owner Evidence Batch Backfill Pack

Generated at: ``$($pack.generatedAtUtc)``

## Summary

- packState: ``$($pack.packState)``
- taskCount: ``$($pack.taskCount)``
- totalMissingFieldCount: ``$($pack.totalMissingFieldCount)``
- globalMissingFieldCount: ``$($pack.globalMissingFieldCount)``
- groupCount: ``$($pack.groupCount)``
- canAutoFillOwnerFields: ``False``
- canPromoteRealModelRuntime: ``False``
- canPromotePackageConsumerRuntime: ``False``
- canCloseReleaseIssue: ``False``

## Backfill Groups

| Group | Title | Field Count | Cases | Can Promote Proof |
| --- | --- | ---: | --- | --- |
$($rows -join "`r`n")

## Owner Commands

$($commandLines -join "`r`n")

## Boundary

$($pack.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "YoloVision owner evidence batch backfill pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackState=$($pack.packState) GroupCount=$($pack.groupCount) TotalMissingFieldCount=$($pack.totalMissingFieldCount)"
