[CmdletBinding()]
param(
  [string]$OutputDirectory,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\final-release"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8FileWithRetry {
  param(
    [string]$LiteralPath,
    [AllowNull()][object]$InputObject,
    [int]$MaxAttempts = 8,
    [int]$DelayMilliseconds = 250
  )

  $directory = Split-Path -Parent $LiteralPath
  if (-not [string]::IsNullOrWhiteSpace($directory)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
  }

  $content = @($InputObject) -join [Environment]::NewLine
  $tempPath = Join-Path $directory (".{0}.{1}.tmp" -f ([IO.Path]::GetFileName($LiteralPath)), [Guid]::NewGuid().ToString("N"))
  [IO.File]::WriteAllText($tempPath, $content + [Environment]::NewLine, $script:utf8)

  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Move-Item -LiteralPath $tempPath -Destination $LiteralPath -Force
      return
    }
    catch [System.IO.IOException] {
      if ($attempt -eq $MaxAttempts) { throw }
      Start-Sleep -Milliseconds $DelayMilliseconds
    }
    catch [System.UnauthorizedAccessException] {
      if ($attempt -eq $MaxAttempts) { throw }
      Start-Sleep -Milliseconds $DelayMilliseconds
    }
  }
}

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
    return $Object.$Name
  }

  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-InputRecord {
  param(
    [string]$Id,
    [string]$Path,
    [AllowNull()][object]$Record,
    [string]$StateProperty,
    [string]$DefaultState
  )

  [pscustomobject]@{
    id = $Id
    path = $Path
    present = $null -ne $Record
    state = [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue $DefaultState)
    failedBlockerCount = [int](Get-PropertyOrDefault -Object $Record -Name "failedBlockerCount" -DefaultValue 0)
    failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $Record -Name "failedActionRequiredCount" -DefaultValue 0)
    canPublishPublicly = [bool](Get-PropertyOrDefault -Object $Record -Name "canPublishPublicly" -DefaultValue $false)
    canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false)
    isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isRuntimeExecutionProof" -DefaultValue $false)
    isPostPublishProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isPostPublishProof" -DefaultValue $false)
    isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isReleaseCloseProof" -DefaultValue $false)
    performsPublish = [bool](Get-PropertyOrDefault -Object $Record -Name "performsPublish" -DefaultValue $false)
  }
}

$inputSpecs = @(
  @{ id = "release-candidate-real-proof-final-freeze"; path = "artifacts\final-release\release-candidate-real-proof-final-freeze-validation.json"; property = "validationState"; default = "missing-release-candidate-real-proof-final-freeze-validation" },
  @{ id = "owner-real-input-import-preflight"; path = "artifacts\final-release\owner-real-input-import-preflight-validation.json"; property = "validationState"; default = "missing-owner-real-input-import-preflight-validation" },
  @{ id = "public-package-hash-cross-check-gate"; path = "artifacts\final-release\public-package-hash-cross-check-gate-validation.json"; property = "validationState"; default = "missing-public-package-hash-cross-check-gate-validation" },
  @{ id = "clean-consumer-runtime-proof-cross-check-gate"; path = "artifacts\final-release\clean-consumer-runtime-proof-cross-check-gate-validation.json"; property = "validationState"; default = "missing-clean-consumer-runtime-proof-cross-check-gate-validation" },
  @{ id = "post-publish-rollback-owner-decision-gate"; path = "artifacts\final-release\post-publish-rollback-owner-decision-gate-validation.json"; property = "validationState"; default = "missing-post-publish-rollback-owner-decision-gate-validation" },
  @{ id = "release-close-final-real-input-admission-pack"; path = "artifacts\final-release\release-close-final-real-input-admission-pack-validation.json"; property = "validationState"; default = "missing-release-close-final-real-input-admission-pack-validation" },
  @{ id = "owner-real-input-json-contract"; path = "artifacts\final-release\owner-real-input-json-contract-validation.json"; property = "validationState"; default = "missing-owner-real-input-json-contract-validation" },
  @{ id = "owner-real-input-json-import"; path = "artifacts\final-release\owner-real-input-json-import-validation.json"; property = "validationState"; default = "missing-owner-real-input-json-import-validation" },
  @{ id = "owner-real-input-hash-and-path-validator"; path = "artifacts\final-release\owner-real-input-hash-and-path-validator-validation.json"; property = "validationState"; default = "missing-owner-real-input-hash-and-path-validator-validation" },
  @{ id = "owner-real-input-forbidden-substitute-validator"; path = "artifacts\final-release\owner-real-input-forbidden-substitute-validator-validation.json"; property = "validationState"; default = "missing-owner-real-input-forbidden-substitute-validator-validation" },
  @{ id = "strict-close-real-input-dry-run"; path = "artifacts\final-release\strict-close-real-input-dry-run-validation.json"; property = "validationState"; default = "missing-strict-close-real-input-dry-run-validation" },
  @{ id = "strict-close-real-input-finding-report"; path = "artifacts\final-release\strict-close-real-input-finding-report-validation.json"; property = "validationState"; default = "missing-strict-close-real-input-finding-report-validation" },
  @{ id = "strict-close-owner-action-pack"; path = "artifacts\final-release\strict-close-owner-action-pack-validation.json"; property = "validationState"; default = "missing-strict-close-owner-action-pack-validation" },
  @{ id = "release-close-real-input-final-blocker-ledger"; path = "artifacts\final-release\release-close-real-input-final-blocker-ledger-validation.json"; property = "validationState"; default = "missing-release-close-real-input-final-blocker-ledger-validation" },
  @{ id = "release-close-real-input-candidate-promotion-readiness"; path = "artifacts\final-release\release-close-real-input-candidate-promotion-readiness-validation.json"; property = "validationState"; default = "missing-release-close-real-input-candidate-promotion-readiness-validation" },
  @{ id = "real-owner-evidence-strict-validator-orchestration"; path = "artifacts\final-release\real-owner-evidence-strict-validator-orchestration-validation.json"; property = "validationState"; default = "missing-real-owner-evidence-strict-validator-orchestration-validation" },
  @{ id = "final-publish-proof-gate"; path = "artifacts\final-release\final-publish-proof-gate-report.json"; property = "validationState"; default = "missing-final-publish-proof-gate-report" },
  @{ id = "release-evidence-classification-audit"; path = "artifacts\final-release\release-evidence-classification-audit.json"; property = "auditState"; default = "missing-release-evidence-classification-audit" },
  @{ id = "release-evidence-bundle"; path = "artifacts\final-release\release-evidence-bundle.json"; property = "bundleState"; default = "missing-release-evidence-bundle" },
  @{ id = "final-owner-strict-close-execution-order"; path = "artifacts\final-release\final-owner-strict-close-execution-order-validation.json"; property = "validationState"; default = "missing-final-owner-strict-close-execution-order-validation" },
  @{ id = "final-release-close-record-real-validator"; path = "artifacts\final-release\final-release-close-record-real-validator-validation.json"; property = "validationState"; default = "missing-final-release-close-record-real-validator-validation" }
)

$records = New-Object System.Collections.Generic.List[object]
foreach ($spec in $inputSpecs) {
  $record = Read-JsonOrNull $spec.path
  $records.Add((New-InputRecord -Id $spec.id -Path $spec.path -Record $record -StateProperty $spec.property -DefaultState $spec.default))
}

$recordArray = @($records.ToArray())
$missingInputCount = @($recordArray | Where-Object { -not $_.present }).Count
$blockedInputCount = @($recordArray | Where-Object { $_.state -match "blocked|missing|owner-action|required|template|incomplete" }).Count
$unexpectedProofFlagCount = @($recordArray | Where-Object {
    $_.canPublishPublicly -or $_.canCloseReleaseIssue -or $_.isRuntimeExecutionProof -or $_.isPostPublishProof -or $_.isReleaseCloseProof -or $_.performsPublish
  }).Count
$failedBlockerCount = ($recordArray | Measure-Object -Property failedBlockerCount -Sum).Sum
if ($null -eq $failedBlockerCount) { $failedBlockerCount = 0 }
$failedActionRequiredCount = ($recordArray | Measure-Object -Property failedActionRequiredCount -Sum).Sum
if ($null -eq $failedActionRequiredCount) { $failedActionRequiredCount = 0 }

$holdItems = @(
  "Owner real input JSON contract 已成为最终冻结输入，但仍需真实 Owner JSON 回填",
  "公开包 hash cross-check 与 clean consumer runtime proof cross-check 已进入冻结面，但仍不能替代 post-publish proof",
  "strict close JSON import/hash/path/forbidden substitute validators 已进入冻结面，但仍只是字段与边界校验",
  "真实 Owner 授权与发布命令结果仍需导入",
  "公开包 clean consumer runtime proof 仍需真实外部记录",
  "Linux runner proof 仍需真实 runner 记录",
  "real-model-runtime proof 仍需真实模型、license、hash、日志和输出",
  "post-publish verification 仍需公开发布后的真实验证记录",
  "failedBlockerCount=0 只能表示结构 blocker 未失败，不能解释为 ready"
)

$boundary = "This dashboard is final quality freeze planning only: not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. failedBlockerCount=0 is not ready."

$dashboard = [pscustomobject]@{
  recordKind = "final-quality-freeze-dashboard"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  freezeState = "blocked-final-quality-freeze-real-proof-required"
  canPublishPublicly = $false
  canExecutePublicPublish = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  performsPublish = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isPackagePush = $false
  inputRecordCount = $recordArray.Count
  ownerRealInputControlRecordCount = @($recordArray | Where-Object { $_.id -match "owner-real-input|strict-close-real-input|release-close-real-input|public-package-hash-cross-check-gate|clean-consumer-runtime-proof-cross-check-gate|release-candidate-real-proof-final-freeze|release-close-final-real-input-admission-pack|post-publish-rollback-owner-decision-gate" }).Count
  missingInputCount = $missingInputCount
  blockedInputCount = $blockedInputCount
  unexpectedProofFlagCount = $unexpectedProofFlagCount
  failedBlockerCount = [int]$failedBlockerCount
  failedActionRequiredCount = [int]$failedActionRequiredCount
  holdItemCount = $holdItems.Count
  holdItems = @($holdItems)
  inputRecords = @($recordArray)
  sourceArtifacts = @($inputSpecs | ForEach-Object { $_.path })
  boundary = $boundary
  nonProofBoundary = @(
    "not runtime proof",
    "not post-publish proof",
    "not publish approval",
    "not release close approval",
    "not package push",
    "failedBlockerCount=0 is not ready"
  )
}

$jsonPath = Join-Path $OutputDirectory "final-quality-freeze-dashboard.json"
$markdownPath = Join-Path $OutputDirectory "final-quality-freeze-dashboard.md"

$dashboard | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Final Quality Freeze Dashboard")
$lines.Add("")
$lines.Add("`final-quality-freeze-dashboard` 聚合发布前最后一层质量冻结状态。它只说明当前仍被真实 Owner 输入和外部 proof 阻断，不执行发布、不生成 proof、不关闭 release issue。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| freezeState | ``$($dashboard.freezeState)`` |")
$lines.Add("| canPublishPublicly | ``$($dashboard.canPublishPublicly)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($dashboard.canCloseReleaseIssue)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($dashboard.isRuntimeExecutionProof)`` |")
$lines.Add("| isPostPublishProof | ``$($dashboard.isPostPublishProof)`` |")
$lines.Add("| isReleaseCloseProof | ``$($dashboard.isReleaseCloseProof)`` |")
$lines.Add("| failedBlockerCount | ``$($dashboard.failedBlockerCount)`` |")
$lines.Add("| failedActionRequiredCount | ``$($dashboard.failedActionRequiredCount)`` |")
$lines.Add("| inputRecordCount | ``$($dashboard.inputRecordCount)`` |")
$lines.Add("| ownerRealInputControlRecordCount | ``$($dashboard.ownerRealInputControlRecordCount)`` |")
$lines.Add("| blockedInputCount | ``$($dashboard.blockedInputCount)`` |")
$lines.Add("")
$lines.Add("## Input Records")
$lines.Add("")
$lines.Add("| Id | Present | State | Action Required |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($inputRecord in $recordArray) {
  $lines.Add("| $(ConvertTo-MarkdownCell $inputRecord.id) | ``$($inputRecord.present)`` | $(ConvertTo-MarkdownCell $inputRecord.state) | ``$($inputRecord.failedActionRequiredCount)`` |")
}
$lines.Add("")
$lines.Add("## Hold Items")
$lines.Add("")
foreach ($item in $holdItems) {
  $lines.Add("- $item")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($boundary)

Write-Utf8FileWithRetry -LiteralPath $markdownPath -InputObject $lines

Write-Host "Final quality freeze dashboard written: $jsonPath"
Write-Host "Final quality freeze dashboard markdown written: $markdownPath"
