[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:OwnerExternalRealProofBoundary = 'not runtime proof; not post-publish proof; not publish approval; not release close approval; not package push'
$script:OwnerExternalRealProofForbiddenSubstitutes = @(
  'local .nupkg',
  'local feed',
  'ProjectReference',
  'direct nupkg',
  'template',
  'draft',
  'dry-run',
  'runbook',
  'dashboard',
  'audit pack',
  'hash slot',
  'candidate',
  'local-only scan',
  'manual handoff',
  'Owner execution package'
)

function Get-OwnerExternalRealProofGateSpec {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)]
    [ValidateSet(
      'owner-external-real-proof-input-contract',
      'owner-external-real-proof-import-validator',
      'post-publish-clean-consumer-real-proof-gate',
      'runtime-compatible-host-real-proof-gate',
      'release-close-real-proof-readiness-gate'
    )]
    [string]$ArtifactId
  )

  $specs = @{
    'owner-external-real-proof-input-contract' = [ordered]@{
      artifactId = 'owner-external-real-proof-input-contract'
      title = 'Owner external real proof input contract'
      documentTitle = 'Owner 外部真实 Proof 输入合同'
      statePropertyName = 'contractState'
      state = 'blocked-owner-external-real-proof-input-contract-owner-input-required'
      countPropertyName = 'fieldCount'
      itemsPropertyName = 'fields'
      itemKind = 'field'
      summary = '定义公开发布、仓库外 clean consumer、兼容主机 runtime proof、post-publish verification 与 release close approval 的真实 Owner 输入字段。'
      requiredOwnerFields = @(
        'ownerName',
        'packageId',
        'packageVersion',
        'nugetOrgPackageUrl',
        'githubPackagesUrl',
        'publishedNupkgSha256',
        'publishedSymbolsSha256',
        'publicPackageDownloadUrl',
        'cleanConsumerProjectPath',
        'cleanConsumerPackageSource',
        'cleanConsumerRestoreLogSha256',
        'cleanConsumerSmokeLogSha256',
        'cleanConsumerProjectReferenceCount',
        'cleanConsumerDirectNupkgReferenceCount',
        'cleanConsumerLocalFeedReferenceCount',
        'runtimeHostId',
        'runtimePackageKey',
        'cudaVersion',
        'tensorRtVersion',
        'driverVersion',
        'runtimeSmokeExitCode',
        'runtimeSmokeLogSha256',
        'versionGuardSummary',
        'postPublishVerificationUrl',
        'rollbackPlanReviewed',
        'ownerPublishDecision',
        'ownerCloseDecision'
      )
      items = @(
        '公开 package URL、package id/version 与下载 hash 必须来自真实公开渠道',
        '仓库外 clean consumer 路径必须位于源码仓库之外',
        'clean consumer restore/smoke 日志 SHA256 必须由 Owner 回填',
        'clean consumer 必须声明 ProjectReference、direct nupkg 与 local feed 均为 0',
        '兼容主机必须回填 host id、GPU driver、CUDA、TensorRT 与 runtime package key',
        'runtime smoke 必须回填 exit code、stdout/stderr 摘要、日志 SHA256 与 version guard 结果',
        'post-publish verification 必须回填公开 URL、download proof、known limitations 与 rollback review',
        'release close 必须回填最终 Owner close decision 且仍需 strict validator 通过'
      )
    }
    'owner-external-real-proof-import-validator' = [ordered]@{
      artifactId = 'owner-external-real-proof-import-validator'
      title = 'Owner external real proof import validator'
      documentTitle = 'Owner 外部真实 Proof 导入校验器'
      statePropertyName = 'validatorState'
      state = 'blocked-owner-external-real-proof-import-validator-owner-input-required'
      countPropertyName = 'checkCount'
      itemsPropertyName = 'checks'
      itemKind = 'check'
      summary = '校验 Owner 真实输入 JSON 的字段完整性、URL/hash/source/host metadata 与禁止替代项；只校验输入，不执行发布或下载。'
      requiredOwnerFields = @(
        'ownerInputJsonPath',
        'ownerInputSha256',
        'publicUrlFields',
        'sha256Fields',
        'publicSourceFields',
        'forbiddenSubstituteCounts',
        'runtimeHostMetadata',
        'validationTranscriptSha256'
      )
      items = @(
        '校验 Owner 输入 JSON 存在且 hash 被记录',
        '校验 public URL 字段是 https 或明确的公开源 URL',
        '校验 nupkg、restore log、smoke log、runtime log SHA256 是 64 位十六进制',
        '校验 package source 不是 local feed、文件夹源或 cache-only 命中',
        '校验 ProjectReference、direct nupkg 与 local feed 替代计数均为 0',
        '校验 runtime host metadata 包含 CUDA、TensorRT、driver 与 runtime package key',
        '校验 Owner publish/close decision 字段存在但不自动批准',
        '输出 blocked validation，直到真实 proof record strict validator 通过'
      )
    }
    'post-publish-clean-consumer-real-proof-gate' = [ordered]@{
      artifactId = 'post-publish-clean-consumer-real-proof-gate'
      title = 'Post-publish clean consumer real proof gate'
      documentTitle = '发布后 Clean Consumer 真实 Proof Gate'
      statePropertyName = 'gateState'
      state = 'blocked-post-publish-clean-consumer-real-proof-gate-owner-proof-required'
      countPropertyName = 'requirementCount'
      itemsPropertyName = 'requirements'
      itemKind = 'requirement'
      summary = '聚焦仓库外 clean consumer proof：公开源 restore、smoke、日志 hash、项目身份与禁止本地替代。'
      requiredOwnerFields = @(
        'consumerProjectPath',
        'consumerProjectOutsideRepository',
        'packageSource',
        'packageId',
        'packageVersion',
        'restoreExitCode',
        'smokeExitCode',
        'restoreLogSha256',
        'smokeLogSha256',
        'resolvedPackagePath',
        'projectReferenceCount',
        'directNupkgReferenceCount',
        'localFeedReferenceCount'
      )
      items = @(
        'consumer project 必须在源码仓库外创建',
        'restore 必须使用公开源和已发布 package id/version',
        'restore exit code、stdout/stderr 摘要与日志 SHA256 必须完整',
        'smoke exit code、stdout/stderr 摘要与日志 SHA256 必须完整',
        'resolved package path 必须来自公开 NuGet cache，不是 local feed 或 direct nupkg',
        'ProjectReference count 必须为 0',
        'direct nupkg reference count 必须为 0',
        'local feed reference count 必须为 0'
      )
    }
    'runtime-compatible-host-real-proof-gate' = [ordered]@{
      artifactId = 'runtime-compatible-host-real-proof-gate'
      title = 'Runtime compatible host real proof gate'
      documentTitle = '兼容主机 Runtime 真实 Proof Gate'
      statePropertyName = 'gateState'
      state = 'blocked-runtime-compatible-host-real-proof-gate-owner-proof-required'
      countPropertyName = 'requirementCount'
      itemsPropertyName = 'requirements'
      itemKind = 'requirement'
      summary = '聚焦兼容 GPU/CUDA/TensorRT 主机 runtime proof：host metadata、version guard、smoke exit code 与日志 hash 必须完整。'
      requiredOwnerFields = @(
        'hostId',
        'osDescription',
        'gpuName',
        'driverVersion',
        'cudaVersion',
        'tensorRtVersion',
        'runtimePackageKey',
        'versionGuardSummary',
        'cudaSmokeExitCode',
        'tensorRtSmokeExitCode',
        'packageConsumerExitCode',
        'runtimeSmokeLogSha256',
        'dependencyProbeOnly',
        'driverBlocked',
        'buildOnly'
      )
      items = @(
        'host metadata 必须包含 OS、GPU、driver、CUDA、TensorRT 与 dotnet 信息',
        'runtime package key 必须与目标包和 native asset 解析路径匹配',
        'version guard summary 必须覆盖 TRT8/TRT10/TRT11 available/skipped 状态',
        'CUDA smoke exit code 必须来自真实兼容主机执行',
        'TensorRT smoke exit code 必须来自真实兼容主机执行',
        'package consumer smoke exit code 必须来自公开包消费路径',
        'runtime smoke log SHA256 必须完整',
        'driver-blocked、DependencyProbe-only 或 build-only 结果不得晋级'
      )
    }
    'release-close-real-proof-readiness-gate' = [ordered]@{
      artifactId = 'release-close-real-proof-readiness-gate'
      title = 'Release close real proof readiness gate'
      documentTitle = 'ReleaseClose 真实 Proof 准入 Gate'
      statePropertyName = 'readinessState'
      state = 'blocked-release-close-real-proof-readiness-gate-owner-proof-required'
      countPropertyName = 'blockerCount'
      itemsPropertyName = 'blockers'
      itemKind = 'blocker'
      summary = '汇总公开发布结果、clean consumer proof、runtime proof、post-publish verification、rollback approval 与 Owner close decision。'
      requiredOwnerFields = @(
        'publicPublishResultRecord',
        'publicPackageDownloadHash',
        'cleanConsumerRealProofRecord',
        'runtimeCompatibleHostRealProofRecord',
        'postPublishVerificationRecord',
        'rollbackPlanReviewed',
        'ownerCloseDecision',
        'releaseIssueCloseRecordValidation',
        'classificationAudit'
      )
      items = @(
        '公开发布结果必须来自真实公开渠道和 Owner 手动执行记录',
        '公开包下载 hash 必须与 freeze/release package hash 交叉核对',
        'clean consumer real proof 必须来自仓库外项目和公开源',
        'runtime compatible host real proof 必须来自兼容 GPU/CUDA/TensorRT 主机',
        'post-publish verification 必须包含公开 URL、download proof、known limitations 与日志 hash',
        'rollback plan 必须由 Owner 审阅',
        'Owner close decision 必须明确且可追溯',
        'release issue close strict validator 通过前 canCloseReleaseIssue 必须保持 false'
      )
    }
  }

  return $specs[$ArtifactId]
}

function New-OwnerExternalRealProofGateArtifact {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)]
    [string]$ArtifactId,

    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
  )

  $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
  $outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $repoRoot $OutputDirectory }
  New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

  $spec = Get-OwnerExternalRealProofGateSpec -ArtifactId $ArtifactId
  $boundary = $script:OwnerExternalRealProofBoundary
  $items = @()
  for ($i = 0; $i -lt $spec.items.Count; $i++) {
    $items += [pscustomobject]@{
      id = ('{0}-{1:00}' -f $spec.itemKind, ($i + 1))
      title = [string]$spec.items[$i]
      status = 'blocked-owner-proof-required'
      passed = $false
      boundary = $boundary
    }
  }

  $result = [ordered]@{
    artifactId = $spec.artifactId
    generatedAt = (Get-Date).ToUniversalTime().ToString('o')
    title = $spec.title
    summary = $spec.summary
    blockedCount = @($items | Where-Object { -not $_.passed }).Count
    passed = $false
    performsPublish = $false
    notExecutedByAutomation = $true
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    isReleaseCloseRecordProof = $false
    requiredOwnerFields = @($spec.requiredOwnerFields)
    boundary = $boundary
    forbiddenSubstitutes = @($script:OwnerExternalRealProofForbiddenSubstitutes)
  }

  $result[$spec.statePropertyName] = $spec.state
  $result[$spec.countPropertyName] = $items.Count
  $result[$spec.itemsPropertyName] = $items

  $jsonPath = Join-Path $outputRoot "$ArtifactId.json"
  $mdPath = Join-Path $outputRoot "$ArtifactId.md"
  $result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

  $lines = @(
    "# $($spec.title)",
    '',
    "- Artifact: ``$ArtifactId``",
    "- State: ``$($spec.state)``",
    '- Passed: `false`',
    '- Performs publish: `false`',
    '- Can publish publicly: `false`',
    '- Can close release issue: `false`',
    "- Boundary: $boundary",
    '',
    '## Summary',
    '',
    $spec.summary,
    '',
    '## Required owner fields',
    ''
  )
  $lines += @($spec.requiredOwnerFields | ForEach-Object { "- ``$_``" })
  $lines += @('', "## $($spec.itemsPropertyName)", '')
  $lines += @($items | ForEach-Object { "- [$($_.status)] $($_.title) - $($_.boundary)" })
  $lines += @('', '## Forbidden substitutes', '')
  $lines += @($result.forbiddenSubstitutes | ForEach-Object { "- $_" })
  $lines | Set-Content -LiteralPath $mdPath -Encoding UTF8

  Write-Host "Wrote $jsonPath"
  Write-Host "Wrote $mdPath"
}

function Test-OwnerExternalRealProofGateArtifact {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)]
    [string]$ArtifactId,

    [switch]$Strict,

    [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
  )

  $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
  $artifactRoot = if ([System.IO.Path]::IsPathRooted($ArtifactDirectory)) { $ArtifactDirectory } else { Join-Path $repoRoot $ArtifactDirectory }
  $jsonPath = Join-Path $artifactRoot "$ArtifactId.json"
  if (-not (Test-Path -LiteralPath $jsonPath)) {
    New-OwnerExternalRealProofGateArtifact -ArtifactId $ArtifactId -OutputDirectory $artifactRoot
  }

  $spec = Get-OwnerExternalRealProofGateSpec -ArtifactId $ArtifactId
  $artifact = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
  $findings = New-Object System.Collections.Generic.List[object]
  function Add-Finding([string]$Id, [string]$Message) {
    $findings.Add([pscustomobject]@{ id = $Id; message = $Message }) | Out-Null
  }

  $stateProperty = $artifact.PSObject.Properties[[string]$spec.statePropertyName]
  $countProperty = $artifact.PSObject.Properties[[string]$spec.countPropertyName]

  if ([string]$artifact.artifactId -ne $ArtifactId) { Add-Finding 'artifact-id' 'Unexpected artifact id.' }
  if ($null -eq $stateProperty -or [string]$stateProperty.Value -ne [string]$spec.state) { Add-Finding 'state' 'Unexpected blocked state.' }
  if ([bool]$artifact.passed) { Add-Finding 'passed' 'Real proof gate artifact must not pass without real owner proof.' }
  if ([bool]$artifact.performsPublish) { Add-Finding 'performs-publish' 'This artifact must not perform package push.' }
  if ([bool]$artifact.canPublishPublicly) { Add-Finding 'can-publish-publicly' 'This artifact must not approve public publishing.' }
  if ([bool]$artifact.canCloseReleaseIssue) { Add-Finding 'can-close-release-issue' 'This artifact must not approve release issue close.' }
  if ([bool]$artifact.canPromoteRuntimeProof) { Add-Finding 'can-promote-runtime-proof' 'This artifact must not promote runtime proof.' }
  if ([bool]$artifact.isRuntimeExecutionProof) { Add-Finding 'runtime-proof' 'This artifact must not be runtime proof.' }
  if ([bool]$artifact.isPostPublishProof) { Add-Finding 'post-publish-proof' 'This artifact must not be post-publish proof.' }
  if ([bool]$artifact.isReleaseCloseProof) { Add-Finding 'release-close-proof' 'This artifact must not be release close proof.' }
  if ([bool]$artifact.isReleaseCloseRecordProof) { Add-Finding 'release-close-record-proof' 'This artifact must not be release close record proof.' }
  foreach ($token in @('not runtime proof', 'not post-publish proof', 'not publish approval', 'not release close approval', 'not package push')) {
    if ([string]$artifact.boundary -notmatch [regex]::Escape($token)) {
      Add-Finding "boundary-$token" "Missing boundary token: $token"
    }
  }
  if ($null -eq $countProperty -or [int]$countProperty.Value -lt @($spec.items).Count) { Add-Finding 'item-count' 'Expected owner proof gate surface is incomplete.' }
  if ($null -eq $countProperty -or [int]$artifact.blockedCount -ne [int]$countProperty.Value) { Add-Finding 'blocked-count' 'All generated items must remain blocked.' }
  if (@($artifact.requiredOwnerFields).Count -lt @($spec.requiredOwnerFields).Count) { Add-Finding 'required-owner-fields' 'Required owner input fields are incomplete.' }
  foreach ($substitute in $script:OwnerExternalRealProofForbiddenSubstitutes) {
    if (@($artifact.forbiddenSubstitutes) -notcontains $substitute) {
      Add-Finding "forbidden-$substitute" "Missing forbidden substitute: $substitute"
    }
  }

  $validationState = if ($findings.Count -eq 0) { 'validation-passed-non-proof-owner-external-real-proof-boundary-intact' } else { 'validation-failed' }
  $validation = [ordered]@{
    artifactId = $ArtifactId
    generatedAt = (Get-Date).ToUniversalTime().ToString('o')
    validationState = $validationState
    findingCount = $findings.Count
    findings = @($findings.ToArray())
    strict = [bool]$Strict
    passed = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    isReleaseCloseRecordProof = $false
    boundary = $script:OwnerExternalRealProofBoundary
  }

  $validationJsonPath = Join-Path $artifactRoot "$ArtifactId-validation.json"
  $validationMdPath = Join-Path $artifactRoot "$ArtifactId-validation.md"
  $validation | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $validationJsonPath -Encoding UTF8
  @(
    "# $($spec.title) validation",
    '',
    "- State: ``$validationState``",
    "- Findings: ``$($findings.Count)``",
    '- Passed: `false`',
    "- Boundary: $($script:OwnerExternalRealProofBoundary)"
  ) | Set-Content -LiteralPath $validationMdPath -Encoding UTF8

  if ($Strict -and $findings.Count -gt 0) {
    $findings | Format-Table -AutoSize | Out-String | Write-Error
  }

  Write-Host "${ArtifactId}: $validationState; FindingCount=$($findings.Count)"
}
