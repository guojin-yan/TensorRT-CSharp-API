[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:OwnerPublicReleaseExecutionBoundary = 'not runtime proof; not post-publish proof; not publish approval; not release close approval; not package push'

function Get-OwnerPublicReleaseExecutionKitSpec {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)]
    [ValidateSet(
      'public-release-owner-execution-package',
      'external-clean-consumer-proof-kit',
      'runtime-proof-compatible-host-kit',
      'post-publish-owner-verification-kit',
      'owner-public-release-execution-readiness-pack'
    )]
    [string]$ArtifactId
  )

  $specs = @{
    'public-release-owner-execution-package' = [ordered]@{
      artifactId = 'public-release-owner-execution-package'
      title = 'Public release owner execution package'
      documentTitle = '真实公开发布 Owner 执行包'
      statePropertyName = 'packageState'
      state = 'blocked-public-release-owner-execution-package-owner-proof-required'
      countPropertyName = 'commandCount'
      itemsPropertyName = 'commands'
      itemKind = 'command'
      summary = '汇总 NuGet.org、GitHub Packages、GitHub Release、包/hash/release notes 核对等最终人工发布命令面，但不执行任何上传。'
      requiredOwnerFields = @(
        'ownerName',
        'packageId',
        'packageVersion',
        'nugetOrgPackageUrl',
        'githubPackagesUrl',
        'publishedNupkgSha256',
        'publishedSymbolsSha256',
        'releaseNotesUrl',
        'rollbackPlanReviewed',
        'ownerPublishDecision'
      )
      items = @(
        '确认 release evidence bundle 当前仍为 canPublishPublicly=false 且 canCloseReleaseIssue=false',
        '由 Owner 在隔离终端手动核对 NuGet.org API key、source、package id 与 version',
        '由 Owner 手动核对 GitHub Packages token、owner、source 与 package id',
        '由 Owner 手动核对 GitHub Release tag、release notes、artifact hash 与 rollback notes',
        '由 Owner 手动执行公开发布命令；本 artifact 只保存命令模板，不执行 dotnet nuget push',
        '发布后采集公开包 URL、下载 hash、source index 与时间戳',
        '刷新 post-publish owner verification kit 并保持所有本地候选项 non-proof',
        '运行 release evidence classification audit，确认禁止替代边界仍为 0 findings'
      )
    }
    'external-clean-consumer-proof-kit' = [ordered]@{
      artifactId = 'external-clean-consumer-proof-kit'
      title = 'External clean consumer proof kit'
      documentTitle = '外部干净 Consumer Proof 采集包'
      statePropertyName = 'kitState'
      state = 'blocked-external-clean-consumer-proof-kit-owner-proof-required'
      countPropertyName = 'stepCount'
      itemsPropertyName = 'steps'
      itemKind = 'step'
      summary = '定义仓库外 clean consumer 项目的真实验证步骤、日志字段、包源字段与不可替代边界。'
      requiredOwnerFields = @(
        'consumerProjectPath',
        'packageSource',
        'packageId',
        'packageVersion',
        'restoreLogSha256',
        'smokeLogSha256',
        'runtimePackageKey',
        'hostMetadata',
        'projectReferenceCount',
        'directNupkgReferenceCount'
      )
      items = @(
        '在仓库外新建 clean consumer 目录，不允许 ProjectReference 或 direct nupkg',
        '清理 NuGet 缓存并使用公开源 restore 指定 package id/version',
        '记录 dotnet restore stdout/stderr、source、resolved package path 与 lock file hash',
        '运行最小 managed API smoke，确认只引用公开包',
        '运行 TensorRT/CUDA 可用路径 smoke，并记录 exit code、stdout/stderr 与日志 hash',
        '采集 dotnet SDK、OS、CPU/GPU、CUDA、TensorRT、driver 版本',
        '生成 post-publish clean consumer proof record owner input',
        '运行 strict validator，禁止 local feed、ProjectReference、direct nupkg、cache-only 命中晋级'
      )
    }
    'runtime-proof-compatible-host-kit' = [ordered]@{
      artifactId = 'runtime-proof-compatible-host-kit'
      title = 'Runtime proof compatible host kit'
      documentTitle = '兼容主机 Runtime Proof 采集包'
      statePropertyName = 'kitState'
      state = 'blocked-runtime-proof-compatible-host-kit-owner-proof-required'
      countPropertyName = 'laneCount'
      itemsPropertyName = 'lanes'
      itemKind = 'lane'
      summary = '把兼容 GPU/CUDA/TensorRT 主机上的 runtime proof 采集任务集中为可执行 lane，但不把本地 precheck 晋级。'
      requiredOwnerFields = @(
        'hostId',
        'runtimePackageKey',
        'cudaVersion',
        'tensorRtVersion',
        'driverVersion',
        'cudaSmokeExitCode',
        'tensorRtSmokeExitCode',
        'packageConsumerExitCode',
        'logSha256',
        'versionGuardSummary'
      )
      items = @(
        '确认主机 GPU driver、CUDA runtime、TensorRT runtime 与目标 runtime package key 匹配',
        '运行 CUDA smoke runner 并采集 exit code、stdout/stderr、日志 hash',
        '运行 TensorRT smoke runner 并采集 version guard 与 native asset resolution 日志',
        '运行 package consumer runner，确认公开包引用路径与 runtime asset 解析路径',
        '覆盖 TRT8/TRT10/TRT11 version guard 的 skipped/available 状态记录',
        '采集 nvidia-smi、CUDA version、TensorRT library version、dotnet --info',
        '生成 external-runtime-proof-record owner input',
        '运行 runtime proof validator，确保 driver-blocked/dependency-probe-only 不晋级'
      )
    }
    'post-publish-owner-verification-kit' = [ordered]@{
      artifactId = 'post-publish-owner-verification-kit'
      title = 'Post-publish owner verification kit'
      documentTitle = '发布后 Owner 验证采集包'
      statePropertyName = 'kitState'
      state = 'blocked-post-publish-owner-verification-kit-owner-proof-required'
      countPropertyName = 'fieldCount'
      itemsPropertyName = 'fields'
      itemKind = 'field'
      summary = '定义真实发布后 Owner 必须回填的公开 URL、hash、clean restore、download proof、runtime proof 与 known limitations 链接。'
      requiredOwnerFields = @(
        'nugetOrgPackageUrl',
        'publishedVersion',
        'publishedNupkgSha256',
        'cleanRestoreProofPath',
        'cleanConsumerProofPath',
        'runtimeProofPath',
        'knownLimitationsUrl',
        'ownerPostPublishDecision'
      )
      items = @(
        '回填 NuGet.org package URL 与 version index 可见性',
        '回填 GitHub Packages URL 或明确不使用 GitHub Packages 的 Owner 决策',
        '回填公开下载 .nupkg/.snupkg SHA256 与本地 freeze hash 对照',
        '回填 clean restore proof 与仓库外 consumer proof record',
        '回填兼容主机 runtime proof record 与日志 hash',
        '回填 known limitations、release notes 与 rollback plan review',
        '回填 post-publish verification owner decision',
        '运行 strict close validation 前保持 canCloseReleaseIssue=false'
      )
    }
    'owner-public-release-execution-readiness-pack' = [ordered]@{
      artifactId = 'owner-public-release-execution-readiness-pack'
      title = 'Owner public release execution readiness pack'
      documentTitle = 'Owner 公开发布执行 Readiness 汇总包'
      statePropertyName = 'readinessState'
      state = 'blocked-owner-public-release-execution-readiness-pack-owner-proof-required'
      countPropertyName = 'blockerCount'
      itemsPropertyName = 'blockers'
      itemKind = 'blocker'
      summary = '汇总公开发布执行、外部 clean consumer、兼容主机 runtime proof、post-publish verification 四条 owner lane 的阻断状态。'
      requiredOwnerFields = @(
        'publicReleaseOwnerExecutionPackage',
        'externalCleanConsumerProofKit',
        'runtimeProofCompatibleHostKit',
        'postPublishOwnerVerificationKit',
        'releaseEvidenceClassificationAudit',
        'releaseIssueCloseStrictValidation'
      )
      items = @(
        'public-release-owner-execution-package 仍需真实 Owner 发布结果',
        'external-clean-consumer-proof-kit 仍需仓库外公开包 restore/smoke 结果',
        'runtime-proof-compatible-host-kit 仍需兼容 GPU/CUDA/TensorRT 主机运行结果',
        'post-publish-owner-verification-kit 仍需公开 URL、hash、download 与 rollback review',
        'release evidence bundle 必须继续保持 canPublishPublicly=false',
        'release issue close 必须继续保持 canCloseReleaseIssue=false',
        'classification audit 必须保持 no promoted substitute proof',
        '最终关闭前必须运行 Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady'
      )
    }
  }

  return $specs[$ArtifactId]
}

function New-OwnerPublicReleaseExecutionArtifact {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)]
    [string]$ArtifactId,

    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
  )

  $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
  $outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $repoRoot $OutputDirectory }
  New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

  $spec = Get-OwnerPublicReleaseExecutionKitSpec -ArtifactId $ArtifactId
  $boundary = $script:OwnerPublicReleaseExecutionBoundary
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
    forbiddenSubstitutes = @(
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
      'manual handoff'
    )
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

function Test-OwnerPublicReleaseExecutionArtifact {
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
    New-OwnerPublicReleaseExecutionArtifact -ArtifactId $ArtifactId -OutputDirectory $artifactRoot
  }

  $spec = Get-OwnerPublicReleaseExecutionKitSpec -ArtifactId $ArtifactId
  $artifact = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
  $findings = New-Object System.Collections.Generic.List[object]
  function Add-Finding([string]$Id, [string]$Message) {
    $findings.Add([pscustomobject]@{ id = $Id; message = $Message }) | Out-Null
  }

  $stateProperty = $artifact.PSObject.Properties[[string]$spec.statePropertyName]
  $countProperty = $artifact.PSObject.Properties[[string]$spec.countPropertyName]

  if ([string]$artifact.artifactId -ne $ArtifactId) { Add-Finding 'artifact-id' 'Unexpected artifact id.' }
  if ($null -eq $stateProperty -or [string]$stateProperty.Value -ne [string]$spec.state) { Add-Finding 'state' 'Unexpected blocked state.' }
  if ([bool]$artifact.passed) { Add-Finding 'passed' 'Owner execution package must not pass without real owner proof.' }
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
  if ($null -eq $countProperty -or [int]$countProperty.Value -lt @($spec.items).Count) { Add-Finding 'item-count' 'Expected owner action surface is incomplete.' }
  if ($null -eq $countProperty -or [int]$artifact.blockedCount -ne [int]$countProperty.Value) { Add-Finding 'blocked-count' 'All generated items must remain blocked.' }
  if (@($artifact.requiredOwnerFields).Count -lt @($spec.requiredOwnerFields).Count) { Add-Finding 'required-owner-fields' 'Required owner input fields are incomplete.' }
  foreach ($substitute in @('local .nupkg', 'local feed', 'ProjectReference', 'direct nupkg', 'template', 'draft', 'dry-run', 'runbook', 'dashboard', 'audit pack', 'hash slot', 'candidate', 'local-only scan', 'manual handoff')) {
    if (@($artifact.forbiddenSubstitutes) -notcontains $substitute) {
      Add-Finding "forbidden-$substitute" "Missing forbidden substitute: $substitute"
    }
  }

  $validationState = if ($findings.Count -eq 0) { 'validation-passed-non-proof-owner-execution-boundary-intact' } else { 'validation-failed' }
  $validation = [ordered]@{
    artifactId = $ArtifactId
    generatedAt = (Get-Date).ToUniversalTime().ToString('o')
    validationState = $validationState
    findingCount = $findings.Count
    findings = @($findings.ToArray())
    strict = [bool]$Strict
    passed = $false
    boundary = $script:OwnerPublicReleaseExecutionBoundary
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
    "- Boundary: $($script:OwnerPublicReleaseExecutionBoundary)"
  ) | Set-Content -LiteralPath $validationMdPath -Encoding UTF8

  if ($Strict -and $findings.Count -gt 0) {
    $findings | Format-Table -AutoSize | Out-String | Write-Error
  }

  Write-Host "${ArtifactId}: $validationState; FindingCount=$($findings.Count)"
}
