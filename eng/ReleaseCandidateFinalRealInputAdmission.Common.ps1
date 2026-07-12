[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:ReleaseCandidateFinalRealInputAdmissionBoundary = 'not runtime proof; not post-publish proof; not publish approval; not release close approval; not package push'
$script:ReleaseCandidateFinalRealInputAdmissionForbiddenSubstitutes = @(
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
  'Owner execution package',
  'real proof readiness gate'
)

function Get-ReleaseCandidateFinalRealInputAdmissionSpec {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)]
    [ValidateSet(
      'release-candidate-real-proof-final-freeze',
      'owner-real-input-import-preflight',
      'public-package-hash-cross-check-gate',
      'clean-consumer-runtime-proof-cross-check-gate',
      'post-publish-rollback-owner-decision-gate',
      'release-close-final-real-input-admission-pack'
    )]
    [string]$ArtifactId
  )

  $specs = @{
    'release-candidate-real-proof-final-freeze' = [ordered]@{
      artifactId = 'release-candidate-real-proof-final-freeze'
      title = 'Release candidate real proof final freeze'
      documentTitle = '发布候选真实 Proof 最终冻结'
      statePropertyName = 'freezeState'
      state = 'blocked-release-candidate-real-proof-final-freeze-owner-input-required'
      countPropertyName = 'freezeRequirementCount'
      itemsPropertyName = 'freezeRequirements'
      itemKind = 'freeze-requirement'
      summary = '汇总真实 proof 准入 gate、release evidence bundle、classification audit 和公开发布 Owner 执行包的最终冻结面；只冻结本地证据状态，不晋级为 proof。'
      requiredOwnerFields = @(
        'releaseEvidenceBundleSha256',
        'classificationAuditSha256',
        'publicReleaseOwnerExecutionPackageSha256',
        'ownerExternalRealProofInputContractSha256',
        'ownerExternalRealProofImportValidatorSha256',
        'postPublishCleanConsumerRealProofGateSha256',
        'runtimeCompatibleHostRealProofGateSha256',
        'releaseCloseRealProofReadinessGateSha256',
        'ownerFreezeReviewDecision'
      )
      items = @(
        '冻结 release-evidence-bundle.json path、exists、sha256 和 gate 状态',
        '冻结 release-evidence-classification-audit.json path、exists、sha256 和 finding count',
        '冻结 public-release-owner-execution-package.json path、exists、sha256 和 blocked 状态',
        '冻结 5 个真实 proof readiness gate 的 json/md/validation 路径和 hash',
        '记录缺失 artifact 数，但缺失或存在均不能自动通过 release close',
        '保持 canPublishPublicly=false 与 canCloseReleaseIssue=false',
        '禁止把本地 hash 一致性解释为 post-publish proof',
        '等待真实 Owner 输入与 strict close validator 后续回填'
      )
    }
    'owner-real-input-import-preflight' = [ordered]@{
      artifactId = 'owner-real-input-import-preflight'
      title = 'Owner real input import preflight'
      documentTitle = 'Owner 真实输入导入预检'
      statePropertyName = 'preflightState'
      state = 'blocked-owner-real-input-import-preflight-owner-input-required'
      countPropertyName = 'checkCount'
      itemsPropertyName = 'checks'
      itemKind = 'check'
      summary = '定义 Owner 真实输入 JSON 导入前的路径、hash、字段合同版本、公开 URL、日志 hash 和 host metadata 预检。'
      requiredOwnerFields = @(
        'ownerInputJsonPath',
        'ownerInputSha256',
        'contractVersion',
        'publicUrlFieldCount',
        'sha256FieldCount',
        'hostMetadataFieldCount',
        'forbiddenSubstituteCounts',
        'ownerImportDecision'
      )
      items = @(
        'Owner 输入 JSON 路径必须由 Owner 明确提供',
        'Owner 输入 JSON SHA256 必须被记录',
        '字段合同版本必须与当前 input contract 对齐',
        '公开 URL 字段必须完整但不能自动下载验证',
        'log hash 字段必须完整且保持 64 位十六进制格式',
        'host metadata 字段必须包含 CUDA、TensorRT、driver 与 runtime package key',
        'local feed、ProjectReference、direct nupkg 等替代项计数必须为 0',
        '没有真实 Owner 输入时保持 blocked'
      )
    }
    'public-package-hash-cross-check-gate' = [ordered]@{
      artifactId = 'public-package-hash-cross-check-gate'
      title = 'Public package hash cross-check gate'
      documentTitle = '公开包 Hash 交叉核对 Gate'
      statePropertyName = 'gateState'
      state = 'blocked-public-package-hash-cross-check-gate-owner-input-required'
      countPropertyName = 'checkCount'
      itemsPropertyName = 'checks'
      itemKind = 'check'
      summary = '聚焦公开包 hash 与本地 freeze/release package hash 的交叉核对；只接收 Owner 回填 hash，不下载包、不访问外网。'
      requiredOwnerFields = @(
        'nugetPackageSource',
        'githubRelease.releaseUrl',
        'githubRelease.tagName',
        'githubRelease.managedAssetPath',
        'githubRelease.managedAssetSha256',
        'githubRelease.runtimeAssetPath',
        'githubRelease.runtimeAssetSha256',
        'managedPackage.packageUrl',
        'managedPackage.publicDownloadUrl',
        'managedPackage.publicDownloadSha256',
        'runtimePackage.packageUrl',
        'runtimePackage.publicDownloadUrl',
        'runtimePackage.publicDownloadSha256',
        'publicPackageUrl',
        'publishedNupkgSha256',
        'publishedSymbolsSha256',
        'localFreezeNupkgSha256',
        'localFreezeSymbolsSha256',
        'hashComparisonResult',
        'ownerHashReviewDecision',
        'ownerReview.reviewer',
        'ownerReview.reviewedAtUtc'
      )
      items = @(
        'NuGet package source 必须来自 Owner 真实公开包输入',
        'GitHub Release URL、tag、managed/runtime asset path 必须可追溯',
        'GitHub Release managed/runtime asset SHA256 必须由 Owner 回填',
        'managed/runtime package URL 与 public download URL 必须来自真实公开渠道',
        'managed/runtime public download SHA256 必须与 Owner 回填一致',
        'published nupkg sha256 必须由 Owner 回填',
        'published symbols sha256 必须由 Owner 回填或明确不发布 symbols',
        '本地 freeze hash 必须可追溯',
        '公开 hash 与本地 hash mismatch 时保持 blocked',
        '缺少公开下载 hash 时保持 blocked',
        'ownerReview reviewer/reviewedAtUtc 必须记录',
        '不下载公开包，不访问外部源',
        'hash match 不能替代 post-publish proof 或 runtime proof'
      )
    }
    'clean-consumer-runtime-proof-cross-check-gate' = [ordered]@{
      artifactId = 'clean-consumer-runtime-proof-cross-check-gate'
      title = 'Clean consumer runtime proof cross-check gate'
      documentTitle = 'Clean Consumer 与 Runtime Proof 交叉核对 Gate'
      statePropertyName = 'gateState'
      state = 'blocked-clean-consumer-runtime-proof-cross-check-gate-owner-input-required'
      countPropertyName = 'checkCount'
      itemsPropertyName = 'checks'
      itemKind = 'check'
      summary = '交叉核对仓库外 clean consumer proof 与兼容主机 runtime proof 的 package id/version/source/runtime package key 一致性。'
      requiredOwnerFields = @(
        'publicPackageOwnerInput.nugetPackageSource',
        'publicPackageOwnerInput.managedPackage.version',
        'publicPackageOwnerInput.runtimePackage.version',
        'publicPackageOwnerInput.runtimePackageKey',
        'publicPackageOwnerInput.cleanExternalConsumer.root',
        'publicPackageOwnerInput.cleanExternalConsumer.projectPath',
        'publicPackageOwnerInput.cleanExternalConsumer.restoreLogSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.buildLogSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.smokeLogSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.stdoutLogSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.stderrLogSha256',
        'publicPackageOwnerInput.hostMetadata',
        'publicPackageOwnerInput.ownerReview',
        'cleanConsumerPackageId',
        'cleanConsumerPackageVersion',
        'cleanConsumerPackageSource',
        'runtimeHostPackageId',
        'runtimeHostPackageVersion',
        'runtimePackageKey',
        'projectReferenceCount',
        'directNupkgReferenceCount',
        'localFeedReferenceCount',
        'dependencyProbeOnly',
        'driverBlocked',
        'buildOnly'
      )
      items = @(
        'public package owner input 与 clean consumer owner input 的 package source 必须一致',
        'public package owner input 与 post-publish owner input 的 package version/runtime key 必须一致',
        'clean external consumer root 必须在仓库外',
        'restore/build/smoke/stdout/stderr log SHA256 必须存在且格式有效',
        'host metadata 必须覆盖 OS、GPU、CUDA、cuDNN、TensorRT 与 runtime package key',
        'ownerReview 必须记录 reviewer 与 reviewedAtUtc',
        'clean consumer package id/version 必须与 runtime host 记录一致',
        'clean consumer package source 必须是公开源',
        'runtime package key 必须匹配实际 native asset 路径',
        'ProjectReference count 必须为 0',
        'direct nupkg reference count 必须为 0',
        'local feed reference count 必须为 0',
        'DependencyProbe-only、driver-blocked 或 build-only 不得晋级',
        '缺少任一真实执行日志 hash 时保持 blocked'
      )
    }
    'post-publish-rollback-owner-decision-gate' = [ordered]@{
      artifactId = 'post-publish-rollback-owner-decision-gate'
      title = 'Post-publish rollback owner decision gate'
      documentTitle = '发布后 Rollback 与 Owner 决策 Gate'
      statePropertyName = 'gateState'
      state = 'blocked-post-publish-rollback-owner-decision-gate-owner-input-required'
      countPropertyName = 'checkCount'
      itemsPropertyName = 'checks'
      itemKind = 'check'
      summary = '聚焦 post-publish verification、known limitations、rollback plan review 与 Owner close decision；没有真实决策时保持 blocked。'
      requiredOwnerFields = @(
        'postPublishVerificationRecord',
        'knownLimitationsUrl',
        'rollbackPlanReviewed',
        'rollbackPlanSha256',
        'ownerPostPublishDecision',
        'ownerCloseDecision',
        'strictCloseValidatorCommand'
      )
      items = @(
        'post-publish verification record 必须来自真实发布后输入',
        'known limitations URL 必须可追溯',
        'rollback plan 必须由 Owner 审阅',
        'rollback plan hash 必须记录',
        'Owner post-publish decision 必须明确',
        'Owner close decision 必须明确',
        'strict close validator 命令必须记录但不自动关闭 issue',
        '没有 rollback review 或 Owner close decision 时保持 blocked'
      )
    }
    'release-close-final-real-input-admission-pack' = [ordered]@{
      artifactId = 'release-close-final-real-input-admission-pack'
      title = 'Release close final real input admission pack'
      documentTitle = 'ReleaseClose 最终真实输入准入包'
      statePropertyName = 'admissionState'
      state = 'blocked-release-close-final-real-input-admission-pack-owner-input-required'
      countPropertyName = 'blockerCount'
      itemsPropertyName = 'blockers'
      itemKind = 'blocker'
      summary = '汇总最终冻结、Owner 真实输入导入预检、公开包 hash、clean consumer/runtime proof 交叉核对和 rollback/Owner 决策 gate。'
      requiredOwnerFields = @(
        'releaseCandidateRealProofFinalFreeze',
        'ownerRealInputImportPreflight',
        'publicPackageHashCrossCheckGate',
        'cleanConsumerRuntimeProofCrossCheckGate',
        'postPublishRollbackOwnerDecisionGate',
        'releaseEvidenceClassificationAudit',
        'releaseIssueCloseStrictValidation'
      )
      items = @(
        'release-candidate-real-proof-final-freeze 仍需真实 Owner freeze review',
        'owner-real-input-import-preflight 仍需真实 Owner 输入 JSON',
        'public-package-hash-cross-check-gate 仍需公开下载 hash',
        'clean-consumer-runtime-proof-cross-check-gate 仍需真实 clean consumer 与 runtime proof',
        'post-publish-rollback-owner-decision-gate 仍需 rollback review 与 Owner close decision',
        'classification audit 必须保持 no promoted substitute proof',
        'release issue close strict validator 真实通过前保持 blocked',
        'canPublishPublicly=false 且 canCloseReleaseIssue=false 必须保持'
      )
    }
  }

  return $specs[$ArtifactId]
}

function Get-AdmissionFreezeSourceArtifacts {
  [CmdletBinding()]
  param()

  return @(
    'artifacts/final-release/release-evidence-bundle.json',
    'artifacts/final-release/release-evidence-classification-audit.json',
    'artifacts/final-release/public-release-owner-execution-package.json',
    'artifacts/final-release/owner-external-real-proof-input-contract.json',
    'artifacts/final-release/owner-external-real-proof-input-contract-validation.json',
    'artifacts/final-release/owner-external-real-proof-import-validator.json',
    'artifacts/final-release/owner-external-real-proof-import-validator-validation.json',
    'artifacts/final-release/post-publish-clean-consumer-real-proof-gate.json',
    'artifacts/final-release/post-publish-clean-consumer-real-proof-gate-validation.json',
    'artifacts/final-release/runtime-compatible-host-real-proof-gate.json',
    'artifacts/final-release/runtime-compatible-host-real-proof-gate-validation.json',
    'artifacts/final-release/release-close-real-proof-readiness-gate.json',
    'artifacts/final-release/release-close-real-proof-readiness-gate-validation.json'
  )
}

function New-AdmissionFreezeItem {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)]
    [string]$RepoRoot,

    [Parameter(Mandatory)]
    [string]$RelativePath
  )

  $path = Join-Path $RepoRoot $RelativePath
  $exists = Test-Path -LiteralPath $path
  $sha256 = $null
  if ($exists) {
    $sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  }

  return [pscustomobject]@{
    path = $RelativePath
    exists = $exists
    sha256 = $sha256
    boundary = $script:ReleaseCandidateFinalRealInputAdmissionBoundary
  }
}

function New-ReleaseCandidateFinalRealInputAdmissionArtifact {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)]
    [string]$ArtifactId,

    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
  )

  $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
  $outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $repoRoot $OutputDirectory }
  New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

  $spec = Get-ReleaseCandidateFinalRealInputAdmissionSpec -ArtifactId $ArtifactId
  $boundary = $script:ReleaseCandidateFinalRealInputAdmissionBoundary
  $items = @()
  for ($i = 0; $i -lt $spec.items.Count; $i++) {
    $items += [pscustomobject]@{
      id = ('{0}-{1:00}' -f $spec.itemKind, ($i + 1))
      title = [string]$spec.items[$i]
      status = 'blocked-owner-input-required'
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
    forbiddenSubstitutes = @($script:ReleaseCandidateFinalRealInputAdmissionForbiddenSubstitutes)
  }

  $result[$spec.statePropertyName] = $spec.state
  $result[$spec.countPropertyName] = $items.Count
  $result[$spec.itemsPropertyName] = $items

  if ($ArtifactId -eq 'release-candidate-real-proof-final-freeze') {
    $freezeItems = @(Get-AdmissionFreezeSourceArtifacts | ForEach-Object { New-AdmissionFreezeItem -RepoRoot $repoRoot -RelativePath $_ })
    $result.freezeItems = $freezeItems
    $result.freezeItemCount = $freezeItems.Count
    $result.missingArtifactCount = @($freezeItems | Where-Object { -not $_.exists }).Count
  }

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
  if ($ArtifactId -eq 'release-candidate-real-proof-final-freeze') {
    $lines += @('', '## Freeze items', '')
    $lines += @($result.freezeItems | ForEach-Object { "- ``$($_.path)`` exists=$($_.exists) sha256=$($_.sha256)" })
  }
  $lines += @('', '## Forbidden substitutes', '')
  $lines += @($result.forbiddenSubstitutes | ForEach-Object { "- $_" })
  $lines | Set-Content -LiteralPath $mdPath -Encoding UTF8

  Write-Host "Wrote $jsonPath"
  Write-Host "Wrote $mdPath"
}

function Test-ReleaseCandidateFinalRealInputAdmissionArtifact {
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
    New-ReleaseCandidateFinalRealInputAdmissionArtifact -ArtifactId $ArtifactId -OutputDirectory $artifactRoot
  }

  $spec = Get-ReleaseCandidateFinalRealInputAdmissionSpec -ArtifactId $ArtifactId
  $artifact = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
  $findings = New-Object System.Collections.Generic.List[object]
  function Add-Finding([string]$Id, [string]$Message) {
    $findings.Add([pscustomobject]@{ id = $Id; message = $Message }) | Out-Null
  }

  $stateProperty = $artifact.PSObject.Properties[[string]$spec.statePropertyName]
  $countProperty = $artifact.PSObject.Properties[[string]$spec.countPropertyName]

  if ([string]$artifact.artifactId -ne $ArtifactId) { Add-Finding 'artifact-id' 'Unexpected artifact id.' }
  if ($null -eq $stateProperty -or [string]$stateProperty.Value -ne [string]$spec.state) { Add-Finding 'state' 'Unexpected blocked state.' }
  if ([bool]$artifact.passed) { Add-Finding 'passed' 'Admission artifact must not pass without real owner input and strict validation.' }
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
  if ($null -eq $countProperty -or [int]$countProperty.Value -lt @($spec.items).Count) { Add-Finding 'item-count' 'Expected admission surface is incomplete.' }
  if ($null -eq $countProperty -or [int]$artifact.blockedCount -ne [int]$countProperty.Value) { Add-Finding 'blocked-count' 'All generated items must remain blocked.' }
  if (@($artifact.requiredOwnerFields).Count -lt @($spec.requiredOwnerFields).Count) { Add-Finding 'required-owner-fields' 'Required owner input fields are incomplete.' }
  foreach ($substitute in $script:ReleaseCandidateFinalRealInputAdmissionForbiddenSubstitutes) {
    if (@($artifact.forbiddenSubstitutes) -notcontains $substitute) {
      Add-Finding "forbidden-$substitute" "Missing forbidden substitute: $substitute"
    }
  }
  if ($ArtifactId -eq 'release-candidate-real-proof-final-freeze') {
    if (@($artifact.freezeItems).Count -lt 8) { Add-Finding 'freeze-items' 'Final freeze must preserve key source artifact paths.' }
  }

  $validationState = if ($findings.Count -eq 0) { 'validation-passed-non-proof-final-real-input-admission-boundary-intact' } else { 'validation-failed' }
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
    boundary = $script:ReleaseCandidateFinalRealInputAdmissionBoundary
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
    "- Boundary: $($script:ReleaseCandidateFinalRealInputAdmissionBoundary)"
  ) | Set-Content -LiteralPath $validationMdPath -Encoding UTF8

  if ($Strict -and $findings.Count -gt 0) {
    $findings | Format-Table -AutoSize | Out-String | Write-Error
  }

  Write-Host "${ArtifactId}: $validationState; FindingCount=$($findings.Count)"
}
