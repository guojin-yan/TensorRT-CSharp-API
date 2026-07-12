[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:StrictCloseRealInputBoundary = 'not runtime proof; not post-publish proof; not publish approval; not release close approval; not package push'
$script:StrictCloseRealInputForbiddenSubstitutes = @(
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
  'real proof readiness gate',
  'pre-publish package',
  'build-only',
  'dependency probe',
  'blocked-by-driver',
  'public package owner input template'
)

function Get-StrictCloseRealInputFinalBlockerLanes {
  [CmdletBinding()]
  param()

  $commonRequiredEvidenceFields = @(
    'stdoutPath',
    'stderrPath',
    'transcriptPath',
    'logPath',
    'logSha256',
    'exitCode',
    'hostIdentity',
    'hostOs',
    'hostArchitecture',
    'runnerIdentity',
    'startedAtUtc',
    'completedAtUtc',
    'ownerReviewer',
    'ownerReviewedAtUtc',
    'nonSubstituteConfirmations'
  )

  return @(
    [ordered]@{
      blockerId = 'owner-authorization'
      title = 'Owner authorization'
      ownerInputFile = 'artifacts/final-release/owner-authorization-owner-input.json'
      validator = 'eng\Test-OwnerAuthorizationOwnerInput.ps1 -Strict'
      releaseCloseTarget = 'release-issue-close-strict-owner-decision-import'
      requiredEvidenceFields = @(
        'ownerIdentity',
        'ownerAuthorizationStatement',
        'authorizedPackageId',
        'authorizedPackageVersion',
        'authorizedPublicPackageUrl',
        'authorizedNupkgSha256',
        'authorizationTimestampUtc',
        'ownerFinalCloseDecision'
      ) + $commonRequiredEvidenceFields
      forbiddenSubstitutes = @('manual handoff', 'runbook', 'dashboard', 'candidate', 'draft', 'template')
      acceptanceRule = 'Owner authorization can unblock only after a real signed owner decision names the public package identity, public URL, SHA256, owner identity, timestamp, final close decision, and validator transcript.'
    }
    [ordered]@{
      blockerId = 'package-consumer-runtime'
      title = 'Package consumer runtime'
      ownerInputFile = 'artifacts/final-release/package-consumer-runtime-proof-owner-input.json'
      validator = 'eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict'
      releaseCloseTarget = 'final-publish-proof-gate-report'
      requiredEvidenceFields = @(
        'cleanConsumerProjectPath',
        'cleanConsumerPackageId',
        'cleanConsumerPackageVersion',
        'cleanConsumerPackageSource',
        'publicPackageUrl',
        'publishedNupkgSha256',
        'restoreCommand',
        'buildCommand',
        'smokeCommand',
        'runtimePackageKey',
        'cudaVersion',
        'tensorrtVersion',
        'driverVersion'
      ) + $commonRequiredEvidenceFields
      forbiddenSubstitutes = @('local feed', 'ProjectReference', 'direct nupkg', 'local .nupkg', 'build-only', 'dependency probe', 'blocked-by-driver')
      acceptanceRule = 'Package consumer runtime proof must come from a repository-external clean consumer using the public package source and cannot use local feed, ProjectReference, direct nupkg, or dependency-probe-only evidence.'
    }
    [ordered]@{
      blockerId = 'linux-runner-proof'
      title = 'Linux runner proof'
      ownerInputFile = 'artifacts/final-release/linux-runner-proof-owner-input.json'
      validator = 'eng\Test-LinuxRunnerProofOwnerInput.ps1 -Strict'
      releaseCloseTarget = 'release-close-real-proof-import-bridge'
      requiredEvidenceFields = @(
        'linuxDistribution',
        'containerImage',
        'glibcVersion',
        'cudaRuntimePackage',
        'tensorrtRuntimePackage',
        'nativeLibraryProbeLogPath',
        'runtimePackageKey',
        'cudaVersion',
        'tensorrtVersion',
        'driverVersion'
      ) + $commonRequiredEvidenceFields
      forbiddenSubstitutes = @('template', 'runbook', 'dashboard', 'local-only scan', 'candidate', 'dry-run')
      acceptanceRule = 'Linux runner proof requires real Linux host or container execution with runtime package metadata, native library load/probe output, exitCode=0, logs, hashes, and host identity.'
    }
    [ordered]@{
      blockerId = 'real-model-runtime'
      title = 'Real model runtime'
      ownerInputFile = 'artifacts/final-release/real-model-runtime-owner-input.json'
      validator = 'eng\Test-RealModelRuntimeOwnerInput.ps1 -Strict'
      releaseCloseTarget = 'real-proof-record-candidate-from-owner-result-import'
      requiredEvidenceFields = @(
        'modelAssetPath',
        'modelAssetSha256',
        'modelLicense',
        'inputAssetPath',
        'inputAssetSha256',
        'outputArtifactPath',
        'outputArtifactSha256',
        'sampleRunner',
        'runtimePackageKey',
        'cudaVersion',
        'tensorrtVersion',
        'driverVersion'
      ) + $commonRequiredEvidenceFields
      forbiddenSubstitutes = @('sample scaffold', 'asset candidate', 'template', 'draft', 'dashboard', 'build-only')
      acceptanceRule = 'Real model runtime proof requires an owner-approved model asset, license, input/output artifacts, sample runner log, hashes, exitCode=0, and strict validator pass.'
    }
    [ordered]@{
      blockerId = 'post-publish-verification'
      title = 'Post-publish verification'
      ownerInputFile = 'artifacts/final-release/post-publish-verification-owner-input.json'
      validator = 'eng\Test-PostPublishVerification.ps1 -Strict'
      releaseCloseTarget = 'release-issue-close-record'
      requiredEvidenceFields = @(
        'publicPackageUrl',
        'publicPackageVersion',
        'publishedNupkgSha256',
        'publishedSymbolsSha256',
        'packageIndexObservedAtUtc',
        'cleanInstallCommand',
        'cleanRestoreCommand',
        'cleanSmokeCommand',
        'postPublishVerificationRecord',
        'rollbackPlanSha256',
        'rollbackPlanReviewed'
      ) + $commonRequiredEvidenceFields
      forbiddenSubstitutes = @('pre-publish package', 'local feed', 'direct nupkg', 'dry-run', 'candidate', 'dashboard', 'owner execution package')
      acceptanceRule = 'Post-publish verification can pass only after the public package is observable from the public channel and a clean external consumer install/run log proves the exact public package hash.'
    }
  )
}

function Get-StrictCloseRealInputSpec {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)]
    [ValidateSet(
      'owner-real-input-json-contract',
      'owner-real-input-json-import',
      'owner-real-input-hash-and-path-validator',
      'owner-real-input-forbidden-substitute-validator',
      'strict-close-real-input-dry-run',
      'strict-close-real-input-finding-report',
      'strict-close-owner-action-pack',
      'release-close-real-input-final-blocker-ledger'
    )]
    [string]$ArtifactId
  )

  $finalBlockerLanes = Get-StrictCloseRealInputFinalBlockerLanes
  $finalBlockerRequiredFields = @($finalBlockerLanes | ForEach-Object { $_.requiredEvidenceFields } | Select-Object -Unique)
  $finalBlockerOwnerInputFiles = @($finalBlockerLanes | ForEach-Object { $_.ownerInputFile })
  $finalBlockerReleaseCloseTargets = @($finalBlockerLanes | ForEach-Object { $_.releaseCloseTarget } | Select-Object -Unique)
  $commonOwnerFields = @(
    'nugetPackageSource',
    'githubRelease.releaseUrl',
    'githubRelease.tagName',
    'githubRelease.managedAssetPath',
    'githubRelease.managedAssetSha256',
    'githubRelease.runtimeAssetPath',
    'githubRelease.runtimeAssetSha256',
    'managedPackage.packageId',
    'managedPackage.version',
    'managedPackage.packageUrl',
    'managedPackage.publicDownloadUrl',
    'managedPackage.publicDownloadSha256',
    'runtimePackage.packageId',
    'runtimePackage.version',
    'runtimePackage.packageUrl',
    'runtimePackage.publicDownloadUrl',
    'runtimePackage.publicDownloadSha256',
    'publicPackageOwnerInput.nugetPackageSource',
    'publicPackageOwnerInput.managedPackage.version',
    'publicPackageOwnerInput.runtimePackage.version',
    'publicPackageOwnerInput.runtimePackageKey',
    'publicPackageOwnerInput.cleanExternalConsumer.root',
    'publicPackageOwnerInput.cleanExternalConsumer.projectPath',
    'publicPackageOwnerInput.cleanExternalConsumer.restoreLogPath',
    'publicPackageOwnerInput.cleanExternalConsumer.restoreLogSha256',
    'publicPackageOwnerInput.cleanExternalConsumer.buildLogPath',
    'publicPackageOwnerInput.cleanExternalConsumer.buildLogSha256',
    'publicPackageOwnerInput.cleanExternalConsumer.smokeLogPath',
    'publicPackageOwnerInput.cleanExternalConsumer.smokeLogSha256',
    'publicPackageOwnerInput.cleanExternalConsumer.stdoutLogPath',
    'publicPackageOwnerInput.cleanExternalConsumer.stdoutLogSha256',
    'publicPackageOwnerInput.cleanExternalConsumer.stderrLogPath',
    'publicPackageOwnerInput.cleanExternalConsumer.stderrLogSha256',
    'publicPackageOwnerInput.hostMetadata',
    'publicPackageOwnerInput.hostMetadata.osDescription',
    'publicPackageOwnerInput.hostMetadata.architecture',
    'publicPackageOwnerInput.hostMetadata.gpuName',
    'publicPackageOwnerInput.hostMetadata.cudaDriverVersion',
    'publicPackageOwnerInput.hostMetadata.cudaRuntimeVersion',
    'publicPackageOwnerInput.hostMetadata.cudnnVersion',
    'publicPackageOwnerInput.hostMetadata.tensorRtVersion',
    'publicPackageOwnerInput.hostMetadata.tensorRtLine',
    'publicPackageOwnerInput.ownerReview',
    'publicPackageOwnerInput.ownerReview.reviewer',
    'publicPackageOwnerInput.ownerReview.reviewedAtUtc',
    'publicPackageUrl',
    'publishedNupkgSha256',
    'publishedSymbolsSha256',
    'cleanConsumerPackageId',
    'cleanConsumerPackageVersion',
    'cleanConsumerPackageSource',
    'cleanConsumerCommand',
    'cleanConsumerLogSha256',
    'runtimeHostPackageId',
    'runtimeHostPackageVersion',
    'runtimePackageKey',
    'cudaVersion',
    'tensorrtVersion',
    'driverVersion',
    'runtimePackageMetadata',
    'postPublishVerificationRecord',
    'postPublishVerificationLogSha256',
    'rollbackReview.reviewedBy',
    'rollbackReview.reviewedAtUtc',
    'rollbackReview.rollbackPlanSha256',
    'rollbackReview.decision',
    'rollbackPlanSha256',
    'rollbackPlanReviewed',
    'finalCloseDecision.decision',
    'finalCloseDecision.decidedAtUtc',
    'finalCloseDecision.ownerReviewer',
    'finalCloseDecision.releaseIssueUrl',
    'ownerFinalCloseDecision'
  )

  $specs = @{
    'owner-real-input-json-contract' = [ordered]@{
      artifactId = 'owner-real-input-json-contract'
      title = 'Owner real input JSON contract'
      documentTitle = 'Owner 真实输入 JSON 合同'
      statePropertyName = 'contractState'
      state = 'blocked-owner-real-input-json-contract-owner-input-required'
      countPropertyName = 'fieldCount'
      itemsPropertyName = 'fields'
      itemKind = 'field'
      summary = '定义真实 Owner 输入 JSON 的字段合同，覆盖 NuGet/GitHub Release 公开包、managed/runtime public download hash、仓库外 clean consumer 执行日志、runtime host metadata、post-publish verification、rollback review 和 final close decision。'
      requiredOwnerFields = @($commonOwnerFields + $finalBlockerRequiredFields | Select-Object -Unique)
      items = @($commonOwnerFields + $finalBlockerRequiredFields | Select-Object -Unique)
    }
    'owner-real-input-json-import' = [ordered]@{
      artifactId = 'owner-real-input-json-import'
      title = 'Owner real input JSON import'
      documentTitle = 'Owner 真实输入 JSON 导入'
      statePropertyName = 'importState'
      state = 'blocked-owner-real-input-json-import-owner-input-required'
      countPropertyName = 'importCheckCount'
      itemsPropertyName = 'importChecks'
      itemKind = 'import-check'
      summary = '导入可选 Owner 输入 JSON；未提供真实输入时生成 blocked surface，提供路径时只读取本地 JSON，不访问网络，也不把导入成功解释为 proof。'
      requiredOwnerFields = @(
        'ownerInputPath',
        'ownerInputSha256',
        'contractVersion',
        'ownerImportDecision',
        'finalBlockerLanes',
        'releaseCloseTargetMapping',
        'nugetPackageSource',
        'githubRelease.releaseUrl',
        'githubRelease.tagName',
        'managedPackage.publicDownloadSha256',
        'runtimePackage.publicDownloadSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.root',
        'publicPackageOwnerInput.cleanExternalConsumer.restoreLogSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.buildLogSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.smokeLogSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.stdoutLogSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.stderrLogSha256',
        'publicPackageOwnerInput.hostMetadata',
        'publicPackageOwnerInput.ownerReview',
        'rollbackReview.reviewedAtUtc',
        'finalCloseDecision.decidedAtUtc'
      ) + $finalBlockerRequiredFields
      items = @(
        'Owner 输入路径必须由 Owner 明确提供',
        'Owner 输入文件存在性必须记录',
        'Owner 输入 JSON 必须可解析',
        'Owner 输入 SHA256 必须记录',
        '字段合同版本必须与当前 contract 对齐',
        'NuGet package source 与 GitHub Release URL/tag 必须来自真实 Owner 输入',
        'managed/runtime public download SHA256 必须在导入面保留',
        'clean external consumer root/project/log SHA256 必须在导入面保留',
        'host metadata 与 ownerReview 必须在导入面保留',
        'rollbackReview 与 finalCloseDecision 必须在导入面保留但不能自动通过',
        '导入成功不能替代 runtime proof',
        '导入成功不能替代 post-publish proof',
        '没有真实 Owner 输入时保持 blocked',
        '五个最终 blocker 必须全部有独立真实输入槽',
        '每个 blocker 必须映射到 ReleaseClose 下游目标',
        '每个 blocker 必须保留 stdout/stderr/transcript/log/hash/exitCode/host identity'
      )
    }
    'owner-real-input-hash-and-path-validator' = [ordered]@{
      artifactId = 'owner-real-input-hash-and-path-validator'
      title = 'Owner real input hash and path validator'
      documentTitle = 'Owner 真实输入 Hash 与路径校验器'
      statePropertyName = 'validatorState'
      state = 'blocked-owner-real-input-hash-and-path-validator-owner-input-required'
      countPropertyName = 'validationCheckCount'
      itemsPropertyName = 'validationChecks'
      itemKind = 'validation-check'
      summary = '校验 Owner 输入中的本地路径、公开 URL、GitHub Release asset、public download SHA256、clean consumer 日志 SHA256 和 host metadata；只做字段级检查，不下载公开 URL。'
      requiredOwnerFields = @(
        'ownerInputJsonPath',
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
        'publicPackageOwnerInput.cleanExternalConsumer.root',
        'publicPackageOwnerInput.cleanExternalConsumer.restoreLogPath',
        'publicPackageOwnerInput.cleanExternalConsumer.restoreLogSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.buildLogPath',
        'publicPackageOwnerInput.cleanExternalConsumer.buildLogSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.smokeLogPath',
        'publicPackageOwnerInput.cleanExternalConsumer.smokeLogSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.stdoutLogPath',
        'publicPackageOwnerInput.cleanExternalConsumer.stdoutLogSha256',
        'publicPackageOwnerInput.cleanExternalConsumer.stderrLogPath',
        'publicPackageOwnerInput.cleanExternalConsumer.stderrLogSha256',
        'publicPackageOwnerInput.hostMetadata',
        'postPublishVerificationLogSha256',
        'rollbackPlanSha256'
      ) + $finalBlockerRequiredFields
      items = @(
        '本地 owner input path 必须存在或明确缺失',
        'NuGet package source 和公开 package URL 字段必须非空',
        'GitHub Release URL/tag 与 managed/runtime asset path 必须可追溯',
        'GitHub Release managed/runtime asset SHA256 必须是 64 位十六进制',
        'managed/runtime public download URL 必须非空',
        'managed/runtime public download SHA256 必须是 64 位十六进制',
        'nupkg hash 必须是 64 位十六进制',
        'symbols hash 必须是 64 位十六进制或明确未发布 symbols',
        'clean external consumer root 必须明确且指向仓库外',
        'restore/build/smoke/stdout/stderr log path 必须可追溯',
        'restore/build/smoke/stdout/stderr log SHA256 必须是 64 位十六进制或明确 no-stderr-emitted',
        'host metadata 必须覆盖 OS、architecture、GPU、CUDA driver/runtime、cuDNN、TensorRT 与 TensorRT line',
        'post-publish verification log hash 必须是 64 位十六进制',
        'rollback plan hash 必须是 64 位十六进制',
        '不得下载公开 URL 或执行发布',
        '每个 blocker 的 stdout/stderr/transcript/log path 必须可追溯',
        '每个 blocker 的 SHA256 字段必须是 64 位十六进制',
        '每个 blocker 的 host identity 必须记录'
      )
    }
    'owner-real-input-forbidden-substitute-validator' = [ordered]@{
      artifactId = 'owner-real-input-forbidden-substitute-validator'
      title = 'Owner real input forbidden substitute validator'
      documentTitle = 'Owner 真实输入禁止替代项校验器'
      statePropertyName = 'validatorState'
      state = 'blocked-owner-real-input-forbidden-substitute-validator-owner-input-required'
      countPropertyName = 'substituteCheckCount'
      itemsPropertyName = 'substituteChecks'
      itemKind = 'substitute-check'
      summary = '检查 Owner 输入是否包含 local feed、ProjectReference、direct nupkg、pre-publish package、template、draft、dry-run、build-only、dependency probe、blocked-by-driver、local-only scan 等禁止替代项。'
      requiredOwnerFields = @(
        'localFeedReferenceCount',
        'projectReferenceCount',
        'directNupkgReferenceCount',
        'prePublishPackageReferenceCount',
        'templateOnlyCount',
        'dryRunOnlyCount',
        'buildOnlyCount',
        'dependencyProbeOnlyCount',
        'blockedByDriverOnlyCount',
        'localOnlyScanCount',
        'publicPackageOwnerInput.cleanExternalConsumer.forbiddenSubstituteCounts',
        'perBlockerForbiddenSubstituteFindings'
      )
      items = @(
        'local feed reference count 必须为 0',
        'ProjectReference count 必须为 0',
        'direct nupkg reference count 必须为 0',
        'pre-publish package reference count 必须为 0',
        'template-only record count 必须为 0',
        'draft/dry-run-only record count 必须为 0',
        'build-only record count 必须为 0',
        'dependency-probe-only record count 必须为 0',
        'blocked-by-driver-only record count 必须为 0',
        'local-only scan count 必须为 0',
        'publicPackageOwnerInput.cleanExternalConsumer 必须拒绝 local feed/ProjectReference/direct nupkg',
        'real proof readiness gate 不能替代真实 proof',
        '发现禁止替代项时保持 blocked',
        'package-consumer-runtime 必须拒绝 local feed/ProjectReference/direct nupkg',
        'post-publish-verification 必须拒绝 pre-publish package/local feed/dry-run',
        'release close 不得把 candidate/dashboard/runbook 晋级为 proof'
      )
    }
    'strict-close-real-input-dry-run' = [ordered]@{
      artifactId = 'strict-close-real-input-dry-run'
      title = 'Strict close real input dry-run'
      documentTitle = 'StrictClose 真实输入 Dry Run'
      statePropertyName = 'dryRunState'
      state = 'blocked-strict-close-real-input-dry-run-owner-input-required'
      countPropertyName = 'dryRunCheckCount'
      itemsPropertyName = 'dryRunChecks'
      itemKind = 'dry-run-check'
      summary = '汇总 Owner 输入合同、导入、hash/path validator、forbidden substitute validator 和最终准入包，输出 strict close 是否仍被阻塞。'
      requiredOwnerFields = @('ownerRealInputJsonContract', 'ownerRealInputJsonImport', 'ownerRealInputHashAndPathValidator', 'ownerRealInputForbiddenSubstituteValidator', 'releaseCloseFinalRealInputAdmissionPack') + $finalBlockerOwnerInputFiles + $finalBlockerReleaseCloseTargets
      items = @(
        'Owner 输入 JSON 合同必须存在',
        'Owner 输入 JSON 导入必须可追溯',
        'hash/path validator findings 必须汇总',
        'forbidden substitute validator findings 必须汇总',
        'final real input admission pack 必须仍可追溯',
        'strict close 不能自动通过',
        'canCloseReleaseIssue 必须保持 false',
        'dry-run 不能替代真实 close validator',
        '五个 blocker 未全部真实回填时 failedActionRequiredCount 必须保持大于 0',
        'failedBlockerCount=0 不能解释为 ready',
        'ReleaseClose 映射只能作为 owner input bridge'
      )
    }
    'strict-close-real-input-finding-report' = [ordered]@{
      artifactId = 'strict-close-real-input-finding-report'
      title = 'Strict close real input finding report'
      documentTitle = 'StrictClose 真实输入 Finding 报告'
      statePropertyName = 'reportState'
      state = 'blocked-strict-close-real-input-finding-report-owner-input-required'
      countPropertyName = 'findingGroupCount'
      itemsPropertyName = 'findingGroups'
      itemKind = 'finding-group'
      summary = '按公开包 hash、clean consumer、runtime host、post-publish verification、rollback review、Owner close decision 和禁止替代项分组汇总 blocker。'
      requiredOwnerFields = @('publicPackageFindings', 'cleanConsumerFindings', 'runtimeHostFindings', 'postPublishFindings', 'rollbackFindings', 'ownerCloseDecisionFindings', 'perBlockerFindings') + $finalBlockerRequiredFields
      items = @(
        '公开包 hash blocker 分组',
        'clean consumer proof blocker 分组',
        'runtime host proof blocker 分组',
        'post-publish verification blocker 分组',
        'rollback review blocker 分组',
        'Owner final close decision blocker 分组',
        '禁止替代项 blocker 分组',
        '每个 finding 必须指向 Owner 下一步',
        '每个最终 blocker 必须独立显示 blocked/ready 原因',
        '每个 ReleaseClose target 必须显示未晋级原因',
        'strict validator 未真实通过时保持 blocked'
      )
    }
    'strict-close-owner-action-pack' = [ordered]@{
      artifactId = 'strict-close-owner-action-pack'
      title = 'Strict close owner action pack'
      documentTitle = 'StrictClose Owner 行动包'
      statePropertyName = 'actionPackState'
      state = 'blocked-strict-close-owner-action-pack-owner-input-required'
      countPropertyName = 'actionCount'
      itemsPropertyName = 'actions'
      itemKind = 'owner-action'
      summary = '给 Owner 一页式行动包，列出需要执行的外部命令、需要复制的 hash、需要保留的日志和需要填写的最终决策。'
      requiredOwnerFields = @('externalCommandLines', 'hashesToCopy', 'logsToPreserve', 'decisionsToFill', 'strictCloseValidatorCommand') + $finalBlockerOwnerInputFiles
      items = @(
        '执行公开包 hash 采集',
        '执行仓库外 clean consumer runtime smoke',
        '执行兼容主机 runtime proof',
        '记录 CUDA/TensorRT/driver/runtime package metadata',
        '执行 post-publish verification',
        '审阅 rollback plan',
        '填写 Owner final close decision',
        '运行 strict close validator 但不自动关闭 issue',
        '按 owner-authorization/package-consumer/linux/real-model/post-publish 五个槽位逐项回填',
        '每个槽位必须保留 stdout/stderr/transcript/log/SHA256/exitCode/host identity',
        '回填后运行 ReleaseClose bridge、FinalPublishProofGate、ReleaseEvidence classification audit'
      )
    }
    'release-close-real-input-final-blocker-ledger' = [ordered]@{
      artifactId = 'release-close-real-input-final-blocker-ledger'
      title = 'Release close real input final blocker ledger'
      documentTitle = 'ReleaseClose 真实输入最终 Blocker 台账'
      statePropertyName = 'ledgerState'
      state = 'blocked-release-close-real-input-final-blocker-ledger-owner-input-required'
      countPropertyName = 'blockerCount'
      itemsPropertyName = 'blockers'
      itemKind = 'blocker'
      summary = '最终 blocker ledger，列出 release close 仍被真实公开发布、post-publish verification、clean consumer runtime proof、runtime compatible host proof 和 Owner close decision 阻塞。'
      requiredOwnerFields = @('publicPublishProof', 'postPublishVerificationProof', 'cleanConsumerRuntimeProof', 'runtimeCompatibleHostProof', 'ownerCloseDecision', 'strictCloseValidatorResult') + $finalBlockerOwnerInputFiles + $finalBlockerReleaseCloseTargets
      items = @(
        '真实公开发布 proof 缺失时 blocked',
        'post-publish verification proof 缺失时 blocked',
        'clean consumer runtime proof 缺失时 blocked',
        'runtime compatible host proof 缺失时 blocked',
        'Owner final close decision 缺失时 blocked',
        'strict close validator 未真实通过时 blocked',
        'classification audit 发现 promoted substitute 时 blocked',
        'canPublishPublicly=false 且 canCloseReleaseIssue=false 必须保持',
        'owner-authorization 未真实通过时 blocked',
        'package-consumer-runtime 未使用公开包源真实通过时 blocked',
        'linux-runner-proof 未真实通过时 blocked',
        'real-model-runtime 未真实通过时 blocked',
        'post-publish-verification 未真实通过时 blocked'
      )
    }
  }

  foreach ($key in @($specs.Keys)) {
    $specs[$key].finalBlockerLanes = $finalBlockerLanes
    $specs[$key].finalBlockerLaneCount = $finalBlockerLanes.Count
    $specs[$key].finalBlockerIds = @($finalBlockerLanes | ForEach-Object { $_.blockerId })
    $specs[$key].releaseCloseTargetMapping = @($finalBlockerLanes | ForEach-Object {
      [pscustomobject]@{
        blockerId = $_.blockerId
        ownerInputFile = $_.ownerInputFile
        validator = $_.validator
        releaseCloseTarget = $_.releaseCloseTarget
        acceptanceRule = $_.acceptanceRule
        boundary = $script:StrictCloseRealInputBoundary
      }
    })
  }

  return $specs[$ArtifactId]
}

function New-StrictCloseRealInputValidationArtifact {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)]
    [string]$ArtifactId,

    [string]$OwnerInputPath,

    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
  )

  $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
  $outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $repoRoot $OutputDirectory }
  New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

  $spec = Get-StrictCloseRealInputSpec -ArtifactId $ArtifactId
  $boundary = $script:StrictCloseRealInputBoundary
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

  $ownerInput = [ordered]@{
    provided = -not [string]::IsNullOrWhiteSpace($OwnerInputPath)
    path = $OwnerInputPath
    exists = $false
    sha256 = $null
    parseState = 'not-provided'
  }

  if ($ownerInput.provided) {
    $inputPath = if ([System.IO.Path]::IsPathRooted($OwnerInputPath)) { $OwnerInputPath } else { Join-Path $repoRoot $OwnerInputPath }
    $ownerInput.path = $inputPath
    $ownerInput.exists = Test-Path -LiteralPath $inputPath
    if ($ownerInput.exists) {
      $ownerInput.sha256 = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
      try {
        Get-Content -LiteralPath $inputPath -Raw | ConvertFrom-Json | Out-Null
        $ownerInput.parseState = 'json-parse-ok-non-proof'
      } catch {
        $ownerInput.parseState = 'json-parse-failed'
      }
    } else {
      $ownerInput.parseState = 'missing-owner-input-file'
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
    finalBlockerLaneCount = [int]$spec.finalBlockerLaneCount
    finalBlockerIds = @($spec.finalBlockerIds)
    finalBlockerLanes = @($spec.finalBlockerLanes | ForEach-Object { [pscustomobject]$_ })
    releaseCloseTargetMapping = @($spec.releaseCloseTargetMapping)
    ownerInput = [pscustomobject]$ownerInput
    boundary = $boundary
    forbiddenSubstitutes = @($script:StrictCloseRealInputForbiddenSubstitutes)
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
    '## Owner input',
    '',
    "- Provided: ``$($ownerInput.provided)``",
    "- Exists: ``$($ownerInput.exists)``",
    "- Parse state: ``$($ownerInput.parseState)``",
    '',
    '## Required owner fields',
    ''
  )
  $lines += @($spec.requiredOwnerFields | ForEach-Object { "- ``$_``" })
  $lines += @('', '## Final blocker lanes', '')
  foreach ($lane in $spec.finalBlockerLanes) {
    $lines += "- ``$($lane.blockerId)`` -> ``$($lane.releaseCloseTarget)`` via ``$($lane.validator)``"
    $lines += "  - Owner input: ``$($lane.ownerInputFile)``"
    $lines += "  - Acceptance: $($lane.acceptanceRule)"
  }
  $lines += @('', "## $($spec.itemsPropertyName)", '')
  $lines += @($items | ForEach-Object { "- [$($_.status)] $($_.title) - $($_.boundary)" })
  $lines += @('', '## Forbidden substitutes', '')
  $lines += @($result.forbiddenSubstitutes | ForEach-Object { "- $_" })
  $lines | Set-Content -LiteralPath $mdPath -Encoding UTF8

  Write-Host "Wrote $jsonPath"
  Write-Host "Wrote $mdPath"
}

function Test-StrictCloseRealInputValidationArtifact {
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
    New-StrictCloseRealInputValidationArtifact -ArtifactId $ArtifactId -OutputDirectory $artifactRoot
  }

  $spec = Get-StrictCloseRealInputSpec -ArtifactId $ArtifactId
  $artifact = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
  $findings = New-Object System.Collections.Generic.List[object]
  function Add-Finding([string]$Id, [string]$Message) {
    $findings.Add([pscustomobject]@{ id = $Id; message = $Message }) | Out-Null
  }

  $stateProperty = $artifact.PSObject.Properties[[string]$spec.statePropertyName]
  $countProperty = $artifact.PSObject.Properties[[string]$spec.countPropertyName]

  if ([string]$artifact.artifactId -ne $ArtifactId) { Add-Finding 'artifact-id' 'Unexpected artifact id.' }
  if ($null -eq $stateProperty -or [string]$stateProperty.Value -ne [string]$spec.state) { Add-Finding 'state' 'Unexpected blocked state.' }
  if ([bool]$artifact.passed) { Add-Finding 'passed' 'Strict close real input artifact must not pass without real owner input and final validator.' }
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
  if ($null -eq $countProperty -or [int]$countProperty.Value -lt @($spec.items).Count) { Add-Finding 'item-count' 'Expected strict close surface is incomplete.' }
  if ($null -eq $countProperty -or [int]$artifact.blockedCount -ne [int]$countProperty.Value) { Add-Finding 'blocked-count' 'All generated items must remain blocked.' }
  if (@($artifact.requiredOwnerFields).Count -lt @($spec.requiredOwnerFields).Count) { Add-Finding 'required-owner-fields' 'Required owner input fields are incomplete.' }
  $finalBlockerLanes = @($artifact.finalBlockerLanes)
  $finalBlockerIds = @($artifact.finalBlockerIds | ForEach-Object { [string]$_ })
  foreach ($blockerId in @('owner-authorization', 'package-consumer-runtime', 'linux-runner-proof', 'real-model-runtime', 'post-publish-verification')) {
    if ($finalBlockerIds -notcontains $blockerId) {
      Add-Finding "final-blocker-$blockerId" "Missing final blocker lane: $blockerId"
    }
  }
  if ([int]$artifact.finalBlockerLaneCount -ne 5 -or $finalBlockerLanes.Count -ne 5) {
    Add-Finding 'final-blocker-lane-count' 'Expected exactly five final blocker lanes.'
  }
  foreach ($lane in $finalBlockerLanes) {
    foreach ($field in @('blockerId', 'ownerInputFile', 'validator', 'releaseCloseTarget', 'requiredEvidenceFields', 'forbiddenSubstitutes', 'acceptanceRule')) {
      if ($null -eq $lane.PSObject.Properties[$field]) {
        Add-Finding "lane-field-$field" "Final blocker lane is missing field $field."
      }
    }
    foreach ($requiredEvidenceField in @('stdoutPath', 'stderrPath', 'transcriptPath', 'logPath', 'logSha256', 'exitCode', 'hostIdentity', 'ownerReviewer', 'nonSubstituteConfirmations')) {
      if (@($lane.requiredEvidenceFields) -notcontains $requiredEvidenceField) {
        Add-Finding "lane-required-$($lane.blockerId)-$requiredEvidenceField" "Lane $($lane.blockerId) is missing required evidence field $requiredEvidenceField."
      }
    }
  }
  if (@($artifact.releaseCloseTargetMapping).Count -ne 5) {
    Add-Finding 'release-close-target-mapping' 'Expected five release close target mappings.'
  }
  foreach ($substitute in $script:StrictCloseRealInputForbiddenSubstitutes) {
    if (@($artifact.forbiddenSubstitutes) -notcontains $substitute) {
      Add-Finding "forbidden-$substitute" "Missing forbidden substitute: $substitute"
    }
  }

  $validationState = if ($findings.Count -eq 0) { 'validation-passed-non-proof-strict-close-real-input-boundary-intact' } else { 'validation-failed' }
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
    boundary = $script:StrictCloseRealInputBoundary
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
    "- Boundary: $($script:StrictCloseRealInputBoundary)"
  ) | Set-Content -LiteralPath $validationMdPath -Encoding UTF8

  if ($Strict -and $findings.Count -gt 0) {
    $findings | Format-Table -AutoSize | Out-String | Write-Error
  }

  Write-Host "${ArtifactId}: $validationState; FindingCount=$($findings.Count)"
}
