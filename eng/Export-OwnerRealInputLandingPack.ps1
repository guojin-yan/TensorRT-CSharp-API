[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function ConvertTo-FlatStringLines {
  param([AllowNull()][object]$Value)

  foreach ($item in @($Value)) {
    if ($null -eq $item) {
      ""
    }
    elseif ($item -is [string]) {
      $item
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { [string]$child }
    }
    else {
      [string]$item
    }
  }
}
function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $directory = Split-Path -Parent $LiteralPath
  if (-not [string]::IsNullOrWhiteSpace($directory)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
  }

  $lines = @(ConvertTo-FlatStringLines -Value $InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$commonForbiddenSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "direct .nupkg file reference",
  "dry-run",
  "dashboard",
  "runbook",
  "candidate",
  "draft",
  "build-only",
  "parse-only",
  "sidecar-only",
  "template",
  "schema-only",
  "precheck-only",
  "dependency-probe-only",
  "collection package",
  "input package",
  "scaffold",
  "synthetic input",
  "GUI screenshot",
  "TensorRtExec report without runtime proof"
)

function New-Blocker {
  param(
    [string]$BlockerId,
    [string]$Title,
    [string[]]$RequiredOwnerInputFiles,
    [string[]]$RequiredFields,
    [string]$StrictValidator,
    [string[]]$AcceptableProofKinds,
    [string]$FirstOwnerCommand,
    [string]$CleanExternalExecution
  )

  [pscustomobject]@{
    blockerId = $BlockerId
    title = $Title
    status = "blocked-owner-real-input-required"
    requiredOwnerInputFiles = @($RequiredOwnerInputFiles)
    requiredFields = @($RequiredFields)
    requiredFieldCount = @($RequiredFields).Count
    strictValidator = $StrictValidator
    acceptableProofKinds = @($AcceptableProofKinds)
    firstOwnerCommand = $FirstOwnerCommand
    cleanExternalExecution = $CleanExternalExecution
    forbiddenSubstitutes = @($commonForbiddenSubstitutes)
    promotionBlockedUntil = "真实 Owner 输入文件存在、stdout/stderr/log/hash/SHA256/exitCode/host identity 均回填，并且 strict validator 通过。"
    nonProofBoundary = "This landing blocker is an input contract and handoff surface only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

$blockers = @(
  New-Blocker `
    -BlockerId "owner-authorization" `
    -Title "Owner authorization" `
    -RequiredOwnerInputFiles @(
      "artifacts/final-release/release-owner-approval-input.json",
      "artifacts/final-release/public-publish-result-owner-input.json",
      "artifacts/final-release/release-issue-close-owner-decision-input.json"
    ) `
    -RequiredFields @(
      "ownerName",
      "ownerEmail",
      "approvalDecision",
      "approvalTimestampUtc",
      "packageVersion",
      "publicChannel",
      "rollbackPlanReviewed",
      "releaseIssueCloseDecision",
      "ownerSignature"
    ) `
    -StrictValidator "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1 -Strict" `
    -AcceptableProofKinds @("real-owner-authorization-record", "signed-owner-release-decision") `
    -FirstOwnerCommand "填写 release-owner-approval-input.json 与 release-issue-close-owner-decision-input.json 后运行 strict validator。" `
    -CleanExternalExecution "Owner 决策记录可在仓库内校验，但必须引用真实公开包和真实 proof validator 输出。"

  New-Blocker `
    -BlockerId "package-consumer-runtime" `
    -Title "Clean package consumer runtime proof" `
    -RequiredOwnerInputFiles @(
      "artifacts/final-release/package-consumer-runtime-proof-owner-input.json",
      "artifacts/final-release/package-consumer-runtime-proof-record.json",
      "artifacts/final-release/owner-external-proof-execution-result.input.json"
    ) `
    -RequiredFields @(
      "consumerProjectPath",
      "consumerProjectCreatedOutsideRepository",
      "packageSourceUrl",
      "managedPackageId",
      "managedPackageVersion",
      "runtimePackageKey",
      "runtimePackageVersion",
      "managedNupkgSha256",
      "runtimeNupkgSha256",
      "smokeCommand",
      "smokeExitCode",
      "stdoutPath",
      "stderrPath",
      "smokeLogPath",
      "smokeLogSha256",
      "hostIdentity"
    ) `
    -StrictValidator "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" `
    -AcceptableProofKinds @("clean-external-consumer-runtime-smoke", "validator-passing-package-consumer-runtime-proof") `
    -FirstOwnerCommand "在仓库外 clean consumer 项目安装公开包并执行 runtime smoke，再回填 proof record。" `
    -CleanExternalExecution "必须在仓库外 clean consumer 目录执行；禁止 ProjectReference、local feed、direct nupkg 和仓库路径泄漏。"

  New-Blocker `
    -BlockerId "linux-runner-proof" `
    -Title "Linux runner proof" `
    -RequiredOwnerInputFiles @(
      "artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence.json",
      "artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-validation.json"
    ) `
    -RequiredFields @(
      "linuxDistribution",
      "kernelVersion",
      "cudaDriverVersion",
      "cudaRuntimeVersion",
      "tensorrtVersion",
      "runtimePackageKey",
      "packageVersion",
      "runnerCommand",
      "exitCode",
      "stdoutPath",
      "stderrPath",
      "logPath",
      "logSha256",
      "hostIdentity"
    ) `
    -StrictValidator "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidence.ps1 -Strict" `
    -AcceptableProofKinds @("real-linux-runner-execution-proof", "validator-passing-linux-host-proof") `
    -FirstOwnerCommand "在真实 Linux CUDA/TensorRT 主机运行 runner proof 命令并回填日志与 hash。" `
    -CleanExternalExecution "必须在真实 Linux runner 主机执行；Windows build、dry-run、template-only 均不可替代。"

  New-Blocker `
    -BlockerId "real-model-runtime" `
    -Title "Real model runtime proof" `
    -RequiredOwnerInputFiles @(
      "artifacts/user-acceptance/sample-run-evidence-record.json",
      "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.json",
      "artifacts/user-acceptance/sample-run-evidence-record-validation.json"
    ) `
    -RequiredFields @(
      "modelFamily",
      "modelVersion",
      "modelSha256",
      "inputAssetPath",
      "inputAssetSha256",
      "enginePath",
      "engineSha256",
      "runtimePackageKey",
      "executionCommand",
      "exitCode",
      "stdoutPath",
      "stderrPath",
      "outputPath",
      "outputSha256",
      "latencySummary",
      "hostIdentity"
    ) `
    -StrictValidator "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -Strict" `
    -AcceptableProofKinds @("real-model-runtime-execution", "validator-passing-yolovision-real-asset-proof") `
    -FirstOwnerCommand "用真实 YOLO/ONNX 模型资产执行 YoloVision 或 TensorRtExec runtime 路径并回填 evidence record。" `
    -CleanExternalExecution "必须使用真实模型、真实输入、真实输出与 hash；parse-only、build-only、sidecar-only 不可替代。"

  New-Blocker `
    -BlockerId "post-publish-verification" `
    -Title "Post-publish verification" `
    -RequiredOwnerInputFiles @(
      "artifacts/final-release/post-publish-verification-owner-input.json",
      "artifacts/final-release/post-publish-verification-record.json",
      "artifacts/final-release/post-publish-verification-validation.json"
    ) `
    -RequiredFields @(
      "publishedPackageUrl",
      "publishedPackageVersion",
      "publishedPackageSha256",
      "nugetPackagePageUrl",
      "githubReleaseUrl",
      "installCommand",
      "cleanConsumerCommand",
      "exitCode",
      "stdoutPath",
      "stderrPath",
      "verificationLogPath",
      "verificationLogSha256",
      "rollbackPlan",
      "ownerVerificationDecision",
      "hostIdentity"
    ) `
    -StrictValidator "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -AcceptableProofKinds @("real-post-publish-channel-verification", "validator-passing-post-publish-proof") `
    -FirstOwnerCommand "发布后从公开渠道重新安装并验证，再回填 post-publish verification record。" `
    -CleanExternalExecution "必须从公开渠道验证；本地 feed、candidate、draft、dashboard 和 runbook 不可替代。"
)

$sourceArtifacts = @(
  "artifacts/final-release/final-publish-proof-gate-report.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/release-close-real-input-candidate-promotion-readiness.json",
  "artifacts/final-release/owner-real-input-json-contract.json",
  "artifacts/final-release/owner-real-input-import-preflight.json"
)

$record = [ordered]@{
  recordKind = "owner-real-input-landing-pack"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  landingState = "blocked-owner-real-input-required"
  blockerCount = $blockers.Count
  blockedBlockerCount = $blockers.Count
  blockers = @($blockers)
  blockerIds = @($blockers | ForEach-Object { $_.blockerId })
  requiredOwnerInputFiles = @($blockers | ForEach-Object { $_.requiredOwnerInputFiles } | Select-Object -Unique)
  strictValidators = @($blockers | ForEach-Object { $_.strictValidator } | Select-Object -Unique)
  forbiddenSubstitutes = @($commonForbiddenSubstitutes)
  nonSubstituteProofKinds = @(
    "validator-passing-real-owner-input",
    "clean-external-consumer-runtime-smoke",
    "real-linux-runner-execution-proof",
    "real-model-runtime-execution",
    "real-post-publish-channel-verification"
  )
  performsPublish = $false
  notExecutedByAutomation = $true
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @($sourceArtifacts)
  boundary = "Owner real input landing pack maps the five final blockers to required files, fields, validators, and forbidden substitutes. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-real-input-landing-pack.json"
$markdownPath = Join-Path $OutputRoot "owner-real-input-landing-pack.md"
$record | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($blocker in $blockers) {
  "| ``$(ConvertTo-MarkdownCell $blocker.blockerId)`` | $(ConvertTo-MarkdownCell $blocker.title) | ``$(ConvertTo-MarkdownCell $blocker.status)`` | ``$($blocker.requiredFieldCount)`` | ``$(ConvertTo-MarkdownCell $blocker.strictValidator)`` | $(ConvertTo-MarkdownCell $blocker.cleanExternalExecution) |"
}

$details = foreach ($blocker in $blockers) {
  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("### $($blocker.blockerId)")
  $lines.Add("")
  $lines.Add(("- 状态：{0}" -f $blocker.status))
  $lines.Add(("- 首个 Owner 动作：{0}" -f $blocker.firstOwnerCommand))
  $lines.Add(("- strict validator：{0}" -f $blocker.strictValidator))
  $lines.Add(("- promotionBlockedUntil：{0}" -f $blocker.promotionBlockedUntil))
  $lines.Add(("- nonProofBoundary：{0}" -f $blocker.nonProofBoundary))
  $lines.Add("- requiredOwnerInputFiles：")
  foreach ($file in $blocker.requiredOwnerInputFiles) {
    $lines.Add(("  - {0}" -f $file))
  }
  $lines.Add("- requiredFields：")
  foreach ($field in $blocker.requiredFields) {
    $lines.Add(("  - {0}" -f $field))
  }
  $lines.Add("")
  $lines -join [Environment]::NewLine
}

$markdown = @(
  "# Owner Real Input Landing Pack",
  "",
  "- landingState: $($record.landingState)",
  "- blockerCount: $($record.blockerCount)",
  "- canPublishPublicly: $($record.canPublishPublicly)",
  "- canCloseReleaseIssue: $($record.canCloseReleaseIssue)",
  "- boundary: $($record.boundary)",
  "",
  "| Blocker | Title | State | Required Fields | Strict Validator | External Boundary |",
  "|---|---|---:|---:|---|---|",
  @($rows),
  "",
  "## Blocker Details",
  "",
  @($details)
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
