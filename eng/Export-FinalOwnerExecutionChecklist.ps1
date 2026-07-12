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
  $lines = @(ConvertTo-FlatStringLines -Value $InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-Step {
  param(
    [int]$Order,
    [string]$StepId,
    [string]$BlockerId,
    [string]$Title,
    [string]$OwnerAction,
    [string]$ExecutionLocation,
    [string]$Command,
    [string[]]$RequiredBackfillFields,
    [string[]]$ExpectedArtifacts,
    [string]$StrictValidator
  )

  [pscustomobject]@{
    order = $Order
    stepId = $StepId
    blockerId = $BlockerId
    title = $Title
    status = "blocked-final-owner-execution-real-input-required"
    ownerAction = $OwnerAction
    executionLocation = $ExecutionLocation
    mustRunOutsideRepository = ($ExecutionLocation -match "仓库外|Linux|公开渠道")
    command = $Command
    requiredBackfillFields = @($RequiredBackfillFields)
    requiredBackfillFieldCount = @($RequiredBackfillFields).Count
    expectedArtifacts = @($ExpectedArtifacts)
    strictValidator = $StrictValidator
    requiredCapture = @("stdout", "stderr", "log", "hash", "SHA256", "exitCode", "host identity")
    forbiddenSubstitutes = @("local feed", "ProjectReference", "direct nupkg", "dry-run", "dashboard", "runbook", "candidate", "draft", "build-only", "parse-only", "sidecar-only", "template")
    performsPublish = $false
    notExecutedByAutomation = $true
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "This checklist step is owner execution guidance only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$steps = @(
  New-Step -Order 1 -StepId "01-owner-authorization" -BlockerId "owner-authorization" -Title "Owner authorization" -OwnerAction "回填 Owner 发布授权、rollback review 和 close decision 草案。" -ExecutionLocation "仓库内记录校验；必须引用真实公开包和真实 proof validator 输出。" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1 -Strict" -RequiredBackfillFields @("ownerName", "ownerEmail", "approvalDecision", "approvalTimestampUtc", "packageVersion", "rollbackPlanReviewed", "releaseIssueCloseDecision", "ownerSignature") -ExpectedArtifacts @("artifacts/final-release/release-owner-approval-input-validation.json", "artifacts/final-release/release-issue-close-owner-decision-input-validation.json") -StrictValidator "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1 -Strict"
  New-Step -Order 2 -StepId "02-clean-external-package-consumer" -BlockerId "package-consumer-runtime" -Title "Clean external package consumer runtime smoke" -OwnerAction "在仓库外新建 clean consumer，安装公开包，执行 runtime smoke 并回填日志/hash。" -ExecutionLocation "仓库外 clean consumer 目录。" -Command "dotnet new console --framework net8.0; dotnet add package JYPPX.TensorRtSharp --version <public-version>; dotnet run -- --runtime-package-key <runtime-key>" -RequiredBackfillFields @("consumerProjectPath", "consumerProjectCreatedOutsideRepository", "packageSourceUrl", "managedNupkgSha256", "runtimeNupkgSha256", "smokeCommand", "smokeExitCode", "stdoutPath", "stderrPath", "smokeLogPath", "smokeLogSha256", "hostIdentity") -ExpectedArtifacts @("artifacts/final-release/package-consumer-runtime-proof-record.json", "artifacts/final-release/package-consumer-runtime-proof-record-validation.json") -StrictValidator "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  New-Step -Order 3 -StepId "03-linux-runner-proof" -BlockerId "linux-runner-proof" -Title "Linux runner proof" -OwnerAction "在真实 Linux CUDA/TensorRT runner 上执行包加载和 smoke，回填 host/package/log hash。" -ExecutionLocation "真实 Linux CUDA/TensorRT 主机。" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Test-LinuxRunnerEvidence.ps1 -Strict" -RequiredBackfillFields @("linuxDistribution", "kernelVersion", "cudaDriverVersion", "cudaRuntimeVersion", "tensorrtVersion", "runtimePackageKey", "runnerCommand", "exitCode", "stdoutPath", "stderrPath", "logPath", "logSha256", "hostIdentity") -ExpectedArtifacts @("artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence.json", "artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-validation.json") -StrictValidator "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidence.ps1 -Strict"
  New-Step -Order 4 -StepId "04-real-model-runtime" -BlockerId "real-model-runtime" -Title "Real model runtime proof" -OwnerAction "用真实模型、真实输入和真实输出跑 YoloVision/TensorRtExec runtime path。" -ExecutionLocation "可访问真实 CUDA/TensorRT 与模型资产的执行主机。" -Command "dotnet run --project samples/YoloVision/YoloVision.csproj -- --model <real-model.onnx|engine> --input <real-input> --runtime-package-key <runtime-key> --write-evidence" -RequiredBackfillFields @("modelFamily", "modelVersion", "modelSha256", "inputAssetSha256", "engineSha256", "executionCommand", "exitCode", "stdoutPath", "stderrPath", "outputPath", "outputSha256", "latencySummary", "hostIdentity") -ExpectedArtifacts @("artifacts/user-acceptance/sample-run-evidence-record.json", "artifacts/user-acceptance/sample-run-evidence-record-validation.json") -StrictValidator "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -Strict"
  New-Step -Order 5 -StepId "05-post-publish-verification" -BlockerId "post-publish-verification" -Title "Post-publish verification" -OwnerAction "发布后从公开渠道重新安装、验证 package page/hash、clean consumer smoke 和 rollback plan。" -ExecutionLocation "公开 NuGet/GitHub channel + 仓库外 clean consumer。" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" -RequiredBackfillFields @("publishedPackageUrl", "publishedPackageVersion", "publishedPackageSha256", "nugetPackagePageUrl", "githubReleaseUrl", "installCommand", "cleanConsumerCommand", "exitCode", "stdoutPath", "stderrPath", "verificationLogPath", "verificationLogSha256", "rollbackPlan", "ownerVerificationDecision", "hostIdentity") -ExpectedArtifacts @("artifacts/final-release/post-publish-verification-record.json", "artifacts/final-release/post-publish-verification-validation.json") -StrictValidator "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof"
  New-Step -Order 6 -StepId "06-final-close-validation" -BlockerId "release-close-final-decision" -Title "Final release close validation" -OwnerAction "所有真实 proof validator 通过后，刷新 final publish proof gate 并运行 strict close validator。" -ExecutionLocation "仓库内聚合校验；输入必须来自前 5 步真实证据。" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalPublishProofGate.ps1 -Strict; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady" -RequiredBackfillFields @("finalPublishProofGateReportPath", "releaseIssueCloseRecordPath", "ownerFinalDecision", "allValidatorOutputPaths", "allValidatorOutputSha256", "closeTimestampUtc", "ownerSignature") -ExpectedArtifacts @("artifacts/final-release/final-publish-proof-gate-report.json", "artifacts/final-release/release-issue-close-record-validation.json") -StrictValidator "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
)

$record = [ordered]@{
  recordKind = "final-owner-execution-checklist"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  checklistState = "blocked-final-owner-execution-checklist-real-owner-input-required"
  stepCount = $steps.Count
  blockedStepCount = $steps.Count
  executionSteps = @($steps)
  commands = @($steps | ForEach-Object { $_.command })
  strictValidators = @($steps | ForEach-Object { $_.strictValidator } | Select-Object -Unique)
  requiredCapture = @("stdout", "stderr", "log", "hash", "SHA256", "exitCode", "host identity")
  externalExecutionStepIds = @($steps | Where-Object { $_.mustRunOutsideRepository } | ForEach-Object { $_.stepId })
  forbiddenSubstitutes = @("local feed", "ProjectReference", "direct nupkg", "dry-run", "dashboard", "runbook", "candidate", "draft", "build-only", "parse-only", "sidecar-only", "template")
  performsPublish = $false
  notExecutedByAutomation = $true
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    "artifacts/final-release/owner-real-input-landing-pack.json",
    "artifacts/final-release/final-publish-proof-gate-report.json",
    "artifacts/final-release/release-evidence-bundle.json"
  )
  boundary = "Final owner execution checklist is the shortest manual execution path and backfill contract only; it does not run dotnet nuget push, does not publish, is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-checklist.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-checklist.md"
$record | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($step in $steps) {
  "| ``$($step.order)`` | ``$(ConvertTo-MarkdownCell $step.stepId)`` | ``$(ConvertTo-MarkdownCell $step.blockerId)`` | $(ConvertTo-MarkdownCell $step.executionLocation) | ``$($step.requiredBackfillFieldCount)`` | ``$(ConvertTo-MarkdownCell $step.strictValidator)`` |"
}

$markdown = @(
  "# Final Owner Execution Checklist",
  "",
  "- checklistState: $($record.checklistState)",
  "- stepCount: $($record.stepCount)",
  "- blockedStepCount: $($record.blockedStepCount)",
  "- performsPublish: False",
  "- canPublishPublicly: False",
  "- boundary: $($record.boundary)",
  "",
  "> 注意：本清单不执行真实发布，不调用 `dotnet nuget push`。公开发布和 release close 必须等待真实 Owner 回填与 strict validator 通过。",
  "",
  "| Order | Step | Blocker | Execution Location | Fields | Strict Validator |",
  "|---:|---|---|---|---:|---|",
  @($rows),
  "",
  "## Required Capture",
  "",
  "- stdout",
  "- stderr",
  "- log",
  "- hash / SHA256",
  "- exitCode",
  "- host identity"
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
