[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputPath = "artifacts\final-release\github-actions-package-validation-audit.json",
  [string]$MarkdownOutputPath = "artifacts\final-release\github-actions-package-validation-audit.md"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepoPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Read-TextOrEmpty {
  param([Parameter(Mandatory = $true)][string]$Path)

  $fullPath = Resolve-RepoPath -Path $Path
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    return ""
  }

  return Get-Content -LiteralPath $fullPath -Raw -Encoding utf8
}

function Invoke-GitText {
  param([Parameter(Mandatory = $true)][string[]]$Arguments)

  $psi = [System.Diagnostics.ProcessStartInfo]::new()
  $psi.FileName = "git"
  $psi.WorkingDirectory = $RepositoryRoot
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  foreach ($argument in $Arguments) {
    $psi.ArgumentList.Add($argument)
  }

  $process = [System.Diagnostics.Process]::Start($psi)
  $stdout = $process.StandardOutput.ReadToEnd()
  $stderr = $process.StandardError.ReadToEnd()
  $process.WaitForExit()

  return [pscustomobject]@{
    exitCode = $process.ExitCode
    stdout = $stdout.Trim()
    stderr = $stderr.Trim()
  }
}

function Test-ContainsAll {
  param(
    [Parameter(Mandatory = $true)][string]$Text,
    [Parameter(Mandatory = $true)][string[]]$Needles
  )

  foreach ($needle in $Needles) {
    if (-not $Text.Contains($needle, [StringComparison]::OrdinalIgnoreCase)) {
      return $false
    }
  }

  return $true
}

function New-Check {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][bool]$Passed,
    [Parameter(Mandatory = $true)][string]$Severity,
    [Parameter(Mandatory = $true)][string]$Detail
  )

  return [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$insideWorkTree = Invoke-GitText -Arguments @("rev-parse", "--is-inside-work-tree")
$headSha = Invoke-GitText -Arguments @("rev-parse", "HEAD")
$branchName = Invoke-GitText -Arguments @("branch", "--show-current")
$status = Invoke-GitText -Arguments @("status", "--porcelain=v1")
$upstream = Invoke-GitText -Arguments @("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
$remoteUrl = Invoke-GitText -Arguments @("remote", "get-url", "origin")

$hasUpstream = $upstream.exitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($upstream.stdout)
$upstreamSha = if ($hasUpstream) { Invoke-GitText -Arguments @("rev-parse", "@{u}") } else { $null }
$headPushedToUpstream = $false
if ($hasUpstream -and $null -ne $upstreamSha -and $headSha.exitCode -eq 0 -and $upstreamSha.exitCode -eq 0) {
  $headPushedToUpstream = $headSha.stdout.Equals($upstreamSha.stdout, [StringComparison]::OrdinalIgnoreCase)
}

$statusLines = @()
if (-not [string]::IsNullOrWhiteSpace($status.stdout)) {
  $statusLines = @($status.stdout -split "\r?\n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}
$dirtyTrackedCount = @($statusLines | Where-Object { $_ -notmatch "^\?\?" }).Count
$untrackedCount = @($statusLines | Where-Object { $_ -match "^\?\?" }).Count
$isWorktreeClean = $status.exitCode -eq 0 -and $statusLines.Count -eq 0

$packageWorkflow = Read-TextOrEmpty -Path ".github\workflows\package-managed.yml"
$releaseQualityWorkflow = Read-TextOrEmpty -Path ".github\workflows\release-quality-gate.yml"
$releaseBundleWorkflow = Read-TextOrEmpty -Path ".github\workflows\release-bundle.yml"
$runtimeWindowsWorkflow = Read-TextOrEmpty -Path ".github\workflows\runtime-windows.yml"
$runtimeLinuxWorkflow = Read-TextOrEmpty -Path ".github\workflows\runtime-linux.yml"

$packageManagedDryRunReady = Test-ContainsAll -Text $packageWorkflow -Needles @(
  "workflow_dispatch",
  "publish_to_nuget",
  "default: false",
  "publish_to_github_packages",
  "dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj",
  "dotnet pack .\pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj",
  "dotnet pack .\samples\YoloVision\YoloVision.csproj",
  "Test-ManagedPackageContent.ps1",
  "Test-YoloVisionManagedPackageDryRun.ps1",
  "actions/upload-artifact"
)

$packageManagedPublishGuarded = Test-ContainsAll -Text $packageWorkflow -Needles @(
  'if: ${{ inputs.publish_to_nuget && inputs.owner_publish_approved && github.repository_owner == ''guojin-yan'' }}',
  "NUGET_API_KEY",
  "Test-PublishPrerequisites.ps1",
  "Push-NuGetPackages.ps1",
  'if: ${{ inputs.publish_to_github_packages && inputs.owner_publish_approved && github.repository_owner == ''guojin-yan'' }}',
  "Package or Release publication requires owner_publish_approved=true",
  "grape-yan repository is validation-only"
)

$releaseQualityHasSourceGate = (Test-ContainsAll -Text $releaseQualityWorkflow -Needles @(
  "workflow_dispatch:",
  "Test-ReleaseQualityGate.ps1 -Strict",
  "Generate-Bindings.ps1",
  "Test-BindingGeneratorOutputs.ps1",
  "Export-InterfaceCoverageMatrix.ps1",
  "dotnet build TensorRtSharp.sln",
  "dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj"
)) -and
  -not $releaseQualityWorkflow.Contains("push:", [StringComparison]::Ordinal) -and
  -not $releaseQualityWorkflow.Contains("pull_request:", [StringComparison]::Ordinal)
$releaseQualitySourceOnlyFilterClean =
  (Test-ContainsAll -Text $releaseQualityWorkflow -Needles @(
    "Run source-only release quality tests",
    "--filter `"FullyQualifiedName~ReleaseAutomationTests|FullyQualifiedName~ReleaseQualityGateWorkflowTests|FullyQualifiedName~PublicationLicenseReadinessTests`""
  )) -and
  -not $releaseQualityWorkflow.Contains("FinalReleaseMarkdownRenderingTests", [StringComparison]::Ordinal) -and
  -not $releaseQualityWorkflow.Contains("RnnV2BorrowedStateDesignGateTests", [StringComparison]::Ordinal) -and
  -not $releaseQualityWorkflow.Contains("EngineAndRnnReadonlyDiagnosticsTests", [StringComparison]::Ordinal) -and
  -not $releaseQualityWorkflow.Contains("RuntimePackageReadinessTests", [StringComparison]::Ordinal)

$releaseQualityHasPackageDryRunAudit = Test-ContainsAll -Text $releaseQualityWorkflow -Needles @(
  "Export-GitHubActionsPackageValidationAudit.ps1",
  "github-actions-package-validation-audit.*",
  "run_package_managed_dry_run",
  "package-managed.yml",
  "publish_to_nuget: false",
  "publish_to_github_packages: false"
)

$releaseBundleRemoteReady = Test-ContainsAll -Text $releaseBundleWorkflow -Needles @(
  "workflow_dispatch",
  "dispatch_workflow `"package-managed`"",
  "dispatch_workflow `"runtime-windows`"",
  "dispatch_workflow `"runtime-linux`"",
  "publish_managed_to_nuget",
  "publish_managed_to_github_packages",
  "publish_runtime_to_github_packages"
)

$runtimeWorkflowsPackageReady = (Test-ContainsAll -Text $runtimeWindowsWorkflow -Needles @(
    "workflow_dispatch",
    "dotnet restore TensorRtSharp.sln",
    "dotnet build TensorRtSharp.sln",
    "dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj",
    "Validate-RuntimeManifest.ps1",
    "Test-BindingGeneratorOutputs.ps1",
    "Invoke-LocalSplitRuntimePackage.ps1",
    "Push-NuGetPackages.ps1",
    "publish_to_github_packages"
  )) -and (Test-ContainsAll -Text $runtimeLinuxWorkflow -Needles @(
    "workflow_dispatch",
    "dotnet test",
    "dotnet pack",
    "Push-NuGetPackages.ps1",
    "publish_to_github_packages"
  ))

$upstreamShaText = ""
if ($null -ne $upstreamSha) {
  $upstreamShaText = $upstreamSha.stdout
}

$checks = @(
  New-Check -Id "git-worktree-present" -Passed ($insideWorkTree.exitCode -eq 0 -and $insideWorkTree.stdout -eq "true") -Severity "blocker" -Detail "RepositoryRoot=$RepositoryRoot"
  New-Check -Id "git-origin-present" -Passed ($remoteUrl.exitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($remoteUrl.stdout)) -Severity "blocker" -Detail $remoteUrl.stdout
  New-Check -Id "git-upstream-present" -Passed $hasUpstream -Severity "blocker" -Detail $(if ($hasUpstream) { $upstream.stdout } else { "No upstream branch is configured for the current branch." })
  New-Check -Id "git-worktree-clean" -Passed $isWorktreeClean -Severity "blocker" -Detail "dirtyTracked=$dirtyTrackedCount; untracked=$untrackedCount"
  New-Check -Id "git-head-pushed-to-upstream" -Passed $headPushedToUpstream -Severity "blocker" -Detail "HEAD=$($headSha.stdout); upstream=$upstreamShaText"
  New-Check -Id "workflow-package-managed-dry-run-contract" -Passed $packageManagedDryRunReady -Severity "blocker" -Detail "package-managed.yml must test, pack, validate package content, upload artifacts, and default publish toggles to false."
  New-Check -Id "workflow-package-managed-publish-guard" -Passed $packageManagedPublishGuarded -Severity "blocker" -Detail "package-managed.yml must guard nuget.org/GitHub Packages publication behind explicit inputs, guojin-yan ownership, and prerequisites."
  New-Check -Id "workflow-release-quality-source-gate" -Passed $releaseQualityHasSourceGate -Severity "blocker" -Detail "release-quality-gate.yml must run source quality, bindings, coverage, build, and tests."
  New-Check -Id "workflow-release-quality-source-only-filter" -Passed $releaseQualitySourceOnlyFilterClean -Severity "blocker" -Detail "release-quality-gate.yml source-quality must not depend on final-release artifact-only test classes."
  New-Check -Id "workflow-release-quality-package-dry-run-audit" -Passed $releaseQualityHasPackageDryRunAudit -Severity "blocker" -Detail "release-quality-gate.yml must expose an opt-in package-managed dry run and archive this audit."
  New-Check -Id "workflow-release-bundle-remote-orchestration" -Passed $releaseBundleRemoteReady -Severity "blocker" -Detail "release-bundle.yml must orchestrate managed/runtime workflows and explicit publish toggles."
  New-Check -Id "workflow-runtime-package-ready" -Passed $runtimeWorkflowsPackageReady -Severity "blocker" -Detail "runtime Windows/Linux workflows must provide package build/test and guarded GitHub Packages publication routes."
)

$failedBlockerCount = @($checks | Where-Object { -not $_.passed -and $_.severity -eq "blocker" }).Count

$record = [pscustomobject]@{
  recordKind = "github-actions-package-validation-audit"
  generatedAt = (Get-Date).ToString("o")
  repositoryRoot = $RepositoryRoot
  branch = $branchName.stdout
  headSha = $headSha.stdout
  origin = $remoteUrl.stdout
  upstream = if ($hasUpstream) { $upstream.stdout } else { "" }
  git = [pscustomobject]@{
    isInsideWorkTree = ($insideWorkTree.exitCode -eq 0 -and $insideWorkTree.stdout -eq "true")
    isWorktreeClean = $isWorktreeClean
    dirtyTrackedCount = $dirtyTrackedCount
    untrackedCount = $untrackedCount
    hasUpstream = $hasUpstream
    headPushedToUpstream = $headPushedToUpstream
    sampleStatusLines = @($statusLines | Select-Object -First 30)
  }
  workflowContracts = [pscustomobject]@{
    packageManagedDryRunReady = $packageManagedDryRunReady
    packageManagedPublishGuarded = $packageManagedPublishGuarded
    releaseQualityHasSourceGate = $releaseQualityHasSourceGate
    releaseQualitySourceOnlyFilterClean = $releaseQualitySourceOnlyFilterClean
    releaseQualityHasPackageDryRunAudit = $releaseQualityHasPackageDryRunAudit
    releaseBundleRemoteReady = $releaseBundleRemoteReady
  runtimeWorkflowsPackageReady = $runtimeWorkflowsPackageReady
  }
  failedBlockerCount = $failedBlockerCount
  canClaimCurrentCodeUploadedToGitHub = $isWorktreeClean -and $headPushedToUpstream
  hasGitHubActionsRunEvidenceForCurrentCode = $false
  canClaimGitHubActionsPackageValidationForCurrentCode = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  performsPublish = $false
  usesPublishToken = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  proofBoundary = "This audit is a source-state and workflow-contract audit only. It does not upload code, does not execute GitHub Actions, does not publish packages, does not run dotnet nuget push, is not runtime proof, is not package-consumer runtime proof, and is not post-publish proof."
  ownerAnswer = if ($isWorktreeClean -and $headPushedToUpstream) {
    "当前工作区 HEAD 与上游一致；仍需用 GitHub Actions run URL/artifact 补充包验证证据。"
  }
  else {
    "当前本地改动尚未形成可证明的已推送 GitHub 状态；不能声称本轮代码已在 GitHub Actions 上完成 NuGet/包验证。"
  }
  nextRequiredActions = @(
    "整理工作区并提交所有应进入发布候选的源码、workflow、脚本、文档和测试改动。",
    "推送到 GitHub 分支或 PR，确保当前 HEAD 可由 GitHub Actions 检出。",
    "运行 release-quality-gate.yml，先保持 run_package_managed_dry_run=true、run_split_package_build=false、run_release_artifact_audit=false。",
    "确认 package-managed.yml 以 owner_publish_approved=false、publish_to_nuget=false、publish_to_github_packages=false 完成基础 managed + YoloVision 双包 pack、surface、clean consumer 和 artifact 上传。",
    "在自托管 runner 上按 runtime key 运行 split/runtime workflow，保持 publish=false，取得 package-consumer smoke artifact。",
    "只有在 public package source、downloaded nupkg SHA256、clean consumer stdout/stderr 和 post-publish proof 都齐备后，才允许进入 Owner 授权发布。"
  )
  checks = $checks
}

$jsonPath = Resolve-RepoPath -Path $OutputPath
$markdownPath = Resolve-RepoPath -Path $MarkdownOutputPath
New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($jsonPath)) -Force | Out-Null
New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($markdownPath)) -Force | Out-Null

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# GitHub Actions / NuGet 打包验证审计")
$lines.Add("")
$lines.Add("- 结论：$($record.ownerAnswer)")
$lines.Add("- 当前分支：``$($record.branch)``")
$lines.Add("- 当前 HEAD：``$($record.headSha)``")
$lines.Add("- 上游分支：``$($record.upstream)``")
$lines.Add("- 工作区干净：``$($record.git.isWorktreeClean)``")
$lines.Add("- HEAD 已推送到上游：``$($record.git.headPushedToUpstream)``")
$lines.Add("- 可声称本轮代码已上传 GitHub：``$($record.canClaimCurrentCodeUploadedToGitHub)``")
$lines.Add("- 可声称 GitHub Actions 已验证当前代码包构建：``$($record.canClaimGitHubActionsPackageValidationForCurrentCode)``")
$lines.Add("")
$lines.Add("## 检查项")
$lines.Add("")
$lines.Add("| ID | 结果 | 级别 | 说明 |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($check in $checks) {
  $detail = ([string]$check.detail).Replace("|", "\|")
  $lines.Add("| $($check.id) | $($check.passed) | $($check.severity) | $detail |")
}
$lines.Add("")
$lines.Add("## 下一步")
$lines.Add("")
foreach ($action in $record.nextRequiredActions) {
  $lines.Add("- $action")
}
$lines.Add("")
$lines.Add("## 边界")
$lines.Add("")
$lines.Add($record.proofBoundary)

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "GitHub Actions package validation audit written to $jsonPath"
Write-Host "GitHub Actions package validation audit written to $markdownPath"
Write-Host "failedBlockerCount=$failedBlockerCount"
