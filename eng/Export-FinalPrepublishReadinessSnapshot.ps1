[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputPath = "artifacts\final-release\final-prepublish-readiness-snapshot.json",
  [string]$MarkdownOutputPath = "artifacts\final-release\final-prepublish-readiness-snapshot.md"
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
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Test-FileExists {
  param([string]$Path)
  return [bool](Test-Path -LiteralPath (Resolve-RepoPath -Path $Path) -PathType Leaf)
}

function Test-DirectoryExists {
  param([string]$Path)
  return [bool](Test-Path -LiteralPath (Resolve-RepoPath -Path $Path) -PathType Container)
}

function Read-Text {
  param([string]$Path)
  $fullPath = Resolve-RepoPath -Path $Path
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    return ""
  }

  return Get-Content -LiteralPath $fullPath -Raw
}

function Read-Json {
  param([string]$Path)
  $fullPath = Resolve-RepoPath -Path $Path
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $fullPath -Raw | ConvertFrom-Json
}

function New-GateStatus {
  param(
    [string]$Id,
    [string]$Path
  )

  $json = Read-Json -Path $Path
  if ($null -eq $json) {
    return [pscustomobject]@{
      id = $Id
      path = $Path
      exists = $false
      recordKind = $null
      state = "missing"
      canPublishPublicly = $false
      canCloseReleaseIssue = $false
      blocked = $true
    }
  }

  $state = $null
  if ($json.PSObject.Properties.Name -contains "validationState") {
    $state = [string]$json.validationState
  } elseif ($json.PSObject.Properties.Name -contains "convergenceState") {
    $state = [string]$json.convergenceState
  } elseif ($json.PSObject.Properties.Name -contains "worklistState") {
    $state = [string]$json.worklistState
  } else {
    $state = "unknown"
  }

  [pscustomobject]@{
    id = $Id
    path = $Path
    exists = $true
    recordKind = [string]$json.recordKind
    state = $state
    canPublishPublicly = [bool]$json.canPublishPublicly
    canCloseReleaseIssue = [bool]$json.canCloseReleaseIssue
    blocked = (-not [bool]$json.canPublishPublicly) -or (-not [bool]$json.canCloseReleaseIssue)
  }
}

function New-Check {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$State,
    [string]$Evidence
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    state = $State
    evidence = $Evidence
  }
}

$gateStatuses = @(
  New-GateStatus -Id "release-proof-owner-backfill-summary-validation" -Path "artifacts/final-release/release-proof-owner-backfill-summary-validation.json"
  New-GateStatus -Id "release-final-blocker-convergence" -Path "artifacts/final-release/release-final-blocker-convergence.json"
  New-GateStatus -Id "owner-real-proof-final-action-worklist" -Path "artifacts/final-release/owner-real-proof-final-action-worklist.json"
)

$directoryBuildProps = Read-Text -Path "Directory.Build.props"
$managedProject = Read-Text -Path "src/JYPPX.TensorRtSharp/JYPPX.TensorRtSharp.csproj"
$solutionText = Read-Text -Path "TensorRtSharp.sln"
$docsIndex = Read-Text -Path "docs/index.md"
$docsToc = Read-Text -Path "docs/toc.yml"
$readmeEn = Read-Text -Path "README.md"
$readmeZh = Read-Text -Path "README.zh-CN.md"

$runtimeSplitProjectCount = 0
$runtimeSplitRoot = Resolve-RepoPath -Path "pack/runtime-split"
if (Test-Path -LiteralPath $runtimeSplitRoot -PathType Container) {
  $runtimeSplitProjectCount = @(Get-ChildItem -LiteralPath $runtimeSplitRoot -Recurse -Filter "*.csproj" -File).Count
}

$runtimeMonolithicProjectCount = 0
$runtimeRoot = Resolve-RepoPath -Path "pack/runtime"
if (Test-Path -LiteralPath $runtimeRoot -PathType Container) {
  $runtimeMonolithicProjectCount = @(Get-ChildItem -LiteralPath $runtimeRoot -Recurse -Filter "*.csproj" -File).Count
}

$packageReadinessChecks = @(
  New-Check -Id "managed-package-id-present" -Passed ($managedProject.Contains("<PackageId>JYPPX.TensorRT.CSharp.API</PackageId>")) -State "ready" -Evidence "src/JYPPX.TensorRtSharp/JYPPX.TensorRtSharp.csproj contains managed PackageId."
  New-Check -Id "managed-description-present" -Passed ($managedProject.Contains("<Description>")) -State "ready" -Evidence "src/JYPPX.TensorRtSharp/JYPPX.TensorRtSharp.csproj contains package description."
  New-Check -Id "repository-url-present" -Passed ($directoryBuildProps.Contains("<RepositoryUrl>")) -State "ready" -Evidence "Directory.Build.props contains RepositoryUrl."
  New-Check -Id "authors-present" -Passed ($directoryBuildProps.Contains("<Authors>")) -State "ready" -Evidence "Directory.Build.props contains Authors."
  New-Check -Id "version-prefix-present" -Passed ($directoryBuildProps.Contains("<VersionPrefix>4.0.0</VersionPrefix>")) -State "ready" -Evidence "Directory.Build.props contains VersionPrefix 4.0.0."
  New-Check -Id "runtime-split-script-present" -Passed (Test-FileExists -Path "eng/Collect-SplitRuntimeAssets.ps1") -State "ready" -Evidence "eng/Collect-SplitRuntimeAssets.ps1 exists."
  New-Check -Id "runtime-split-package-script-present" -Passed (Test-FileExists -Path "eng/Invoke-LocalSplitRuntimePackage.ps1") -State "ready" -Evidence "eng/Invoke-LocalSplitRuntimePackage.ps1 exists."
  New-Check -Id "runtime-split-projects-present" -Passed ($runtimeSplitProjectCount -gt 0) -State "ready" -Evidence "pack/runtime-split contains $runtimeSplitProjectCount project file(s)."
  New-Check -Id "runtime-monolithic-projects-present" -Passed ($runtimeMonolithicProjectCount -gt 0) -State "ready" -Evidence "pack/runtime contains $runtimeMonolithicProjectCount project file(s)."
)

$docsReadinessChecks = @(
  New-Check -Id "docs-index-final-gates-linked" -Passed ($docsIndex.Contains("owner-real-proof-final-action-worklist.md") -and $docsIndex.Contains("release-final-blocker-convergence.md") -and $docsIndex.Contains("release-proof-owner-backfill-summary")) -State "ready" -Evidence "docs/index.md links final gates and artifacts."
  New-Check -Id "docs-toc-final-gates-linked" -Passed ($docsToc.Contains("owner-real-proof-final-action-worklist.md") -and $docsToc.Contains("release-final-blocker-convergence.md")) -State "ready" -Evidence "docs/toc.yml links final gate articles."
  New-Check -Id "readme-en-nonproof-boundary" -Passed ($readmeEn.Contains("not release proof") -and $readmeEn.Contains("not public package proof")) -State "ready" -Evidence "README.md states documentation/build-only outputs are not release proof."
  New-Check -Id "readme-zh-nonproof-boundary" -Passed ($readmeZh.Contains("不是 release proof") -and $readmeZh.Contains("不是 public package proof")) -State "ready" -Evidence "README.zh-CN.md states documentation/build-only outputs are not release proof."
)

$sampleRenameChecks = @(
  New-Check -Id "yolovision-project-present" -Passed (Test-FileExists -Path "samples/YoloVision/YoloVision.csproj") -State "ready" -Evidence "samples/YoloVision/YoloVision.csproj exists."
  New-Check -Id "yolovision-program-present" -Passed (Test-FileExists -Path "samples/YoloVision/Program.cs") -State "ready" -Evidence "samples/YoloVision/Program.cs exists."
  New-Check -Id "yolovision-readme-present" -Passed (Test-FileExists -Path "samples/YoloVision/README.md") -State "ready" -Evidence "samples/YoloVision/README.md exists."
  New-Check -Id "retired-sample-project-absent" -Passed (-not (Test-FileExists -Path "samples/YoloDet/YoloDet.csproj")) -State "ready" -Evidence "The retired detection-only sample project is absent."
  New-Check -Id "solution-references-yolovision" -Passed ($solutionText.Contains("samples\YoloVision\YoloVision.csproj")) -State "ready" -Evidence "TensorRtSharp.sln references samples/YoloVision."
  New-Check -Id "solution-no-retired-sample-reference" -Passed (-not $solutionText.Contains("samples\YoloDet\YoloDet.csproj")) -State "ready" -Evidence "TensorRtSharp.sln has no retired detection-only sample project reference."
)

$ownerEvidenceChecks = @(
  New-Check -Id "owner-evidence-real-model-runtime-missing" -Passed (-not (Test-DirectoryExists -Path "artifacts/final-release/owner-evidence/real-model-runtime")) -State "blocked-owner-action-required" -Evidence "real-model-runtime owner evidence directory is missing."
  New-Check -Id "owner-evidence-package-consumer-runtime-missing" -Passed (-not (Test-DirectoryExists -Path "artifacts/final-release/owner-evidence/package-consumer-runtime")) -State "blocked-owner-action-required" -Evidence "package-consumer-runtime owner evidence directory is missing."
  New-Check -Id "owner-evidence-post-publish-verification-missing" -Passed (-not (Test-DirectoryExists -Path "artifacts/final-release/owner-evidence/post-publish-verification")) -State "blocked-owner-action-required" -Evidence "post-publish-verification owner evidence directory is missing."
  New-Check -Id "owner-evidence-release-issue-close-missing" -Passed (-not (Test-DirectoryExists -Path "artifacts/final-release/owner-evidence/release-issue-close")) -State "blocked-owner-action-required" -Evidence "release-issue-close owner evidence directory is missing."
)

$failedPackageReadinessCount = @($packageReadinessChecks | Where-Object { -not $_.passed }).Count
$failedDocsReadinessCount = @($docsReadinessChecks | Where-Object { -not $_.passed }).Count
$failedSampleRenameReadinessCount = @($sampleRenameChecks | Where-Object { -not $_.passed }).Count
$ownerEvidenceMissingCount = @($ownerEvidenceChecks | Where-Object { $_.passed }).Count

$result = [pscustomobject]@{
  recordKind = "final-prepublish-readiness-snapshot"
  generatedAtLocal = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
  snapshotState = "blocked-owner-real-proof-missing"
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  publishBlockedReason = "owner-real-proof-missing"
  performsPublish = $false
  approvesPublicRelease = $false
  proofBoundary = "This snapshot audits packaging, docs, samples, and final gates. It is not proof, does not publish packages, and cannot override missing Owner real proof."
  finalGates = @($gateStatuses)
  packageReadiness = [pscustomobject]@{
    state = $(if ($failedPackageReadinessCount -eq 0) { "ready-for-owner-proof" } else { "needs-repair" })
    failedReadinessCount = [int]$failedPackageReadinessCount
    checks = @($packageReadinessChecks)
  }
  docsReadiness = [pscustomobject]@{
    state = $(if ($failedDocsReadinessCount -eq 0) { "ready-for-owner-proof" } else { "needs-repair" })
    failedReadinessCount = [int]$failedDocsReadinessCount
    checks = @($docsReadinessChecks)
  }
  sampleRenameReadiness = [pscustomobject]@{
    state = $(if ($failedSampleRenameReadinessCount -eq 0) { "ready-for-owner-proof" } else { "needs-repair" })
    failedReadinessCount = [int]$failedSampleRenameReadinessCount
    legacySampleIdentityRemoved = $true
    currentSampleName = "YoloVision"
    checks = @($sampleRenameChecks)
  }
  ownerEvidenceReadiness = [pscustomobject]@{
    state = "blocked-owner-action-required"
    missingOwnerEvidenceDirectoryCount = [int]$ownerEvidenceMissingCount
    checks = @($ownerEvidenceChecks)
  }
  forbiddenProofSubstitutes = @(
    "template",
    "report",
    "matrix",
    "article",
    "command pack",
    "summary pack",
    "dry-run",
    "build-only",
    "sidecar-only",
    "screenshot-only",
    "Skipped=True",
    "blocked-by-cuda-driver",
    "local feed",
    "ProjectReference",
    "direct .nupkg"
  )
  nextRequiredAction = "Owner must provide real proof evidence and strict validator output before public publish or release issue close can be reconsidered."
}

$outputFullPath = Resolve-RepoPath -Path $OutputPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputFullPath) | Out-Null
$result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $outputFullPath -Encoding utf8

$markdownFullPath = Resolve-RepoPath -Path $MarkdownOutputPath
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('# Final Prepublish Readiness Snapshot')
$lines.Add('')
$lines.Add('Generated: 2026-07-08')
$lines.Add('')
$lines.Add('- Record kind: `final-prepublish-readiness-snapshot`')
$lines.Add('- Snapshot state: `blocked-owner-real-proof-missing`')
$lines.Add('- Can publish publicly: `false`')
$lines.Add('- Can close release issue: `false`')
$lines.Add('- Publish blocked reason: `owner-real-proof-missing`')
$lines.Add('')
$lines.Add('## Final Gates')
$lines.Add('')
$lines.Add('| Gate | State | Can publish | Can close |')
$lines.Add('|---|---|---:|---:|')
foreach ($gate in $gateStatuses) {
  $lines.Add(('| `{0}` | `{1}` | `{2}` | `{3}` |' -f $gate.id, $gate.state, $gate.canPublishPublicly.ToString().ToLowerInvariant(), $gate.canCloseReleaseIssue.ToString().ToLowerInvariant()))
}
$lines.Add('')
$lines.Add('## Readiness Summary')
$lines.Add('')
$lines.Add(('- Package readiness: `{0}`' -f $result.packageReadiness.state))
$lines.Add(('- Docs readiness: `{0}`' -f $result.docsReadiness.state))
$lines.Add(('- Sample rename readiness: `{0}`' -f $result.sampleRenameReadiness.state))
$lines.Add(('- Owner evidence readiness: `{0}`' -f $result.ownerEvidenceReadiness.state))
$lines.Add('')
$lines.Add('## Sample Rename')
$lines.Add('')
$lines.Add('`samples/YoloVision` is the active YOLO-family sample. The retired detection-only sample identity is no longer referenced by the solution.')
$lines.Add('')
$lines.Add('## Boundary')
$lines.Add('')
$lines.Add('This snapshot is not release proof. It does not publish packages, cannot close release issues, and cannot override missing Owner real proof.')
$lines | Set-Content -LiteralPath $markdownFullPath -Encoding utf8

$result | ConvertTo-Json -Depth 12
