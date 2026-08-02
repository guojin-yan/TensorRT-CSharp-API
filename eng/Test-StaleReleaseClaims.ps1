[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [switch]$WarnOnly
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function ConvertTo-RelativePath {
  param([string]$Path)

  return $Path.Substring($RepositoryRoot.Length).TrimStart('\')
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Test-AllowedClaimContext {
  param(
    [string]$RuleId,
    [string]$RelativePath,
    [string]$Line,
    [string]$LeadingContext = ""
  )

  $path = $RelativePath.Replace('/', '\')
  $text = $Line.Trim()
  $lower = $text.ToLowerInvariant()
  $contextLower = $LeadingContext.ToLowerInvariant()

  if ($path -like "tests\*") {
    return $true
  }

  if ($path -like "eng\*.ps1") {
    if ($RuleId -in @(
        "can-publish-publicly-true",
        "can-close-release-issue-true",
        "classification-passed-without-boundary",
        "yolovision-passed-without-boundary",
        "yolodet-project-name",
        "yolodet-project-file",
        "yolodet-sample-path",
        "yolodet-sample-path-windows")) {
      return $true
    }
  }

  if ($path -like "artifacts\*") {
    if ($lower.Contains("requires") -or
        $lower.Contains("required") -or
        $lower.Contains("must") -or
        $lower.Contains("template") -or
        $lower.Contains("example") -or
        $lower.Contains("not proof") -or
        $lower.Contains("not publication")) {
      return $true
    }
  }

  if ($RuleId -in @("classification-passed-without-boundary", "yolovision-passed-without-boundary")) {
    if ($path.EndsWith(".cs", [System.StringComparison]::OrdinalIgnoreCase) -or
        $path.EndsWith(".json", [System.StringComparison]::OrdinalIgnoreCase)) {
      return $true
    }

    if ($path -in @(
        "docs\articles\zh-cn\classification-real-asset-walkthrough.md",
        "docs\articles\zh-cn\real-model-owner-backfill-checklist.md",
        "docs\articles\zh-cn\yolovision-real-asset-walkthrough.md",
        "docs\articles\zh-cn\yolovision-multi-output-metadata-guide.md",
        "samples\Classification\README.md",
        "samples\YoloVision\README.md",
        "samples\assets\README.md")) {
      return $true
    }

    if ($lower.Contains("expected") -or
        $lower.Contains("walkthrough") -or
        $lower.Contains("checklist") -or
        $lower.Contains("guide") -or
        $lower.Contains("readme") -or
        $lower.Contains("record") -or
        $lower.Contains("run") -or
        $lower.Contains("log") -or
        $lower.Contains("真实") -or
        $lower.Contains("最后") -or
        $lower.Contains("最后才") -or
        $lower.Contains("附模型") -or
        $lower.Contains("owner") -or
        $lower.Contains("不要") -or
        $lower.Contains("not") -or
        $lower.Contains("不能") -or
        $lower.Contains("不是") -or
        $lower.Contains("until") -or
        $lower.Contains("only") -or
        $lower.Contains("requires") -or
        $lower.Contains("required") -or
        $lower.Contains("evidence") -or
        $lower.Contains("proof") -or
        $lower.Contains("sha256")) {
      return $true
    }

    if ($contextLower.Contains("实测结果") -or
        $contextLower.Contains("真实日志") -or
        $contextLower.Contains("正例结果") -or
        $contextLower.Contains("最终日志")) {
      return $true
    }
  }

  if ($RuleId -eq "yolodet-project-name") {
    if ($text -match "YoloDetection|YoloDetector|YoloDetections|YoloDetectionDecoder|YoloObbDetection") {
      return $true
    }

    if ($lower.Contains("must not") -or
        $lower.Contains("不得") -or
        $lower.Contains("不应") -or
        $lower.Contains("stale") -or
        $lower.Contains("renamed") -or
        $lower.Contains("防回归")) {
      return $true
    }
  }

  if ($RuleId -in @("can-publish-publicly-true", "can-close-release-issue-true")) {
    if ($path -in @(
        "docs\articles\zh-cn\release-owner-approval-input.md",
        "docs\articles\zh-cn\post-publish-verification-record.md",
        "docs\articles\zh-cn\release-evidence-bundle.md")) {
      return $true
    }

    if ($lower.Contains("requires") -or
        $lower.Contains("required") -or
        $lower.Contains("must") -or
        $lower.Contains("explicit") -or
        $lower.Contains("real") -or
        $lower.Contains("owner") -or
        $lower.Contains("真实") -or
        $lower.Contains("必须") -or
        $lower.Contains("需要") -or
        $lower.Contains("至少填写") -or
        $lower.Contains("填写真实输入") -or
        $lower.Contains("并至少填写") -or
        $lower.Contains("before")) {
      return $true
    }
  }

  if ($RuleId -in @(
      "runtime-proof-complete",
      "release-ready-to-publish",
      "release-ready-to-publish-hyphen",
      "post-publish-verified",
      "post-publish-cn-verified",
      "published-to-nuget",
      "nuget-published-en",
      "package-consumer-runtime-passed",
      "package-consumer-runtime-cn-passed",
      "real-model-runtime-passed",
      "real-model-runtime-cn-passed",
      "build-only-release-proof",
      "parse-only-implemented",
      "sidecar-only-runtime-proof",
      "compatible-host-runbook-runtime-proof",
      "compatible-host-runbook-cn-runtime-proof",
      "compatible-host-collection-runtime-proof",
      "compatible-host-collection-cn-runtime-proof",
      "compatible-host-collection-public-release",
      "compatible-host-collection-cn-public-release",
      "collection-bundle-can-promote-true",
      "collection-bundle-runtime-evidence-true",
      "collection-bundle-performs-publish-true",
      "collection-bundle-approves-release-true",
      "can-publish-publicly-true",
      "can-close-release-issue-true")) {
    if ($lower.Contains("not") -or
        $lower.Contains("does not") -or
        $lower.Contains("cannot") -or
        $lower.Contains("不能") -or
        $lower.Contains("不是") -or
        $lower.Contains("不得") -or
        $lower.Contains("不代表") -or
        $lower.Contains("不要")) {
      return $true
    }
  }

  if ($RuleId -eq "runtime-proof-complete" -and
      $lower.Contains("callback runtime proof")) {
    return $true
  }

  return $false
}

$patterns = @(
  [pscustomobject]@{ id = "old-bilingual-281"; pattern = "findingCount=281"; description = "Old bilingual documentation finding count must not appear in current release docs." }
  [pscustomobject]@{ id = "old-bilingual-1109"; pattern = "findingCount=1109"; description = "Old bilingual documentation finding count must not appear in current release docs." }
  [pscustomobject]@{ id = "old-bilingual-1839"; pattern = "findingCount=1839"; description = "Old bilingual documentation finding count must not appear in current release docs." }
  [pscustomobject]@{ id = "runtime-100-percent-cn"; pattern = "已 100% runtime 可用"; description = "Release-facing articles must not claim full runtime availability while driver/callback/Linux proof remains pending." }
  [pscustomobject]@{ id = "runtime-100-percent-en"; pattern = "100% runtime ready"; description = "Release-facing articles must not claim full runtime readiness while driver/callback/Linux proof remains pending." }
  [pscustomobject]@{ id = "all-smoke-passed-cn"; pattern = "所有 smoke 已通过"; description = "Do not claim all smoke passed while CUDA 13.2 smoke is driver-blocked." }
  [pscustomobject]@{ id = "nuget-published-cn"; pattern = "已经发布到 nuget.org"; description = "Do not claim nuget.org publication from local dry-run evidence." }
  [pscustomobject]@{ id = "linux-overclaim-cn"; pattern = "Linux 已验证"; description = "Linux handoff must not be written as completed Linux validation." }
  [pscustomobject]@{ id = "linux-hosted-overclaim"; pattern = "validated through hosted runtime packaging"; description = "Hosted or Windows-generated handoff must not be phrased as completed Linux proof." }
  [pscustomobject]@{ id = "linux-validator-template-proof"; pattern = "validationState=template-only means Linux runner proof"; description = "Linux validator template-only output must not be written as real runner proof." }
  [pscustomobject]@{ id = "linux-validator-template-cn-proof"; pattern = "validationState=template-only 表示 Linux runner proof"; description = "Linux validator template-only output must not be written as real runner proof in Chinese release docs." }
  [pscustomobject]@{ id = "linux-validator-handoff-proof"; pattern = "linux-runner-evidence-validation.md 是 Linux runner proof"; description = "The validation artifact is only proof when it reports real-linux-runner-proof." }
  [pscustomobject]@{ id = "callback-proof-overclaim-cn"; pattern = "callback proof 已完成"; description = "Callback proof must not be claimed complete without InvocationCount>0 evidence." }
  [pscustomobject]@{ id = "managed-readiness-runtime-proof"; pattern = "managed-readiness means runtime proof"; description = "Managed readiness snapshots must not be written as runtime proof." }
  [pscustomobject]@{ id = "managed-readiness-cn-runtime-proof"; pattern = "managed-readiness 表示 runtime proof"; description = "Managed readiness snapshots must not be written as runtime proof in Chinese release docs." }
  [pscustomobject]@{ id = "callback-allocator-readiness-proof"; pattern = "CallbackAllocatorReadinessSnapshot means runtime proof"; description = "CallbackAllocatorReadinessSnapshot is readiness evidence only, not runtime proof." }
  [pscustomobject]@{ id = "callback-allocator-readiness-cn-proof"; pattern = "CallbackAllocatorReadinessSnapshot 表示 runtime proof"; description = "CallbackAllocatorReadinessSnapshot must not be written as runtime proof in Chinese release docs." }
  [pscustomobject]@{ id = "precheck-only-runtime-proof"; pattern = "precheck-only means runtime proof"; description = "Precheck-only records must not be written as runtime proof." }
  [pscustomobject]@{ id = "dry-run-only-runtime-proof"; pattern = "dry-run-only means runtime proof"; description = "Dry-run-only records must not be written as runtime proof." }
  [pscustomobject]@{ id = "schema-only-runtime-proof"; pattern = "schema-only means runtime proof"; description = "Schema-only records must not be written as runtime proof." }
  [pscustomobject]@{ id = "driver-blocker-passed"; pattern = "blocked-by-cuda-driver passed"; description = "Driver blockers must not be written as passed smoke." }
  [pscustomobject]@{ id = "driver-blocker-ready"; pattern = "blocked-by-cuda-driver ready"; description = "Driver blockers must not be written as ready smoke." }
  [pscustomobject]@{ id = "runtime-smoke-overclaim-cn"; pattern = "blocked-by-cuda-driver 是 smoke passed"; description = "Driver blockers must not be written as passed smoke in Chinese release docs." }
  [pscustomobject]@{ id = "allow-runtime-smoke-blocked-ready"; pattern = "allowRuntimeSmokeBlocked=true means ready"; description = "The allowRuntimeSmokeBlocked flag records dry-run intent only and must not be written as ready." }
  [pscustomobject]@{ id = "allow-runtime-smoke-blocked-passed"; pattern = "allowRuntimeSmokeBlocked=true means smoke passed"; description = "The allowRuntimeSmokeBlocked flag must not be written as smoke passed." }
  [pscustomobject]@{ id = "allow-runtime-smoke-blocked-cn-ready"; pattern = "allowRuntimeSmokeBlocked=true 表示 ready"; description = "The allowRuntimeSmokeBlocked flag must not be written as ready in Chinese release docs." }
  [pscustomobject]@{ id = "allow-runtime-smoke-blocked-cn-passed"; pattern = "allowRuntimeSmokeBlocked=true 表示 smoke passed"; description = "The allowRuntimeSmokeBlocked flag must not be written as smoke passed in Chinese release docs." }
  [pscustomobject]@{ id = "dry-run-release-complete-cn"; pattern = "ready-needs-manual-approval 等于发布完成"; description = "Final dry run manual approval state must not be equated with release completion." }
  [pscustomobject]@{ id = "ready-needs-manual-approval-public-release"; pattern = "ready-needs-manual-approval means public release approved"; description = "Manual approval dry-run status must not be written as public release approval." }
  [pscustomobject]@{ id = "dependency-probe-runtime-proof"; pattern = "dependency-probe-passed means runtime execution proof"; description = "Dependency probes must not be written as runtime execution proof." }
  [pscustomobject]@{ id = "dependency-probe-cn-runtime-proof"; pattern = "dependency-probe-passed 是 runtime proof"; description = "Dependency probes must not be written as runtime execution proof in Chinese release docs." }
  [pscustomobject]@{ id = "dependency-probe-only-runtime-proof"; pattern = "IsDependencyProbeOnly=True means runtime execution proof"; description = "Dependency-probe-only package consumer evidence must not be written as runtime execution proof." }
  [pscustomobject]@{ id = "dependency-probe-only-cn-runtime-proof"; pattern = "IsDependencyProbeOnly=True 表示 runtime proof"; description = "Dependency-probe-only package consumer evidence must not be written as runtime execution proof in Chinese release docs." }
  [pscustomobject]@{ id = "runtime-smoke-driver-blocked-passed"; pattern = "runtime-smoke-driver-blocked means smoke passed"; description = "Driver-blocked runtime smoke classification must not be written as smoke passed." }
  [pscustomobject]@{ id = "runtime-smoke-driver-blocked-cn-passed"; pattern = "runtime-smoke-driver-blocked 表示 smoke passed"; description = "Driver-blocked runtime smoke classification must not be written as smoke passed in Chinese release docs." }
  [pscustomobject]@{ id = "callback-proof-false-ready"; pattern = "IsRealCallbackRuntimeProof=False means callback proof complete"; description = "False callback proof fields must not be written as completed callback proof." }
  [pscustomobject]@{ id = "callback-proof-false-cn-ready"; pattern = "IsRealCallbackRuntimeProof=False 表示 callback proof 已完成"; description = "False callback proof fields must not be written as completed callback proof in Chinese release docs." }
  [pscustomobject]@{ id = "runtime-proof-complete"; pattern = "runtime proof complete"; description = "Runtime proof must not be claimed complete until external/package-consumer runtime proof is real and validated." }
  [pscustomobject]@{ id = "release-ready-to-publish"; pattern = "ready to publish"; description = "Do not claim the release is ready to publish before owner authorization, package-consumer proof, and post-publish proof are real." }
  [pscustomobject]@{ id = "release-ready-to-publish-hyphen"; pattern = "ready-to-publish"; description = "Do not use ready-to-publish as current state before all close gates pass." }
  [pscustomobject]@{ id = "post-publish-verified"; pattern = "post-publish verified"; description = "Post-publish verification must not be claimed before a real channel publish and clean consumer validation." }
  [pscustomobject]@{ id = "post-publish-cn-verified"; pattern = "post-publish 已验证"; description = "Post-publish verification must not be claimed in Chinese release docs before a real channel publish and clean consumer validation." }
  [pscustomobject]@{ id = "published-to-nuget"; pattern = "published to NuGet"; description = "Do not claim NuGet publication without explicit owner authorization and channel evidence." }
  [pscustomobject]@{ id = "nuget-published-en"; pattern = "NuGet published"; description = "Do not claim NuGet publication without explicit owner authorization and channel evidence." }
  [pscustomobject]@{ id = "package-consumer-runtime-passed"; pattern = "package-consumer-runtime passed"; description = "Package-consumer runtime proof must not be claimed from template, handoff, build-only, dependency probe, or driver-blocked evidence." }
  [pscustomobject]@{ id = "package-consumer-runtime-cn-passed"; pattern = "package-consumer-runtime 已通过"; description = "Package-consumer runtime proof must not be claimed in Chinese release docs from template, handoff, build-only, dependency probe, or driver-blocked evidence." }
  [pscustomobject]@{ id = "real-model-runtime-passed"; pattern = "real-model-runtime passed"; description = "Real-model runtime proof must not be claimed until model/hash/license/input/log evidence is complete and validated." }
  [pscustomobject]@{ id = "real-model-runtime-cn-passed"; pattern = "real-model-runtime 已通过"; description = "Real-model runtime proof must not be claimed in Chinese release docs until model/hash/license/input/log evidence is complete and validated." }
  [pscustomobject]@{ id = "build-only-release-proof"; pattern = "build-only means release proof"; description = "Build-only reports are conversion evidence and must not be written as release proof." }
  [pscustomobject]@{ id = "parse-only-implemented"; pattern = "parse-only means implemented"; description = "Parse-only option coverage must not be written as implemented TensorRT behavior." }
  [pscustomobject]@{ id = "sidecar-only-runtime-proof"; pattern = "sidecar-only means runtime proof"; description = "Evidence sidecars are diagnostics/handoff records and must not be written as runtime proof." }
  [pscustomobject]@{ id = "sidecar-cn-close-release-issue"; pattern = "sidecar 可以关闭 release issue"; description = "Evidence sidecars are diagnostics/handoff records and must not be written as able to close release issues." }
  [pscustomobject]@{ id = "sidecar-en-close-release-issue"; pattern = "sidecar can close release issue"; description = "Evidence sidecars are diagnostics/handoff records and must not be written as able to close release issues." }
  [pscustomobject]@{ id = "compatible-host-runbook-runtime-proof"; pattern = "compatible-host-runtime-proof-runbook means runtime proof"; description = "Compatible host runbooks are owner-action guidance and must not be written as runtime execution proof." }
  [pscustomobject]@{ id = "compatible-host-runbook-cn-runtime-proof"; pattern = "compatible-host-runtime-proof-runbook 表示 runtime proof"; description = "Compatible host runbooks must not be written as runtime execution proof in Chinese release docs." }
  [pscustomobject]@{ id = "compatible-host-collection-runtime-proof"; pattern = "compatible-host-runtime-proof-collection-bundle means runtime proof"; description = "Compatible host collection bundles are executable guidance and must not be written as runtime execution proof." }
  [pscustomobject]@{ id = "compatible-host-collection-cn-runtime-proof"; pattern = "compatible-host-runtime-proof-collection-bundle 表示 runtime proof"; description = "Compatible host collection bundles must not be written as runtime execution proof in Chinese release docs." }
  [pscustomobject]@{ id = "compatible-host-collection-public-release"; pattern = "compatible-host-runtime-proof-collection-bundle means public release approved"; description = "Compatible host collection bundles must not be written as public release approval." }
  [pscustomobject]@{ id = "compatible-host-collection-cn-public-release"; pattern = "compatible-host-runtime-proof-collection-bundle 表示公开发布批准"; description = "Compatible host collection bundles must not be written as public release approval in Chinese release docs." }
  [pscustomobject]@{ id = "collection-bundle-can-promote-true"; pattern = "compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof=True"; description = "Collection bundle canPromoteRuntimeProof must remain false unless real external runtime proof is validated." }
  [pscustomobject]@{ id = "collection-bundle-runtime-evidence-true"; pattern = "compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence=True"; description = "Collection bundle runtime execution evidence must remain false; only validated external runtime proof can promote." }
  [pscustomobject]@{ id = "collection-bundle-performs-publish-true"; pattern = "compatibleHostRuntimeProofCollectionBundlePerformsPublish=True"; description = "Collection bundles must not perform package publication." }
  [pscustomobject]@{ id = "collection-bundle-approves-release-true"; pattern = "compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease=True"; description = "Collection bundles must not approve public release." }
  [pscustomobject]@{ id = "can-publish-publicly-true"; pattern = "canPublishPublicly=true"; description = "Release-facing docs must not show publication approval as current state; only owner input requirements may mention this." }
  [pscustomobject]@{ id = "can-close-release-issue-true"; pattern = "canCloseReleaseIssue=true"; description = "Release-facing docs must not show issue closure readiness as current state; only real post-publish proof requirements may mention this." }
  [pscustomobject]@{ id = "classification-passed-without-boundary"; pattern = "Classification Passed=True"; description = "Classification Passed=True must appear only as expected real-log evidence or guarded owner-action guidance." }
  [pscustomobject]@{ id = "yolovision-passed-without-boundary"; pattern = "YoloVision Passed=True"; description = "YoloVision Passed=True must appear only as expected real-log evidence or guarded owner-action guidance." }
  [pscustomobject]@{ id = "yolodet-project-name"; pattern = "YoloDet"; description = "YoloDet must not reappear as a current sample project name; use YoloVision for the unified YOLO-family sample." }
  [pscustomobject]@{ id = "yolodet-project-file"; pattern = "YoloDet.csproj"; description = "YoloDet.csproj must not reappear; the unified YOLO-family sample project is YoloVision.csproj." }
  [pscustomobject]@{ id = "yolodet-sample-path"; pattern = "samples/YoloDet"; description = "samples/YoloDet must not reappear as a live sample path; use samples/YoloVision." }
  [pscustomobject]@{ id = "yolodet-sample-path-windows"; pattern = "samples\\YoloDet"; description = "samples\\YoloDet must not reappear as a live sample path; use samples\\YoloVision." }
)

$scanTargets = @(
  "README.md",
  "README.zh-CN.md",
  "docs\index.md",
  "docs\toc.yml",
  "docs\articles\zh-cn",
  "samples",
  "tests\JYPPX.ProjectQuality.Tests",
  "artifacts\final-release\release-owner-decision-template.md",
  "artifacts\final-release\release-owner-decision-record.md",
  "artifacts\final-release\release-evidence-bundle.md",
  "artifacts\final-release\release-package-proof-bundle.md",
  "artifacts\final-release\docs-publish-readiness-bundle.md",
  "artifacts\final-release\release-owner-approval-input-template.md",
  "artifacts\final-release\release-owner-approval-input-validation.md",
  "artifacts\final-release\external-runtime-proof-record-template.md",
  "artifacts\final-release\external-runtime-proof-validation.md",
  "artifacts\final-release\release-publish-execution-checklist.md",
  "artifacts\final-release\post-publish-verification-record-template.md",
  "artifacts\final-release\post-publish-verification-validation.md",
  "artifacts\final-release\release-promotion-issue-record.md",
  "artifacts\linux-dry-run"
)

$files = New-Object System.Collections.Generic.List[string]
foreach ($target in $scanTargets) {
  $path = Join-Path $RepositoryRoot $target
  if (Test-Path -LiteralPath $path -PathType Leaf) {
    $files.Add($path)
  }
  elseif (Test-Path -LiteralPath $path -PathType Container) {
    Get-ChildItem -LiteralPath $path -Recurse -File -Include *.md,*.yml,*.ps1,*.cs,*.csproj,*.json |
      Where-Object { $_.FullName -notlike "*\docs\_site\*" } |
      ForEach-Object { $files.Add($_.FullName) }
  }
}

$findings = New-Object System.Collections.Generic.List[object]
foreach ($file in ($files | Sort-Object -Unique)) {
  $lines = @(Get-Content -LiteralPath $file -Encoding utf8)
  $relativePath = ConvertTo-RelativePath -Path $file
  for ($i = 0; $i -lt $lines.Count; $i++) {
    foreach ($rule in $patterns) {
      if (-not $lines[$i].Contains($rule.pattern, [System.StringComparison]::Ordinal)) {
        continue
      }

      $leadingContext = ""
      if ($rule.id -in @("classification-passed-without-boundary", "yolovision-passed-without-boundary")) {
        $contextStart = [Math]::Max(0, $i - 10)
        $leadingContext = ($lines[$contextStart..$i] -join "`n")
      }

      if (-not (Test-AllowedClaimContext -RuleId $rule.id -RelativePath $relativePath -Line $lines[$i] -LeadingContext $leadingContext)) {
        $findings.Add([pscustomobject]@{
            ruleId = $rule.id
            pattern = $rule.pattern
            description = $rule.description
            file = $relativePath
            line = $i + 1
            text = $lines[$i].Trim()
          })
      }
    }
  }
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "stale-release-claims-audit.json"
$markdownPath = Join-Path $outputRoot "stale-release-claims-audit.md"

$publicRules = @(
  foreach ($rule in $patterns) {
    switch ($rule.id) {
      "yolodet-project-name" {
        [pscustomobject]@{
          id = "retired-sample-project-name"
          pattern = "<retired-sample-name>"
          description = "The retired detection-only sample name must not reappear as a current project identity."
        }
        break
      }
      "yolodet-project-file" {
        [pscustomobject]@{
          id = "retired-sample-project-file"
          pattern = "<retired-sample-project-file>"
          description = "The retired detection-only project file must not reappear."
        }
        break
      }
      "yolodet-sample-path" {
        [pscustomobject]@{
          id = "retired-sample-path"
          pattern = "<retired-sample-path>"
          description = "The retired detection-only sample path must not reappear."
        }
        break
      }
      "yolodet-sample-path-windows" {
        [pscustomobject]@{
          id = "retired-sample-path-windows"
          pattern = "<retired-sample-path-windows>"
          description = "The retired Windows sample path must not reappear."
        }
        break
      }
      default {
        $rule
      }
    }
  }
)

[pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  findingCount = $findings.Count
  scannedFileCount = @($files | Sort-Object -Unique).Count
  rules = $publicRules
  findings = @($findings.ToArray())
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$linesOut = New-Object System.Collections.Generic.List[string]
$linesOut.Add("# Stale Release Claims Audit")
$linesOut.Add("")
$linesOut.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$linesOut.Add("")
$linesOut.Add("- scanned files: $(@($files | Sort-Object -Unique).Count)")
$linesOut.Add("- findings: $($findings.Count)")
$linesOut.Add("")
$linesOut.Add("## Rules")
$linesOut.Add("")
$linesOut.Add("| Rule | Pattern | Description |")
$linesOut.Add("| --- | --- | --- |")
foreach ($rule in $publicRules) {
  $linesOut.Add("| $($rule.id) | ``$($rule.pattern)`` | $(ConvertTo-MarkdownCell $rule.description) |")
}
$linesOut.Add("")
$linesOut.Add("## Findings")
$linesOut.Add("")
if ($findings.Count -eq 0) {
  $linesOut.Add("- none")
}
else {
  $linesOut.Add("| Rule | File | Line | Text |")
  $linesOut.Add("| --- | --- | ---: | --- |")
  foreach ($finding in $findings) {
    $linesOut.Add("| $($finding.ruleId) | ``$($finding.file)`` | $($finding.line) | $(ConvertTo-MarkdownCell $finding.text) |")
  }
}

$linesOut | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Stale release claims audit written to $jsonPath"
Write-Host "Stale release claims audit written to $markdownPath"

if ($findings.Count -gt 0) {
  $message = "Found $($findings.Count) stale or over-claimed release statement(s)."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
