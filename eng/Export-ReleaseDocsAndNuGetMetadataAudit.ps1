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

function Resolve-RepositoryPath {
  param([string]$RelativePath)
  return Join-Path $RepositoryRoot $RelativePath
}

function ConvertTo-RelativePath {
  param([string]$Path)
  return $Path.Substring($RepositoryRoot.Length).TrimStart('\', '/')
}

function Read-TextOrEmpty {
  param([string]$RelativePath)
  $path = Resolve-RepositoryPath -RelativePath $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return ""
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8
}

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Resolve-RepositoryPath -RelativePath $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Test-TextContains {
  param([AllowNull()][string]$Text, [string]$Needle)
  if ([string]::IsNullOrWhiteSpace($Text)) {
    return $false
  }

  return $Text.Contains($Needle, [StringComparison]::OrdinalIgnoreCase)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-AuditItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Test-AllowedBoundaryContext {
  param([string]$Line)

  $lower = $Line.ToLowerInvariant()
  foreach ($marker in @(
      "not proof",
      "not runtime proof",
      "not post-publish",
      "not package-consumer",
      "does not",
      "cannot",
      "must not",
      "blocked",
      "requires",
      "required",
      "boundary",
      "forbidden",
      "old",
      "legacy",
      "stale",
      "不是",
      "不能",
      "不得",
      "不可",
      "不代表",
      "不允许",
      "避免",
      "禁止",
      "需要",
      "必须",
      "阻断",
      "边界",
      "旧",
      "过时"
    )) {
    if ($lower.Contains($marker)) {
      return $true
    }
  }

  return $false
}

$readme = Read-TextOrEmpty "README.md"
$readmeZh = Read-TextOrEmpty "README.zh-CN.md"
$samplesReadme = Read-TextOrEmpty "samples\README.md"
$runtimeReadme = Read-TextOrEmpty "pack\runtime\README.md"
$runtimeSplitReadme = Read-TextOrEmpty "pack\runtime-split\README.md"
$packageCsproj = Read-TextOrEmpty "pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj"
$directoryProps = Read-TextOrEmpty "Directory.Build.props"

$yoloMatrix = Read-JsonOrNull "samples\YoloVision\yolo-model-matrix.json"
$onnxToEngineMatrix = Read-JsonOrNull "samples\OnnxToEngine\trtexec-parity-matrix.json"
$tensorRtExecMatrix = Read-JsonOrNull "applications\TensorRtExec\tensor-rt-exec-trtexec-parity-matrix.json"
$splitRuntimeManifest = Read-JsonOrNull "pack\runtime-split\split-runtime-packages.manifest.json"
$publicDocsGate = Read-JsonOrNull "artifacts\final-release\public-docs-package-metadata-gate.json"

$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-AuditItem "readme-managed-package-id" (Test-TextContains $readme "JYPPX.TensorRT.CSharp.API") "blocker" "README.md must expose the managed package id.")) | Out-Null
$items.Add((New-AuditItem "readme-zh-managed-package-id" (Test-TextContains $readmeZh "JYPPX.TensorRT.CSharp.API") "blocker" "README.zh-CN.md must expose the managed package id.")) | Out-Null
$items.Add((New-AuditItem "readme-boundary-markers" ((Test-TextContains $readme "post-publish") -and (Test-TextContains $readme "package-consumer") -and (Test-TextContains $readme "not proof")) "blocker" "README.md must keep post-publish/package-consumer/not-proof boundaries visible.")) | Out-Null
$items.Add((New-AuditItem "readme-zh-boundary-markers" ((Test-TextContains $readmeZh "post-publish") -and (Test-TextContains $readmeZh "package-consumer") -and ((Test-TextContains $readmeZh "not proof") -or (Test-TextContains $readmeZh "不是 proof"))) "blocker" "README.zh-CN.md must keep post-publish/package-consumer/not-proof boundaries visible.")) | Out-Null
$items.Add((New-AuditItem "samples-readme-current-samples" ((Test-TextContains $samplesReadme "YoloVision") -and (Test-TextContains $samplesReadme "OnnxToEngine")) "blocker" "samples/README.md must name YoloVision and OnnxToEngine as current sample surfaces.")) | Out-Null
$runtimeDocsReady = (Test-TextContains $runtimeReadme "NuGet package") -and ((Test-TextContains $runtimeSplitReadme "split") -or (Test-TextContains $runtimeSplitReadme "Bridge"))
$items.Add((New-AuditItem "runtime-docs-dual-route" $runtimeDocsReady "blocker" "Runtime package docs must describe managed/runtime and split package delivery surfaces.")) | Out-Null

$yoloFamilies = @((Get-PropertyOrDefault -Object $yoloMatrix -Name "families" -DefaultValue @()) | ForEach-Object { [string]$_ })
$yoloTasks = @((Get-PropertyOrDefault -Object $yoloMatrix -Name "tasks" -DefaultValue @()) | ForEach-Object { [string]$_ })
$requiredYoloFamilies = @("yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov26", "custom")
$requiredYoloTasks = @("det", "cls", "seg", "obb", "pose", "sem")
$missingYoloFamilies = @($requiredYoloFamilies | Where-Object { $yoloFamilies -notcontains $_ })
$missingYoloTasks = @($requiredYoloTasks | Where-Object { $yoloTasks -notcontains $_ })
$items.Add((New-AuditItem "yolovision-family-matrix-complete" ($missingYoloFamilies.Count -eq 0) "blocker" ("Missing YOLO families: " + ($missingYoloFamilies -join ", ")))) | Out-Null
$items.Add((New-AuditItem "yolovision-task-matrix-complete" ($missingYoloTasks.Count -eq 0) "blocker" ("Missing YOLO tasks: " + ($missingYoloTasks -join ", ")))) | Out-Null
$items.Add((New-AuditItem "yolovision-non-proof-boundary" (Test-TextContains ([string](Get-PropertyOrDefault -Object $yoloMatrix -Name "proofBoundary" -DefaultValue "")) "not post-publish proof") "blocker" "YoloVision matrix must keep sample support separate from post-publish proof.")) | Out-Null

$onnxBoundary = [string](Get-PropertyOrDefault -Object $onnxToEngineMatrix -Name "proofBoundary" -DefaultValue "")
$onnxMarkers = @((Get-PropertyOrDefault -Object $onnxToEngineMatrix -Name "requiredBoundaryMarkers" -DefaultValue @()) | ForEach-Object { [string]$_ })
$items.Add((New-AuditItem "onnx-to-engine-boundary" ((Test-TextContains $onnxBoundary "not runtime proof") -and (Test-TextContains $onnxBoundary "not full trtexec replacement proof") -and ($onnxMarkers -contains "build-only")) "blocker" "OnnxToEngine parity matrix must remain build/report boundary evidence, not runtime or full trtexec replacement proof.")) | Out-Null

$tensorRtExecModes = @((Get-PropertyOrDefault -Object $tensorRtExecMatrix -Name "modes" -DefaultValue @()) | ForEach-Object { [string]$_ })
$tensorRtExecEntries = @((Get-PropertyOrDefault -Object $tensorRtExecMatrix -Name "entries" -DefaultValue @()))
$pluginEntry = @($tensorRtExecEntries | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "plugin-library-boundary" })
$items.Add((New-AuditItem "tensorrtexec-cli-winforms-modes" (($tensorRtExecModes -contains "CLI") -and ($tensorRtExecModes -contains "WinForms")) "blocker" "TensorRtExec parity matrix must include CLI and WinForms modes.")) | Out-Null
$pluginBoundary = if ($pluginEntry.Count -gt 0) { [string](Get-PropertyOrDefault -Object $pluginEntry[0] -Name "proofBoundary" -DefaultValue "") } else { "" }
$items.Add((New-AuditItem "tensorrtexec-plugin-boundary" ($pluginEntry.Count -gt 0 -and ((Test-TextContains $pluginBoundary "does not prove plugin library load") -or (Test-TextContains $pluginBoundary "do not prove plugin library load"))) "blocker" "TensorRtExec plugin path parity must remain diagnostic-only, not plugin load/register proof.")) | Out-Null

$items.Add((New-AuditItem "managed-pack-csproj-package-id" (Test-TextContains $packageCsproj "<PackageId>JYPPX.TensorRT.CSharp.API</PackageId>") "blocker" "Managed package project must set PackageId to JYPPX.TensorRT.CSharp.API.")) | Out-Null
$items.Add((New-AuditItem "managed-pack-csproj-readme-project-url" ((Test-TextContains $packageCsproj "<PackageReadmeFile>README.md</PackageReadmeFile>") -and (Test-TextContains $packageCsproj '<PackageProjectUrl>$(RepositoryUrl)</PackageProjectUrl>') -and (Test-TextContains $packageCsproj '<RepositoryUrl>$(RepositoryUrl)</RepositoryUrl>')) "blocker" "Managed package project must include readme, project URL, and repository URL metadata.")) | Out-Null
$items.Add((New-AuditItem "managed-pack-csproj-description" (Test-TextContains $packageCsproj "<Description>") "blocker" "Managed package project must include a description.")) | Out-Null
$items.Add((New-AuditItem "directory-build-repository-and-frameworks" ((Test-TextContains $directoryProps "https://github.com/guojin-yan/TensorRT-CSharp-API") -and (Test-TextContains $directoryProps "net8.0") -and (Test-TextContains $directoryProps "net10.0")) "blocker" "Directory.Build.props must carry repository URL and broad target framework metadata.")) | Out-Null

$splitPackages = @((Get-PropertyOrDefault -Object $splitRuntimeManifest -Name "packages" -DefaultValue @()))
$splitPackageIds = @($splitPackages | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "packageId" -DefaultValue "") } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique)
$splitRoles = @($splitPackages | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "role" -DefaultValue "") } | Sort-Object -Unique)
$splitTensorRtLines = @($splitPackages | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "tensorRtLine" -DefaultValue "") } | Sort-Object -Unique)
$splitCudaLines = @($splitPackages | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "cudaLine" -DefaultValue "") } | Sort-Object -Unique)
$missingSplitRoles = @(@("bridge", "cuda-cudnn", "tensorrt") | Where-Object { $splitRoles -notcontains $_ })
$missingTensorRtLines = @(@("8", "10", "11") | Where-Object { $splitTensorRtLines -notcontains $_ })
$missingCudaLines = @(@("11", "12", "13") | Where-Object { $splitCudaLines -notcontains $_ })
$invalidSplitPackageIds = @($splitPackages | Where-Object { -not (Test-TextContains ([string](Get-PropertyOrDefault -Object $_ -Name "packageId" -DefaultValue "")) "JYPPX.TensorRT.CSharp.API.Runtime.") })
$items.Add((New-AuditItem "runtime-split-package-roles" ($missingSplitRoles.Count -eq 0) "blocker" ("Missing split runtime roles: " + ($missingSplitRoles -join ", ")))) | Out-Null
$items.Add((New-AuditItem "runtime-split-version-lines" ($missingTensorRtLines.Count -eq 0 -and $missingCudaLines.Count -eq 0) "blocker" ("Missing TensorRT lines: " + ($missingTensorRtLines -join ", ") + "; missing CUDA lines: " + ($missingCudaLines -join ", ")))) | Out-Null
$items.Add((New-AuditItem "runtime-split-package-ids" ($splitPackages.Count -ge 10 -and $invalidSplitPackageIds.Count -eq 0) "blocker" "Split runtime package manifest must expose concrete JYPPX.TensorRT.CSharp.API.Runtime package ids.")) | Out-Null

$scanRoots = @("README.md", "README.zh-CN.md", "docs", "samples", "applications", "src", "pack", ".github")
$scanExtensions = @(".md", ".yml", ".yaml", ".json", ".props", ".targets", ".csproj", ".cs", ".ps1", ".xml", ".txt")
$scanFiles = New-Object System.Collections.Generic.List[string]
foreach ($root in $scanRoots) {
  $path = Resolve-RepositoryPath -RelativePath $root
  if (-not (Test-Path -LiteralPath $path)) {
    continue
  }

  if (Test-Path -LiteralPath $path -PathType Leaf) {
    $scanFiles.Add((Resolve-Path -LiteralPath $path).Path) | Out-Null
    continue
  }

  Get-ChildItem -LiteralPath $path -Recurse -File |
    Where-Object {
      $relative = (ConvertTo-RelativePath -Path $_.FullName).Replace('/', '\')
      $relative -notmatch "(^|\\)(bin|obj)(\\|$)" -and
      $relative -notlike "docs\_site\*" -and
      $scanExtensions -contains $_.Extension.ToLowerInvariant()
    } |
    ForEach-Object { $scanFiles.Add($_.FullName) | Out-Null }
}

$yoloDetBlockedMatches = New-Object System.Collections.Generic.List[object]
foreach ($file in @($scanFiles | Sort-Object -Unique)) {
  $relative = ConvertTo-RelativePath -Path $file
  $lines = Get-Content -LiteralPath $file -Encoding utf8
  for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = [string]$lines[$i]
    if ($line -match "\bYoloDet\b" -and -not (Test-AllowedBoundaryContext -Line $line)) {
      $yoloDetBlockedMatches.Add([pscustomobject]@{
        file = $relative
        line = $i + 1
        text = $line.Trim()
      }) | Out-Null
    }
  }
}
$items.Add((New-AuditItem "no-live-yolodet-name" ($yoloDetBlockedMatches.Count -eq 0) "blocker" "No live public docs/source/package metadata should use the old YoloDet name outside explicit boundary/regression context.")) | Out-Null

$publicDocsGateFailedBlockers = [int](Get-PropertyOrDefault -Object $publicDocsGate -Name "failedBlockerCount" -DefaultValue -1)
$items.Add((New-AuditItem "public-docs-gate-companion-passed" ($null -ne $publicDocsGate -and $publicDocsGateFailedBlockers -eq 0 -and -not [bool](Get-PropertyOrDefault -Object $publicDocsGate -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $publicDocsGate -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $publicDocsGate -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "The negative public docs/package metadata gate must be present and free of blocker findings.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$auditState = if ($failedBlockers.Count -eq 0) {
  "release-docs-and-nuget-metadata-audit-ready-non-proof"
}
else {
  "failed-release-docs-and-nuget-metadata-audit"
}

$record = [ordered]@{
  recordKind = "release-docs-and-nuget-metadata-audit"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditState = $auditState
  auditItemCount = $items.Count
  failedBlockerCount = $failedBlockers.Count
  scannedPathCount = @($scanFiles | Sort-Object -Unique).Count
  yoloDetBlockedMatchCount = $yoloDetBlockedMatches.Count
  yoloDetBlockedMatches = @($yoloDetBlockedMatches.ToArray())
  managedPackageId = "JYPPX.TensorRT.CSharp.API"
  requiredYoloFamilies = $requiredYoloFamilies
  requiredYoloTasks = $requiredYoloTasks
  splitRuntimePackageCount = $splitPackages.Count
  splitRuntimePackageIds = $splitPackageIds
  splitRuntimeRoles = $splitRoles
  splitRuntimeTensorRtLines = $splitTensorRtLines
  splitRuntimeCudaLines = $splitCudaLines
  publicDocsPackageMetadataGateState = [string](Get-PropertyOrDefault -Object $publicDocsGate -Name "gateState" -DefaultValue "missing-public-docs-package-metadata-gate")
  publicDocsPackageMetadataGateFailedBlockerCount = $publicDocsGateFailedBlockers
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  canPromotePublicProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  sourceArtifacts = @(
    "README.md",
    "README.zh-CN.md",
    "samples/README.md",
    "samples/YoloVision/yolo-model-matrix.json",
    "samples/OnnxToEngine/trtexec-parity-matrix.json",
    "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json",
    "pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj",
    "pack/runtime-split/split-runtime-packages.manifest.json",
    "artifacts/final-release/public-docs-package-metadata-gate.json"
  )
  safetyBoundary = "Release docs and NuGet metadata audit is a positive/negative documentation and package metadata gate only. It is not public package download proof, does not publish packages, does not download public packages, does not run runtime smoke, does not promote runtime/public/post-publish proof, and cannot close release issues."
}

$jsonPath = Join-Path $OutputRoot "release-docs-and-nuget-metadata-audit.json"
$markdownPath = Join-Path $OutputRoot "release-docs-and-nuget-metadata-audit.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$itemRows = foreach ($item in $items) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$yoloRows = foreach ($match in $yoloDetBlockedMatches) {
  "| ``$(ConvertTo-MarkdownCell $match.file)`` | ``$($match.line)`` | $(ConvertTo-MarkdownCell $match.text) |"
}

$markdown = @"
# Release Docs And NuGet Metadata Audit

Generated at: ``$($record.generatedAtUtc)``

## Summary

- recordKind: ``$($record.recordKind)``
- auditState: ``$($record.auditState)``
- auditItemCount: ``$($record.auditItemCount)``
- failedBlockerCount: ``$($record.failedBlockerCount)``
- scannedPathCount: ``$($record.scannedPathCount)``
- splitRuntimePackageCount: ``$($record.splitRuntimePackageCount)``
- publicDocsPackageMetadataGateState: ``$($record.publicDocsPackageMetadataGateState)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromoteRuntimeProof: ``False``

## Validation Items

| Id | Passed | Severity | Detail |
| --- | --- | --- | --- |
$($itemRows -join "`r`n")

## Blocked YoloDet Matches

| File | Line | Text |
| --- | ---: | --- |
$(if ($yoloDetBlockedMatches.Count -eq 0) { "| none |  |  |" } else { $yoloRows -join "`r`n" })

## Boundary

$($record.safetyBoundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release docs and NuGet metadata audit written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "AuditState=$($record.auditState) FailedBlockers=$($record.failedBlockerCount) Items=$($record.auditItemCount)"
