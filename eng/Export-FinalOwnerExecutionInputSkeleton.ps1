[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)

  $directory = Split-Path -Parent $LiteralPath
  if ([string]::IsNullOrWhiteSpace($directory)) {
    $directory = "."
  }
  New-Item -ItemType Directory -Path $directory -Force | Out-Null

  $lines = New-Object System.Collections.Generic.List[string]
  foreach ($item in @($InputObject)) {
    if ($null -eq $item) {
      $lines.Add("") | Out-Null
    }
    elseif ($item -is [string]) {
      $lines.Add($item) | Out-Null
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { $lines.Add([string]$child) | Out-Null }
    }
    else {
      $lines.Add([string]$item) | Out-Null
    }
  }

  $content = (($lines.ToArray() -join [Environment]::NewLine) + [Environment]::NewLine)
  $fileName = Split-Path -Leaf $LiteralPath
  $tempPath = Join-Path $directory (".{0}.{1}.tmp" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  $backupPath = Join-Path $directory (".{0}.{1}.bak" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  try {
    [System.IO.File]::WriteAllText($tempPath, $content, $script:utf8)
    for ($attempt = 1; $attempt -le 10; $attempt++) {
      try {
        if (Test-Path -LiteralPath $LiteralPath -PathType Leaf) {
          [System.IO.File]::Replace($tempPath, $LiteralPath, $backupPath)
          Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
        }
        else {
          [System.IO.File]::Move($tempPath, $LiteralPath)
        }

        return
      }
      catch {
        if ($attempt -eq 10) { throw }
        Start-Sleep -Milliseconds ([Math]::Min(250, 25 * $attempt))
      }
    }
  }
  finally {
    if (Test-Path -LiteralPath $tempPath -PathType Leaf) {
      Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $backupPath -PathType Leaf) {
      Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
    }
  }
}

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-OwnerField {
  param(
    [string]$Id,
    [string]$Group,
    [string]$FieldPath,
    [string]$Kind,
    [string]$Description,
    [string[]]$AcceptedEvidence,
    [string[]]$Rejects
  )

  [pscustomobject]@{
    id = $Id
    group = $Group
    fieldPath = $FieldPath
    kind = $Kind
    required = $true
    value = "<owner-real-input-required>"
    placeholder = $true
    status = "missing owner input"
    readyForImport = $false
    ownerActionRequired = $true
    description = $Description
    acceptedEvidence = @($AcceptedEvidence)
    rejects = @($Rejects)
  }
}

function New-FieldGroup {
  param([string]$Id, [string]$Title, [object[]]$Fields)
  [pscustomobject]@{
    id = $Id
    title = $Title
    fieldCount = @($Fields).Count
    missingFieldCount = @($Fields).Count
    placeholderFieldCount = @($Fields).Count
    readyFieldCount = 0
    fields = @($Fields)
    ownerActionRequired = $true
    readyForImport = $false
  }
}

$oneScreen = Read-JsonOrNull "artifacts\final-release\final-owner-execution-one-screen-pack.json"
$oneScreenValidation = Read-JsonOrNull "artifacts\final-release\final-owner-execution-one-screen-pack-validation.json"
$ownerRuntimeSmokeFieldAlignment = Read-JsonOrNull "artifacts\final-release\package-consumer-owner-runtime-smoke-field-alignment.json"
$ownerRuntimeSmokeFieldAlignmentValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-owner-runtime-smoke-field-alignment-validation.json"
$gaps = @(Convert-ToArray (Get-PropertyOrDefault -Object $oneScreen -Name "ownerInputGapTable" -DefaultValue @()))
$ownerRuntimeSmokeFieldAlignmentState = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "alignmentState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment")
$ownerRuntimeSmokeFieldAlignmentValidationState = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignmentValidation -Name "validationState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment-validation")
$ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "runtimeSmokeStatus" -DefaultValue "Smoke=missing")
$ownerRuntimeSmokeFieldAlignmentFieldCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "fieldCount" -DefaultValue 0)
$ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "missingRequiredFieldCount" -DefaultValue -1)
$ownerRuntimeSmokeFieldAlignmentFailedBlockerCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignmentValidation -Name "failedBlockerCount" -DefaultValue -1)

$commonRejects = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "pre-publish smoke reused as post-publish proof",
  "build-only",
  "dependency-probe",
  "dashboard",
  "runbook",
  "draft",
  "candidate",
  "template",
  "Skipped=True"
)

$groups = @(
  New-FieldGroup -Id "clean-consumer" -Title "Clean external package consumer" -Fields @(
    New-OwnerField -Id "clean-consumer-project-root" -Group "clean-consumer" -FieldPath "cleanConsumer.projectRoot" -Kind "path" -Description "Repository-external clean consumer project root." -AcceptedEvidence @("absolute or repository-external path created by Owner") -Rejects $commonRejects
    New-OwnerField -Id "clean-consumer-restore-log-path" -Group "clean-consumer" -FieldPath "cleanConsumer.restoreLog.path" -Kind "path" -Description "Restore transcript from clean external consumer." -AcceptedEvidence @("existing restore log path") -Rejects $commonRejects
    New-OwnerField -Id "clean-consumer-build-log-path" -Group "clean-consumer" -FieldPath "cleanConsumer.buildLog.path" -Kind "path" -Description "Build transcript from clean external consumer." -AcceptedEvidence @("existing build log path") -Rejects $commonRejects
  )
  New-FieldGroup -Id "package-source" -Title "Package source and package identity" -Fields @(
    New-OwnerField -Id "package-source-url" -Group "package-source" -FieldPath "packageSource.url" -Kind "url" -Description "Owner-approved package source URL." -AcceptedEvidence @("public or owner-approved package source URL") -Rejects $commonRejects
    New-OwnerField -Id "managed-package-id" -Group "package-source" -FieldPath "managedPackage.id" -Kind "identity" -Description "Managed NuGet package id." -AcceptedEvidence @("published managed package id") -Rejects $commonRejects
    New-OwnerField -Id "managed-package-version" -Group "package-source" -FieldPath "managedPackage.version" -Kind "identity" -Description "Managed NuGet package version." -AcceptedEvidence @("published managed package version") -Rejects $commonRejects
    New-OwnerField -Id "managed-package-sha256" -Group "package-source" -FieldPath "managedPackage.sha256" -Kind "sha256" -Description "SHA256 of the managed package file." -AcceptedEvidence @("64-character SHA256") -Rejects $commonRejects
    New-OwnerField -Id "runtime-package-id" -Group "package-source" -FieldPath "runtimePackage.id" -Kind "identity" -Description "Runtime NuGet package id." -AcceptedEvidence @("runtime package id") -Rejects $commonRejects
    New-OwnerField -Id "runtime-package-version" -Group "package-source" -FieldPath "runtimePackage.version" -Kind "identity" -Description "Runtime NuGet package version." -AcceptedEvidence @("runtime package version") -Rejects $commonRejects
    New-OwnerField -Id "runtime-package-key" -Group "package-source" -FieldPath "runtimePackage.key" -Kind "identity" -Description "Runtime package key such as win-x64-trt11.0-cuda13.2-cudnn9.22." -AcceptedEvidence @("runtime package key") -Rejects $commonRejects
    New-OwnerField -Id "runtime-package-sha256" -Group "package-source" -FieldPath "runtimePackage.sha256" -Kind "sha256" -Description "SHA256 of the runtime package file." -AcceptedEvidence @("64-character SHA256") -Rejects $commonRejects
  )
  New-FieldGroup -Id "native-assets" -Title "Native asset listing" -Fields @(
    New-OwnerField -Id "native-asset-listing-path" -Group "native-assets" -FieldPath "nativeAssetListing.path" -Kind "path" -Description "Native asset listing captured from package consumer." -AcceptedEvidence @("existing native asset listing file") -Rejects $commonRejects
    New-OwnerField -Id "native-asset-listing-sha256" -Group "native-assets" -FieldPath "nativeAssetListing.sha256" -Kind "sha256" -Description "SHA256 for native asset listing file." -AcceptedEvidence @("64-character SHA256") -Rejects $commonRejects
  )
  New-FieldGroup -Id "runtime-logs" -Title "Restore build run logs and runtime status" -Fields @(
    New-OwnerField -Id "stdout-path" -Group "runtime-logs" -FieldPath "runtimeSmoke.stdoutPath" -Kind "path" -Description "Runtime smoke stdout log." -AcceptedEvidence @("existing stdout file") -Rejects $commonRejects
    New-OwnerField -Id "stderr-path" -Group "runtime-logs" -FieldPath "runtimeSmoke.stderrPath" -Kind "path" -Description "Runtime smoke stderr log." -AcceptedEvidence @("existing stderr file") -Rejects $commonRejects
    New-OwnerField -Id "merged-transcript-path" -Group "runtime-logs" -FieldPath "runtimeSmoke.mergedTranscriptPath" -Kind "path" -Description "Merged restore/build/run transcript." -AcceptedEvidence @("existing merged transcript file") -Rejects $commonRejects
    New-OwnerField -Id "stdout-sha256" -Group "runtime-logs" -FieldPath "runtimeSmoke.stdoutSha256" -Kind "sha256" -Description "SHA256 of stdout log." -AcceptedEvidence @("64-character SHA256") -Rejects $commonRejects
    New-OwnerField -Id "stderr-sha256" -Group "runtime-logs" -FieldPath "runtimeSmoke.stderrSha256" -Kind "sha256" -Description "SHA256 of stderr log." -AcceptedEvidence @("64-character SHA256") -Rejects $commonRejects
    New-OwnerField -Id "merged-transcript-sha256" -Group "runtime-logs" -FieldPath "runtimeSmoke.mergedTranscriptSha256" -Kind "sha256" -Description "SHA256 of merged transcript." -AcceptedEvidence @("64-character SHA256") -Rejects $commonRejects
    New-OwnerField -Id "exit-code" -Group "runtime-logs" -FieldPath "runtimeSmoke.exitCode" -Kind "integer" -Description "Runtime command exit code." -AcceptedEvidence @("0 from real external runtime command") -Rejects $commonRejects
    New-OwnerField -Id "smoke-status" -Group "runtime-logs" -FieldPath "runtimeSmoke.smokeStatus" -Kind "status" -Description "Runtime smoke status." -AcceptedEvidence @("passed from real external runtime command") -Rejects $commonRejects
  )
  New-FieldGroup -Id "host-metadata" -Title "Host metadata" -Fields @(
    New-OwnerField -Id "host-os" -Group "host-metadata" -FieldPath "host.os" -Kind "identity" -Description "Operating system." -AcceptedEvidence @("real owner host metadata") -Rejects $commonRejects
    New-OwnerField -Id "host-arch" -Group "host-metadata" -FieldPath "host.arch" -Kind "identity" -Description "CPU architecture." -AcceptedEvidence @("real owner host metadata") -Rejects $commonRejects
    New-OwnerField -Id "host-rid" -Group "host-metadata" -FieldPath "host.rid" -Kind "identity" -Description "Runtime identifier." -AcceptedEvidence @("real owner host metadata") -Rejects $commonRejects
    New-OwnerField -Id "gpu-name" -Group "host-metadata" -FieldPath "host.gpuName" -Kind "identity" -Description "GPU name." -AcceptedEvidence @("real GPU metadata") -Rejects $commonRejects
    New-OwnerField -Id "nvidia-driver" -Group "host-metadata" -FieldPath "host.nvidiaDriver" -Kind "identity" -Description "NVIDIA driver version." -AcceptedEvidence @("real driver version") -Rejects $commonRejects
    New-OwnerField -Id "cuda-runtime-toolkit" -Group "host-metadata" -FieldPath "host.cudaRuntimeToolkit" -Kind "identity" -Description "CUDA runtime/toolkit version." -AcceptedEvidence @("real CUDA runtime/toolkit version") -Rejects $commonRejects
    New-OwnerField -Id "tensorrt-version" -Group "host-metadata" -FieldPath "host.tensorrt" -Kind "identity" -Description "TensorRT version." -AcceptedEvidence @("real TensorRT version") -Rejects $commonRejects
    New-OwnerField -Id "cudnn-version" -Group "host-metadata" -FieldPath "host.cudnn" -Kind "identity" -Description "cuDNN version." -AcceptedEvidence @("real cuDNN version") -Rejects $commonRejects
  )
  New-FieldGroup -Id "owner-review" -Title "Owner review" -Fields @(
    New-OwnerField -Id "owner-name" -Group "owner-review" -FieldPath "owner.name" -Kind "identity" -Description "Owner reviewer name." -AcceptedEvidence @("real owner reviewer") -Rejects $commonRejects
    New-OwnerField -Id "owner-machine" -Group "owner-review" -FieldPath "owner.machine" -Kind "identity" -Description "Owner machine name." -AcceptedEvidence @("real owner machine") -Rejects $commonRejects
    New-OwnerField -Id "reviewed-at-utc" -Group "owner-review" -FieldPath "owner.reviewedAtUtc" -Kind "datetime" -Description "Owner review timestamp in UTC." -AcceptedEvidence @("ISO-8601 UTC timestamp") -Rejects $commonRejects
    New-OwnerField -Id "owner-note" -Group "owner-review" -FieldPath "owner.note" -Kind "text" -Description "Owner review note." -AcceptedEvidence @("owner review note") -Rejects $commonRejects
  )
  New-FieldGroup -Id "post-publish" -Title "Post-publish proof" -Fields @(
    New-OwnerField -Id "post-publish-downloaded-package-hash" -Group "post-publish" -FieldPath "postPublish.downloadedPackageSha256" -Kind "sha256" -Description "SHA256 of package downloaded from public channel after publish." -AcceptedEvidence @("64-character SHA256 from post-publish package") -Rejects $commonRejects
    New-OwnerField -Id "post-publish-proof-log-path" -Group "post-publish" -FieldPath "postPublish.proofLogPath" -Kind "path" -Description "Post-publish clean consumer proof log." -AcceptedEvidence @("existing post-publish proof log") -Rejects $commonRejects
    New-OwnerField -Id "post-publish-proof-log-sha256" -Group "post-publish" -FieldPath "postPublish.proofLogSha256" -Kind "sha256" -Description "SHA256 of post-publish proof log." -AcceptedEvidence @("64-character SHA256") -Rejects $commonRejects
  )
  New-FieldGroup -Id "dual-package-route-proof" -Title "Dual-package route proof" -Fields @(
    New-OwnerField -Id "dual-package-nuget-owner-authorization-url" -Group "dual-package-route-proof" -FieldPath "dualPackageRoutes.nugetSmallBridgeCore.ownerAuthorizationUrl" -Kind "url" -Description "Owner authorization record for the NuGet small core/bridge route." -AcceptedEvidence @("owner-approved public NuGet publish authorization URL or signed record") -Rejects $commonRejects
    New-OwnerField -Id "dual-package-nuget-public-download-url" -Group "dual-package-route-proof" -FieldPath "dualPackageRoutes.nugetSmallBridgeCore.publicPackageDownloadUrl" -Kind "url" -Description "Public NuGet package download URL captured after publication." -AcceptedEvidence @("public NuGet package URL from the published channel") -Rejects $commonRejects
    New-OwnerField -Id "dual-package-nuget-clean-consumer-log-path" -Group "dual-package-route-proof" -FieldPath "dualPackageRoutes.nugetSmallBridgeCore.cleanConsumerProofLogPath" -Kind "path" -Description "Repository-external clean consumer proof log for the NuGet small core/bridge route." -AcceptedEvidence @("existing clean consumer restore/build/run log from public NuGet packages") -Rejects $commonRejects
    New-OwnerField -Id "dual-package-nuget-post-publish-proof-log-sha256" -Group "dual-package-route-proof" -FieldPath "dualPackageRoutes.nugetSmallBridgeCore.postPublishProofLogSha256" -Kind "sha256" -Description "SHA256 of the NuGet-route post-publish clean consumer proof log." -AcceptedEvidence @("64-character SHA256 from the real post-publish proof log") -Rejects $commonRejects
    New-OwnerField -Id "dual-package-github-owner-authorization-url" -Group "dual-package-route-proof" -FieldPath "dualPackageRoutes.githubPackagesFullRuntime.ownerAuthorizationUrl" -Kind "url" -Description "Owner authorization record for the GitHub Packages full runtime route." -AcceptedEvidence @("owner-approved GitHub Packages publish authorization URL or signed record") -Rejects $commonRejects
    New-OwnerField -Id "dual-package-github-restore-source-url" -Group "dual-package-route-proof" -FieldPath "dualPackageRoutes.githubPackagesFullRuntime.restoreSourceUrl" -Kind "url" -Description "GitHub Packages restore source URL used by the external consumer." -AcceptedEvidence @("credentialed GitHub Packages source URL from the real restore") -Rejects $commonRejects
    New-OwnerField -Id "dual-package-github-runtime-dll-resolution-report-path" -Group "dual-package-route-proof" -FieldPath "dualPackageRoutes.githubPackagesFullRuntime.runtimeDllResolutionReportPath" -Kind "path" -Description "Runtime DLL resolution report from the GitHub Packages full runtime route." -AcceptedEvidence @("existing runtime DLL resolution report from external consumer") -Rejects $commonRejects
    New-OwnerField -Id "dual-package-github-clean-runtime-smoke-log-sha256" -Group "dual-package-route-proof" -FieldPath "dualPackageRoutes.githubPackagesFullRuntime.cleanRuntimeSmokeLogSha256" -Kind "sha256" -Description "SHA256 of the GitHub Packages full runtime clean smoke log." -AcceptedEvidence @("64-character SHA256 from the real GitHub Packages runtime smoke log") -Rejects $commonRejects
  )
  New-FieldGroup -Id "release-close" -Title "Rollback review and final close" -Fields @(
    New-OwnerField -Id "rollback-review" -Group "release-close" -FieldPath "rollback.review" -Kind "text" -Description "Rollback review and decision." -AcceptedEvidence @("owner rollback review") -Rejects $commonRejects
    New-OwnerField -Id "final-close-decision" -Group "release-close" -FieldPath "finalClose.decision" -Kind "decision" -Description "Final release close decision." -AcceptedEvidence @("owner final close approval") -Rejects $commonRejects
  )
  New-FieldGroup -Id "strict-validators" -Title "Strict validator chain" -Fields @(
    New-OwnerField -Id "strict-validator-output-path" -Group "strict-validators" -FieldPath "strictValidators.outputPath" -Kind "path" -Description "Accepted strict validator output path." -AcceptedEvidence @("existing strict validator output") -Rejects $commonRejects
    New-OwnerField -Id "strict-validator-output-sha256" -Group "strict-validators" -FieldPath "strictValidators.outputSha256" -Kind "sha256" -Description "SHA256 of strict validator output." -AcceptedEvidence @("64-character SHA256") -Rejects $commonRejects
    New-OwnerField -Id "strict-validator-chain-state" -Group "strict-validators" -FieldPath "strictValidators.chainState" -Kind "status" -Description "Strict validator chain state." -AcceptedEvidence @("all required strict validators passed") -Rejects $commonRejects
  )
)

$fields = @($groups | ForEach-Object { $_.fields })

$record = [pscustomobject]@{
  recordKind = "final-owner-execution-input-skeleton"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  skeletonState = "blocked-final-owner-real-input-required"
  sourceOneScreenPackState = [string](Get-PropertyOrDefault -Object $oneScreen -Name "packState" -DefaultValue "missing-final-owner-execution-one-screen-pack")
  sourceOneScreenPackValidationState = [string](Get-PropertyOrDefault -Object $oneScreenValidation -Name "validationState" -DefaultValue "missing-final-owner-execution-one-screen-pack-validation")
  sourceOwnerInputGapCount = $gaps.Count
  packageConsumerOwnerRuntimeSmokeFieldAlignmentState = $ownerRuntimeSmokeFieldAlignmentState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState = $ownerRuntimeSmokeFieldAlignmentValidationState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = $ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount = $ownerRuntimeSmokeFieldAlignmentFieldCount
  packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = $ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount = $ownerRuntimeSmokeFieldAlignmentFailedBlockerCount
  fieldGroupCount = $groups.Count
  requiredFieldCount = $fields.Count
  missingFieldCount = $fields.Count
  placeholderFieldCount = $fields.Count
  readyForImportFieldCount = 0
  fieldGroups = @($groups)
  ownerInputGapIds = @($gaps | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  forbiddenSubstitutes = @($commonRejects)
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  nonSubstituteProofKinds = @($commonRejects + @("final owner execution input skeleton", "owner fillable input skeleton", "placeholder owner input", "dual package route proof", "dual package final close lanes"))
  sourceArtifacts = @(
    "artifacts/final-release/final-owner-execution-one-screen-pack.json",
    "artifacts/final-release/final-owner-execution-one-screen-pack-validation.json",
    "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json",
    "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json"
  )
  boundary = "Final Owner execution input skeleton is fillable owner input only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. All fields intentionally default to placeholder/missing until real Owner evidence is supplied and strict validators accept it."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-input-skeleton.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-input-skeleton.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 20)
$rows = foreach ($field in $fields) {
  "| ``$(ConvertTo-MarkdownCell $field.id)`` | ``$(ConvertTo-MarkdownCell $field.group)`` | ``$(ConvertTo-MarkdownCell $field.fieldPath)`` | ``$(ConvertTo-MarkdownCell $field.kind)`` | ``$($field.placeholder)`` | ``$($field.readyForImport)`` |"
}

$markdown = @"
# Final Owner Execution Input Skeleton

该 skeleton 把 `final-owner-execution-one-screen-pack` 的 Owner 缺口展开成可填写字段。所有字段默认 placeholder/missing，不能作为 proof、publish approval 或 close approval。

| Field | Value |
|---|---|
| skeletonState | ``$($record.skeletonState)`` |
| sourceOwnerInputGapCount | ``$($record.sourceOwnerInputGapCount)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentState | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentState)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount)`` |
| fieldGroupCount | ``$($record.fieldGroupCount)`` |
| requiredFieldCount | ``$($record.requiredFieldCount)`` |
| missingFieldCount | ``$($record.missingFieldCount)`` |
| placeholderFieldCount | ``$($record.placeholderFieldCount)`` |
| readyForImportFieldCount | ``$($record.readyForImportFieldCount)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Fields

| ID | Group | Field Path | Kind | Placeholder | Ready |
|---|---|---|---|---:|---:|
$($rows -join "`r`n")

## Boundary

$($record.boundary)
"@
Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown

Write-Host "Final owner execution input skeleton written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "SkeletonState=$($record.skeletonState) Fields=$($record.requiredFieldCount) Ready=0"
