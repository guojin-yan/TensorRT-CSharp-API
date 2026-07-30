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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$forbiddenSubstitutes = @(
  "project-reference substitute",
  "local package-feed substitute",
  "direct nupkg substitute",
  "template-only record",
  "dashboard-only record",
  "build-only report",
  "dry-run record",
  "screenshot-only evidence"
)

$routes = @(
  [pscustomobject]@{
    routeId = "github-release-managed-plus-bridge-assets"
    routeState = "blocked-owner-public-package-runtime-evidence-required"
    packageSource = "Public GitHub Release assets for the managed API package and the matching project-owned bridge package; NVIDIA TensorRT/CUDA/cuDNN/NVRTC libraries remain machine-installed prerequisites."
    intendedUse = "Release-asset route for consumers who want immutable GitHub URLs and SHA256 digests without redistributing NVIDIA vendor runtime libraries."
    requiredArtifacts = @(
      "public GitHub Release URL",
      "managed package Release asset URL and GitHub digest",
      "bridge package Release asset URL and GitHub digest",
      "matching managed and bridge nuspec repository commit",
      "verified public asset download transcript",
      "clean external consumer project",
      "bridge native asset copy log",
      "machine-installed NVIDIA dependency metadata",
      "consumer restore/build/run transcript",
      "smoke output JSON/log",
      "independent public Release bridge consumer validation report"
    )
    requiredHashes = @(
      "managed Release asset GitHub SHA256",
      "downloaded managed package SHA256",
      "bridge Release asset GitHub SHA256",
      "downloaded bridge package SHA256",
      "runtime consumer report SHA256",
      "runtime stdout and stderr SHA256",
      "smoke output log SHA256"
    )
    requiredEnvironmentMetadata = @(
      "Windows version",
      "GPU name",
      "NVIDIA driver version",
      "CUDA runtime version",
      "TensorRT version",
      "cuDNN version",
      ".NET SDK version"
    )
    cleanConsumerCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-PublicReleaseBridgePackageConsumer.ps1 -ManagedReleaseTag <managed-tag> -BridgeReleaseTag <bridge-tag> -SourceRuntimeKey win-x64-trt11.0-cuda13.2-cudnn9.22"
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicReleaseBridgePackageConsumer.ps1 -InputPath <public-release-consumer-report> -RequireReferencedFiles -Strict -FailOnNotEvidence"
    promotionRecordValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
    firstOwnerCommand = "Run the public Release bridge consumer outside the repository, verify both GitHub digests and package identities, then record machine-installed dependency metadata and the runtime smoke logs."
    publicSourceRequirement = "Must reference public immutable GitHub Release asset URLs for both managed and bridge packages; both nuspec files must name the formal repository and the same source commit; verified download staging must contain no locally built nupkg."
    downloadedAssetStagingPolicy = "A downloaded public Release nupkg may be placed in isolated NuGet restore staging only after URL, GitHub digest, package id, package version, and bridge-only content validation. PackageReference remains mandatory; direct file references remain forbidden."
    forbiddenSubstitutes = @($forbiddenSubstitutes)
    ownerInputArtifact = "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json"
    requiredRecord = "artifacts/final-release/package-consumer-runtime-proof-record.json"
    canPromotePackageConsumerRuntime = $false
    canPromoteRuntimeProof = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  },
  [pscustomobject]@{
    routeId = "nuget-managed-plus-bridge-packages"
    routeState = "blocked-owner-public-package-runtime-evidence-required"
    packageSource = "Public NuGet-compatible source for the managed C# API package and matching project-owned bridge package; TensorRT/CUDA/cuDNN/NVRTC remain machine-installed prerequisites."
    intendedUse = "Normal PackageReference route for consumers who install NVIDIA dependencies separately."
    requiredArtifacts = @(
      "public NuGet package URL",
      "NuGet restore transcript",
      "managed package metadata",
      "native bridge asset list",
      "matching managed and bridge nuspec repository commit",
      "clean external consumer project",
      "external NVIDIA runtime metadata",
      "consumer restore/build/run transcript",
      "smoke output JSON/log"
    )
    requiredHashes = @(
      "managed package SHA256",
      "native bridge SHA256",
      "runtime package key metadata hash",
      "smoke output log SHA256"
    )
    requiredEnvironmentMetadata = @(
      "Windows version",
      "GPU name",
      "NVIDIA driver version",
      "CUDA runtime version",
      "TensorRT version",
      "cuDNN version",
      ".NET SDK version",
      "PATH/library probing summary"
    )
    cleanConsumerCommand = "dotnet new console -n TensorRtSharpPublicConsumer; dotnet add package JYPPX.TensorRT.CSharp.API --version <managed-version> --source <public-nuget-source>; dotnet add package <bridge-package-id> --version <bridge-version> --source <public-nuget-source>; dotnet restore; dotnet build -c Release; dotnet run -c Release -- --runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22"
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
    promotionRecordValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
    firstOwnerCommand = "Create a clean consumer outside the repository, restore only from the public NuGet source, then record package metadata, environment metadata, exit code, and smoke log."
    publicSourceRequirement = "Must reference a public NuGet source and a published version."
    forbiddenSubstitutes = @($forbiddenSubstitutes)
    ownerInputArtifact = "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json"
    requiredRecord = "artifacts/final-release/package-consumer-runtime-proof-record.json"
    canPromotePackageConsumerRuntime = $false
    canPromoteRuntimeProof = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
)

$plan = [pscustomobject]@{
  recordKind = "package-consumer-dual-route-proof-plan"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  planState = "blocked-owner-public-package-runtime-evidence-required"
  routeCount = @($routes).Count
  routes = @($routes)
  sharedOwnerInputArtifact = "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json"
  sharedValidator = "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  requiredDecision = "Owner must choose one bridge-only public package consumption route and provide real external consumer logs, hashes, host metadata, package source, and reviewer decision. NVIDIA vendor runtime packages are not a supported route."
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  boundary = "This plan is an owner action map for managed plus bridge-only delivery. It does not publish packages, redistribute NVIDIA runtime libraries, run a consumer, or promote package consumer runtime evidence."
}

$jsonPath = Join-Path $OutputRoot "package-consumer-dual-route-proof-plan.json"
$markdownPath = Join-Path $OutputRoot "package-consumer-dual-route-proof-plan.md"
$plan | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$routeRows = foreach ($route in $routes) {
  "| ``$(ConvertTo-MarkdownCell $route.routeId)`` | $(ConvertTo-MarkdownCell $route.packageSource) | ``$(ConvertTo-MarkdownCell $route.validatorCommand)`` | ``False`` |"
}

$markdown = @"
# Package Consumer Dual Route Proof Plan

Generated at: ``$($plan.generatedAtUtc)``

## Summary

- planState: ``$($plan.planState)``
- routeCount: ``$($plan.routeCount)``
- sharedOwnerInputArtifact: ``$($plan.sharedOwnerInputArtifact)``
- sharedValidator: ``$($plan.sharedValidator)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromotePackageConsumerRuntime: ``False``

## Routes

| Route | Package Source | Validator | Can Publish |
| --- | --- | --- | --- |
$($routeRows -join "`r`n")

## Owner Decision

$($plan.requiredDecision)

## Boundary

$($plan.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Package consumer dual route proof plan written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PlanState=$($plan.planState) RouteCount=$($plan.routeCount)"
