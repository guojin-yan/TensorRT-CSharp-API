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
    routeId = "github-full-dependency-package"
    routeState = "blocked-owner-public-package-runtime-evidence-required"
    packageSource = "GitHub Release asset that carries managed package, native bridge, and TensorRT/CUDA/cuDNN runtime dependency bundle."
    intendedUse = "Full dependency route for owners who want a single downloadable package set with large runtime assets."
    requiredArtifacts = @(
      "public GitHub Release URL",
      "release asset download transcript",
      "full dependency package file",
      "extraction or install transcript",
      "clean external consumer project",
      "runtime asset copy log",
      "consumer restore/build/run transcript",
      "smoke output JSON/log"
    )
    requiredHashes = @(
      "GitHub release asset SHA256",
      "managed package SHA256",
      "native bridge SHA256",
      "runtime dependency bundle SHA256",
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
    cleanConsumerCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\New-PackageConsumerExternalSmokeScaffold.ps1; dotnet restore --source <public-github-package-source>; dotnet build -c Release; dotnet run -c Release -- --runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22"
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
    firstOwnerCommand = "Download the public GitHub Release asset into a clean directory outside the repository and record URL, SHA256, install log, and smoke log."
    publicSourceRequirement = "Must reference a public GitHub Release channel and immutable asset URL."
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
    routeId = "nuget-core-api-plus-bridge-package"
    routeState = "blocked-owner-public-package-runtime-evidence-required"
    packageSource = "Public NuGet package for the C# API and C++ bridge; TensorRT/CUDA/cuDNN remain machine-installed prerequisites."
    intendedUse = "Small NuGet route for normal package consumers who install NVIDIA dependencies separately."
    requiredArtifacts = @(
      "public NuGet package URL",
      "NuGet restore transcript",
      "managed package metadata",
      "native bridge asset list",
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
    cleanConsumerCommand = "dotnet new console -n TensorRtSharpPublicConsumer; dotnet add package JYPPX.TensorRT.CSharp.API --version <published-version> --source <public-nuget-source>; dotnet restore; dotnet build -c Release; dotnet run -c Release -- --runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22"
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
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
  requiredDecision = "Owner must choose one public package consumption route and provide real external consumer logs, hashes, host metadata, package source, and reviewer decision."
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  boundary = "This plan is an owner action map only. It does not publish packages, does not run a consumer, and does not promote package consumer runtime evidence."
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
