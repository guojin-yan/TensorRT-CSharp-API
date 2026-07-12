[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$ManagedPackageId = "JYPPX.TensorRT.CSharp.API",
  [string]$ManagedPackageVersion = "<owner-fill-managed-package-version>",
  [string]$RuntimePackageId = "",
  [string]$RuntimePackageVersion = "<owner-fill-runtime-package-version>",
  [string]$PublicPackageSource = "<owner-fill-public-package-source-url-or-id>",
  [string]$RepositoryRoot
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

if ([string]::IsNullOrWhiteSpace($RuntimePackageId)) {
  $RuntimePackageId = "JYPPX.TensorRT.CSharp.API.runtime.$RuntimePackageKey"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path ([IO.Path]::GetTempPath()) "TensorRtSharp4.PackageConsumerSmoke.Template"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-IsInsideRepository {
  param([string]$Path)

  try {
    $repositoryFullPath = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $candidateFullPath = [IO.Path]::GetFullPath($Path).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    return $candidateFullPath.StartsWith($repositoryFullPath, [StringComparison]::OrdinalIgnoreCase)
  }
  catch {
    return $true
  }
}

function Test-PublicPackageSourceIsLocal {
  param([AllowNull()][object]$Value)

  $text = ([string]$Value).Trim()
  if (Test-IsPlaceholder -Value $text) {
    return $true
  }

  if ($text -match "^[a-zA-Z]:[\\/]" -or $text.StartsWith("\\", [StringComparison]::Ordinal) -or $text.StartsWith("./", [StringComparison]::Ordinal) -or $text.StartsWith("../", [StringComparison]::Ordinal)) {
    return $true
  }

  return $text.Contains("local", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase)
}

function Get-ProjectScan {
  param([string]$ProjectPath)

  $result = [ordered]@{
    projectPath = $ProjectPath
    projectExists = $false
    projectReferenceCount = 0
    usesProjectReference = $false
    usesLocalFeed = $false
    usesDirectNupkg = $false
    packageReferences = @()
    runtimePackageKey = $RuntimePackageKey
    canBePublicProof = $false
  }

  if (-not (Test-Path -LiteralPath $ProjectPath -PathType Leaf)) {
    return [pscustomobject]$result
  }

  $result.projectExists = $true
  $content = Get-Content -LiteralPath $ProjectPath -Raw -Encoding utf8
  $projectReferenceMatches = [regex]::Matches($content, "<ProjectReference\b", [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
  $packageReferenceMatches = [regex]::Matches($content, "<PackageReference\s+Include=""([^""]+)""\s+Version=""([^""]+)""", [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)

  $result.projectReferenceCount = $projectReferenceMatches.Count
  $result.usesProjectReference = $projectReferenceMatches.Count -gt 0
  $result.usesLocalFeed = $content.Contains("RestoreSources", [StringComparison]::OrdinalIgnoreCase) -and
    ($content.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or $content.Contains("local", [StringComparison]::OrdinalIgnoreCase))
  $result.usesDirectNupkg = $content.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase)
  $result.packageReferences = @($packageReferenceMatches | ForEach-Object {
    [pscustomobject]@{
      id = $_.Groups[1].Value
      version = $_.Groups[2].Value
    }
  })
  $result.canBePublicProof = $result.projectExists -and
    -not $result.usesProjectReference -and
    -not $result.usesLocalFeed -and
    -not $result.usesDirectNupkg -and
    -not (Test-IsPlaceholder -Value $ManagedPackageVersion) -and
    -not (Test-IsPlaceholder -Value $RuntimePackageVersion) -and
    -not (Test-PublicPackageSourceIsLocal -Value $PublicPackageSource)

  return [pscustomobject]$result
}

$outputFullPath = [IO.Path]::GetFullPath($OutputRoot)
$insideRepository = Test-IsInsideRepository -Path $outputFullPath
New-Item -ItemType Directory -Force -Path $outputFullPath | Out-Null

$projectPath = Join-Path $outputFullPath "TensorRtSharp.PackageConsumerSmoke.csproj"
$programPath = Join-Path $outputFullPath "Program.cs"
$readmePath = Join-Path $outputFullPath "README.md"

$projectXml = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="$ManagedPackageId" Version="$ManagedPackageVersion" />
    <PackageReference Include="$RuntimePackageId" Version="$RuntimePackageVersion" />
  </ItemGroup>
</Project>
"@

$programText = @"
using System;

string runtimePackageKey = "$RuntimePackageKey";
for (int i = 0; i < args.Length - 1; i++)
{
    if (string.Equals(args[i], "--runtime-package-key", StringComparison.OrdinalIgnoreCase))
    {
        runtimePackageKey = args[i + 1];
    }
}

Console.WriteLine($"TensorRtSharp package consumer smoke placeholder. runtimePackageKey={runtimePackageKey}");
Console.WriteLine("Replace this placeholder with the real TensorRT smoke entry once the public packages are available on the target host.");
"@

$readmeText = @"
# TensorRtSharp Package Consumer Smoke Scaffold

This scaffold is an owner-fill external consumer template. It must live outside the repository and must restore from the real public package source before it can be evidence.

## Command Shape

```powershell
dotnet restore "$projectPath" --source "$PublicPackageSource"
dotnet build "$projectPath" -c Release
dotnet run --project "$projectPath" -c Release -- --runtime-package-key $RuntimePackageKey
```

## Non-Proof Boundary

- This scaffold is not proof by itself.
- ProjectReference, local feed, and direct .nupkg references are not public package proof.
- The real proof record must capture package hashes, host metadata, smoke log SHA256, stdout/stderr summaries, and strict validator output.
"@

$projectXml | Set-Content -LiteralPath $projectPath -Encoding utf8
$programText | Set-Content -LiteralPath $programPath -Encoding utf8
$readmeText | Set-Content -LiteralPath $readmePath -Encoding utf8

$scan = Get-ProjectScan -ProjectPath $projectPath
$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$summaryPath = Join-Path $artifactRoot "package-consumer-external-smoke-scaffold.json"
$summaryMarkdownPath = Join-Path $artifactRoot "package-consumer-external-smoke-scaffold.md"

$summary = [pscustomobject]@{
  recordKind = "package-consumer-external-smoke-scaffold"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  outputRoot = $outputFullPath
  outputRootIsOutsideRepository = -not $insideRepository
  projectPath = $projectPath
  programPath = $programPath
  readmePath = $readmePath
  runtimePackageKey = $RuntimePackageKey
  managedPackageId = $ManagedPackageId
  managedPackageVersion = $ManagedPackageVersion
  runtimePackageId = $RuntimePackageId
  runtimePackageVersion = $RuntimePackageVersion
  publicPackageSource = $PublicPackageSource
  publicPackageSourceIsLocal = Test-PublicPackageSourceIsLocal -Value $PublicPackageSource
  scan = $scan
  isProof = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Scaffold and scan only. It cannot prove runtime execution, publish packages, or close the release issue."
}

$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $summaryPath -Encoding utf8

$markdown = @"
# Package Consumer External Smoke Scaffold

生成时间：$($summary.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| outputRoot | ``$($summary.outputRoot)`` |
| outputRootIsOutsideRepository | ``$($summary.outputRootIsOutsideRepository)`` |
| publicPackageSourceIsLocal | ``$($summary.publicPackageSourceIsLocal)`` |
| projectReferenceCount | ``$($summary.scan.projectReferenceCount)`` |
| usesLocalFeed | ``$($summary.scan.usesLocalFeed)`` |
| usesDirectNupkg | ``$($summary.scan.usesDirectNupkg)`` |
| canBePublicProof | ``$($summary.scan.canBePublicProof)`` |
| performsPublish | ``$($summary.performsPublish)`` |
| canPublishPublicly | ``$($summary.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($summary.canCloseReleaseIssue)`` |

## Safety Boundary

$($summary.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $summaryMarkdownPath -Encoding utf8

Write-Host "Package consumer external smoke scaffold written:"
Write-Host "  OutputRoot=$outputFullPath"
Write-Host "  Summary=$summaryPath"
Write-Host "  Markdown=$summaryMarkdownPath"
Write-Host "OutputRootIsOutsideRepository=$($summary.outputRootIsOutsideRepository) CanBePublicProof=$($summary.scan.canBePublicProof) IsProof=$($summary.isProof)"
