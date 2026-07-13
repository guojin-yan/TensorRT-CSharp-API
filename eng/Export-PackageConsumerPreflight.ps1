[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

function Resolve-RepoPath {
  param([string]$RelativePath)
  return Join-Path $RepositoryRoot $RelativePath
}

function Read-ProjectXml {
  param([string]$RelativePath)
  $path = Resolve-RepoPath $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Required project file is missing: $RelativePath"
  }

  [xml](Get-Content -LiteralPath $path -Raw -Encoding utf8)
}

function Get-FirstPropertyValue {
  param(
    [xml]$Project,
    [string]$Name
  )

  foreach ($propertyGroup in @($Project.Project.PropertyGroup)) {
    $node = $propertyGroup.SelectSingleNode($Name)
    if ($null -ne $node -and -not [string]::IsNullOrWhiteSpace($node.InnerText)) {
      return [string]$node.InnerText
    }
  }

  return ""
}

function Get-ProjectReferences {
  param([xml]$Project)
  $items = New-Object System.Collections.Generic.List[string]
  foreach ($itemGroup in @($Project.Project.ItemGroup)) {
    foreach ($reference in @($itemGroup.ProjectReference)) {
      $include = [string]$reference.Include
      if (-not [string]::IsNullOrWhiteSpace($include)) {
        $items.Add($include)
      }
    }
  }

  return @($items.ToArray())
}

$managedPackProject = "pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj"
$runtimePropsProject = "pack\runtime\Directory.Build.props"
$runtimeSplitPropsProject = "pack\runtime-split\Directory.Build.props"
$packageConsumerScript = "eng\Test-PackageConsumer.ps1"
$finalOwnerGate = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-owner-next-decision-gate.json"

$managed = Read-ProjectXml $managedPackProject
$runtimeProps = Read-ProjectXml $runtimePropsProject
$runtimeSplitProps = Read-ProjectXml $runtimeSplitPropsProject
$rootProps = Read-ProjectXml "Directory.Build.props"

$managedProjectReferences = @(Get-ProjectReferences -Project $managed)
$packageConsumerScriptText = Get-Content -LiteralPath (Resolve-RepoPath $packageConsumerScript) -Raw -Encoding utf8
$forbiddenSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "repo-local bin",
  "sidecar-only report",
  "TensorRtExec report",
  "dry-run",
  "queued workflow",
  "missing self-hosted runner"
)

$metadataChecks = @(
  [pscustomobject]@{ id = "managed-package-id"; propertyName = "PackageId"; passed = (Get-FirstPropertyValue -Project $managed -Name "PackageId") -eq "JYPPX.TensorRT.CSharp.API"; detail = "Managed package id must remain public package identity." },
  [pscustomobject]@{ id = "managed-readme"; propertyName = "PackageReadmeFile"; passed = (Get-FirstPropertyValue -Project $managed -Name "PackageReadmeFile") -eq "README.md"; detail = "Managed package must include README.md." },
  [pscustomobject]@{ id = "managed-repository-url"; propertyName = "RepositoryUrl"; passed = -not [string]::IsNullOrWhiteSpace((Get-FirstPropertyValue -Project $managed -Name "RepositoryUrl")); detail = "Managed package must declare RepositoryUrl." },
  [pscustomobject]@{ id = "managed-project-url"; propertyName = "PackageProjectUrl"; passed = -not [string]::IsNullOrWhiteSpace((Get-FirstPropertyValue -Project $managed -Name "PackageProjectUrl")); detail = "Managed package must declare PackageProjectUrl." },
  [pscustomobject]@{ id = "managed-description"; propertyName = "Description"; passed = -not [string]::IsNullOrWhiteSpace((Get-FirstPropertyValue -Project $managed -Name "Description")); detail = "Managed package must declare Description." },
  [pscustomobject]@{ id = "root-authors"; propertyName = "Authors"; passed = -not [string]::IsNullOrWhiteSpace((Get-FirstPropertyValue -Project $rootProps -Name "Authors")); detail = "Root package metadata must declare Authors." },
  [pscustomobject]@{ id = "runtime-readme"; propertyName = "PackageReadmeFile"; passed = (Get-FirstPropertyValue -Project $runtimeProps -Name "PackageReadmeFile") -eq "README.md"; detail = "Runtime package props must include README.md." },
  [pscustomobject]@{ id = "runtime-split-readme"; propertyName = "PackageReadmeFile"; passed = (Get-FirstPropertyValue -Project $runtimeSplitProps -Name "PackageReadmeFile") -eq "README.md"; detail = "Split runtime package props must include README.md." },
  [pscustomobject]@{ id = "runtime-assets-dir"; propertyName = "JYPPXRuntimeAssetsDir"; passed = -not [string]::IsNullOrWhiteSpace((Get-FirstPropertyValue -Project $runtimeProps -Name "JYPPXRuntimeAssetsDir")); detail = "Runtime package props must define runtime asset directory." },
  [pscustomobject]@{ id = "runtime-split-assets-dir"; propertyName = "JYPPXRuntimeAssetsDir"; passed = -not [string]::IsNullOrWhiteSpace((Get-FirstPropertyValue -Project $runtimeSplitProps -Name "JYPPXRuntimeAssetsDir")); detail = "Split runtime package props must define runtime asset directory." }
)

$consumerBoundaryChecks = @(
  [pscustomobject]@{ id = "consumer-script-exists"; passed = (Test-Path -LiteralPath (Resolve-RepoPath $packageConsumerScript) -PathType Leaf); detail = "Test-PackageConsumer.ps1 must exist." },
  [pscustomobject]@{ id = "consumer-script-uses-nuget-config"; passed = $packageConsumerScriptText.Contains("New-NuGetConfigContent"); detail = "Consumer validation must restore from explicit NuGet sources." },
  [pscustomobject]@{ id = "consumer-script-builds-clean-output"; passed = $packageConsumerScriptText.Contains("build-out\package-consumer"); detail = "Consumer validation must build in a clean output root." },
  [pscustomobject]@{ id = "consumer-script-records-forbidden-substitutes"; passed = $packageConsumerScriptText.Contains("ForbiddenProofSubstitutes") -and $packageConsumerScriptText.Contains("ProjectReference") -and $packageConsumerScriptText.Contains("direct .nupkg"); detail = "Consumer validation must record forbidden proof substitutes." },
  [pscustomobject]@{ id = "consumer-script-keeps-non-proof"; passed = $packageConsumerScriptText.Contains('$isPackageConsumerRuntimeProof = $false') -and $packageConsumerScriptText.Contains('CanPublishPublicly = $false') -and $packageConsumerScriptText.Contains('CanCloseReleaseIssue = $false'); detail = "Consumer validation must not promote local checks to public proof." }
)

$projectReferenceBoundary = [pscustomobject]@{
  managedPackProject = $managedPackProject
  projectReferenceCount = [int]$managedProjectReferences.Count
  projectReferences = @($managedProjectReferences)
  referenceOutputAssemblyFalseCount = [int]([regex]::Matches((Get-Content -LiteralPath (Resolve-RepoPath $managedPackProject) -Raw -Encoding utf8), 'ReferenceOutputAssembly="false"').Count)
  boundary = "ProjectReference is allowed only inside the pack project as a build input with ReferenceOutputAssembly=false; it is forbidden as package-consumer-runtime proof."
  canPromotePackageConsumerRuntimeProof = $false
}

$runtimeProjectCount = [int]@(Get-ChildItem -LiteralPath (Resolve-RepoPath "pack\runtime") -Recurse -Filter "*.csproj").Count
$runtimeSplitProjectCount = [int]@(Get-ChildItem -LiteralPath (Resolve-RepoPath "pack\runtime-split") -Recurse -Filter "*.csproj").Count
$failedMetadataChecks = @($metadataChecks | Where-Object { -not [bool]$_.passed })
$failedConsumerChecks = @($consumerBoundaryChecks | Where-Object { -not [bool]$_.passed })
$failedBlockerCount = [int]($failedMetadataChecks.Count + $failedConsumerChecks.Count)
$preflightState = if ($failedBlockerCount -eq 0) { "package-consumer-preflight-ready-non-proof" } else { "blocked-package-consumer-preflight-invalid" }

$record = [pscustomobject]@{
  recordKind = "package-consumer-preflight"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  preflightState = $preflightState
  failedBlockerCount = $failedBlockerCount
  metadataCheckCount = [int]$metadataChecks.Count
  consumerBoundaryCheckCount = [int]$consumerBoundaryChecks.Count
  runtimeProjectCount = $runtimeProjectCount
  runtimeSplitProjectCount = $runtimeSplitProjectCount
  managedPackageProject = $managedPackProject
  packageConsumerScript = $packageConsumerScript
  finalOwnerGateState = [string](Get-PropertyOrDefault -Object $finalOwnerGate -Name "gateState" -DefaultValue "missing-final-owner-next-decision-gate")
  recommendedOwnerDefault = [string](Get-PropertyOrDefault -Object $finalOwnerGate -Name "recommendedDefault" -DefaultValue "missing-final-owner-next-decision-gate")
  metadataChecks = @($metadataChecks)
  consumerBoundaryChecks = @($consumerBoundaryChecks)
  projectReferenceBoundary = $projectReferenceBoundary
  forbiddenProofSubstitutes = @($forbiddenSubstitutes)
  ownerActionRequired = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  canPromotePackageConsumerRuntimeProof = $false
  boundary = "Package consumer preflight only. This is not package publication and not public package-consumer proof. It checks metadata and local validation guardrails; it never pushes packages, publishes GitHub Packages, dispatches workflows, proves public package consumption, promotes runtime proof, or closes release issues."
}

$jsonPath = Join-Path $OutputRoot "package-consumer-preflight.json"
$mdPath = Join-Path $OutputRoot "package-consumer-preflight.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Package Consumer Preflight") | Out-Null
$md.Add("") | Out-Null
$md.Add("- preflightState: ``$preflightState``") | Out-Null
$md.Add("- failedBlockerCount: ``$failedBlockerCount``") | Out-Null
$md.Add("- runtimeProjectCount: ``$runtimeProjectCount``") | Out-Null
$md.Add("- runtimeSplitProjectCount: ``$runtimeSplitProjectCount``") | Out-Null
$md.Add("- finalOwnerGateState: ``$($record.finalOwnerGateState)``") | Out-Null
$md.Add("- recommendedOwnerDefault: ``$($record.recommendedOwnerDefault)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("## Forbidden Proof Substitutes") | Out-Null
$md.Add("") | Out-Null
foreach ($item in $forbiddenSubstitutes) {
  $md.Add("- ``$item``") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md

Write-Host "PackageConsumerPreflightState=$preflightState FailedBlockers=$failedBlockerCount RuntimeProjects=$runtimeProjectCount RuntimeSplitProjects=$runtimeSplitProjectCount"
