[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$OutputRoot,
  [switch]$RequirePackageInventory,
  [switch]$RequireClassificationAudit,
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\release-quality-gate"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$checks = New-Object System.Collections.Generic.List[object]

function Add-Check {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][bool]$Passed,
    [Parameter(Mandatory = $true)][bool]$Required,
    [Parameter(Mandatory = $true)][string]$Detail
  )

  $checks.Add([pscustomobject]@{
    id = $Id
    passed = $Passed
    required = $Required
    detail = $Detail
  }) | Out-Null
}

function Read-JsonOrNull {
  param([Parameter(Mandatory = $true)][string]$Path)

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-TextFindings {
  param(
    [Parameter(Mandatory = $true)][string[]]$Roots,
    [Parameter(Mandatory = $true)][string]$Pattern
  )

  $extensions = @(".md", ".json", ".yml", ".yaml", ".txt", ".csproj", ".sln", ".cs", ".ps1")
  $findings = New-Object System.Collections.Generic.List[string]
  foreach ($root in $Roots) {
    if (-not (Test-Path -LiteralPath $root)) {
      continue
    }

    $files = if (Test-Path -LiteralPath $root -PathType Leaf) {
      @(Get-Item -LiteralPath $root)
    }
    else {
      @(Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue |
          Where-Object {
            $extensions -contains $_.Extension.ToLowerInvariant() -and
            $_.FullName -notmatch "\\(bin|obj)\\"
          })
    }

    foreach ($file in $files) {
      $matches = @(Select-String -LiteralPath $file.FullName -Pattern $Pattern -AllMatches -ErrorAction SilentlyContinue)
      foreach ($match in $matches) {
        $relative = [System.IO.Path]::GetRelativePath($RepositoryRoot, $file.FullName)
        $findings.Add("${relative}:$($match.LineNumber)") | Out-Null
      }
    }
  }

  return @($findings)
}

$workflowPath = Join-Path $RepositoryRoot ".github\workflows\release-quality-gate.yml"
$workflow = if (Test-Path -LiteralPath $workflowPath -PathType Leaf) {
  Get-Content -LiteralPath $workflowPath -Raw
}
else {
  ""
}

Add-Check -Id "workflow-present" -Passed (-not [string]::IsNullOrWhiteSpace($workflow)) -Required $true -Detail $workflowPath
Add-Check -Id "workflow-read-only-permissions" -Passed ($workflow -match "permissions:\s*\r?\n\s+contents:\s+read") -Required $true -Detail "Workflow must use contents: read."
Add-Check -Id "workflow-push-current-branch" -Passed (
  $workflow.Contains("push:", [StringComparison]::Ordinal) -and
  $workflow.Contains("- TensorRtSharp4.0", [StringComparison]::Ordinal)
) -Required $true -Detail "Workflow must run automatically on push to the TensorRtSharp4.0 release branch."
Add-Check -Id "workflow-source-gate" -Passed ($workflow.Contains("Test-ReleaseQualityGate.ps1 -Strict", [StringComparison]::Ordinal)) -Required $true -Detail "Source gate must execute the strict quality summary."
Add-Check -Id "workflow-bindings-and-coverage" -Passed (
  $workflow.Contains("Generate-Bindings.ps1", [StringComparison]::Ordinal) -and
  $workflow.Contains("Test-BindingGeneratorOutputs.ps1", [StringComparison]::Ordinal) -and
  $workflow.Contains("Export-InterfaceCoverageMatrix.ps1", [StringComparison]::Ordinal) -and
  $workflow.Contains("Export-DeferredReadOnlyApiCandidatePlan.ps1", [StringComparison]::Ordinal)
) -Required $true -Detail "Workflow must regenerate bindings, coverage, and deferred triage."
Add-Check -Id "workflow-build-and-tests" -Passed (
  $workflow.Contains("dotnet build TensorRtSharp.sln", [StringComparison]::Ordinal) -and
  $workflow.Contains("dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj", [StringComparison]::Ordinal)
) -Required $true -Detail "Workflow must build the solution and run source-only quality tests."
Add-Check -Id "workflow-public-api-documentation" -Passed (
  $workflow.Contains("Enforce public API documentation", [StringComparison]::Ordinal) -and
  $workflow.Contains("Test-PublicApiBilingualDocumentation.ps1", [StringComparison]::Ordinal) -and
  -not $workflow.Contains("Test-PublicApiBilingualDocumentation.ps1 -SkipBuild", [StringComparison]::Ordinal)
) -Required $true -Detail "Workflow must fail on compiler-reported missing XML documentation and non-bilingual public documentation."
Add-Check -Id "workflow-source-only-test-filter" -Passed (
  $workflow.Contains("Run source-only release quality tests", [StringComparison]::Ordinal) -and
  $workflow.Contains("--filter ""FullyQualifiedName~ReleaseAutomationTests|FullyQualifiedName~ReleaseQualityGateWorkflowTests""", [StringComparison]::Ordinal) -and
  -not $workflow.Contains("FinalReleaseMarkdownRenderingTests", [StringComparison]::Ordinal) -and
  -not $workflow.Contains("RnnV2BorrowedStateDesignGateTests", [StringComparison]::Ordinal) -and
  -not $workflow.Contains("EngineAndRnnReadonlyDiagnosticsTests", [StringComparison]::Ordinal) -and
  -not $workflow.Contains("RuntimePackageReadinessTests", [StringComparison]::Ordinal)
) -Required $true -Detail "Push/PR source-quality must run deterministic source-only tests and leave artifact-only release-close tests to opt-in gates."
Add-Check -Id "workflow-project-quality-shard-smoke" -Passed (
  $workflow.Contains("Invoke-ProjectQualityTestShards.ps1", [StringComparison]::Ordinal) -and
  $workflow.Contains("-Shard N-S", [StringComparison]::Ordinal) -and
  $workflow.Contains("PluginInventorySourceOnlySmoke|PublicApiDocumentationClosure|PublicApiHandleExposureAudit|ReleaseQualityGateWorkflow", [StringComparison]::Ordinal) -and
  $workflow.Contains("artifacts/test-analysis/project-quality-shards/**", [StringComparison]::Ordinal)
) -Required $true -Detail "Workflow must execute a bounded ProjectQuality shard smoke and archive shard evidence."
Add-Check -Id "workflow-opt-in-large-jobs" -Passed (
  $workflow.Contains("run_split_package_build", [StringComparison]::Ordinal) -and
  $workflow.Contains("run_release_artifact_audit", [StringComparison]::Ordinal) -and
  ([regex]::Matches($workflow, "default:\s+false")).Count -ge 2
) -Required $true -Detail "Large split-package and release-artifact jobs must default to false."
Add-Check -Id "workflow-split-all-contract" -Passed (
  $workflow.Contains("Invoke-LocalSplitRuntimePackage.ps1", [StringComparison]::Ordinal) -and
  $workflow.Contains("-SplitPackageRole all", [StringComparison]::Ordinal) -and
  $workflow.Contains("-IncludeMetaPackage", [StringComparison]::Ordinal)
) -Required $true -Detail "Opt-in split job must build all component roles plus meta."
Add-Check -Id "workflow-strict-release-audits" -Passed (
  $workflow.Contains("Test-ReleaseEvidenceClassificationAudit.ps1 -Strict", [StringComparison]::Ordinal) -and
  $workflow.Contains("Test-PublicProofClaimBoundaryAudit.ps1 -Strict", [StringComparison]::Ordinal)
) -Required $true -Detail "Opt-in artifact audit must run strict classification and proof-claim scans."

$forbiddenWorkflowPatterns = @(
  "dotnet\s+nuget\s+push",
  "Push-NuGetPackages",
  "NUGET_API_KEY",
  "GITHUB_PACKAGES_TOKEN",
  "gh\s+release\s+create",
  "gh\s+release\s+upload"
)
$forbiddenWorkflowHits = @(
  foreach ($pattern in $forbiddenWorkflowPatterns) {
    if ($workflow -match $pattern) {
      $pattern
    }
  }
)
Add-Check -Id "workflow-no-publish-side-effects" -Passed ($forbiddenWorkflowHits.Count -eq 0) -Required $true -Detail ("Forbidden workflow patterns: " + ($forbiddenWorkflowHits -join ", "))

$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$splitManifest = Read-JsonOrNull $splitManifestPath
$targetSplitPackages = if ($null -eq $splitManifest) {
  @()
}
else {
  @($splitManifest.packages | Where-Object { [string]$_.sourceRuntimeKey -eq $RuntimePackageKey })
}
$targetRoles = @($targetSplitPackages | ForEach-Object { [string]$_.role } | Sort-Object -Unique)
$requiredComponentRoles = @("bridge")
$missingComponentRoles = @($requiredComponentRoles | Where-Object { $targetRoles -notcontains $_ })
$invalidSplitPackages = @($targetSplitPackages | Where-Object {
  [string]::IsNullOrWhiteSpace([string]$_.packageId) -or
  @($_.assets).Count -eq 0
})
Add-Check -Id "split-manifest-component-roles" -Passed ($missingComponentRoles.Count -eq 0) -Required $true -Detail ("Missing roles: " + ($missingComponentRoles -join ", "))
Add-Check -Id "split-manifest-package-contract" -Passed (@($targetSplitPackages | Where-Object { [string]$_.role -eq "bridge" }).Count -eq 1 -and $invalidSplitPackages.Count -eq 0) -Required $true -Detail "Each target requires exactly one active bridge package; legacy vendor entries are cleanup identities only."

$objectArrayFindings = Get-TextFindings -Roots @(
  (Join-Path $RepositoryRoot "docs"),
  (Join-Path $RepositoryRoot "artifacts")
) -Pattern "System\.Object\[\]"
Add-Check -Id "public-markdown-array-rendering" -Passed ($objectArrayFindings.Count -eq 0) -Required $true -Detail ("Findings: " + ($objectArrayFindings -join ", "))

$legacySampleFindings = Get-TextFindings -Roots @(
  (Join-Path $RepositoryRoot "README.md"),
  (Join-Path $RepositoryRoot "README.zh-CN.md"),
  (Join-Path $RepositoryRoot "docs"),
  (Join-Path $RepositoryRoot "samples"),
  (Join-Path $RepositoryRoot "applications"),
  (Join-Path $RepositoryRoot "artifacts")
) -Pattern "samples[/\\]YoloDet|YoloDet\.csproj"
Add-Check -Id "legacy-yolodet-public-paths" -Passed ($legacySampleFindings.Count -eq 0) -Required $true -Detail ("Findings: " + ($legacySampleFindings -join ", "))

$inventoryPath = Join-Path $RepositoryRoot "artifacts\final-release\release-candidate-package-inventory.json"
$inventory = Read-JsonOrNull $inventoryPath
$inventoryReady = $false
$inventoryDetail = "inventory-not-present"
if ($null -ne $inventory) {
  $missingRoles = @($inventory.missingSplitRoles)
  $inventoryReady =
    [string]$inventory.recordKind -eq "release-candidate-package-inventory" -and
    [bool]$inventory.splitBridgePackageReady -and
    [bool]$inventory.splitRuntimePackagesReady -and
    [bool]$inventory.packageSetReady -and
    [bool]$inventory.sha256Ready -and
    $missingRoles.Count -eq 0 -and
    -not [bool]$inventory.performsPublish -and
    -not [bool]$inventory.canPublishPublicly -and
    -not [bool]$inventory.canCloseReleaseIssue
  $inventoryDetail = "packageSetReady=$($inventory.packageSetReady); sha256Ready=$($inventory.sha256Ready); missingRoles=$($missingRoles.Count)"
}
Add-Check -Id "release-package-inventory" -Passed ($inventoryReady -or -not $RequirePackageInventory) -Required ([bool]$RequirePackageInventory) -Detail $inventoryDetail

$classificationPath = Join-Path $RepositoryRoot "artifacts\final-release\release-evidence-classification-audit.json"
$classification = Read-JsonOrNull $classificationPath
$classificationReady = $false
$classificationDetail = "classification-audit-not-present"
if ($null -ne $classification) {
  $classificationReady =
    [bool]$classification.auditPassed -and
    [int]$classification.findingCount -eq 0 -and
    [string]$classification.auditState -eq "classification-audit-passed-non-proof-boundaries-intact"
  $classificationDetail = "auditState=$($classification.auditState); findingCount=$($classification.findingCount)"
}
Add-Check -Id "release-classification-audit" -Passed ($classificationReady -or -not $RequireClassificationAudit) -Required ([bool]$RequireClassificationAudit) -Detail $classificationDetail

$requiredFailures = @($checks | Where-Object { $_.required -and -not $_.passed })
$sourceFailures = @($checks | Where-Object {
  $_.required -and
  $_.id -notin @("release-package-inventory", "release-classification-audit") -and
  -not $_.passed
})
$state = if ($requiredFailures.Count -eq 0) {
  "release-quality-gate-passed"
}
else {
  "release-quality-gate-failed"
}

$record = [ordered]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-quality-gate-summary"
  state = $state
  runtimePackageKey = $RuntimePackageKey
  sourceGatePassed = $sourceFailures.Count -eq 0
  packageInventoryRequired = [bool]$RequirePackageInventory
  packageInventoryReady = $inventoryReady
  classificationAuditRequired = [bool]$RequireClassificationAudit
  classificationAuditReady = $classificationReady
  checkCount = $checks.Count
  requiredFailureCount = $requiredFailures.Count
  performsPublish = $false
  usesPublishToken = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  checks = $checks.ToArray()
  boundary = "This workflow gate validates source, build, package contracts, and non-proof classification. It does not publish packages and cannot substitute compatible-host runtime proof, public package proof, post-publish proof, owner authorization, or release close approval."
}

$jsonPath = Join-Path $OutputRoot "release-quality-gate-summary.json"
$markdownPath = Join-Path $OutputRoot "release-quality-gate-summary.md"
[System.IO.File]::WriteAllText($jsonPath, ($record | ConvertTo-Json -Depth 8), $utf8)

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Quality Gate Summary")
$lines.Add("")
$lines.Add("- state: ``$state``")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- source gate passed: ``$($record.sourceGatePassed)``")
$lines.Add("- package inventory required/ready: ``$($record.packageInventoryRequired)`` / ``$inventoryReady``")
$lines.Add("- classification audit required/ready: ``$($record.classificationAuditRequired)`` / ``$classificationReady``")
$lines.Add("- performs publish: ``False``")
$lines.Add("- runtime proof: ``False``")
$lines.Add("- can publish publicly: ``False``")
$lines.Add("- can close release issue: ``False``")
$lines.Add("")
$lines.Add("## Checks")
$lines.Add("")
$lines.Add("| Check | Required | Passed | Detail |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($check in $checks) {
  $detail = ([string]$check.detail).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
  $lines.Add("| $($check.id) | $($check.required) | $($check.passed) | $detail |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)
[System.IO.File]::WriteAllLines($markdownPath, $lines, $utf8)

Write-Host "Release quality gate summary written: $jsonPath"
Write-Host "Release quality gate markdown written: $markdownPath"
Write-Host "State=$state RequiredFailureCount=$($requiredFailures.Count)"

if ($Strict -and $requiredFailures.Count -ne 0) {
  throw "Release quality gate failed with $($requiredFailures.Count) required finding(s)."
}
