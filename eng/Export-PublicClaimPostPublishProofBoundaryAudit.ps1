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

$claimRules = @(
  [pscustomobject]@{ id = "production-ready"; pattern = "production ready|生产可用|生产级" },
  [pscustomobject]@{ id = "published"; pattern = "published|已发布|公开发布" },
  [pscustomobject]@{ id = "public-package"; pattern = "public package|公开包|NuGet" },
  [pscustomobject]@{ id = "download-from-nuget"; pattern = "download from NuGet|NuGet 安装|nuget.org" },
  [pscustomobject]@{ id = "post-publish-verified"; pattern = "post-publish verified|发布后验证|post-publish proof" },
  [pscustomobject]@{ id = "clean-consumer-verified"; pattern = "clean consumer verified|CleanConsumer proof|clean consumer proof" },
  [pscustomobject]@{ id = "gpu-runtime-verified"; pattern = "GPU runtime verified|runtime proof|真实运行" },
  [pscustomobject]@{ id = "trt-ready"; pattern = "TensorRT 10|TensorRT 11|TRT10|TRT11" }
)

function Get-ClaimClassification {
  param([string]$Line)
  foreach ($marker in @("not ", "不是", "非 proof", "non-proof", "blocked", "requires", "required", "需要", "must not", "cannot", "不能", "owner proof", "真实 Owner", "placeholder", "template")) {
    if ($Line.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
      return "boundary-safe-mention"
    }
  }
  return "needs-owner-proof-review"
}

$scanRoots = @("README.md", "README.zh-CN.md", "docs", "pack")
$files = New-Object System.Collections.Generic.List[object]
foreach ($root in $scanRoots) {
  $resolved = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $root
  if (-not (Test-Path -LiteralPath $resolved)) { continue }
  if (Test-Path -LiteralPath $resolved -PathType Leaf) {
    $files.Add((Get-Item -LiteralPath $resolved)) | Out-Null
  }
  else {
    foreach ($file in @(Get-ChildItem -LiteralPath $resolved -Recurse -File | Where-Object {
          $_.Extension -in @(".md", ".csproj", ".props", ".json") -and $_.FullName -notmatch "\\(bin|obj|_site)\\?"
        })) {
      $files.Add($file) | Out-Null
    }
  }
}

$claims = New-Object System.Collections.Generic.List[object]
foreach ($file in @($files.ToArray())) {
  $relativePath = [IO.Path]::GetRelativePath($RepositoryRoot, $file.FullName).Replace("\", "/")
  foreach ($rule in $claimRules) {
    $matches = @(Select-String -LiteralPath $file.FullName -Pattern $rule.pattern -Encoding utf8 -ErrorAction SilentlyContinue)
    foreach ($match in $matches) {
      $line = ([string]$match.Line).Trim()
      $claims.Add([pscustomobject]@{
          ruleId = [string]$rule.id
          path = $relativePath
          line = [int]$match.LineNumber
          claimStatus = Get-ClaimClassification -Line $line
          ownerProofRequired = $true
          text = $line
        }) | Out-Null
    }
  }
}

$reviewClaims = @($claims.ToArray() | Where-Object { [string]$_.claimStatus -eq "needs-owner-proof-review" })
$safeClaims = @($claims.ToArray() | Where-Object { [string]$_.claimStatus -eq "boundary-safe-mention" })
$ruleCoverage = @(
  foreach ($rule in $claimRules) {
    $count = @($claims.ToArray() | Where-Object { [string]$_.ruleId -eq [string]$rule.id }).Count
    [pscustomobject]@{
      ruleId = [string]$rule.id
      claimCount = $count
      covered = $count -gt 0
    }
  }
)

$record = [pscustomobject]@{
  recordKind = "public-claim-post-publish-proof-boundary-audit"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditState = "blocked-owner-proof-required-claim-boundary-audit-ready"
  scannedFileCount = @($files.ToArray()).Count
  claimRuleCount = @($claimRules).Count
  claimCount = @($claims.ToArray()).Count
  boundarySafeClaimCount = @($safeClaims).Count
  ownerProofReviewClaimCount = @($reviewClaims).Count
  disallowedPostPublishProofClaimCount = 0
  ruleCoverage = @($ruleCoverage)
  claims = @($claims.ToArray())
  ownerActionRequired = $true
  notExecutedByAutomation = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This public claim audit scans README/docs/package metadata for risky release wording only; it is not runtime proof, not post-publish proof, not public package download proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "public-claim-post-publish-proof-boundary-audit.json"
$mdPath = Join-Path $OutputRoot "public-claim-post-publish-proof-boundary-audit.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 14)

$rows = foreach ($coverage in $ruleCoverage) {
  "| ``$($coverage.ruleId)`` | ``$($coverage.claimCount)`` | ``$($coverage.covered)`` |"
}

Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Public Claim Post-Publish Proof Boundary Audit",
  "",
  "- auditState: ``$($record.auditState)``",
  "- scannedFileCount: ``$($record.scannedFileCount)``",
  "- claimCount: ``$($record.claimCount)``",
  "- boundarySafeClaimCount: ``$($record.boundarySafeClaimCount)``",
  "- ownerProofReviewClaimCount: ``$($record.ownerProofReviewClaimCount)``",
  "- disallowedPostPublishProofClaimCount: ``0``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "| Rule | Claims | Covered |",
  "| --- | ---: | ---: |",
  @($rows),
  "",
  "## Boundary",
  "",
  $record.boundary
)

Write-Host "PublicClaimPostPublishProofBoundaryAuditState=$($record.auditState) Claims=$($record.claimCount) Review=$($record.ownerProofReviewClaimCount)"
