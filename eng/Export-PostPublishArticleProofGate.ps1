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

function New-ClaimGate {
  param(
    [string]$Id,
    [string]$Claim,
    [string[]]$RequiredEvidence,
    [string[]]$ForbiddenSubstitutes
  )

  [pscustomobject]@{
    id = $Id
    claim = $Claim
    requiredEvidence = @($RequiredEvidence)
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    gateState = "blocked-owner-post-publish-proof-required"
    ownerActionRequired = $true
    passed = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isProof = $false
  }
}

$articleMatrix = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\public-article-readiness-matrix-validation.json"
$postPublish = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-strict-cross-check-pack-validation.json"
$closure = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\release-close-strict-evidence-closure-validation.json"

$articleMatrixState = [string](Get-PropertyOrDefault -Object $articleMatrix -Name "validationState" -DefaultValue "missing-public-article-readiness-matrix-validation")
$postPublishState = [string](Get-PropertyOrDefault -Object $postPublish -Name "validationState" -DefaultValue "missing-post-publish-strict-cross-check-pack-validation")
$closureState = [string](Get-PropertyOrDefault -Object $closure -Name "validationState" -DefaultValue "missing-release-close-strict-evidence-closure-validation")

$forbidden = @("local feed", "ProjectReference", "direct .nupkg", "dashboard", "dry-run", "manual approval", "queued GitHub Actions run", "missing self-hosted runner", "sidecar-only", "TensorRtExec report")
$claimGates = @(
  New-ClaimGate "nuget-published" "文章宣称已发布到 NuGet 或可从 NuGet 安装" @("publicPackageUrl", "managedPackageUrl", "runtimePackageUrl", "downloaded package SHA256", "Owner publish transcript") $forbidden
  New-ClaimGate "github-packages-published" "文章宣称 GitHub Packages 全量 runtime 包已经公开可用" @("GitHub Packages package page URL", "runtime package download URL", "runtime package SHA256") $forbidden
  New-ClaimGate "clean-consumer-verified" "文章宣称公开渠道 clean consumer restore/build/smoke 已通过" @("clean consumer root outside repository", "restore/build/smoke logs", "no local feed evidence", "PostPublish validation") $forbidden
  New-ClaimGate "release-closed" "文章宣称 release 已经最终关闭" @("release close strict evidence closure pass", "rollback review", "Owner close decision") $forbidden
  New-ClaimGate "runtime-proof" "文章宣称真实 GPU/runtime proof 已完成" @("runtime smoke report", "host runtime identity", "public package source", "strict validator transcript") $forbidden
)

$record = [pscustomobject]@{
  recordKind = "post-publish-article-proof-gate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  gateState = "blocked-post-publish-article-proof-owner-evidence-required"
  claimGateCount = $claimGates.Count
  blockedClaimGateCount = @($claimGates | Where-Object { [string]$_.gateState -like "blocked*" }).Count
  claimGates = @($claimGates)
  sourceStates = [pscustomobject]@{
    publicArticleReadinessMatrixValidationState = $articleMatrixState
    postPublishStrictCrossCheckValidationState = $postPublishState
    releaseCloseStrictEvidenceClosureValidationState = $closureState
  }
  forbiddenNonProofSubstitutes = @($forbidden)
  ownerActionRequired = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "PostPublish article proof gate only. It blocks public article claims about NuGet/GitHub package availability, clean consumer success, runtime proof, or release close until Owner public publish and PostPublish evidence pass strict validators."
}

$jsonPath = Join-Path $OutputRoot "post-publish-article-proof-gate.json"
$mdPath = Join-Path $OutputRoot "post-publish-article-proof-gate.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# PostPublish Article Proof Gate") | Out-Null
$md.Add("") | Out-Null
$md.Add("- gateState: ``$($record.gateState)``") | Out-Null
$md.Add("- claimGateCount: ``$($record.claimGateCount)``") | Out-Null
$md.Add("- blockedClaimGateCount: ``$($record.blockedClaimGateCount)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Claim Gate | State | Claim |") | Out-Null
$md.Add("| --- | --- | --- |") | Out-Null
foreach ($gate in $claimGates) {
  $md.Add("| ``$($gate.id)`` | ``$($gate.gateState)`` | $(ConvertTo-MarkdownCell $gate.claim) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "PostPublishArticleProofGateState=$($record.gateState) Claims=$($record.claimGateCount) Blocked=$($record.blockedClaimGateCount)"
