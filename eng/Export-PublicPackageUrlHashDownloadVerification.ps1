[CmdletBinding()]
param(
  [string]$ImportPath = "artifacts\final-release\owner-post-publish-docs-article-sample-real-input-import.json",
  [string]$PublicPackageValidatorPath = "artifacts\final-release\public-package-url-hash-proof-validator-validation.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$resolvedImportPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $ImportPath
if (-not (Test-Path -LiteralPath $resolvedImportPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Import-OwnerPostPublishDocsArticleSampleRealInput.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$resolvedValidatorPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $PublicPackageValidatorPath
if (-not (Test-Path -LiteralPath $resolvedValidatorPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-PublicPackageUrlHashProofValidator.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
  & (Join-Path $RepositoryRoot "eng\Test-PublicPackageUrlHashProofValidator.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot -Strict
}

$import = Get-Content -LiteralPath $resolvedImportPath -Raw -Encoding utf8 | ConvertFrom-Json
$validator = Get-Content -LiteralPath $resolvedValidatorPath -Raw -Encoding utf8 | ConvertFrom-Json
$realOwnerInputPresent = [bool](Get-PropertyOrDefault -Object $import -Name "realOwnerInputPresent" -DefaultValue $false)
$validatorAccepted = [bool](Get-PropertyOrDefault -Object $validator -Name "ownerEvidenceAccepted" -DefaultValue $false)
$blockedReasons = New-Object System.Collections.Generic.List[string]
if (-not $realOwnerInputPresent) { $blockedReasons.Add("real-owner-input-file-missing") | Out-Null }
if (-not $validatorAccepted) { $blockedReasons.Add("public-package-url-hash-proof-validator-not-accepted") | Out-Null }

$downloadSpecs = @(
  [pscustomobject]@{ id = "nuget-managed-package"; urlField = "nugetManagedPackageUrl"; shaField = "nugetManagedPackageSha256" },
  [pscustomobject]@{ id = "github-runtime-package"; urlField = "githubRuntimePackageUrl"; shaField = "githubRuntimePackageSha256" }
)

$checks = New-Object System.Collections.Generic.List[object]
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("TensorRtSharp-owner-package-download-{0}" -f [System.Guid]::NewGuid().ToString("N"))
$downloadAllowed = $realOwnerInputPresent -and $validatorAccepted
try {
  if ($downloadAllowed) {
    New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
  }

  foreach ($spec in $downloadSpecs) {
    $url = Get-OwnerPostPublishImportFieldValue -ImportRecord $import -LaneId "public-package-urls-and-hashes" -FieldName $spec.urlField
    $expectedSha = (Get-OwnerPostPublishImportFieldValue -ImportRecord $import -LaneId "public-package-urls-and-hashes" -FieldName $spec.shaField).ToLowerInvariant()
    $downloadAttempted = $false
    $downloadSucceeded = $false
    $actualSha = ""
    $hashMatched = $false
    $errorMessage = ""

    if ($downloadAllowed) {
      $downloadAttempted = $true
      try {
        $uri = [System.Uri]::new($url)
        if ($uri.Scheme -ne "https") { throw "Only HTTPS package URLs are allowed." }
        $targetPath = Join-Path $tempRoot ("{0}.nupkg" -f $spec.id)
        $client = [System.Net.Http.HttpClient]::new()
        try {
          $bytes = $client.GetByteArrayAsync($uri).GetAwaiter().GetResult()
          [System.IO.File]::WriteAllBytes($targetPath, $bytes)
          $actualSha = Get-FileSha256Text -Path $targetPath
          $hashMatched = $actualSha.Equals($expectedSha, [StringComparison]::OrdinalIgnoreCase)
          $downloadSucceeded = $true
        }
        finally {
          $client.Dispose()
        }
      }
      catch {
        $errorMessage = [string]$_.Exception.Message
      }
    }

    $checks.Add([pscustomobject]@{
        id = [string]$spec.id
        url = $url
        expectedSha256 = $expectedSha
        downloadAttempted = $downloadAttempted
        downloadSucceeded = $downloadSucceeded
        actualSha256 = $actualSha
        hashMatched = $hashMatched
        errorMessage = $errorMessage
      }) | Out-Null
  }
}
finally {
  if (Test-Path -LiteralPath $tempRoot) {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
  }
}

$packageChecks = @($checks.ToArray())
$attemptedCount = @($packageChecks | Where-Object { [bool]$_.downloadAttempted }).Count
$succeededCount = @($packageChecks | Where-Object { [bool]$_.downloadSucceeded }).Count
$hashMatchedCount = @($packageChecks | Where-Object { [bool]$_.hashMatched }).Count
if ($downloadAllowed -and $hashMatchedCount -lt $packageChecks.Count) { $blockedReasons.Add("public-package-download-hash-mismatch-or-download-failure") | Out-Null }
$downloadVerificationReady = $downloadAllowed -and $packageChecks.Count -gt 0 -and $hashMatchedCount -eq $packageChecks.Count
$verificationState = if ($downloadVerificationReady) { "public-package-url-hash-download-verification-matched" } else { "blocked-public-package-url-hash-download-verification-real-owner-input-required" }

$record = [pscustomobject]@{
  recordKind = "public-package-url-hash-download-verification"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  verificationState = $verificationState
  realOwnerInputPresent = $realOwnerInputPresent
  publicPackageValidatorAccepted = $validatorAccepted
  downloadAllowed = $downloadAllowed
  packageCheckCount = $packageChecks.Count
  downloadAttemptedCount = $attemptedCount
  downloadSucceededCount = $succeededCount
  hashMatchedCount = $hashMatchedCount
  downloadVerificationReady = $downloadVerificationReady
  packageChecks = @($packageChecks)
  blockedReasonCount = $blockedReasons.Count
  blockedReasons = @($blockedReasons.ToArray())
  ownerActionRequired = -not $downloadVerificationReady
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Public package URL/hash download verification downloads only Owner-provided HTTPS package URLs after strict Owner input acceptance. It does not publish, does not use tokens, does not access private feeds, does not close the release issue, and is not post-publish proof or package push by itself."
}

Write-Utf8File -LiteralPath (Join-Path $OutputRoot "public-package-url-hash-download-verification.json") -InputObject ($record | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "public-package-url-hash-download-verification.md") -InputObject @("# Public Package URL/Hash Download Verification", "", "- verificationState: ``$verificationState``", "- downloadAllowed: ``$downloadAllowed``", "- downloadAttemptedCount: ``$attemptedCount``", "- hashMatchedCount: ``$hashMatchedCount/$($packageChecks.Count)``", "- canPublishPublicly: ``False``", "- canCloseReleaseIssue: ``False``", "", $record.boundary)
Write-Host "PublicPackageUrlHashDownloadVerificationState=$verificationState DownloadAttempted=$attemptedCount HashMatched=$hashMatchedCount/$($packageChecks.Count)"
