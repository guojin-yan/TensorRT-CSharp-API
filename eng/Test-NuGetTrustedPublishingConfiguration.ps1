[CmdletBinding()]
param(
  [AllowEmptyString()]
  [string]$NuGetUser = $env:NUGET_USER,
  [string]$Repository,
  [switch]$RequireUser
)

$ErrorActionPreference = "Stop"
$source = if ($PSBoundParameters.ContainsKey("NuGetUser")) { "input" } else { "environment" }

# Actions passes the resolved variable explicitly, including an empty value.
# Only local callers without an explicit value need to query GitHub variables.
if (-not $PSBoundParameters.ContainsKey("NuGetUser") -and
    [string]::IsNullOrEmpty($NuGetUser) -and
    -not [string]::IsNullOrWhiteSpace($Repository)) {
  $source = "gh-variable-get"
  try {
    $value = @(& gh variable get NUGET_USER --repo $Repository 2>$null)
    if ($LASTEXITCODE -eq 0) {
      $NuGetUser = $value -join ""
    }
    else {
      $source = "query-failed"
    }
  }
  catch {
    $source = "query-failed"
  }
}

$userConfigured = -not [string]::IsNullOrWhiteSpace($NuGetUser) -and $NuGetUser -notmatch '[\s@]'
$detail = if ($userConfigured) {
  "NUGET_USER is configured. Trusted Publishing policy is not verified by this read-only check; NuGet/login must verify it in the publishing job."
}
else {
  "NUGET_USER is missing or invalid. Set the repository Actions variable NUGET_USER to the NuGet account username (not an email address). Configure the matching NuGet Trusted Publishing policy before publishing."
}

if ($RequireUser.IsPresent -and -not $userConfigured) {
  throw $detail
}

[pscustomobject]@{
  authentication = "trusted-publishing"
  userConfigured = [bool]$userConfigured
  policyVerified = $false
  source = $source
  detail = $detail
}
