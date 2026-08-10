[CmdletBinding()]
param(
  [AllowEmptyString()]
  [string]$ReadyMarker = ""
)

if (-not [string]::Equals($ReadyMarker.Trim(), "true", [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "The production-release Environment is not configured. Create the production-release Environment with protection rules and set the PRODUCTION_RELEASE_READY environment secret to true before any public publication."
}

Write-Host "production-release Environment readiness marker passed."
