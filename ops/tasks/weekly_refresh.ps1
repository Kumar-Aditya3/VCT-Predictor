$ErrorActionPreference = "Stop"

Write-Host "[vct] starting weekly pipeline"

$repoRoot = Resolve-Path "$PSScriptRoot\..\.."
$backendDir = Join-Path $repoRoot "backend"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $backendDir)) {
	throw "[vct] backend directory not found: $backendDir"
}

if (-not (Test-Path $venvPython)) {
	throw "[vct] venv python not found: $venvPython"
}

function Invoke-CheckedStep {
	param(
		[Parameter(Mandatory = $true)]
		[string]$Description,
		[Parameter(Mandatory = $true)]
		[string[]]$Args
	)

	Write-Host "[vct] $Description"
	& $venvPython @Args
	if ($LASTEXITCODE -ne 0) {
		throw "[vct] step failed: $Description (exit code $LASTEXITCODE)"
	}
}

Push-Location $backendDir

Invoke-CheckedStep -Description "running weekly update" -Args @("-m", "scripts.weekly_update")
Invoke-CheckedStep -Description "running VLR validation" -Args @("-m", "scripts.validate_vlr_ground_truth")

Pop-Location
Write-Host "[vct] weekly pipeline complete"
