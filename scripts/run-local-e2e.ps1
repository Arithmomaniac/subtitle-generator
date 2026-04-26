[CmdletBinding()]
param(
    [int]$Port = $(if ($env:E2E_PORT) { [int]$env:E2E_PORT } else { 8742 }),
    [string]$BaseUrl = $(if ($env:BASE_URL) { $env:BASE_URL } else { "http://127.0.0.1:$Port" }),
    [string]$ArtifactDir = $(if ($env:E2E_ARTIFACT_DIR) { $env:E2E_ARTIFACT_DIR } else { "test-results/local-e2e" })
)

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null
$ServerLog = Join-Path $ArtifactDir "server.log"

function Write-ServerLog {
    if (Test-Path $ServerLog) {
        Get-Content $ServerLog
    }
}

function Test-ServerReady {
    try {
        Invoke-WebRequest -Uri "$BaseUrl/" -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Save-Screenshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    $captureScript = @'
import asyncio
import os
from pathlib import Path

from playwright.async_api import async_playwright


async def main() -> None:
    path = Path(os.environ["E2E_SCREENSHOT_PATH"])
    url = os.environ["E2E_SCREENSHOT_URL"]
    path.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        await page.screenshot(path=str(path), full_page=True)
        await browser.close()


asyncio.run(main())
'@

    $capturePath = Join-Path $ArtifactDir "capture_screenshot.py"
    Set-Content -Path $capturePath -Value $captureScript -Encoding UTF8

    $env:E2E_SCREENSHOT_PATH = $Path
    $env:E2E_SCREENSHOT_URL = $Url
    & uv run python $capturePath
    if ($LASTEXITCODE -ne 0) {
        throw "Screenshot capture failed for $Url"
    }
}

function Invoke-E2ETest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TestPath,
        [Parameter(Mandatory = $true)]
        [string]$FailureScreenshot,
        [Parameter(Mandatory = $true)]
        [string]$FailureUrl
    )

    $env:BASE_URL = $BaseUrl
    & uv run python $TestPath
    if ($LASTEXITCODE -ne 0) {
        Save-Screenshot -Path $FailureScreenshot -Url $FailureUrl
        throw "$TestPath failed. Artifacts are in $ArtifactDir."
    }
}

Write-Host "Starting subtitle-generator local server on $BaseUrl"
$ServerJob = Start-Job -Name "subtitle-generator-e2e-server" -ScriptBlock {
    param($RootDir, $Port, $ServerLog)
    Set-Location $RootDir
    uv run subtitle-gen serve --no-open --port $Port *> $ServerLog
} -ArgumentList $RootDir, $Port, $ServerLog

try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if (Test-ServerReady) {
            $ready = $true
            break
        }

        $job = Get-Job -Id $ServerJob.Id
        if ($job.State -ne "Running") {
            Receive-Job -Id $ServerJob.Id -Keep | Out-Host
            Write-Host "Local server exited before becoming ready. Server log:"
            Write-ServerLog
            exit 1
        }

        Start-Sleep -Seconds 1
    }

    if (-not $ready) {
        Write-Host "Timed out waiting for $BaseUrl. Server log:"
        Write-ServerLog
        exit 1
    }

    Save-Screenshot -Path (Join-Path $ArtifactDir "home-before.png") -Url "$BaseUrl/"
    Invoke-E2ETest `
        -TestPath "tests/test_e2e.py" `
        -FailureScreenshot (Join-Path $ArtifactDir "home-failure.png") `
        -FailureUrl "$BaseUrl/"
    Save-Screenshot -Path (Join-Path $ArtifactDir "home-after.png") -Url "$BaseUrl/"

    Invoke-E2ETest `
        -TestPath "tests/test_e2e_spot_check.py" `
        -FailureScreenshot (Join-Path $ArtifactDir "spot-check-failure.png") `
        -FailureUrl "$BaseUrl/spot-check.html"
    Save-Screenshot -Path (Join-Path $ArtifactDir "spot-check-after.png") -Url "$BaseUrl/spot-check.html"

    Write-Host "Local e2e tests passed. Artifacts are in $ArtifactDir."
}
finally {
    if ($ServerJob) {
        Stop-Job -Id $ServerJob.Id -ErrorAction SilentlyContinue
        Remove-Job -Id $ServerJob.Id -Force -ErrorAction SilentlyContinue
    }
}
