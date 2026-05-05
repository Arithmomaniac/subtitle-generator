[CmdletBinding()]
param(
    [int]$Port = $(if ($env:E2E_PORT) { [int]$env:E2E_PORT } else { 8742 }),
    [string]$BaseUrl = $(if ($env:BASE_URL) { $env:BASE_URL } else { "http://127.0.0.1:$Port" }),
    [string]$ArtifactDir = $(if ($env:E2E_ARTIFACT_DIR) { $env:E2E_ARTIFACT_DIR } else { "test-results/local-e2e" })
)

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

$ArtifactPath = Join-Path $RootDir $ArtifactDir
$ServerLog = Join-Path $ArtifactPath "server.log"
$ScreenshotScript = Join-Path $ArtifactPath "capture_screenshot.py"

function Initialize-E2EArtifacts {
    New-Item -ItemType Directory -Force -Path $ArtifactPath | Out-Null
    if (Test-Path $ServerLog) {
        Remove-Item $ServerLog -Force
    }
}

function Write-E2EServerLog {
    if (Test-Path $ServerLog) {
        Get-Content $ServerLog | Write-Host
    }
}

function Test-E2EServerReady {
    try {
        Invoke-WebRequest -Uri "$BaseUrl/" -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function New-ScreenshotScript {
    $script = @'
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
        await page.goto(url, wait_until="domcontentloaded")
        await page.screenshot(path=str(path), full_page=True)
        await browser.close()


asyncio.run(main())
'@
    Set-Content -Path $ScreenshotScript -Value $script -Encoding UTF8
}

function Save-E2EScreenshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    $env:E2E_SCREENSHOT_PATH = Join-Path $ArtifactPath $Name
    $env:E2E_SCREENSHOT_URL = $Url
    & uv run python $ScreenshotScript
    if ($LASTEXITCODE -ne 0) {
        throw "Screenshot capture failed for $Url"
    }
}

function Invoke-E2EPythonTest {
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
        Save-E2EScreenshot -Name $FailureScreenshot -Url $FailureUrl
        throw "$TestPath failed. Artifacts are in $ArtifactPath."
    }
}

function Start-E2EServer {
    Write-Host "Starting subtitle-generator local server on $BaseUrl"
    Start-Job -Name "subtitle-generator-e2e-server" -ScriptBlock {
        param($WorkingDirectory, $ServerPort, $LogPath)
        Set-Location $WorkingDirectory
        uv run subtitle-gen serve --no-open --port $ServerPort *> $LogPath
    } -ArgumentList $RootDir, $Port, $ServerLog
}

function Wait-E2EServerReady {
    param(
        [Parameter(Mandatory = $true)]
        [System.Management.Automation.Job]$ServerJob
    )

    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if (Test-E2EServerReady) {
            return
        }

        $job = Get-Job -Id $ServerJob.Id
        if ($job.State -ne "Running") {
            Receive-Job -Id $ServerJob.Id -Keep | Write-Host
            Write-Host "Local server exited before becoming ready. Server log:"
            Write-E2EServerLog
            exit 1
        }

        Start-Sleep -Seconds 1
    }

    Write-Host "Timed out waiting for $BaseUrl. Server log:"
    Write-E2EServerLog
    exit 1
}

Initialize-E2EArtifacts
New-ScreenshotScript
$ServerJob = Start-E2EServer

try {
    Wait-E2EServerReady -ServerJob $ServerJob

    Save-E2EScreenshot -Name "home-before.png" -Url "$BaseUrl/"
    Invoke-E2EPythonTest `
        -TestPath "tests/test_e2e.py" `
        -FailureScreenshot "home-failure.png" `
        -FailureUrl "$BaseUrl/"
    Save-E2EScreenshot -Name "home-after.png" -Url "$BaseUrl/"

    Invoke-E2EPythonTest `
        -TestPath "tests/test_e2e_spot_check.py" `
        -FailureScreenshot "spot-check-failure.png" `
        -FailureUrl "$BaseUrl/spot-check.html"
    Save-E2EScreenshot -Name "spot-check-after.png" -Url "$BaseUrl/spot-check.html"

    Write-Host "Local e2e tests passed. Artifacts are in $ArtifactPath."
}
finally {
    if ($ServerJob) {
        Stop-Job -Id $ServerJob.Id -ErrorAction SilentlyContinue
        Remove-Job -Id $ServerJob.Id -Force -ErrorAction SilentlyContinue
    }
}
