param(
    [string[]]$Steps = @("Inventory"),

    [string]$FullDb = "data\db\subtitles.db",
    [string]$ApiDb = "api\data\db\subtitles.db",
    [string]$MiniDb = "api\data\subtitles.mini.db",
    [string]$ExportDir = "api\data",
    [string]$BookModelDir = "generated-artifacts\book-model",
    [string]$MetadataCsv = "",
    [int]$Samples = 12,
    [int]$RandomSeed = 20260505,
    [switch]$ApplyPopularityCalibration,
    [switch]$ReviewGates,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$validSteps = @(
    "All",
    "Inventory",
    "Metadata",
    "Features",
    "Baseline",
    "Torch",
    "CalibratePopularity",
    "PopulatePopularity",
    "Distill",
    "Shadow",
    "DeploymentGate",
    "CategorizationGate",
    "InstallScores",
    "ExportData",
    "BuildDb",
    "Validate"
)
$Steps = @($Steps | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$invalidSteps = @($Steps | Where-Object { $_ -notin $validSteps })
if ($invalidSteps.Count -gt 0) {
    throw "Invalid step(s): $($invalidSteps -join ', '). Valid steps: $($validSteps -join ', ')"
}

if ($Steps -contains "All") {
    $Steps = @(
        "Inventory",
        "Metadata",
        "Features",
        "Baseline",
        "Torch",
        "CalibratePopularity",
        "PopulatePopularity",
        "Distill",
        "Shadow",
        "DeploymentGate",
        "CategorizationGate",
        "InstallScores",
        "ExportData",
        "BuildDb",
        "Validate"
    )
}

function Format-CommandLine {
    param([string]$Command, [string[]]$Arguments)
    $quoted = $Arguments | ForEach-Object {
        if ($_ -match "\s") { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }
    return (@($Command) + $quoted) -join " "
}

function Invoke-Step {
    param(
        [string]$Name,
        [string]$Command,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host "[$Name] $(Format-CommandLine -Command $Command -Arguments $Arguments)"
    if (-not $PlanOnly) {
        & $Command @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Step '$Name' failed with exit code $LASTEXITCODE."
        }
    }
}

function Has-Step {
    param([string]$Name)
    return $Steps -contains $Name
}

$featuresPath = Join-Path $BookModelDir "book_features.csv"
$labelsPath = Join-Path $BookModelDir "book_labels.csv"
$teacherPredictions = Join-Path $BookModelDir "torch-all-spacy\book_torch_predictions.csv"
$currentStudentPredictions = Join-Path $BookModelDir "distill-export-current\book_torch_predictions.csv"
$slotStudentPredictions = Join-Path $BookModelDir "distill-export-slot\book_torch_predictions.csv"
$currentRollups = Join-Path $BookModelDir "shadow-rollups\filler_book_rollups_export-current.csv"
$slotRollups = Join-Path $BookModelDir "shadow-rollups\filler_book_rollups_export-slot.csv"

if (Has-Step "Inventory") {
    Invoke-Step "Inventory" "uv" @(
        "run", "subtitle-gen", "book-model-inventory",
        "--db", $FullDb,
        "--mini-db", $MiniDb,
        "--api-db", $ApiDb,
        "--export-dir", $ExportDir,
        "--output", "generated-artifacts\book_model_inventory.md"
    )
}

if (Has-Step "Metadata") {
    Invoke-Step "Metadata" "uv" @(
        "run", "subtitle-gen", "build-book-metadata",
        "--db", $FullDb,
        "--output-dir", $BookModelDir
    )
}

if (Has-Step "Features") {
    $featureArgs = @(
        "run", "subtitle-gen", "build-book-features",
        "--db", $FullDb,
        "--output-dir", $BookModelDir
    )
    if ($MetadataCsv) {
        $featureArgs += @("--metadata-csv", $MetadataCsv)
    }
    Invoke-Step "Features" "uv" $featureArgs
}

if (Has-Step "Baseline") {
    Invoke-Step "Baseline" "uv" @(
        "run", "subtitle-gen", "train-book-model",
        "--features", $featuresPath,
        "--labels", $labelsPath,
        "--output-dir", $BookModelDir
    )
}

if (Has-Step "Torch") {
    Invoke-Step "Torch" "uv" @(
        "run", "subtitle-gen", "train-book-model-torch",
        "--features", $featuresPath,
        "--labels", $labelsPath,
        "--output-dir", (Join-Path $BookModelDir "torch-all-spacy"),
        "--feature-set", "all",
        "--semantic-vectors", "spacy"
    )
}

if (Has-Step "CalibratePopularity") {
    $calibrationArgs = @(
        "run", "subtitle-gen", "calibrate-popularity-weights",
        "--features", $featuresPath,
        "--teacher-predictions", $teacherPredictions,
        "--output-dir", (Join-Path $BookModelDir "popularity-calibration"),
        "--db", $FullDb
    )
    if ($ApplyPopularityCalibration) {
        $calibrationArgs += "--apply"
    }
    Invoke-Step "CalibratePopularity" "uv" $calibrationArgs
}

if (Has-Step "PopulatePopularity") {
    Invoke-Step "PopulatePopularity" "uv" @(
        "run", "subtitle-gen", "populate-popularity",
        "--db", $FullDb,
        "--skip-data-model"
    )
    $featureArgs = @(
        "run", "subtitle-gen", "build-book-features",
        "--db", $FullDb,
        "--output-dir", $BookModelDir
    )
    if ($MetadataCsv) {
        $featureArgs += @("--metadata-csv", $MetadataCsv)
    }
    Invoke-Step "Features after popularity calibration" "uv" $featureArgs
}

if (Has-Step "Distill") {
    Invoke-Step "Distill export-current" "uv" @(
        "run", "subtitle-gen", "distill-book-model",
        "--features", $featuresPath,
        "--labels", $labelsPath,
        "--teacher-predictions", $teacherPredictions,
        "--output-dir", (Join-Path $BookModelDir "distill-export-current"),
        "--feature-set", "export-current"
    )
    Invoke-Step "Distill export-slot" "uv" @(
        "run", "subtitle-gen", "distill-book-model",
        "--features", $featuresPath,
        "--labels", $labelsPath,
        "--teacher-predictions", $teacherPredictions,
        "--output-dir", (Join-Path $BookModelDir "distill-export-slot"),
        "--feature-set", "export-slot"
    )
}

if (Has-Step "Shadow") {
    Invoke-Step "Shadow" "uv" @(
        "run", "subtitle-gen", "shadow-book-model",
        "--db", $FullDb,
        "--prediction", "export-current", $currentStudentPredictions,
        "--prediction", "export-slot", $slotStudentPredictions,
        "--output-dir", (Join-Path $BookModelDir "shadow-rollups"),
        "--samples", [string]$Samples,
        "--random-seed", [string]$RandomSeed
    )
}

if (Has-Step "DeploymentGate") {
    $gateArgs = @(
        "run", "subtitle-gen", "deployment-gate",
        "--rollup", "export-current", $currentRollups,
        "--rollup", "export-slot", $slotRollups,
        "--output-dir", (Join-Path $BookModelDir "deployment-gate"),
        "--samples", [string]($Samples * 2),
        "--random-seed", [string]$RandomSeed
    )
    if (-not $ReviewGates) {
        $gateArgs += "--dry-run"
    }
    Invoke-Step "DeploymentGate" "uv" $gateArgs
}

if (Has-Step "CategorizationGate") {
    $categorizationArgs = @(
        "run", "subtitle-gen", "categorization-gate",
        "--rollup", "export-current", $currentRollups,
        "--rollup", "export-slot", $slotRollups,
        "--output-dir", (Join-Path $BookModelDir "categorization-gate"),
        "--samples-per-tier", [string]$Samples,
        "--random-seed", [string]$RandomSeed
    )
    if (-not $ReviewGates) {
        $categorizationArgs += "--dry-run"
    }
    Invoke-Step "CategorizationGate" "uv" $categorizationArgs
}

if (Has-Step "InstallScores") {
    Invoke-Step "InstallScores" "uv" @(
        "run", "subtitle-gen", "install-book-model-scores",
        "--input", $slotRollups
    )
}

if (Has-Step "ExportData") {
    Invoke-Step "ExportData" "uv" @(
        "run", "subtitle-gen", "export-data",
        "--output-dir", $ExportDir
    )
}

if (Has-Step "BuildDb") {
    Invoke-Step "BuildDb" "uv" @(
        "run", "subtitle-gen", "build-db",
        "--data-dir", $ExportDir,
        "--output", $MiniDb
    )
}

if (Has-Step "Validate") {
    Invoke-Step "Validate" "uv" @(
        "run", "subtitle-gen", "validate-pipeline",
        "--db", $FullDb
    )
}
