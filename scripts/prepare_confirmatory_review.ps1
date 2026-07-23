param(
    [string]$RunId = "domain-confirmatory-v1",
    [string]$PythonPath = "",
    [switch]$ValidateOnly,
    [switch]$SkipClosure
)

$ConfirmProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$ConfirmRunRoot = Join-Path $ConfirmProjectRoot "data\runs\$RunId"
$ConfirmProtocolDir = Join-Path $ConfirmRunRoot "protocol"
$ConfirmConfigPath = Join-Path $ConfirmProtocolDir "config_snapshot.json"
$ConfirmProtocolPath = Join-Path $ConfirmProtocolDir "protocol_v1.md"
$ConfirmAnalysisPath = Join-Path $ConfirmProtocolDir "analysis_plan.md"
$ConfirmGapDir = Join-Path $ConfirmRunRoot "gaps"
$ConfirmAuditDir = Join-Path $ConfirmRunRoot "audits"

function Stop-ConfirmatoryRun([string]$Message) {
    throw "[confirmatory-review] $Message"
}

function Require-ConfirmatoryFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Stop-ConfirmatoryRun "Missing required file: $Path"
    }
}

function Resolve-ConfirmatoryPython([string]$RequestedPath) {
    if ($RequestedPath) {
        Require-ConfirmatoryFile $RequestedPath
        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }

    $ConfirmPythonCandidates = @()
    $ConfirmLocalPrograms = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path -LiteralPath $ConfirmLocalPrograms) {
        $ConfirmPythonCandidates = @(
            Get-ChildItem -LiteralPath $ConfirmLocalPrograms -Filter python.exe -File -Recurse |
                Sort-Object FullName -Descending
        )
    }
    if ($ConfirmPythonCandidates.Count -gt 0) {
        return $ConfirmPythonCandidates[0].FullName
    }

    $ConfirmPythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($ConfirmPythonCommand) {
        return $ConfirmPythonCommand.Source
    }
    Stop-ConfirmatoryRun "Python was not found. Pass -PythonPath with a Python 3.11+ executable."
}

function Invoke-ConfirmatoryPython([string[]]$Arguments) {
    & $script:ConfirmPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        Stop-ConfirmatoryRun "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Set-Location -LiteralPath $ConfirmProjectRoot

Require-ConfirmatoryFile $ConfirmConfigPath
Require-ConfirmatoryFile $ConfirmProtocolPath
Require-ConfirmatoryFile $ConfirmAnalysisPath

$script:ConfirmPython = Resolve-ConfirmatoryPython $PythonPath
Write-Output "[confirmatory-review] Python: $script:ConfirmPython"
Invoke-ConfirmatoryPython @("--version")

$ConfirmConfig = Get-Content -Raw -LiteralPath $ConfirmConfigPath | ConvertFrom-Json
if ($ConfirmConfig.run_id -ne $RunId) {
    Stop-ConfirmatoryRun "Config run_id '$($ConfirmConfig.run_id)' does not match requested '$RunId'."
}
if ($ConfirmConfig.status -ne "locked") {
    Stop-ConfirmatoryRun "config_snapshot.json status must be 'locked'; current value is '$($ConfirmConfig.status)'."
}

$ConfirmRequiredValues = [ordered]@{
    "domain.name" = $ConfirmConfig.domain.name
    "domain.retrieval_query" = $ConfirmConfig.domain.retrieval_query
    "domain.retrieval_start_utc" = $ConfirmConfig.domain.retrieval_start_utc
    "domain.retrieval_end_utc" = $ConfirmConfig.domain.retrieval_end_utc
    "domain.analysis_cutoff_year" = $ConfirmConfig.domain.analysis_cutoff_year
    "corpus.manifest_path" = $ConfirmConfig.corpus.manifest_path
    "corpus.manifest_sha256" = $ConfirmConfig.corpus.manifest_sha256
    "corpus.retained_record_count" = $ConfirmConfig.corpus.retained_record_count
    "corpus.inclusion_rule_version" = $ConfirmConfig.corpus.inclusion_rule_version
    "models.configured_llm_model" = $ConfirmConfig.models.configured_llm_model
    "temporal.cutoff_year" = $ConfirmConfig.temporal.cutoff_year
    "integrity.protocol_v1_sha256" = $ConfirmConfig.integrity.protocol_v1_sha256
    "integrity.analysis_plan_sha256" = $ConfirmConfig.integrity.analysis_plan_sha256
}
foreach ($ConfirmEntry in $ConfirmRequiredValues.GetEnumerator()) {
    if ($null -eq $ConfirmEntry.Value -or [string]::IsNullOrWhiteSpace([string]$ConfirmEntry.Value)) {
        Stop-ConfirmatoryRun "Required locked value is missing: $($ConfirmEntry.Key)"
    }
}
if ([int]$ConfirmConfig.domain.analysis_cutoff_year -ne [int]$ConfirmConfig.temporal.cutoff_year) {
    Stop-ConfirmatoryRun "domain.analysis_cutoff_year and temporal.cutoff_year must match."
}

$ConfirmProtocolHash = (Get-FileHash -LiteralPath $ConfirmProtocolPath -Algorithm SHA256).Hash.ToLowerInvariant()
$ConfirmAnalysisHash = (Get-FileHash -LiteralPath $ConfirmAnalysisPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ConfirmProtocolHash -ne [string]$ConfirmConfig.integrity.protocol_v1_sha256) {
    Stop-ConfirmatoryRun "protocol_v1.md hash does not match config_snapshot.json."
}
if ($ConfirmAnalysisHash -ne [string]$ConfirmConfig.integrity.analysis_plan_sha256) {
    Stop-ConfirmatoryRun "analysis_plan.md hash does not match config_snapshot.json."
}

$ConfirmTodoCount = @(
    Select-String -LiteralPath $ConfirmProtocolPath,$ConfirmAnalysisPath -SimpleMatch "[[FILL_REQUIRED]]"
).Count
if ($ConfirmTodoCount -gt 0) {
    Stop-ConfirmatoryRun "Protocol files still contain $ConfirmTodoCount FILL_REQUIRED marker(s)."
}

$ConfirmSystemFiles = [ordered]@{
    "kgtabi" = "kgtabi_gaps.json"
    "direct" = "direct_llm_gaps.json"
    "full_grounds" = "full_grounds_gaps.json"
    "concept_only" = "concept_only_gaps.json"
    "abstract_bundle" = "abstract_bundle_gaps.json"
    "shuffled_grounds" = "shuffled_grounds_gaps.json"
}

$ConfirmCandidateCounts = [ordered]@{}
foreach ($ConfirmSystem in $ConfirmSystemFiles.Keys) {
    $ConfirmCandidatePath = Join-Path $ConfirmGapDir $ConfirmSystemFiles[$ConfirmSystem]
    Require-ConfirmatoryFile $ConfirmCandidatePath
    $ConfirmRows = @(Get-Content -Raw -LiteralPath $ConfirmCandidatePath | ConvertFrom-Json)
    if ($ConfirmRows.Count -eq 0) {
        Stop-ConfirmatoryRun "System '$ConfirmSystem' has zero candidates. Record the null/underpowered result; do not relax gates."
    }
    $ConfirmRowIndex = 0
    foreach ($ConfirmRow in $ConfirmRows) {
        $ConfirmRowIndex += 1
        foreach ($ConfirmField in @("Grounds", "Claim", "Warrant", "Bucket")) {
            if ([string]::IsNullOrWhiteSpace([string]$ConfirmRow.$ConfirmField)) {
                Stop-ConfirmatoryRun "Missing $ConfirmField in $ConfirmCandidatePath row $ConfirmRowIndex."
            }
        }
        if ($ConfirmRow.Bucket -notin @("near_term_feasible", "long_term_or_speculative")) {
            Stop-ConfirmatoryRun "Invalid Bucket in $ConfirmCandidatePath row $ConfirmRowIndex."
        }
    }
    $ConfirmCandidateCounts[$ConfirmSystem] = $ConfirmRows.Count
}

$ConfirmUniqueCounts = @($ConfirmCandidateCounts.Values | Sort-Object -Unique)
if ($ConfirmUniqueCounts.Count -ne 1) {
    Stop-ConfirmatoryRun "Candidate counts are not balanced: $($ConfirmCandidateCounts | ConvertTo-Json -Compress)"
}
$ConfirmBudget = [int]$ConfirmConfig.candidate_generation.candidate_budget_per_system_per_domain
$ConfirmObservedCount = [int]$ConfirmUniqueCounts[0]
if ($ConfirmObservedCount -gt $ConfirmBudget) {
    Stop-ConfirmatoryRun "Each system has $ConfirmObservedCount candidates, above locked budget $ConfirmBudget. Apply the predeclared deterministic selection rule first."
}
if ($ConfirmObservedCount -lt $ConfirmBudget) {
    Write-Warning "Underpowered candidate set: $ConfirmObservedCount per system versus locked target $ConfirmBudget."
}

Write-Output "[confirmatory-review] Protocol and $($ConfirmSystemFiles.Count) candidate inputs validated."
Write-Output ($ConfirmCandidateCounts | ConvertTo-Json)

if ($ValidateOnly) {
    Write-Output "[confirmatory-review] Validation-only mode complete; no closure or packet files were written."
    exit 0
}

New-Item -ItemType Directory -Force -Path $ConfirmAuditDir | Out-Null

if (-not $SkipClosure) {
    foreach ($ConfirmSystem in $ConfirmSystemFiles.Keys) {
        $ConfirmCandidatePath = Join-Path $ConfirmGapDir $ConfirmSystemFiles[$ConfirmSystem]
        $ConfirmClosureAudit = Join-Path $ConfirmGapDir "${ConfirmSystem}_closure_search_audit.json"
        $ConfirmClosureManifest = Join-Path $ConfirmGapDir "${ConfirmSystem}_closure_search_manifest.json"
        Write-Output "[confirmatory-review] Running deterministic closure: $ConfirmSystem"
        Invoke-ConfirmatoryPython @(
            "-B", "-m", "src.closure_search",
            "--input", $ConfirmCandidatePath,
            "--output", $ConfirmClosureAudit,
            "--manifest", $ConfirmClosureManifest,
            "--limit", [string]$ConfirmConfig.closure.paper_search_limit_per_query,
            "--citation-limit", [string]$ConfirmConfig.closure.citation_limit_per_direction,
            "--citation-candidate-limit", [string]$ConfirmConfig.closure.citation_candidate_limit,
            "--deterministic-only"
        )
    }
} else {
    Write-Warning "Closure was skipped. Do not collect novelty ratings until equivalent source bundles are complete."
}

$ConfirmPacketPath = Join-Path $ConfirmAuditDir "candidate_blind_packet.csv"
$ConfirmKeyPath = Join-Path $ConfirmAuditDir "candidate_unblinding_key.csv"
$ConfirmPacketArguments = @("-B", "-m", "src.prepare_candidate_blind_review")
foreach ($ConfirmSystem in $ConfirmSystemFiles.Keys) {
    $ConfirmCandidatePath = Join-Path $ConfirmGapDir $ConfirmSystemFiles[$ConfirmSystem]
    $ConfirmPacketArguments += @("--input", "${ConfirmSystem}=$ConfirmCandidatePath")
}
$ConfirmPacketArguments += @(
    "--packet", $ConfirmPacketPath,
    "--key", $ConfirmKeyPath,
    "--seed", [string]$ConfirmConfig.review.packet_seed
)
Invoke-ConfirmatoryPython $ConfirmPacketArguments

$ConfirmHashRecords = @(
    foreach ($ConfirmHashPath in @($ConfirmConfigPath,$ConfirmPacketPath,$ConfirmKeyPath)) {
        $ConfirmHash = Get-FileHash -LiteralPath $ConfirmHashPath -Algorithm SHA256
        $ConfirmResolvedHashPath = (Resolve-Path -LiteralPath $ConfirmHashPath).Path
        $ConfirmRelativeHashPath = $ConfirmResolvedHashPath.Substring($ConfirmProjectRoot.Length).TrimStart("\").Replace("\", "/")
        [ordered]@{
            path = $ConfirmRelativeHashPath
            sha256 = $ConfirmHash.Hash.ToLowerInvariant()
            bytes = (Get-Item -LiteralPath $ConfirmHashPath).Length
        }
    }
)
$ConfirmHashOutput = Join-Path $ConfirmAuditDir "review_packet_hashes.json"
$ConfirmHashRecords | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ConfirmHashOutput -Encoding utf8

Write-Output "[confirmatory-review] Blind packet: $ConfirmPacketPath"
Write-Output "[confirmatory-review] PRIVATE key: $ConfirmKeyPath"
Write-Output "[confirmatory-review] Hash inventory: $ConfirmHashOutput"
Write-Warning "Never send the private unblinding key to reviewers."
