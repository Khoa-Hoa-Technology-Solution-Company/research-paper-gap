param(
    [string]$RunId = "domain-confirmatory-v1"
)

$LockProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$LockProtocolDir = Join-Path $LockProjectRoot "data\runs\$RunId\protocol"
$LockProtocolPath = Join-Path $LockProtocolDir "protocol_v1.md"
$LockAnalysisPath = Join-Path $LockProtocolDir "analysis_plan.md"
$LockConfigPath = Join-Path $LockProtocolDir "config_snapshot.json"
$LockRecordPath = Join-Path $LockProtocolDir "protocol_lock.json"

foreach ($LockRequiredPath in @($LockProtocolPath,$LockAnalysisPath,$LockConfigPath)) {
    if (-not (Test-Path -LiteralPath $LockRequiredPath -PathType Leaf)) {
        throw "[protocol-lock] Missing required file: $LockRequiredPath"
    }
}

$LockMarkers = @(
    Select-String -LiteralPath $LockProtocolPath,$LockAnalysisPath -SimpleMatch "[[FILL_REQUIRED]]"
)
if ($LockMarkers.Count -gt 0) {
    throw "[protocol-lock] Replace all $($LockMarkers.Count) FILL_REQUIRED marker(s) before locking."
}

$LockConfig = Get-Content -Raw -LiteralPath $LockConfigPath | ConvertFrom-Json
if ($LockConfig.run_id -ne $RunId) {
    throw "[protocol-lock] Config run_id '$($LockConfig.run_id)' does not match '$RunId'."
}
if ($LockConfig.status -eq "locked") {
    throw "[protocol-lock] Protocol is already locked. Create a new run ID to make changes."
}

$LockRequiredValues = [ordered]@{
    "domain.name" = $LockConfig.domain.name
    "domain.owner" = $LockConfig.domain.owner
    "domain.retrieval_query" = $LockConfig.domain.retrieval_query
    "domain.retrieval_start_utc" = $LockConfig.domain.retrieval_start_utc
    "domain.retrieval_end_utc" = $LockConfig.domain.retrieval_end_utc
    "domain.analysis_cutoff_year" = $LockConfig.domain.analysis_cutoff_year
    "corpus.manifest_path" = $LockConfig.corpus.manifest_path
    "corpus.manifest_sha256" = $LockConfig.corpus.manifest_sha256
    "corpus.retrieved_record_count" = $LockConfig.corpus.retrieved_record_count
    "corpus.retained_record_count" = $LockConfig.corpus.retained_record_count
    "corpus.inclusion_rule_version" = $LockConfig.corpus.inclusion_rule_version
    "corpus.language_policy" = $LockConfig.corpus.language_policy
    "models.configured_llm_model" = $LockConfig.models.configured_llm_model
    "temporal.cutoff_year" = $LockConfig.temporal.cutoff_year
    "closure.retrieval_end_utc" = $LockConfig.closure.retrieval_end_utc
    "closure.source_bundle_schema_version" = $LockConfig.closure.source_bundle_schema_version
}
foreach ($LockEntry in $LockRequiredValues.GetEnumerator()) {
    if ($null -eq $LockEntry.Value -or [string]::IsNullOrWhiteSpace([string]$LockEntry.Value)) {
        throw "[protocol-lock] Required value is missing: $($LockEntry.Key)"
    }
}

if ([int]$LockConfig.domain.analysis_cutoff_year -ne [int]$LockConfig.temporal.cutoff_year) {
    throw "[protocol-lock] domain.analysis_cutoff_year and temporal.cutoff_year must match."
}

$LockProtocolHash = (Get-FileHash -LiteralPath $LockProtocolPath -Algorithm SHA256).Hash.ToLowerInvariant()
$LockAnalysisHash = (Get-FileHash -LiteralPath $LockAnalysisPath -Algorithm SHA256).Hash.ToLowerInvariant()
$LockConfig.integrity.protocol_v1_sha256 = $LockProtocolHash
$LockConfig.integrity.analysis_plan_sha256 = $LockAnalysisHash
$LockConfig.status = "locked"
$LockConfig | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $LockConfigPath -Encoding utf8

$LockConfigHash = (Get-FileHash -LiteralPath $LockConfigPath -Algorithm SHA256).Hash.ToLowerInvariant()
$LockUtc = [DateTimeOffset]::UtcNow.ToString("o")
$LockRecord = [ordered]@{
    schema_version = "kgtabi-protocol-lock-v1"
    run_id = $RunId
    locked_at_utc = $LockUtc
    files = @(
        [ordered]@{path="protocol/protocol_v1.md"; sha256=$LockProtocolHash},
        [ordered]@{path="protocol/analysis_plan.md"; sha256=$LockAnalysisHash},
        [ordered]@{path="protocol/config_snapshot.json"; sha256=$LockConfigHash}
    )
    instruction = "Do not edit locked files under this run ID; create a new run ID for any change."
}
$LockRecord | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $LockRecordPath -Encoding utf8

Write-Output "[protocol-lock] Locked run: $RunId"
Write-Output "[protocol-lock] Lock record: $LockRecordPath"
Write-Output "[protocol-lock] Config SHA-256: $LockConfigHash"
