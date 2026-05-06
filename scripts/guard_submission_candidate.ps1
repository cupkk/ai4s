param(
  [Parameter(Mandatory = $true)]
  [string]$Candidate,
  [string]$Reference = "outputs/output_nwp_unconstrained_online5117.csv",
  [string]$CandidateName = "",
  [string]$Manifest = ""
)

$name = $CandidateName
if ([string]::IsNullOrWhiteSpace($name)) {
  $name = [System.IO.Path]::GetFileNameWithoutExtension($Candidate)
}

$argsList = @(
  "-m", "src.guard_submission_candidate",
  "--candidate", $Candidate,
  "--reference", $Reference,
  "--candidate-name", $name,
  "--diff-output", "outputs/action_diff_safe5117_vs_$name.csv",
  "--summary-output", "outputs/guard_summary_$name.csv"
)

if (-not [string]::IsNullOrWhiteSpace($Manifest)) {
  $argsList += @("--manifest", $Manifest)
}

python @argsList
