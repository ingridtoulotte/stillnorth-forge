# Launch the StillNorth Forge web UI.
#   .\scripts\run.ps1              # open UI in the browser
#   .\scripts\run.ps1 --no-browser
Set-Location (Join-Path $PSScriptRoot "..")
python -m stillnorth @args
