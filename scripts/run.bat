@echo off
rem Launch the StillNorth Forge web UI.
cd /d "%~dp0\.."
python -m stillnorth %*
