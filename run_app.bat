@echo off
rem FringeLab launcher for Windows: double-click this file.
rem The first run creates a private Python environment in .venv and installs the requirements.
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
if exist ".venv\installed.txt" goto run
echo First start: installing FringeLab. This takes a few minutes and happens only once.
%PY% -m venv .venv || goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
echo ok> ".venv\installed.txt"
:run
".venv\Scripts\python.exe" launch.py
goto end
:fail
echo.
echo Setup failed. Check that Python 3.10 or newer is installed from python.org
echo with "Add python.exe to PATH" ticked, then start this file again.
pause
:end
