@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  AlgoForge — full test suite
REM  Run from the algoforge/ root OR scripts/ directory.
REM  Usage:
REM    scripts\test-all.bat            (run all)
REM    scripts\test-all.bat unit       (backend unit tests only)
REM    scripts\test-all.bat frontend   (frontend tests only)
REM ============================================================

REM Resolve the algoforge root regardless of where we're called from
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%.."
pushd "%ROOT%"

set FAILURES=0
set MODE=%1

echo.
echo ============================================================
echo  AlgoForge Test Suite
echo ============================================================

REM ── Backend unit tests (no DB / Redis required) ───────────────
if /i "%MODE%"=="" goto run_unit
if /i "%MODE%"=="unit" goto run_unit
if /i "%MODE%"=="integration" goto run_integration
if /i "%MODE%"=="frontend" goto run_frontend
echo Unknown mode: %MODE%
echo Usage: test-all.bat [unit^|integration^|frontend]
exit /b 1

:run_unit
echo.
echo [1/3] Backend unit tests (no DB required)...
cd backend
call pytest tests/unit/ -v --tb=short 2>&1
if errorlevel 1 (
    echo [FAIL] Backend unit tests failed.
    set /a FAILURES+=1
) else (
    echo [PASS] Backend unit tests passed.
)
cd ..
if /i "%MODE%"=="unit" goto done

:run_integration
echo.
echo [2/3] Backend integration tests (requires PostgreSQL + Redis)...
cd backend
call pytest tests/ --ignore=tests/unit -v --tb=short 2>&1
if errorlevel 1 (
    echo [FAIL] Backend integration tests failed.
    set /a FAILURES+=1
) else (
    echo [PASS] Backend integration tests passed.
)
cd ..
if /i "%MODE%"=="integration" goto done

:run_frontend
echo.
echo [3/3] Frontend tests...
cd web
call npm test -- --passWithNoTests 2>&1
if errorlevel 1 (
    echo [FAIL] Frontend tests failed.
    set /a FAILURES+=1
) else (
    echo [PASS] Frontend tests passed.
)
cd ..

:done
echo.
echo ============================================================
if %FAILURES%==0 (
    echo  All tests passed. Safe to release.
) else (
    echo  %FAILURES% test suite(s) failed. Do NOT release.
)
echo ============================================================
echo.

popd
exit /b %FAILURES%
