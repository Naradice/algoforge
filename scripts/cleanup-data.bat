@echo off
setlocal EnableDelayedExpansion

set API=http://localhost:8000/api/v1
set ARTIFACTS=..\backend\artifacts

echo ============================================================
echo  AlgoForge -- Data Cleanup
echo  API: %API%
echo ============================================================
echo.

:: ----------------------------------------------------------------
:: Confirm before proceeding
:: ----------------------------------------------------------------
set /p CONFIRM=This will delete ALL data (datasets, datasources, collection jobs, strategies, models). Continue? [y/N]:
if /i not "%CONFIRM%"=="y" (
    echo Aborted.
    exit /b 0
)
echo.

:: ----------------------------------------------------------------
:: Helper: delete all items from a collection endpoint
:: Usage: call :delete_all /endpoint "Label"
:: ----------------------------------------------------------------
goto :main

:delete_all
set ENDPOINT=%~1
set LABEL=%~2

for /f "usebackq delims=" %%I in (`curl -s "%API%%ENDPOINT%?page_size=100" ^| python -c "import sys,json; d=json.load(sys.stdin); [print(x['id']) for x in d.get('data',[])]" 2^>nul`) do (
    curl -s -o nul -X DELETE "%API%%ENDPOINT%/%%I"
    echo   Deleted %LABEL% %%I
)
exit /b 0

:main

:: ----------------------------------------------------------------
:: 1. Strategies (runs are cascade-deleted by the DB)
:: ----------------------------------------------------------------
echo [1/5] Strategies...
call :delete_all /strategies "strategy"

:: ----------------------------------------------------------------
:: 2. Models (training runs cascade)
:: ----------------------------------------------------------------
echo [2/5] Models...
call :delete_all /models "model"

:: ----------------------------------------------------------------
:: 3. Collection jobs
:: ----------------------------------------------------------------
echo [3/5] Collection jobs...
call :delete_all /collection-jobs "collection-job"

:: ----------------------------------------------------------------
:: 4. Datasets
:: ----------------------------------------------------------------
echo [4/5] Datasets...
call :delete_all /datasets "dataset"

:: ----------------------------------------------------------------
:: 5. Datasources
:: ----------------------------------------------------------------
echo [5/5] Datasources...
call :delete_all /datasources "datasource"

:: ----------------------------------------------------------------
:: 6. Artifact files
:: ----------------------------------------------------------------
echo.
echo [6/6] Artifact files...
if exist "%ARTIFACTS%\datasets" (
    rmdir /s /q "%ARTIFACTS%\datasets"
    echo   Removed artifacts\datasets\
)
if exist "%ARTIFACTS%\models" (
    rmdir /s /q "%ARTIFACTS%\models"
    echo   Removed artifacts\models\
)

echo.
echo Done. All data has been cleared.
endlocal
