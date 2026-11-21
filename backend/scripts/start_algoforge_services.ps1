# PowerShell script to start AlgoForge services cleanly
$root = "c:\Users\ksato\workspace\finance\algoforge"
$backend = "$root\backend"
$web = "$root\web"
$python = "C:\Users\ksato\AppData\Local\Programs\Python\Python313\python.exe"
$sitePackages = "C:\Users\ksato\AppData\Local\Programs\Python\Python313\Lib\site-packages"
$financeClient = "c:\Users\ksato\workspace\finance\finance_client\src"
$pythonPath = "$sitePackages;$backend;$financeClient"

Write-Host "=== Starting AlgoForge Services ==="

# 1. Check Docker Containers
Write-Host "1. Checking Docker Containers (Postgres & Redis)..."
docker start infra-postgres-1 infra-redis-1
Start-Sleep -Seconds 2

# 2. Start FastAPI Backend (Port 8000)
Write-Host "2. Starting FastAPI Backend (Port 8000)..."
$apiCmd = @"
`$env:PYTHONPATH='$pythonPath'
cd '$backend'
& '$python' -m uvicorn main:app --host 0.0.0.0 --port 8000 > '$backend\uvicorn_api.log' 2>&1
"@
Start-Process powershell -ArgumentList "-NoProfile", "-WindowStyle", "Hidden", "-Command", $apiCmd

# 3. Start Celery General Worker (collection, characteristics, training, backtest, colab)
Write-Host "3. Starting Celery General Worker..."
$workerGeneralCmd = @"
`$env:PYTHONPATH='$pythonPath'
cd '$backend'
& '$python' -m celery -A celery_worker worker -Q collection,characteristics,training,backtest,colab --pool=solo -n general@%COMPUTERNAME% --loglevel=info > '$backend\celery_general.log' 2>&1
"@
Start-Process powershell -ArgumentList "-NoProfile", "-WindowStyle", "Hidden", "-Command", $workerGeneralCmd

# 4. Start Celery Beat Scheduler
Write-Host "4. Starting Celery Beat..."
$beatCmd = @"
`$env:PYTHONPATH='$pythonPath'
cd '$backend'
& '$python' -m celery -A celery_worker beat --loglevel=info > '$backend\celery_beat.log' 2>&1
"@
Start-Process powershell -ArgumentList "-NoProfile", "-WindowStyle", "Hidden", "-Command", $beatCmd

# 5. Start Web UI (Next.js Port 3000)
Write-Host "5. Starting Next.js Web UI (Port 3000)..."
$webCmd = @"
cd '$web'
npm run dev > '$web\web.log' 2>&1
"@
Start-Process powershell -ArgumentList "-NoProfile", "-WindowStyle", "Hidden", "-Command", $webCmd

Write-Host "=== Services started! Waiting 5 seconds to verify... ==="
Start-Sleep -Seconds 5
