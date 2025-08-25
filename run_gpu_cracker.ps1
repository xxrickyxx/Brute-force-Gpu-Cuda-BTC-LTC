# Configurazione completa ambiente CUDA + Visual Studio per PyCUDA
Write-Host "Configurazione ambiente CUDA + Visual Studio..." -ForegroundColor Yellow

# 1. Configura Visual Studio Build Tools 2019
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $vsPath)) {
    Write-Host "Visual Studio 2019 Build Tools non trovato!" -ForegroundColor Red
    exit 1
}

# 2. Configura variabili ambiente CUDA
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"
$env:CUDA_LIB_PATH = "$env:CUDA_PATH\lib\x64"

# 3. Aggiorna PATH con CUDA e Visual Studio 2019
$vsCompilerPath = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
$env:PATH = "$env:CUDA_PATH\bin;$vsCompilerPath;$env:PATH"

# 4. Configura PyCUDA
$env:PYCUDA_CACHE_DIR = "$env:TEMP\pycuda-cache"
$env:PYCUDA_COMPILER_FLAGS = ""
$env:NVCC_OPTIONS = "--compiler-options -nologo"

# 5. Pulisci cache PyCUDA
if (Test-Path $env:PYCUDA_CACHE_DIR) {
    Remove-Item $env:PYCUDA_CACHE_DIR -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cache PyCUDA pulita" -ForegroundColor Green
}

# 6. Verifica compilatori
Write-Host "`n=== VERIFICA AMBIENTE ===" -ForegroundColor Cyan

Write-Host "Cerco cl.exe..." -ForegroundColor Yellow
$clPath = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($clPath) {
    Write-Host "cl.exe trovato: $($clPath.Source)" -ForegroundColor Green
} else {
    Write-Host "cl.exe NON TROVATO nel PATH!" -ForegroundColor Red
    Write-Host "PATH corrente: $env:PATH" -ForegroundColor Gray
    exit 1
}

Write-Host "Cerco nvcc.exe..." -ForegroundColor Yellow
$nvccPath = Get-Command nvcc.exe -ErrorAction SilentlyContinue
if ($nvccPath) {
    Write-Host "nvcc.exe trovato: $($nvccPath.Source)" -ForegroundColor Green
} else {
    Write-Host "nvcc.exe NON TROVATO nel PATH!" -ForegroundColor Red
    exit 1
}

# 7. Mostra versioni
Write-Host "`n=== VERSIONI COMPILATORI ===" -ForegroundColor Cyan
try {
    Write-Host "Visual C++ Compiler:" -ForegroundColor Yellow
    & cl.exe 2>&1 | Select-Object -First 2
    
    Write-Host "`nNVIDIA CUDA Compiler:" -ForegroundColor Yellow
    & nvcc.exe --version | Select-Object -Last 4
} catch {
    Write-Host "Errore verifica versioni: $_" -ForegroundColor Orange
}

# 8. Test rapido PyCUDA
Write-Host "`n=== TEST PYCUDA ===" -ForegroundColor Cyan
try {
    Write-Host "Test importazione PyCUDA..." -ForegroundColor Yellow
    $testScript = @"
try:
    import pycuda.driver as cuda
    print('PyCUDA driver importato')
    cuda.init()
    print(f'GPU count: {cuda.Device.count()}')
    if cuda.Device.count() > 0:
        dev = cuda.Device(0)
        print(f'GPU: {dev.name()}')
        print(f'Compute: {dev.compute_capability()}')
except Exception as e:
    print(f'Errore PyCUDA: {e}')
    exit(1)
"@
    
    & G:\wallet_decrypt\.venv-pycuda\Scripts\python.exe -c $testScript
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PyCUDA funziona correttamente!" -ForegroundColor Green
    } else {
        Write-Host "PyCUDA ha problemi!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Errore test PyCUDA: $_" -ForegroundColor Red
    exit 1
}

# 9. Controllo PATH e presenza cl.exe prima di avviare il cracker
Write-Host "PATH effettivo:" -ForegroundColor Gray
Write-Host $env:PATH -ForegroundColor Gray
Write-Host "Controllo presenza cl.exe nel PATH..." -ForegroundColor Yellow
$clTest = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($clTest) {
    Write-Host "cl.exe trovato: $($clTest.Source)" -ForegroundColor Green
} else {
    Write-Host "cl.exe NON TROVATO nel PATH!" -ForegroundColor Red
    exit 1
}

# 10. Avvia il cracker GPU
Write-Host "`nAVVIO GPU BITCOIN CRACKER..." -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green

try {
    & G:\wallet_decrypt\.venv-pycuda\Scripts\python.exe G:\wallet_decrypt\true_gpu_cracker_real.py G:\wallet_decrypt\1.001.dat
} catch {
    Write-Host "Errore esecuzione cracker: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`nCracker GPU terminato." -ForegroundColor Cyan
