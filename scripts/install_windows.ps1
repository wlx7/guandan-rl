param([switch]$Quiet, [switch]$ReuseCache)
# 掼蛋 AI 一键安装：uv + Python 3.12 虚拟环境 + PyTorch（自动选择 CUDA 版本）+ guandan-rl
$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$log = Join-Path $PSScriptRoot 'install_log.txt'
Start-Transcript -Path $log -Force | Out-Null
function Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Yellow }
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$tuna = 'https://pypi.tuna.tsinghua.edu.cn/simple'
# Everything stays inside this folder: uv, the Python interpreter, the download cache and the
# training data. Put the folder on any drive (e.g. D:\guandan) and nothing lands on C:.
$uvHome = Join-Path $PSScriptRoot '.uv'
# -ReuseCache: take packages from the user's existing uv cache (used once when moving an
# older installation, so that PyTorch is copied instead of downloaded again)
if (-not $ReuseCache) { $env:UV_CACHE_DIR = Join-Path $uvHome 'cache' }
$env:UV_PYTHON_INSTALL_DIR = Join-Path $uvHome 'python'
$env:UV_INSTALL_DIR = Join-Path $uvHome 'bin'
$env:UV_NO_MODIFY_PATH = '1'
$env:GUANDAN_HOME = Join-Path $PSScriptRoot 'guandan_data'
try {
  $env:Path = "$env:UV_INSTALL_DIR;$env:USERPROFILE\.local\bin;$env:Path"
  Step '1/4 安装 uv（快速的 Python 包管理器）'
  if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    try { Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression } catch { Write-Host $_ }
  }
  if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $zip = Join-Path $env:TEMP 'uv.zip'
    Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip
    New-Item -ItemType Directory -Force -Path $env:UV_INSTALL_DIR | Out-Null
    Expand-Archive $zip -DestinationPath $env:UV_INSTALL_DIR -Force
  }
  uv --version 2>&1 | ForEach-Object { "$_" } | Out-Host
  if ($LASTEXITCODE -ne 0) { throw 'uv 安装失败' }

  Step '2/4 创建 Python 3.12 环境（.venv）'
  if (-not (Test-Path $py)) {
    uv venv .venv --python 3.12 --python-preference only-managed --no-progress 2>&1 | ForEach-Object { "$_" } | Out-Host
    if (-not (Test-Path $py)) {
      Write-Host '官方源下载失败，改用国内镜像'
      $env:UV_PYTHON_INSTALL_MIRROR = 'https://registry.npmmirror.com/-/binary/python-build-standalone'
      uv venv .venv --python 3.12 --python-preference only-managed --no-progress 2>&1 | ForEach-Object { "$_" } | Out-Host
    }
  }
  if (-not (Test-Path $py)) { throw 'Python 环境创建失败' }
  & $py --version 2>&1 | ForEach-Object { "$_" } | Out-Host

  Step '3/4 安装 PyTorch'
  $ok = $false
  & $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
  if ($LASTEXITCODE -eq 0) { $ok = $true; Write-Host '已安装 CUDA 版 PyTorch，跳过' }
  $hasGpu = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
  if ($hasGpu) { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | ForEach-Object { "$_" } | Out-Host }
  if (-not $ok -and $hasGpu) {
    foreach ($cu in @('cu126', 'cu124', 'cu121', 'cu118')) {
      Write-Host "尝试安装 PyTorch ($cu)，文件较大（约 2.5GB），请耐心等待…"
      uv pip install --python "$py" --reinstall-package torch torch --index-url "https://download.pytorch.org/whl/$cu" 2>&1 | ForEach-Object { "$_" } | Out-Host
      if ($LASTEXITCODE -ne 0) {
        Write-Host "官方源失败，尝试阿里云镜像 ($cu)"
        uv pip install --python "$py" --reinstall-package torch torch -f "https://mirrors.aliyun.com/pytorch-wheels/$cu/" -i $tuna 2>&1 | ForEach-Object { "$_" } | Out-Host
      }
      & $py -c "import torch; assert torch.cuda.is_available(); x = torch.randn(512, 512, device='cuda'); print('CUDA OK', torch.__version__, torch.cuda.get_device_name(0), float((x @ x).sum()))" 2>&1 | ForEach-Object { "$_" } | Out-Host
      if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    }
  }
  if (-not $ok) {
    Write-Host '安装 CPU 版 PyTorch'
    uv pip install --python "$py" torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | ForEach-Object { "$_" } | Out-Host
    if ($LASTEXITCODE -ne 0) { uv pip install --python "$py" torch -i $tuna }
  }

  Step '4/4 安装 掼蛋 AI'
  # pick the highest version among GitHub release assets and wheels already in this folder
  $cands = @()
  try {
    $stamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $rels = Invoke-RestMethod "https://api.github.com/repos/wlx7/guandan-rl/releases?per_page=20&t=$stamp" -Headers @{ 'User-Agent' = 'guandan-installer'; 'Cache-Control' = 'no-cache' }
    foreach ($r in @($rels)) { foreach ($a in @($r.assets)) { if ($a.name -like '*win_amd64.whl') { $cands += [pscustomobject]@{ Name = $a.name; Url = $a.browser_download_url } } } }
  } catch { Write-Host "无法访问 GitHub：$_" }
  foreach ($f in @(Get-ChildItem -Path $PSScriptRoot -Filter 'guandan_rl-*win_amd64.whl' -ErrorAction SilentlyContinue)) { $cands += [pscustomobject]@{ Name = $f.Name; Url = $null } }
  $best = $cands | Where-Object { $_.Name -match 'guandan_rl-(\d+\.\d+\.\d+)' } | Sort-Object { [version]([regex]::Match($_.Name, 'guandan_rl-(\d+\.\d+\.\d+)').Groups[1].Value) } -Descending | Select-Object -First 1
  if (-not $best) { throw '找不到 Windows 安装包（guandan_rl-*-win_amd64.whl）' }
  $src = Join-Path $PSScriptRoot $best.Name
  if (-not (Test-Path $src)) {
    Write-Host "下载 $($best.Url)"
    Invoke-WebRequest $best.Url -OutFile $src
  }
  Write-Host "安装 $src"
  uv pip install --python "$py" --reinstall-package guandan-rl "$src" 2>&1 | ForEach-Object { "$_" } | Out-Host
  if ($LASTEXITCODE -ne 0) { uv pip install --python "$py" --reinstall-package guandan-rl "$src" -i $tuna }
  & $py -m guandan info 2>&1 | ForEach-Object { "$_" } | Out-Host
  if ($LASTEXITCODE -ne 0) { throw '安装验证失败' }
  Write-Host "`nINSTALL_OK  安装完成！双击 start_windows.bat 启动。" -ForegroundColor Green
} catch {
  Write-Host "INSTALL_FAILED: $_" -ForegroundColor Red
}
Stop-Transcript | Out-Null
if (-not $Quiet) { Read-Host '按回车键关闭窗口' }
