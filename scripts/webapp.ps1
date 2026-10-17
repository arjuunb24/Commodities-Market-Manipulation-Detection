<#
.SYNOPSIS
  Start, stop or check the Veridex web app hosted from this PC.

.EXAMPLE
  scripts\webapp start                          # this PC only:  http://localhost:8000
  scripts\webapp start -Lan                     # also phones/laptops on the same Wi-Fi
  scripts\webapp start -Public -Password s3cret # an internet link (Cloudflare quick tunnel)
  scripts\webapp status
  scripts\webapp stop

.NOTES
  The server and tunnel run hidden in the background; closing this window does not stop them.
  Logs and state live in .webapp\ (git-ignored). Works in Windows PowerShell 5.1 and PowerShell 7.
#>
param(
    [Parameter(Position = 0)][ValidateSet("start", "stop", "restart", "status")][string]$Action = "status",
    [switch]$Lan,
    [switch]$Public,
    [string]$Password = "",
    [int]$Port = 8000,
    [switch]$Build
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$StateDir = Join-Path $Root ".webapp"
$StateFile = Join-Path $StateDir "state.json"
$Python = Join-Path $Root ".venv\Scripts\python.exe"

function Get-AppState {
    if (Test-Path $StateFile) { return Get-Content $StateFile -Raw | ConvertFrom-Json }
    return $null
}

function Test-Alive($id) {
    if (-not $id) { return $false }
    return [bool](Get-Process -Id $id -ErrorAction SilentlyContinue)
}

function Show-Status {
    $s = Get-AppState
    if (-not $s -or -not (Test-Alive $s.server)) {
        Write-Host "Veridex web app: OFF" -ForegroundColor DarkGray
        return
    }
    Write-Host "Veridex web app: ON" -ForegroundColor Green
    Write-Host "  This PC : $($s.local)"
    if ($s.lan) { Write-Host "  Wi-Fi   : $($s.lan)" }
    if ($s.public) {
        if (Test-Alive $s.tunnel) { Write-Host "  Internet: $($s.public)" -ForegroundColor Cyan }
        else { Write-Host "  Internet: tunnel stopped (see .webapp\tunnel.log)" -ForegroundColor Yellow }
    }
    if ($s.password) { Write-Host "  Password: required (any user name)" }
    Write-Host "  Since   : $($s.started)"
}

function Stop-App {
    $s = Get-AppState
    if (-not $s) { Write-Host "Veridex web app is not running."; return }
    foreach ($id in @($s.tunnel, $s.server)) {
        if (Test-Alive $id) { & taskkill.exe /PID $id /T /F | Out-Null }
    }
    Remove-Item $StateFile -Force
    Write-Host "Veridex web app stopped." -ForegroundColor DarkGray
}

function Find-Cloudflared {
    $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @("${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe", "$env:ProgramFiles\cloudflared\cloudflared.exe",
                     "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cloudflared.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Start-App {
    $s = Get-AppState
    if ($s -and (Test-Alive $s.server)) {
        Write-Host "Already running. Use 'scripts\webapp restart' to change options." -ForegroundColor Yellow
        Show-Status
        return
    }
    if (-not (Test-Path $Python)) { throw "Python venv not found at $Python" }

    $cloudflared = $null
    if ($Public) {
        if (-not $Password) {
            throw "-Public needs -Password: anyone with the link could otherwise use the analyst chat on your LLM key."
        }
        $cloudflared = Find-Cloudflared
        if (-not $cloudflared) {
            throw "cloudflared is not installed. Install it once with:  winget install --id Cloudflare.cloudflared   then open a new terminal."
        }
    }

    $busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($busy) { throw "Port $Port is already in use (another uvicorn?). Stop it or pass -Port 8010." }

    $dist = Join-Path $Root "web\dist\index.html"
    if ($Build -or -not (Test-Path $dist)) {
        Write-Host "Building the web app..."
        Push-Location (Join-Path $Root "web")
        try {
            if (-not (Test-Path "node_modules")) { & npm install --no-audit --no-fund | Out-Host }
            & npm run build | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
        } finally { Pop-Location }
    }

    New-Item -ItemType Directory -Force $StateDir | Out-Null
    $bind = "127.0.0.1"
    if ($Lan) { $bind = "0.0.0.0" }

    # The child process inherits this variable; clear it again right after launch.
    $env:VERIDEX_PASSWORD = $Password
    try {
        $server = Start-Process -FilePath $Python -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
            -ArgumentList @("-m", "uvicorn", "api.main:app", "--host", $bind, "--port", "$Port") `
            -RedirectStandardOutput (Join-Path $StateDir "server.log") -RedirectStandardError (Join-Path $StateDir "server.err.log")
    } finally { Remove-Item Env:VERIDEX_PASSWORD -ErrorAction SilentlyContinue }

    Write-Host "Starting server" -NoNewline
    $up = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Milliseconds 700
        Write-Host "." -NoNewline
        if (-not (Test-Alive $server.Id)) { break }
        if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { $up = $true; break }
    }
    Write-Host ""
    if (-not $up) {
        Get-Content (Join-Path $StateDir "server.err.log") -Tail 15 -ErrorAction SilentlyContinue | Out-Host
        if (Test-Alive $server.Id) { & taskkill.exe /PID $server.Id /T /F | Out-Null }
        throw "The server did not start (log above, full log in .webapp\server.err.log)."
    }

    $state = [ordered]@{
        server   = $server.Id
        tunnel   = $null
        local    = "http://localhost:$Port"
        lan      = $null
        public   = $null
        password = [bool]$Password
        started  = (Get-Date).ToString("yyyy-MM-dd HH:mm")
    }

    if ($Lan) {
        $ip = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -in @("Dhcp", "Manual") } |
            Select-Object -First 1 -ExpandProperty IPAddress
        if ($ip) { $state.lan = "http://${ip}:$Port" }
    }

    if ($Public) {
        $log = Join-Path $StateDir "tunnel.log"
        Remove-Item $log -ErrorAction SilentlyContinue
        $tunnel = Start-Process -FilePath $cloudflared -WindowStyle Hidden -PassThru `
            -ArgumentList @("tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:$Port") `
            -RedirectStandardError $log -RedirectStandardOutput (Join-Path $StateDir "tunnel.out.log")
        $state.tunnel = $tunnel.Id
        Write-Host "Opening tunnel" -NoNewline
        for ($i = 0; $i -lt 60 -and -not $state.public; $i++) {
            Start-Sleep -Milliseconds 700
            Write-Host "." -NoNewline
            $m = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($m) { $state.public = $m.Matches[0].Value }
        }
        Write-Host ""
        if (-not $state.public) { Write-Host "Tunnel did not report a URL yet; check .webapp\tunnel.log" -ForegroundColor Yellow }
    }

    $state | ConvertTo-Json | Set-Content $StateFile
    Show-Status
    if ($Lan) { Write-Host "  If other devices cannot connect, allow Python through Windows Firewall (Private networks)." -ForegroundColor DarkGray }
    Write-Host "  Stop with: scripts\webapp stop" -ForegroundColor DarkGray
}

try {
    switch ($Action) {
        "start"   { Start-App }
        "stop"    { Stop-App }
        "restart" { Stop-App; Start-Sleep -Seconds 1; Start-App }
        "status"  { Show-Status }
    }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
