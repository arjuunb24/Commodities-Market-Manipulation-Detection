@echo off
rem Wrapper so "scripts\webapp start" works without changing the PowerShell execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0webapp.ps1" %*
