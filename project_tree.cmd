@echo off
setlocal EnableExtensions

rem ============== project_tree.cmd =================
rem Uso:
rem   project_tree           -> crea project_tree_YYYY-MM-DD_HHMMSS.txt
rem   project_tree /open     -> además lo abre en Notepad
rem   project_tree /notime   -> usa nombre fijo project_tree.txt
rem =================================================

set "do_open=0"
set "use_ts=1"
for %%A in (%*) do (
  if /i "%%~A"=="/OPEN"   set "do_open=1"
  if /i "%%~A"=="/NOTIME" set "use_ts=0"
)

set "out=project_tree.txt"

> "%out%" (
  echo Project tree for: %CD%
  echo Generated on: %date% %time%
  echo.
  rem /f = incluye archivos, /a = ASCII
  tree /f /a
)

echo Created "%out%"
if "%do_open%"=="1" start "" notepad "%out%"
endlocal & exit /b
