# Windows entry point.  .\run.ps1 quick | full | test
param([string]$what = "quick")
$env:PYTHONUTF8 = "1"
$py = ".\.venv\Scripts\python.exe"
switch ($what) {
  "quick" { & $py -m eval.run_all --quick }
  "full"  { & $py -m eval.run_all }
  "test"  { & $py -m pytest -q tests }
}
