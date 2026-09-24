# Configure and build Gufo on Windows with the TheRock ROCm SDK and vcpkg.
# See docs/WINDOWS.md.
param(
  [string]$Rocm = "C:/rocm-sdk/rocm",
  [string]$Vcpkg = "C:/vcpkg",
  [string]$VisualStudio = "",  # install root; empty = newest found by vswhere
  [string]$BuildDir = "build/win-release",
  [int]$Jobs = 8,
  [switch]$ConfigureOnly
)
$ErrorActionPreference = "Stop"

if (-not $VisualStudio) {
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found; pass -VisualStudio <install root>" }
  $VisualStudio = & $vswhere -latest -prerelease -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (-not $VisualStudio) { throw "no Visual Studio with the C++ x64 tools found; pass -VisualStudio" }
}
$env:PATH = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer;$env:PATH"
$cmake = "$VisualStudio\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = "$VisualStudio\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

# clang targets the MSVC ABI and needs the MSVC/SDK library environment.
$vcvars = "$VisualStudio\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}
$env:HIP_PATH = $Rocm
$env:PATH = "$Rocm/bin;$Rocm/lib/llvm/bin;$Vcpkg/installed/x64-windows/bin;$env:PATH"

$devlib = "$Rocm/lib/llvm/amdgcn/bitcode"
& $cmake -S . -B $BuildDir -G Ninja `
  "-DCMAKE_MAKE_PROGRAM=$ninja" `
  "-DCMAKE_BUILD_TYPE=Release" `
  "-DCMAKE_TOOLCHAIN_FILE=$Vcpkg/scripts/buildsystems/vcpkg.cmake" `
  "-DVCPKG_TARGET_TRIPLET=x64-windows" `
  "-DCMAKE_C_COMPILER=$Rocm/lib/llvm/bin/clang.exe" `
  "-DCMAKE_CXX_COMPILER=$Rocm/lib/llvm/bin/clang++.exe" `
  "-DCMAKE_HIP_COMPILER=$Rocm/lib/llvm/bin/clang++.exe" `
  "-DCMAKE_HIP_COMPILER_ROCM_ROOT=$Rocm" `
  "-DCMAKE_HIP_FLAGS=--rocm-device-lib-path=$devlib" `
  "-DCMAKE_PREFIX_PATH=$Rocm" `
  "-DCMAKE_HIP_ARCHITECTURES=gfx1151" `
  "-DGUFO_ENABLE_WARNINGS=OFF" `
  "-DENGINE_ENABLE_HIP=ON"
if ($LASTEXITCODE -ne 0) { throw "configure failed" }
if ($ConfigureOnly) { return }
& $cmake --build $BuildDir --target gufo -j $Jobs
if ($LASTEXITCODE -ne 0) { throw "build failed" }
