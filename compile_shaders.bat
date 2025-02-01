@echo off
setlocal enabledelayedexpansion

:: Define paths
set SHADER_SRC=shaders/source
set SHADER_OUT=shaders/compiled

:: Ensure the output directory exists
if not exist "%SHADER_OUT%" mkdir "%SHADER_OUT%"

echo Compiling shaders...

:: Loop through shader files and compile them
for %%F in (%SHADER_SRC%\vertex\*.vert) do (
    set FILE=%%~nxF
    echo Compiling vertex shader: !FILE!
    glslc %%F -o %SHADER_OUT%\!FILE!.spv
)

for %%F in (%SHADER_SRC%\fragment\*.frag) do (
    set FILE=%%~nxF
    echo Compiling fragment shader: !FILE!
    glslc %%F -o %SHADER_OUT%\!FILE!.spv
)

for %%F in (%SHADER_SRC%\compute\*.comp) do (
    set FILE=%%~nxF
    echo Compiling compute shader: !FILE!
    glslc %%F -o %SHADER_OUT%\!FILE!.spv
)

echo Shader compilation complete!
