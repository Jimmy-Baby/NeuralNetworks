-- premake5.lua
workspace "NeuralNetworks"
   architecture "x64"
   configurations { "Debug", "Release", "Dist" }
   startproject "NeuralNetworks"

   -- Workspace-wide build options for MSVC
   filter "system:windows"
      buildoptions { "/EHsc", "/Zc:preprocessor", "/Zc:__cplusplus" }

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

include "Build-External.lua"
include "NeuralNetworks/Build-NeuralNetworks.lua"