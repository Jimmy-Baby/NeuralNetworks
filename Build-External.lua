-- VULKAN_SDK = os.getenv("VULKAN_SDK")

IncludeDir = {}
-- Example include dir
-- IncludeDir["VulkanSDK"] = "%{VULKAN_SDK}/Include"

LibraryDir = {}
-- Example library directory declaration
-- LibraryDir["VulkanSDK"] = "%{VULKAN_SDK}/Lib"

Library = {}
-- Example library declaration
-- Library["Vulkan"] = "%{LibraryDir.VulkanSDK}/vulkan-1.lib"

group "Dependencies"
   include "vendor/ExampleLib"
   include "vendor/Eigen3"
group ""

group "Core"
    -- Any core library luas
group ""