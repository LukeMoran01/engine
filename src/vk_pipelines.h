//
// Created by Luke Moran on 23/11/2024.
//

#pragma once
#include <vulkan/vulkan_core.h>

namespace vkutil {
    bool loadShaderModule(const char* filepath, VkDevice device, VkShaderModule* shaderModule);
}
