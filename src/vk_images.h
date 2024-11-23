//
// Created by Luke Moran on 23/11/2024.
//

#pragma once

#include <vulkan/vulkan.h>

namespace vkutil {
    void transitionImage(VkCommandBuffer buffer, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);
}
