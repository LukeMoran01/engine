//
// Created by Luke Moran on 23/11/2024.
//

#pragma once

#include <vulkan/vulkan.h>
#include "vk_engine.h"
#include "fastgltf/types.hpp"

std::optional<AllocatedImage> loadImage(VulkanRenderer* engine, fastgltf::Asset& asset, fastgltf::Image& image);

namespace vkutil {
    void transitionImage(VkCommandBuffer buffer, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);
    void copyImageToImage(VkCommandBuffer buffer, VkImage source, VkImage destination, VkExtent2D srcSize,
                          VkExtent2D dstSize);
}
