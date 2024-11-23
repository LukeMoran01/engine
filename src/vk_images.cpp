//
// Created by Luke Moran on 23/11/2024.
//

#include <vk_images.h>

#include "vk_initializers.h"

void vkutil::transitionImage(VkCommandBuffer buffer, VkImage image, VkImageLayout currentLayout,
                             VkImageLayout newLayout) {
    VkImageMemoryBarrier2 imageBarrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    imageBarrier.pNext = nullptr;

    // TODO: BAD AND INEFFICIENT SWITCH TO SPECIFIC TRANSITIONS LATER
    // https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples
    imageBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    imageBarrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    imageBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

    imageBarrier.oldLayout = currentLayout;
    imageBarrier.newLayout = newLayout;

    VkImageAspectFlags aspectMask = newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                                        ? VK_IMAGE_ASPECT_DEPTH_BIT
                                        : VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange = vkinit::createImageSubresourceRange(aspectMask);
    imageBarrier.image            = image;

    VkDependencyInfo dependencyInfo{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependencyInfo.pNext                   = nullptr;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers    = &imageBarrier;

    vkCmdPipelineBarrier2(buffer, &dependencyInfo);
}
