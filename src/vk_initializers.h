//
// Created by Luke Moran on 20/11/2024.
//

#pragma once

#include <vk_types.h>

namespace vkinit {
    VkCommandPoolCreateInfo createCommandPoolCreateInfo(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags);

    VkCommandBufferAllocateInfo createCommandBufferAllocInfo(VkCommandPool commandPool, uint32_t commandBufferCount);
    VkCommandBufferBeginInfo createCommandBufferBeginInfo(VkCommandBufferUsageFlags flags);
    VkCommandBufferSubmitInfo createCommandBufferSubmitInfo(VkCommandBuffer commandBuffer);

    VkFenceCreateInfo createFenceCreateInfo(VkFenceCreateFlags flags);
    VkSemaphoreCreateInfo createSemaphoreCreateInfo(VkSemaphoreCreateFlags flags);
    VkSemaphoreSubmitInfo createSemaphoreSubmitInfo(VkPipelineStageFlags2 stageMask, VkSemaphore semaphore);

    VkSubmitInfo2 createSubmitInfo(VkCommandBufferSubmitInfo* bufferInfo,
                                   VkSemaphoreSubmitInfo* signalSemaphoreInfo,
                                   VkSemaphoreSubmitInfo* waitSemaphoreInfo);

    VkImageSubresourceRange createImageSubresourceRange(VkImageAspectFlags flags);

    VkImageCreateInfo createImageCreateInfo(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent);
    VkImageViewCreateInfo createImageViewCreateInfo(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags);
    VkRenderingAttachmentInfo createRenderAttachmentInfo(VkImageView view, VkClearValue* clear, VkImageLayout layout);
    VkRenderingInfo createRenderInfo(VkExtent2D extent, VkRenderingAttachmentInfo* colorAttachment,
                                     VkRenderingAttachmentInfo* depthAttachment);

    VkPipelineShaderStageCreateInfo createPipelineShaderStageCreateInfo(VkShaderStageFlagBits stage,
                                                                        VkShaderModule shaderModule, const char* entry);

    VkPipelineLayoutCreateInfo createPipelineLayoutCreateInfo();
}
