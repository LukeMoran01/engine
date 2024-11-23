//
// Created by Luke Moran on 20/11/2024.
//

#include <vk_initializers.h>

VkCommandPoolCreateInfo vkinit::createCommandPoolCreateInfo(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags) {
    VkCommandPoolCreateInfo commandPoolInfo{};
    commandPoolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.flags            = flags;
    commandPoolInfo.queueFamilyIndex = queueFamilyIndex;
    commandPoolInfo.pNext            = nullptr;

    return commandPoolInfo;
}

VkCommandBufferAllocateInfo vkinit::createCommandBufferAllocInfo(VkCommandPool commandPool,
                                                                 uint32_t commandBufferCount) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.pNext              = nullptr;
    allocInfo.commandPool        = commandPool;
    allocInfo.commandBufferCount = commandBufferCount;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    return allocInfo;
}

VkCommandBufferBeginInfo vkinit::createCommandBufferBeginInfo(VkCommandBufferUsageFlags flags) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.pNext            = nullptr;
    beginInfo.pInheritanceInfo = nullptr;
    beginInfo.flags            = flags;

    return beginInfo;
}

VkCommandBufferSubmitInfo vkinit::createCommandBufferSubmitInfo(VkCommandBuffer commandBuffer) {
    VkCommandBufferSubmitInfo submitInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    submitInfo.pNext         = nullptr;
    submitInfo.commandBuffer = commandBuffer;
    submitInfo.deviceMask    = 0;

    return submitInfo;
}

VkFenceCreateInfo vkinit::createFenceCreateInfo(VkFenceCreateFlags flags) {
    VkFenceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = flags;

    return createInfo;
}

VkSemaphoreCreateInfo vkinit::createSemaphoreCreateInfo(VkSemaphoreCreateFlags flags) {
    VkSemaphoreCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = flags;

    return createInfo;
}

VkSemaphoreSubmitInfo vkinit::createSemaphoreSubmitInfo(VkPipelineStageFlags2 stageMask, VkSemaphore semaphore) {
    VkSemaphoreSubmitInfo submitInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    submitInfo.pNext       = nullptr;
    submitInfo.semaphore   = semaphore;
    submitInfo.stageMask   = stageMask;
    submitInfo.deviceIndex = 0;
    submitInfo.value       = 1;

    return submitInfo;
}

VkImageSubresourceRange vkinit::createImageSubresourceRange(VkImageAspectFlags aspectMask) {
    VkImageSubresourceRange subresourceRange{};
    subresourceRange.aspectMask     = aspectMask;
    subresourceRange.baseMipLevel   = 0;
    subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

    return subresourceRange;
}

VkSubmitInfo2 vkinit::createSubmitInfo(VkCommandBufferSubmitInfo* bufferInfo,
                                       VkSemaphoreSubmitInfo* signalSemaphoreInfo,
                                       VkSemaphoreSubmitInfo* waitSemaphoreInfo) {
    VkSubmitInfo2 submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submitInfo.pNext = nullptr;

    submitInfo.waitSemaphoreInfoCount = waitSemaphoreInfo ? 1 : 0;
    submitInfo.pWaitSemaphoreInfos    = waitSemaphoreInfo;

    submitInfo.signalSemaphoreInfoCount = signalSemaphoreInfo ? 1 : 0;
    submitInfo.pSignalSemaphoreInfos    = signalSemaphoreInfo;

    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos    = bufferInfo;

    return submitInfo;
}

