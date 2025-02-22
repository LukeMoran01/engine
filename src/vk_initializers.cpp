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

VkImageCreateInfo vkinit::createImageCreateInfo(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent) {
    VkImageCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    createInfo.pNext     = nullptr;
    createInfo.imageType = VK_IMAGE_TYPE_2D;

    createInfo.format = format;
    createInfo.extent = extent;

    createInfo.mipLevels   = 1;
    createInfo.arrayLayers = 1;

    createInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    createInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    createInfo.usage  = usageFlags;

    return createInfo;
}

VkImageViewCreateInfo
vkinit::createImageViewCreateInfo(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags) {
    VkImageViewCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    createInfo.pNext = nullptr;

    createInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format                          = format;
    createInfo.image                           = image;
    createInfo.subresourceRange.baseMipLevel   = 0;
    createInfo.subresourceRange.levelCount     = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount     = 1;
    createInfo.subresourceRange.aspectMask     = aspectFlags;

    return createInfo;
}

VkRenderingAttachmentInfo vkinit::createRenderAttachmentInfo(VkImageView view, VkClearValue* clear,
                                                             VkImageLayout layout) {
    VkRenderingAttachmentInfo colorAttachment{.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.pNext = nullptr;

    colorAttachment.imageView   = view;
    colorAttachment.imageLayout = layout;
    colorAttachment.loadOp      = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    if (clear) {
        colorAttachment.clearValue = *clear;
    }
    return colorAttachment;
}

VkRenderingAttachmentInfo vkinit::createDepthAttachmentInfo(VkImageView view, VkImageLayout layout) {
    VkRenderingAttachmentInfo depthAttachment{.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.pNext = nullptr;

    depthAttachment.imageView                     = view;
    depthAttachment.imageLayout                   = layout;
    depthAttachment.loadOp                        = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp                       = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil.depth = 0.f;
    return depthAttachment;
}

VkRenderingInfo vkinit::createRenderInfo(VkExtent2D extent, VkRenderingAttachmentInfo* colorAttachment,
                                         VkRenderingAttachmentInfo* depthAttachment) {
    VkRenderingInfo renderInfo{.sType = VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderInfo.pNext = nullptr;

    renderInfo.renderArea           = VkRect2D{VkOffset2D{0, 0}, extent};
    renderInfo.layerCount           = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments    = colorAttachment;
    renderInfo.pDepthAttachment     = depthAttachment;
    renderInfo.pStencilAttachment   = nullptr;

    return renderInfo;
}

VkPipelineShaderStageCreateInfo vkinit::createPipelineShaderStageCreateInfo(
    VkShaderStageFlagBits stage, VkShaderModule shaderModule, const char* entry) {
    VkPipelineShaderStageCreateInfo info{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.pNext = nullptr;

    info.stage  = stage;
    info.module = shaderModule;
    info.pName  = entry;
    return info;
}

VkPipelineLayoutCreateInfo vkinit::createPipelineLayoutCreateInfo() {
    VkPipelineLayoutCreateInfo info{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    info.pNext = nullptr;

    info.flags                  = 0;
    info.setLayoutCount         = 0;
    info.pSetLayouts            = nullptr;
    info.pushConstantRangeCount = 0;
    info.pPushConstantRanges    = nullptr;
    return info;
}
