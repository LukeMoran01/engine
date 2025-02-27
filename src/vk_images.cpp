//
// Created by Luke Moran on 23/11/2024.
//

#include <vk_images.h>

#include "vk_initializers.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::optional<AllocatedImage> loadImage(VulkanRenderer* engine, fastgltf::Asset& asset, fastgltf::Image& image) {
    AllocatedImage newImage{};

    int width, height, nrChannels;

    std::visit(
        fastgltf::visitor{
            [](auto& arg) {
            },
            [&](fastgltf::sources::URI& filePath) {
                assert(filePath.fileByteOffset == 0);
                assert(filePath.uri.isLocalPath());

                const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());
                unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
                if (data) {
                    VkExtent3D imageSize;
                    imageSize.width  = width;
                    imageSize.height = height;
                    imageSize.depth  = 1;

                    newImage = engine->createImage(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM,
                                                   VK_IMAGE_USAGE_SAMPLED_BIT, false);
                    stbi_image_free(data);
                }
            },
            [&](fastgltf::sources::Array& array) {
                // bytes* to unsigned char* is well-defined
                unsigned char* data = stbi_load_from_memory(reinterpret_cast<stbi_uc const*>(array.bytes.data()),
                                                            static_cast<int>(array.bytes.size()),
                                                            &width, &height, &nrChannels, 4
                );
                if (data) {
                    VkExtent3D imageSize;
                    imageSize.width  = width;
                    imageSize.height = height;
                    imageSize.depth  = 1;

                    newImage = engine->createImage(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM,
                                                   VK_IMAGE_USAGE_SAMPLED_BIT, false);
                    stbi_image_free(data);
                }
            },
            [&](fastgltf::sources::BufferView& view) {
                auto& bufferView = asset.bufferViews[view.bufferViewIndex];
                auto& buffer     = asset.buffers[bufferView.bufferIndex];

                std::visit(fastgltf::visitor{
                               [](auto& arg) {
                               },
                               [&](fastgltf::sources::Array& array) {
                                   unsigned char* data = stbi_load_from_memory(
                                       reinterpret_cast<stbi_uc const*>(array.bytes.data()) + bufferView.byteOffset,
                                       static_cast<int>(bufferView.byteLength),
                                       &width, &height, &nrChannels, 4
                                   );
                                   if (data) {
                                       VkExtent3D imageSize;
                                       imageSize.width  = width;
                                       imageSize.height = height;
                                       imageSize.depth  = 1;

                                       newImage = engine->createImage(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM,
                                                                      VK_IMAGE_USAGE_SAMPLED_BIT, false);
                                       stbi_image_free(data);
                                   }
                               }
                           }, buffer.data);
            },
        }, image.data);
    if (newImage.image == VK_NULL_HANDLE) {
        return {};
    }
    return newImage;
}


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

// TODO: apparently blit and copy should be replaced by own version later on that can do extra logic on a fragment buffer
void vkutil::copyImageToImage(VkCommandBuffer buffer, VkImage source, VkImage destination, VkExtent2D srcSize,
                              VkExtent2D dstSize) {
    VkImageBlit2 blitRegion{.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2};

    blitRegion.srcOffsets[1].x = static_cast<int32_t>(srcSize.width);
    blitRegion.srcOffsets[1].y = static_cast<int32_t>(srcSize.height);
    blitRegion.srcOffsets[1].z = 1;

    blitRegion.dstOffsets[1].x = static_cast<int32_t>(dstSize.width);
    blitRegion.dstOffsets[1].y = static_cast<int32_t>(dstSize.height);
    blitRegion.dstOffsets[1].z = 1;

    blitRegion.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.baseArrayLayer = 0;
    blitRegion.srcSubresource.layerCount     = 1;
    blitRegion.srcSubresource.mipLevel       = 0;

    blitRegion.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.baseArrayLayer = 0;
    blitRegion.dstSubresource.layerCount     = 1;
    blitRegion.dstSubresource.mipLevel       = 0;

    VkBlitImageInfo2 blitInfo{.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2};
    blitInfo.srcImage       = source;
    blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    blitInfo.dstImage       = destination;
    blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    blitInfo.filter         = VK_FILTER_LINEAR;
    blitInfo.regionCount    = 1;
    blitInfo.pRegions       = &blitRegion;

    vkCmdBlitImage2(buffer, &blitInfo);
}
