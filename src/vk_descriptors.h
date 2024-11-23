//
// Created by Luke Moran on 23/11/2024.
//

#pragma once
#include <span>
#include <vector>
#include <vulkan/vulkan_core.h>

struct DescriptorLayoutBuilder {
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    void addBinding(uint32_t binding, VkDescriptorType type);
    void clear();
    VkDescriptorSetLayout build(VkDevice device, VkShaderStageFlags shaderStages, void* pNext,
                                VkDescriptorSetLayoutCreateFlags flags);
};

/*
 * One very important thing to do with pools is that when you reset a pool, it destroys all the descriptor sets
 * allocated from it. This is very useful for things like per-frame descriptors. That way we can have descriptors that
 * are used just for one frame, allocated dynamically, and then before we start the frame we completely delete all of
 * them in one go. This is confirmed to be a fast path by GPU vendors, and recommended to use when you need to
 * handle per-frame descriptor sets. - https://vkguide.dev/docs/new_chapter_2/vulkan_shader_code/
 */
struct DescriptorAllocator {
    struct PoolSizeRatio {
        VkDescriptorType type;
        float ratio;
    };

    VkDescriptorPool pool;

    void initPool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios);
    void clearDescriptors(VkDevice device);
    void destroyPool(VkDevice device);

    VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);
};