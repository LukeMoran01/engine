//
// Created by Luke Moran on 23/11/2024.
//

#include "vk_descriptors.h"

#include "vk_types.h"

void DescriptorLayoutBuilder::addBinding(uint32_t binding, VkDescriptorType type) {
    VkDescriptorSetLayoutBinding newBind{};
    newBind.binding         = binding;
    newBind.descriptorCount = 1;
    newBind.descriptorType  = type;

    bindings.push_back(newBind);
}

void DescriptorLayoutBuilder::clear() {
    bindings.clear();
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device, VkShaderStageFlags shaderStages, void* pNext,
                                                     VkDescriptorSetLayoutCreateFlags flags) {
    for (auto& binding : bindings) {
        binding.stageFlags |= shaderStages;
    }
    VkDescriptorSetLayoutCreateInfo info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.pNext = pNext;

    info.bindingCount = bindings.size();
    info.pBindings    = bindings.data();
    info.flags        = flags;

    VkDescriptorSetLayout layout = nullptr;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout));

    return layout;
}

void DescriptorAllocator::initPool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios) {
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (PoolSizeRatio poolRatio : poolRatios) {
        poolSizes.push_back(VkDescriptorPoolSize{
            .type = poolRatio.type,
            .descriptorCount = static_cast<uint32_t>(poolRatio.ratio * static_cast<float>(maxSets))
        });
    }

    VkDescriptorPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags         = 0;
    poolInfo.maxSets       = maxSets;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes    = poolSizes.data();

    vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool);
}

void DescriptorAllocator::clearDescriptors(VkDevice device) {
    vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroyPool(VkDevice device) {
    vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo allocInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.pNext              = nullptr;
    allocInfo.descriptorPool     = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &layout;

    VkDescriptorSet descriptorSet{nullptr};
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

    return descriptorSet;
}







