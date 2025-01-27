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


void DescriptorAllocatorGrowable::init(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios) {
    ratios.clear();

    for (PoolSizeRatio r : poolRatios) {
        ratios.push_back(r);
    }

    VkDescriptorPool newPool = createPool(device, maxSets, ratios);
    setsPerPool              = static_cast<uint32_t>(static_cast<float>(maxSets) * 1.5f);
    readyPools.push_back(newPool);
}


VkDescriptorPool DescriptorAllocatorGrowable::getPool(VkDevice device) {
    VkDescriptorPool newPool;
    if (!readyPools.empty()) {
        newPool = readyPools.back();
        readyPools.pop_back();
    } else {
        newPool     = createPool(device, setsPerPool, ratios);
        setsPerPool = static_cast<uint32_t>(static_cast<float>(setsPerPool) * 1.5f);
        if (setsPerPool > 4092) setsPerPool = 4092;
    }
    return newPool;
}

VkDescriptorPool DescriptorAllocatorGrowable::createPool(VkDevice device, uint32_t setCount,
                                                         std::span<PoolSizeRatio> poolRatios) {
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (PoolSizeRatio ratio : poolRatios) {
        poolSizes.push_back(VkDescriptorPoolSize{
            .type = ratio.type,
            .descriptorCount = static_cast<uint32_t>(ratio.ratio * static_cast<float>(setCount))
        });
    }
    VkDescriptorPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags         = 0;
    poolInfo.maxSets       = setCount;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes    = poolSizes.data();

    VkDescriptorPool newPool;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &newPool);
    return newPool;
}

void DescriptorAllocatorGrowable::clearPools(VkDevice device) {
    for (auto p : readyPools) {
        vkResetDescriptorPool(device, p, 0);
    }
    for (auto p : fullPools) {
        vkResetDescriptorPool(device, p, 0);
        readyPools.push_back(p);
    }
    fullPools.clear();
}

void DescriptorAllocatorGrowable::destroyPools(VkDevice device) {
    for (auto p : readyPools) {
        vkDestroyDescriptorPool(device, p, nullptr);
    }
    readyPools.clear();
    for (auto p : fullPools) {
        vkDestroyDescriptorPool(device, p, nullptr);
    }
    fullPools.clear();
}

VkDescriptorSet DescriptorAllocatorGrowable::allocate(VkDevice device, VkDescriptorSetLayout layout, void* pNext) {
    auto poolToUse = getPool(device);

    VkDescriptorSetAllocateInfo allocInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.pNext              = pNext;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &layout;
    allocInfo.descriptorPool     = poolToUse;

    VkDescriptorSet descriptorSet;
    VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);
    if (result != VK_SUCCESS) {
        fullPools.push_back(poolToUse);
        poolToUse                = getPool(device);
        allocInfo.descriptorPool = poolToUse;

        VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
    }

    readyPools.push_back(poolToUse);
    return descriptorSet;
}


void DescriptorWriter::writeBuffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type) {
    VkDescriptorBufferInfo& info = bufferInfos.emplace_back(VkDescriptorBufferInfo{
        .buffer = buffer,
        .offset = offset,
        .range = size
    });

    VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstBinding      = binding;
    write.dstSet          = nullptr;
    write.descriptorCount = 1;
    write.descriptorType  = type;
    write.pBufferInfo     = &info;

    writes.push_back(write);
}

void DescriptorWriter::writeImage(int binding, VkImageView imageView, VkSampler sampler, VkImageLayout layout,
                                  VkDescriptorType type) {
    VkDescriptorImageInfo& info = imageInfos.emplace_back(VkDescriptorImageInfo{
        .sampler = sampler,
        .imageView = imageView,
        .imageLayout = layout
    });

    VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstBinding      = binding;
    write.dstSet          = nullptr;
    write.descriptorCount = 1;
    write.descriptorType  = type;
    write.pImageInfo      = &info;

    writes.push_back(write);
}

void DescriptorWriter::clear() {
    imageInfos.clear();
    bufferInfos.clear();
    writes.clear();
}

void DescriptorWriter::updateSet(VkDevice device, VkDescriptorSet set) {
    for (auto& write : writes) {
        write.dstSet = set;
    }

    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
}
