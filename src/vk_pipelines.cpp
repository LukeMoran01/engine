//
// Created by Luke Moran on 23/11/2024.
//

#include <vk_pipelines.h>
#include <fstream>
#include <vk_initializers.h>

bool vkutil::loadShaderModule(const char* filepath, VkDevice device, VkShaderModule* shaderModule) {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        return false;
    }
    // Cursor is at end of file, so location is exact size in bytes
    size_t fileSize = file.tellg();
    // spirv expects uint32 so reserve int vector big enough
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);

    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.pNext = nullptr;

    createInfo.codeSize = buffer.size() * sizeof(uint32_t);
    createInfo.pCode    = buffer.data();

    VkShaderModule tmpShaderModule{nullptr};
    if (vkCreateShaderModule(device, &createInfo, nullptr, &tmpShaderModule) != VK_SUCCESS) {
        return false;
    }
    *shaderModule = tmpShaderModule;
    return true;
}
