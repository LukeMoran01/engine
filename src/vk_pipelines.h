//
// Created by Luke Moran on 23/11/2024.
//

#pragma once
#include <vulkan/vulkan_core.h>
#include <vk_types.h>

namespace vkutil {
    bool loadShaderModule(const char* filepath, VkDevice device, VkShaderModule* shaderModule);
}

class PipelineBuilder {
public:
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly;
    VkPipelineRasterizationStateCreateInfo rasterizer;
    VkPipelineColorBlendAttachmentState colorBlendAttachment;
    VkPipelineMultisampleStateCreateInfo multisampling;
    VkPipelineLayout pipelineLayout;
    VkPipelineDepthStencilStateCreateInfo depthStencil;
    VkPipelineRenderingCreateInfo renderInfo;
    VkFormat colorAttachmentFormat;

    PipelineBuilder() { clear(); }

    void clear();

    VkPipeline buildPipeline(VkDevice device);
    void setShaders(VkShaderModule vertexShader, VkShaderModule fragmentShader);
    void setInputTopology(VkPrimitiveTopology topology);
    void setPolygonMode(VkPolygonMode mode);
    void setCullMode(VkCullModeFlags cullMode, VkFrontFace frontFace);
    void setMultisamplingNone();
    void disableBlending();
    void enableBlendingAdditive();
    void enableBlendingAlphaBlend();
    void setColorAttachmentFormat(VkFormat format);
    void setDepthFormat(VkFormat format);
    void disableDepthTest();
    void enableDepthTest(bool depthWriteEnable, VkCompareOp op);
};
