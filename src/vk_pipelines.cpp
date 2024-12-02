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

    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(fileSize));
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

void PipelineBuilder::clear() {
    inputAssembly        = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    rasterizer           = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    colorBlendAttachment = {};
    multisampling        = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    pipelineLayout       = {};
    depthStencil         = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    renderInfo           = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    shaderStages.clear();
}

VkPipeline PipelineBuilder::buildPipeline(VkDevice device) {
    VkPipelineViewportStateCreateInfo viewportState{.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.pNext = nullptr;

    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineColorBlendStateCreateInfo colorBlending{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO
    };
    colorBlending.pNext = nullptr;

    colorBlending.logicOpEnable   = VK_FALSE;
    colorBlending.logicOp         = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments    = &colorBlendAttachment;

    // Clear because we have no need.. ?
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
    };

    VkGraphicsPipelineCreateInfo pipelineInfo{.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pNext = &renderInfo;

    pipelineInfo.stageCount          = shaderStages.size();
    pipelineInfo.pStages             = shaderStages.data();
    pipelineInfo.pVertexInputState   = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pColorBlendState    = &colorBlending;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.layout              = pipelineLayout;

    std::array states = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamicInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicInfo.pDynamicStates                   = states.data();
    dynamicInfo.dynamicStateCount                = states.size();

    pipelineInfo.pDynamicState = &dynamicInfo;

    VkPipeline newPipeline;
    if (vkCreateGraphicsPipelines(device, nullptr, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
        fmt::println("failed to create pipeline");
        return nullptr;
    }

    return newPipeline;
}

void PipelineBuilder::setShaders(VkShaderModule vertexShader, VkShaderModule fragmentShader) {
    shaderStages.clear();
    shaderStages.push_back(
        vkinit::createPipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertexShader, "main")
    );
    shaderStages.push_back(
        vkinit::createPipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader, "main")
    );
}

void PipelineBuilder::setInputTopology(VkPrimitiveTopology topology) {
    inputAssembly.topology = topology;
    // TODO can improve efficiency when used - for triangle and line strips
    inputAssembly.primitiveRestartEnable = VK_FALSE;
}

void PipelineBuilder::setPolygonMode(VkPolygonMode mode) {
    rasterizer.polygonMode = mode;
    rasterizer.lineWidth   = 1.f;
}

void PipelineBuilder::setCullMode(VkCullModeFlags cullMode, VkFrontFace frontFace) {
    rasterizer.cullMode  = cullMode;
    rasterizer.frontFace = frontFace;
}

void PipelineBuilder::setMultisamplingNone() {
    multisampling.sampleShadingEnable   = VK_FALSE;
    multisampling.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading      = 1.0f;
    multisampling.pSampleMask           = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable      = VK_FALSE;
}

void PipelineBuilder::disableBlending() {
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
        | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
}

void PipelineBuilder::enableBlendingAdditive() {
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
        | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable         = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;
}

void PipelineBuilder::enableBlendingAlphaBlend() {
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
        | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable         = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;
}

void PipelineBuilder::setColorAttachmentFormat(VkFormat format) {
    colorAttachmentFormat              = format;
    renderInfo.colorAttachmentCount    = 1;
    renderInfo.pColorAttachmentFormats = &colorAttachmentFormat;
}

void PipelineBuilder::setDepthFormat(VkFormat format) {
    renderInfo.depthAttachmentFormat = format;
}

void PipelineBuilder::disableDepthTest() {
    depthStencil.depthTestEnable       = VK_FALSE;
    depthStencil.depthWriteEnable      = VK_FALSE;
    depthStencil.depthCompareOp        = VK_COMPARE_OP_NEVER;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable     = VK_FALSE;
    depthStencil.front                 = {};
    depthStencil.back                  = {};
    depthStencil.minDepthBounds        = 0.f;
    depthStencil.maxDepthBounds        = 1.f;
}

void PipelineBuilder::enableDepthTest(bool depthWriteEnable, VkCompareOp op) {
    depthStencil.depthTestEnable       = VK_TRUE;
    depthStencil.depthWriteEnable      = depthWriteEnable;
    depthStencil.depthCompareOp        = op;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable     = VK_FALSE;
    depthStencil.front                 = {};
    depthStencil.back                  = {};
    depthStencil.minDepthBounds        = 0.0f;
    depthStencil.maxDepthBounds        = 1.0f;
}

