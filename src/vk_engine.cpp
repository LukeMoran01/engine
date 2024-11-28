//
// Created by Luke Moran on 20/11/2024.
//

#include "vk_engine.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>
#include <vk_images.h>

#include <chrono>
#include <thread>

#include "VkBootstrap.h"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "vk_pipelines.h"

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

constexpr bool useValidationLayers    = true;
constexpr std::array validationLayers = {"VK_LAYER_KHRONOS_validation"};

VulkanEngine* loadedEngine = nullptr;

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::init() {
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        SDL_GetError();
        return;
    }

    SDL_SetAppMetadata("Engine", "0.1.0", nullptr);
    // Hints are configuration variables, there are many more https://wiki.libsdl.org/SDL3/CategoryHints
    SDL_SetHint(SDL_HINT_EVENT_LOGGING, "1");
    auto windowFlags = SDL_WINDOW_INPUT_FOCUS | SDL_WINDOW_VULKAN;
    window = SDL_CreateWindow("Engine", static_cast<int>(windowExtent.width), static_cast<int>(windowExtent.height),
                              windowFlags);

    initVulkan();
    initSwapchain();
    initCommands();
    initSyncStructures();
    initDescriptors();
    initPipelines();

    initImgui();

    // everything went fine
    isInitialized = true;
}

void VulkanEngine::initVulkan() {
    vkb::InstanceBuilder builder;
    auto vkbInstance = builder.set_app_name("Windows Engine").request_validation_layers(useValidationLayers).
                               use_default_debug_messenger().require_api_version(1, 3, 0).build().value();

    instance       = vkbInstance.instance;
    debugMessenger = vkbInstance.debug_messenger;

    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
        SDL_Log(SDL_GetError());
        throw std::runtime_error("failed to create surface");
    }
    SDL_Log("Surface created");

    VkPhysicalDeviceVulkan13Features features13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing  = VK_TRUE;

    vkb::PhysicalDeviceSelector selector{vkbInstance};
    vkb::PhysicalDevice physicalDevice = selector.set_minimum_version(1, 3).set_required_features_13(features13).
                                                  set_required_features_12(features12).set_surface(surface).select().
                                                  value();

    vkb::DeviceBuilder deviceBuilder{physicalDevice};
    vkb::Device vkbDevice = deviceBuilder.build().value();

    device    = vkbDevice.device;
    chosenGPU = physicalDevice.physical_device;

    graphicsQueue       = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.device         = device;
    allocatorInfo.physicalDevice = chosenGPU;
    allocatorInfo.instance       = instance;
    allocatorInfo.flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &allocator);

    mainDeletionQueue.pushTask([&]() {
        vmaDestroyAllocator(allocator);
    });
}

void VulkanEngine::initImgui() {
    std::array poolSizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}
    };

    VkDescriptorPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets       = 1000;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes    = poolSizes.data();

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &imguiPool));

    ImGui::CreateContext();
    ImGui_ImplSDL3_InitForVulkan(window);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance            = instance;
    initInfo.PhysicalDevice      = chosenGPU;
    initInfo.Device              = device;
    initInfo.Queue               = graphicsQueue;
    initInfo.DescriptorPool      = imguiPool;
    initInfo.MinImageCount       = 3;
    initInfo.ImageCount          = 3;
    initInfo.UseDynamicRendering = true;

    initInfo.PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    initInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchainImageFormat;
    // TODO: Update when we add sampling
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&initInfo);
    ImGui_ImplVulkan_CreateFontsTexture();

    mainDeletionQueue.pushTask([&]() {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(device, imguiPool, nullptr);
    });
}

void VulkanEngine::drawImgui(VkCommandBuffer commandBuffer, VkImageView targetImageView) {
    auto colorAttachment = vkinit::createRenderAttachmentInfo(targetImageView, nullptr,
                                                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    auto renderInfo = vkinit::createRenderInfo(swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(commandBuffer, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

    vkCmdEndRendering(commandBuffer);
}


void VulkanEngine::createSwapchain(uint32_t width, uint32_t height) {
    vkb::SwapchainBuilder builder{chosenGPU, device, surface};
    swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    auto surfaceFormat   = VkSurfaceFormatKHR{
        .format = swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
    };
    auto vkbSwapchain = builder.set_desired_format(surfaceFormat).set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR).
                                set_desired_extent(width, height).add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                                .build().value();

    swapchainExtent     = vkbSwapchain.extent;
    swapchain           = vkbSwapchain.swapchain;
    swapchainImages     = vkbSwapchain.get_images().value();
    swapchainImageViews = vkbSwapchain.get_image_views().value();

    VkExtent3D drawImageExtent = {
        .width = windowExtent.width,
        .height = windowExtent.height,
        .depth = 1
    };

    drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    drawImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    auto imageInfo = vkinit::createImageCreateInfo(drawImage.imageFormat, drawImageUsages, drawImageExtent);

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage         = VMA_MEMORY_USAGE_GPU_ONLY;
    allocInfo.requiredFlags = static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vmaCreateImage(allocator, &imageInfo, &allocInfo, &drawImage.image, &drawImage.allocation, nullptr);

    auto imageViewInfo = vkinit::createImageViewCreateInfo(drawImage.imageFormat, drawImage.image,
                                                           VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(device, &imageViewInfo, nullptr, &drawImage.imageView));

    mainDeletionQueue.pushTask([&]() {
        vkDestroyImageView(device, drawImage.imageView, nullptr);
        vmaDestroyImage(allocator, drawImage.image, drawImage.allocation);
    });
}


void VulkanEngine::initSwapchain() {
    createSwapchain(windowExtent.width, windowExtent.height);
}

void VulkanEngine::destroySwapchain() {
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    for (auto& swapchainImageView : swapchainImageViews) {
        vkDestroyImageView(device, swapchainImageView, nullptr);
    }
}

void VulkanEngine::initCommands() {
    auto commandPoolInfo = vkinit::createCommandPoolCreateInfo(graphicsQueueFamily,
                                                               VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &frames[i].commandPool));

        auto allocInfo = vkinit::createCommandBufferAllocInfo(
            frames[i].commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &frames[i].mainCommandBuffer));
    }
}

void VulkanEngine::initSyncStructures() {
    auto fenceCreateInfo     = vkinit::createFenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    auto semaphoreCreateInfo = vkinit::createSemaphoreCreateInfo(0);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &frames[i].renderFence));

        VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frames[i].swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frames[i].renderSemaphore));
    }
}

void VulkanEngine::initDescriptors() {
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}
    };

    globalDescriptorAllocator.initPool(device, 10, sizes);

    DescriptorLayoutBuilder builder;
    builder.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    drawImageDescriptorLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT, nullptr, 0);

    drawImageDescriptors = globalDescriptorAllocator.allocate(device, drawImageDescriptorLayout);

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfo.imageView   = drawImage.imageView;

    VkWriteDescriptorSet drawImageWrite{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    drawImageWrite.pNext           = nullptr;
    drawImageWrite.dstBinding      = 0;
    drawImageWrite.dstSet          = drawImageDescriptors;
    drawImageWrite.descriptorCount = 1;
    drawImageWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    drawImageWrite.pImageInfo      = &imageInfo;

    vkUpdateDescriptorSets(device, 1, &drawImageWrite, 0, nullptr);

    mainDeletionQueue.pushTask([&]() {
        globalDescriptorAllocator.destroyPool(device);
        vkDestroyDescriptorSetLayout(device, drawImageDescriptorLayout, nullptr);
    });
}

void VulkanEngine::initPipelines() {
    initBackgroundPipelines();
    initTrianglePipeline();
}

void VulkanEngine::initBackgroundPipelines() {
    VkPipelineLayoutCreateInfo computeLayout{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    computeLayout.pNext          = nullptr;
    computeLayout.pSetLayouts    = &drawImageDescriptorLayout;
    computeLayout.setLayoutCount = 1;

    VkPushConstantRange pushConstant{};
    pushConstant.offset     = 0;
    pushConstant.size       = sizeof(ComputePushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    computeLayout.pPushConstantRanges    = &pushConstant;
    computeLayout.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(device, &computeLayout, nullptr, &gradientPipelineLayout));

    VkShaderModule gradientShader;
    if (!vkutil::loadShaderModule("../shaders/compiled/gradient_color.comp.spv", device, &gradientShader)) {
        fmt::print("Failed to build compute shader\n");
    }

    VkShaderModule skyShader;
    if (!vkutil::loadShaderModule("../shaders/compiled/sky.comp.spv", device, &skyShader)) {
        fmt::print("Failed to build compute shader\n");
    }

    VkPipelineShaderStageCreateInfo stageInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.pNext  = nullptr;
    stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = gradientShader;
    // This is the function to call in the shader - nice!
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo computeCreateInfo{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    computeCreateInfo.pNext  = nullptr;
    computeCreateInfo.layout = gradientPipelineLayout;
    computeCreateInfo.stage  = stageInfo;

    ComputeEffect gradient{};
    gradient.layout = gradientPipelineLayout;
    gradient.name   = "gradient";
    gradient.data   = {};

    gradient.data.data1 = glm::vec4(1, 0, 0, 1);
    gradient.data.data2 = glm::vec4(0, 0, 1, 1);

    VK_CHECK(vkCreateComputePipelines(device, nullptr, 1, &computeCreateInfo, nullptr, &gradient.pipeline));

    computeCreateInfo.stage.module = skyShader;

    ComputeEffect sky{};
    sky.layout = gradientPipelineLayout;
    sky.name   = "sky";
    sky.data   = {};

    sky.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

    VK_CHECK(vkCreateComputePipelines(device, nullptr, 1, &computeCreateInfo, nullptr, &sky.pipeline));

    backgroundEffects.push_back(gradient);
    backgroundEffects.push_back(sky);

    vkDestroyShaderModule(device, gradientShader, nullptr);
    vkDestroyShaderModule(device, skyShader, nullptr);
    mainDeletionQueue.pushTask([&]() {
        vkDestroyPipelineLayout(device, gradientPipelineLayout, nullptr);
        vkDestroyPipeline(device, gradient.pipeline, nullptr);
        vkDestroyPipeline(device, sky.pipeline, nullptr);
    });
}

// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap12.html#resources-image-layouts
void VulkanEngine::draw() {
    VK_CHECK(vkWaitForFences(device, 1, &getCurrentFrame().renderFence, VK_TRUE, 1000000000));
    getCurrentFrame().deletionQueue.flush();
    VK_CHECK(vkResetFences(device, 1, &getCurrentFrame().renderFence));

    uint32_t swapchainImageIndex = 0;
    VK_CHECK(
        vkAcquireNextImageKHR(device, swapchain, 1000000000, getCurrentFrame().swapchainSemaphore, nullptr,
            &swapchainImageIndex));

    auto cmdBuffer = getCurrentFrame().mainCommandBuffer;
    VK_CHECK(vkResetCommandBuffer(cmdBuffer, 0));

    auto beginInfo = vkinit::createCommandBufferBeginInfo(
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    drawExtent.width  = drawImage.imageExtent.width;
    drawExtent.height = drawImage.imageExtent.height;

    VK_CHECK(vkBeginCommandBuffer(cmdBuffer, &beginInfo));

    vkutil::transitionImage(cmdBuffer, drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    drawBackground(cmdBuffer);

    vkutil::transitionImage(cmdBuffer, drawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    drawGeometry(cmdBuffer);

    vkutil::transitionImage(cmdBuffer, drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    vkutil::transitionImage(cmdBuffer, swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vkutil::copyImageToImage(cmdBuffer, drawImage.image, swapchainImages[swapchainImageIndex], drawExtent,
                             swapchainExtent);

    vkutil::transitionImage(cmdBuffer, swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    drawImgui(cmdBuffer, swapchainImageViews[swapchainImageIndex]);

    vkutil::transitionImage(cmdBuffer, swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cmdBuffer));

    auto bufferInfo = vkinit::createCommandBufferSubmitInfo(cmdBuffer);

    auto waitInfo = vkinit::createSemaphoreSubmitInfo(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
                                                      getCurrentFrame().swapchainSemaphore);
    auto signalInfo = vkinit::createSemaphoreSubmitInfo(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
                                                        getCurrentFrame().renderSemaphore);

    auto submitInfo = vkinit::createSubmitInfo(&bufferInfo, &signalInfo, &waitInfo);

    VK_CHECK(vkQueueSubmit2(graphicsQueue, 1, &submitInfo, getCurrentFrame().renderFence));

    VkPresentInfoKHR presentInfo{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    presentInfo.pNext          = nullptr;
    presentInfo.pSwapchains    = &swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = &getCurrentFrame().renderSemaphore;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VK_CHECK(vkQueuePresentKHR(graphicsQueue, &presentInfo));

    frameNumber++;
}

void VulkanEngine::drawBackground(VkCommandBuffer commandBuffer) {
    ComputeEffect& effect = backgroundEffects[currentBackgroundEffect];

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gradientPipelineLayout, 0, 1,
                            &drawImageDescriptors, 0, nullptr);

    vkCmdPushConstants(commandBuffer, gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(ComputePushConstants), &effect.data);

    vkCmdDispatch(commandBuffer, std::ceil(drawExtent.width / 16.0), std::ceil(drawExtent.height / 16.0), 1);
}

void VulkanEngine::drawGeometry(VkCommandBuffer commandBuffer) {
    auto colorAttachment = vkinit::createRenderAttachmentInfo(drawImage.imageView, nullptr,
                                                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    auto renderInfo = vkinit::createRenderInfo(drawExtent, &colorAttachment, nullptr);
    vkCmdBeginRendering(commandBuffer, &renderInfo);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, trianglePipeline);

    VkViewport viewport{};
    viewport.x        = 0;
    viewport.y        = 0;
    viewport.width    = static_cast<float>(drawExtent.width);
    viewport.height   = static_cast<float>(drawExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset        = {0, 0};
    scissor.extent.width  = drawExtent.width;
    scissor.extent.height = drawExtent.height;

    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    vkCmdEndRendering(commandBuffer);
}

void VulkanEngine::run() {
    SDL_Event event;
    bool running = true;

    // main loop
    while (running) {
        // Handle events on queue
        while (SDL_PollEvent(&event) != 0) {
            // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
            switch (event.type) {
                case SDL_EVENT_QUIT: running = false;
                    break;
                case SDL_EVENT_WINDOW_MINIMIZED: {
                    stopRendering = true;
                    break;
                case SDL_EVENT_WINDOW_RESTORED: stopRendering = false;
                    break;
                }
            }

            ImGui_ImplSDL3_ProcessEvent(&event);
        }

        // do not draw if we are minimized
        if (stopRendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        if (ImGui::Begin("background")) {
            ComputeEffect& selected = backgroundEffects[currentBackgroundEffect];

            ImGui::Text("Selected effect: ", selected.name);

            ImGui::SliderInt("Effect Index", &currentBackgroundEffect, 0,
                             static_cast<int>(backgroundEffects.size() - 1));

            ImGui::InputFloat4("data1", reinterpret_cast<float*>(&selected.data.data1));
            ImGui::InputFloat4("data2", reinterpret_cast<float*>(&selected.data.data2));
            ImGui::InputFloat4("data3", reinterpret_cast<float*>(&selected.data.data3));
            ImGui::InputFloat4("data4", reinterpret_cast<float*>(&selected.data.data4));
        }

        ImGui::End();
        ImGui::Render();

        draw();
    }
}

void VulkanEngine::initTrianglePipeline() {
    VkShaderModule triangleFragShader;
    if (!vkutil::loadShaderModule("../shaders/compiled/colored_triangle.frag.spv", device, &triangleFragShader)) {
        fmt::print("Failed to build the triangle fragment shader module");
    }

    VkShaderModule triangleVertShader;
    if (!vkutil::loadShaderModule("../shaders/compiled/colored_triangle.vert.spv", device, &triangleVertShader)) {
        fmt::print("Failed to build the triangle vertex shader module");
    }

    auto pipelineInfo = vkinit::createPipelineLayoutCreateInfo();
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineInfo, nullptr, &trianglePipelineLayout));

    PipelineBuilder pipelineBuilder;

    pipelineBuilder.pipelineLayout = trianglePipelineLayout;
    pipelineBuilder.setShaders(triangleVertShader, triangleFragShader);
    pipelineBuilder.setInputTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.setPolygonMode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.setMultisamplingNone();
    pipelineBuilder.disableBlending();
    pipelineBuilder.disableDepthtest();

    pipelineBuilder.setColorAttachmentFormat(drawImage.imageFormat);
    pipelineBuilder.setDepthFormat(VK_FORMAT_UNDEFINED);

    trianglePipeline = pipelineBuilder.buildPipeline(device);

    vkDestroyShaderModule(device, triangleVertShader, nullptr);
    vkDestroyShaderModule(device, triangleFragShader, nullptr);

    mainDeletionQueue.pushTask([&]() {
        vkDestroyPipelineLayout(device, trianglePipelineLayout, nullptr);
        vkDestroyPipeline(device, trianglePipeline, nullptr);
    });
}

// ReSharper disable once CppMemberFunctionMayBeConst
void VulkanEngine::cleanup() {
    if (isInitialized) {
        vkDeviceWaitIdle(device);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyCommandPool(device, frames[i].commandPool, nullptr);

            vkDestroyFence(device, frames[i].renderFence, nullptr);
            vkDestroySemaphore(device, frames[i].renderSemaphore, nullptr);
            vkDestroySemaphore(device, frames[i].swapchainSemaphore, nullptr);

            frames[i].deletionQueue.flush();
        }

        mainDeletionQueue.flush();

        destroySwapchain();
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyDevice(device, nullptr);
        vkb::destroy_debug_utils_messenger(instance, debugMessenger);
        vkDestroyInstance(instance, nullptr);
        SDL_DestroyWindow(window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}
