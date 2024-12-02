//
// Created by Luke Moran on 20/11/2024.
//

#pragma once

#include <vk_types.h>

#include <ranges>

#include "vk_descriptors.h"
#include "vk_loader.h"

// TODO: Inefficient but general to use lambdas, could use arrays of specific vulkan handles
struct DeletionQueue {
    std::deque<std::function<void()>> deletionTasks;

    void pushTask(std::function<void()>&& task) {
        deletionTasks.push_back(task);
    }

    void flush() {
        for (auto& deletionTask : std::ranges::reverse_view(deletionTasks)) {
            deletionTask();
        }
        deletionTasks.clear();
    }
};

struct FrameData {
    VkCommandPool commandPool;
    VkCommandBuffer mainCommandBuffer;

    VkSemaphore swapchainSemaphore, renderSemaphore;
    VkFence renderFence;

    DeletionQueue deletionQueue;
};

struct ComputePushConstants {
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

struct ComputeEffect {
    const char* name;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    ComputePushConstants data;
};

constexpr unsigned int MAX_FRAMES_IN_FLIGHT = 2;

class VulkanEngine {
public:
    bool isInitialized{false};
    int frameNumber{0};
    bool stopRendering{false};
    bool resizeRequested{false};
    VkExtent2D windowExtent{1700, 900};
    float renderScale{1.0f};

    struct SDL_Window* window{nullptr};

    DeletionQueue mainDeletionQueue;

    VmaAllocator allocator{nullptr};

    VkInstance instance{nullptr};
    VkDebugUtilsMessengerEXT debugMessenger{nullptr};
    VkPhysicalDevice chosenGPU{nullptr};
    VkDevice device{nullptr};
    VkSurfaceKHR surface{nullptr};

    VkSwapchainKHR swapchain{nullptr};
    VkFormat swapchainImageFormat{VK_FORMAT_UNDEFINED};

    std::vector<VkImage> swapchainImages{};
    std::vector<VkImageView> swapchainImageViews{};
    VkExtent2D swapchainExtent{};

    AllocatedImage drawImage{nullptr};
    AllocatedImage depthImage{nullptr};
    VkExtent2D drawExtent{};

    std::array<FrameData, MAX_FRAMES_IN_FLIGHT> frames;
    FrameData& getCurrentFrame() { return frames[frameNumber % MAX_FRAMES_IN_FLIGHT]; }

    VkQueue graphicsQueue{};
    uint32_t graphicsQueueFamily{0};

    DescriptorAllocator globalDescriptorAllocator{nullptr};
    VkDescriptorSet drawImageDescriptors{nullptr};
    VkDescriptorSetLayout drawImageDescriptorLayout{nullptr};

    VkPipeline gradientPipeline{nullptr};
    VkPipelineLayout gradientPipelineLayout{nullptr};

    std::vector<ComputeEffect> backgroundEffects{};
    int currentBackgroundEffect{0};

    VkPipelineLayout meshPipelineLayout;
    VkPipeline meshPipeline;

    VkFence immFence;
    VkCommandBuffer immCommandBuffer;
    VkCommandPool immCommandPool;

    std::vector<std::shared_ptr<MeshAsset>> testMeshes;

    static VulkanEngine& Get();

    //initializes everything in the engine
    void init();

    //shuts down the engine
    void cleanup();

    //draw loop
    void draw();

    //run main loop
    void run();

    void drawBackground(VkCommandBuffer);

    void immediateSubmit(std::function<void(VkCommandBuffer cmdBuffer)>&& function);

    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

private:
    void initVulkan();

    void createSwapchain(uint32_t width, uint32_t height);
    void destroySwapchain();
    void initSwapchain();
    void resizeSwapchain();

    void initCommands();
    void initSyncStructures();

    void initDescriptors();

    void initPipelines();
    void initBackgroundPipelines();

    void initDefaultData();

    void drawGeometry(VkCommandBuffer commandBuffer);

    void initImgui();
    void drawImgui(VkCommandBuffer commandBuffer, VkImageView targetImageView);

    void initMeshPipeline();

    AllocatedBuffer createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    void destroyBuffer(AllocatedBuffer buffer);
};
