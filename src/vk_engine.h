//
// Created by Luke Moran on 20/11/2024.
//

#pragma once

#include <vk_types.h>

#include <ranges>

#include "vk_descriptors.h"
#include "vk_loader.h"

#include <camera.h>

#include "event_queue.h"

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
    DescriptorAllocatorGrowable frameDescriptors;
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

struct GPUSceneData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection;
    glm::vec4 sunlightColor;
};

struct GLTFMetallic_Roughness {
    MaterialPipeline opaquePipeline;
    MaterialPipeline transparentPipeline;

    VkDescriptorSetLayout materialLayout;

    struct MaterialConstants {
        glm::vec4 colorFactors;
        glm::vec4 metal_rough_factors;
        // Padding that is needed for uniform buffers TODO research
        // Binding a uniform buffer requires a min requirement for alignment. 256 bytes is a good default.
        glm::vec4 extra[14];
    };

    struct MaterialResources {
        AllocatedImage colorImage;
        VkSampler colorSampler;
        AllocatedImage metalRoughImage;
        VkSampler metalRoughSampler;
        VkBuffer dataBuffer;
        uint32_t dataBufferOffset;
    };

    DescriptorWriter writer;

    void buildPipelines(VulkanRenderer* engine);
    void clearResources(VkDevice device) const;

    MaterialInstance writeMaterial(VkDevice device, MaterialPass pass, const MaterialResources& resources,
                                   DescriptorAllocatorGrowable& descriptorAllocator);
};

// The core of our rendering. Renderer will take the array of objects from the context and execute a single draw for each
struct RenderObject {
    uint32_t indexCount;
    uint32_t firstIndex;
    VkBuffer indexBuffer;

    MaterialInstance* material;
    Bounds bounds;
    glm::mat4 transform;
    VkDeviceAddress vertexBufferAddress;
};

struct DrawContext {
    std::vector<RenderObject> OpaqueSurfaces;
    std::vector<RenderObject> TransparentSurfaces;
};

struct MeshNode : Node {
    std::shared_ptr<MeshAsset> mesh;

    void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

// struct PrimitiveNode : Node {
//     void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
// };

struct EngineStats {
    float frameTime;
    int triangleCount;
    int drawcallCount;
    float sceneUpdateTime;
    float meshDrawTime;
};

constexpr unsigned int MAX_FRAMES_IN_FLIGHT = 2;

class VulkanRenderer {
public:
    VulkanRenderer(WindowEventQueue* weq) {
        windowEventQueue = weq;
    }

    EngineStats stats{};

    bool isInitialized{false};
    int frameNumber{0};
    bool stopRendering{false};
    bool resizeRequested{false};
    VkExtent2D windowExtent{1700, 900};
    float renderScale{1.0f};

    SDL_Window* window{nullptr};

    Camera mainCamera;

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
    VkExtent2D drawExtent{1, 1};

    std::array<FrameData, MAX_FRAMES_IN_FLIGHT> frames;
    FrameData& getCurrentFrame() { return frames[frameNumber % MAX_FRAMES_IN_FLIGHT]; }

    VkQueue graphicsQueue{};
    uint32_t graphicsQueueFamily{0};

    DescriptorAllocatorGrowable globalDescriptorAllocator;
    VkDescriptorSet drawImageDescriptors{nullptr};
    VkDescriptorSetLayout drawImageDescriptorLayout{nullptr};

    VkPipeline gradientPipeline{nullptr};
    VkPipelineLayout gradientPipelineLayout{nullptr};

    std::vector<ComputeEffect> backgroundEffects{};
    int currentBackgroundEffect{0};

    VkPipelineLayout meshPipelineLayout{};
    VkPipeline meshPipeline{};

    VkFence immFence{};
    VkCommandBuffer immCommandBuffer{};
    VkCommandPool immCommandPool{};

    GPUSceneData sceneData{};
    VkDescriptorSetLayout gpuSceneDataDescriptorLayout{};

    std::vector<std::shared_ptr<MeshAsset>> testMeshes;

    AllocatedImage whiteImage{};
    AllocatedImage blackImage{};
    AllocatedImage greyImage{};
    AllocatedImage errorCheckerboardImage{};

    VkSampler defaultSamplerLinear{};
    VkSampler defaultSamplerNearest{};

    VkDescriptorSetLayout singleImageDescriptorLayout{};

    MaterialInstance defaultData{};
    GLTFMetallic_Roughness metalRoughMaterial;

    DrawContext mainDrawContext;
    std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;;
    std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;

    static VulkanRenderer& Get();

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

    void updateScene();

    AllocatedBuffer createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    void destroyBuffer(AllocatedBuffer buffer);

    AllocatedImage createImage(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped);
    AllocatedImage createImage(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped);
    void destroyImage(const AllocatedImage& image);

private:
    WindowEventQueue* windowEventQueue = nullptr;

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
};
