//
// Created by Luke Moran on 20/11/2024.
//

#pragma once

#include <vk_types.h>

#include <ranges>

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


constexpr unsigned int MAX_FRAMES_IN_FLIGHT = 2;

class VulkanEngine {
public:
    bool isInitialized{false};
    int frameNumber{0};
    bool stopRendering{false};
    VkExtent2D windowExtent{1700, 900};

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
    VkExtent2D swapchainExtent{0};

    AllocatedImage drawImage{nullptr};
    VkExtent2D drawExtent{0};

    std::array<FrameData, MAX_FRAMES_IN_FLIGHT> frames;
    FrameData& getCurrentFrame() { return frames[frameNumber % MAX_FRAMES_IN_FLIGHT]; }

    VkQueue graphicsQueue{nullptr};
    uint32_t graphicsQueueFamily{0};

    static VulkanEngine& Get();

    //initializes everything in the engine
    void init();

    //shuts down the engine
    void cleanup();

    //draw loop
    void draw();

    void drawBackground(VkCommandBuffer);

    //run main loop
    void run();

private:
    void initVulkan();

    void createSwapchain(uint32_t width, uint32_t height);
    void destroySwapchain();
    void initSwapchain();

    void initCommands();
    void initSyncStructures();
};
