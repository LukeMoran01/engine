//
// Created by Luke Moran on 20/11/2024.
//

#pragma once

#include <vk_types.h>

struct FrameData {
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
};

constexpr unsigned int MAX_FRAMES_IN_FLIGHT = 2;

class VulkanEngine {
public:
    bool isInitialized{false};
    int frameNumber{0};
    bool stopRendering{false};
    VkExtent2D windowExtent{1700, 900};

    struct SDL_Window* window{nullptr};

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
