//
// Created by Luke Moran on 20/11/2024.
//

#include "vk_engine.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>

#include <chrono>
#include <thread>

#include "VkBootstrap.h"

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
}

void VulkanEngine::initSyncStructures() {
}

// ReSharper disable once CppMemberFunctionMayBeConst
void VulkanEngine::cleanup() {
    if (isInitialized) {
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

void VulkanEngine::draw() {
    // nothing yet
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
        }

        // do not draw if we are minimized
        if (stopRendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        draw();
    }
}
