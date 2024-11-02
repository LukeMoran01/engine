#include <iostream>
#include <array>
#include <vector>
#include <cstring>
#include <optional>
#include <set>
#include <algorithm>
#include <limits>

#include <SDL3/SDL_init.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>

#include <vulkan/vulkan.h>

typedef uint64_t uint64;
typedef uint32_t uint32;
typedef uint16_t uint16;
typedef uint8_t uint8;

#define global static

/*
 * Likely that present and graphics queues are the same, but we can treat them as separate
 */
struct QueueFamilyIndices {
    std::optional<uint32> graphicsFamily;
    std::optional<uint32> presentFamily;

    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

const std::array<const char*, 1> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::array<const char*, 1> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

bool checkValidationLayerSupport() {
    uint32 layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) {
            return false;
        }
    }

    return true;
}

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

global bool RUNNING = false;

int runSDL2Program() {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_EVENTS)) {
        SDL_GetError();
        return 1;
    }
    SDL_SetAppMetadata("Engine", "0.0.1", nullptr);
    // Hints are configuration variables, there are many more https://wiki.libsdl.org/SDL3/CategoryHints
    SDL_SetHint(SDL_HINT_EVENT_LOGGING, "1");

    SDL_Window *window = SDL_CreateWindow("Engine", 1980, 1080, SDL_WINDOW_INPUT_FOCUS|SDL_WINDOW_ALWAYS_ON_TOP);
    RUNNING = true;
    while(RUNNING) {
        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            switch(event.type) {
                case SDL_EVENT_QUIT: RUNNING = false; break;
            }
        }
    }

    SDL_Quit();
    return 0;
}

class HelloTriangleApplication {
    public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

    private:
    VkInstance instance = nullptr;
    VkSurfaceKHR surface = nullptr;
    VkPhysicalDevice physicalDevice = nullptr;
    VkDevice logicalDevice = nullptr;
    VkSwapchainKHR swapChain = nullptr;
    VkFormat swapChainImageFormat = {};
    VkExtent2D swapChainExtent = {};


    std::vector<VkImage> swapChainImages;

    VkQueue graphicsQueue = nullptr;
    VkQueue presentQueue = nullptr;

    SDL_Window *window = nullptr;

    void createInstance() {

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Engine";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // page 45 for MACOS error handling and see zig project that successfully created vk instance
        uint32 sdlExtensionCount;
        const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);
        createInfo.enabledExtensionCount = sdlExtensionCount;
        createInfo.ppEnabledExtensionNames = sdlExtensions;

        if (enableValidationLayers) {
            SDL_Log("Validation layers enabled");
            if (!checkValidationLayerSupport()) {
                throw std::runtime_error("Validation layers requested but not supported");
            }
            createInfo.enabledLayerCount = validationLayers.size();
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance");
        }

        SDL_Log("Instance created");
    }

    void createSurface() {
        if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
            SDL_Log(SDL_GetError());
            throw std::runtime_error("failed to create surface");
        }
        SDL_Log("Surface created");
    }

    void initWindow() {
        if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_EVENTS)) {
            SDL_GetError();
            return;
        }
        SDL_SetAppMetadata("Engine", "0.0.1", nullptr);
        // Hints are configuration variables, there are many more https://wiki.libsdl.org/SDL3/CategoryHints
        SDL_SetHint(SDL_HINT_EVENT_LOGGING, "1");
        window = SDL_CreateWindow("Engine", 1980, 1080, SDL_WINDOW_INPUT_FOCUS|SDL_WINDOW_VULKAN);
    }

    void pickPhysicalDevice() {
        uint32 deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find a GPU");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == nullptr) {
            throw std::runtime_error("failed to find a GPU with Vulkan support");
        }
        SDL_Log("Physical device selected");
    }

    void createLogicalDevice() {
        // Set up what queues we want to use, only care about graphics right now
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;
        for (uint32 queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // Would include things like geometry shaders, come back to it later
        VkPhysicalDeviceFeatures deviceFeatures{};

        // Now make the device creation info and pass the above structs to it
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = queueCreateInfos.size();
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = deviceExtensions.size();
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // Up-to-date implementations of Vulkan ignore validation layers defined here - test if okay

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device");
        }

        // queueIndex is the index within the queue family to retrieve
        vkGetDeviceQueue(logicalDevice, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(logicalDevice, indices.presentFamily.value(), 0, &presentQueue);

        SDL_Log("Logical device created");
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);

        swapChainExtent = chooseSwapExtent(swapChainSupport.capabilities);
        swapChainImageFormat = surfaceFormat.format;

        // How many images in the chain (frame buffers.. I think)
        uint32 imageCount = std::clamp(swapChainSupport.capabilities.minImageCount + 1,
            swapChainSupport.capabilities.minImageCount, swapChainSupport.capabilities.maxImageCount);

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = swapChainImageFormat;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = swapChainExtent;

        // Always 1 unless making stereoscopic 3D
        createInfo.imageArrayLayers = 1;

        // Means we are rendering directly to the images. If we wanted to perform post-processing we could use
        // VK_IMAGE_USAGE_TRANSFER_DST_BIT then transfer the rendered image to a swap chain image
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32 queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        // Are the graphics and presentation queues the same family, usually the case so go with best performance with
        // exclusive
        if (indices.graphicsFamily.value() != indices.presentFamily.value()) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        // We can apply a transform to all images in the swap chain, currentTransform to not apply any
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

        // Almost always want to ignore this apparently, used if want to use alpha channel for blending with other
        // windows - sounds like motion blur. This value ignores this.
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;

        // If true we ignore/don't care about pixels that are obscured like behind another window
        createInfo.clipped = VK_TRUE;

        // When we create a new swap chain like say on window resize, we need to reference the previous one
        createInfo.oldSwapchain = nullptr;

        if (vkCreateSwapchainKHR(logicalDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain");
        }
        SDL_Log("Swap chain created");

        // We only specify min number of images so we need to retrieve actual number
        vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, swapChainImages.data());
    }

    /*
     * We have to create an instance with enabled required extensions and optional validation layers
     * We then have to create a surface which as we using an SDL window, SDL provides a useful function for
     * We then have to find a suitable physical device (GPU) that supports what we need including queue families
     * We then we have to create a logical device which we use to interface with the physical device which cares about
     * and uses what we asked our physical device to support
     * We then have to create the swap chain which involves the frame buffers or where we render images into
     */
    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
    }

    void mainLoop() {
        RUNNING = true;
        while(RUNNING) {
            SDL_Event event;
            while(SDL_PollEvent(&event)) {
                switch(event.type) {
                    case SDL_EVENT_QUIT: RUNNING = false; break;
                }
            }
        }
    }

    /*
     * Feels largely pointless and think about removing in actual implementation
     * When we close the application we don't want the user to wait for us to needlessly clean up memory that the OS
     * will handle.
     */
    void cleanup() {
        vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);
        vkDestroyDevice(logicalDevice, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    // Basic suitability check for GPU. Expect to only be developing against M1 Pro or 4070ti
    bool isDeviceSuitable(VkPhysicalDevice physicalDevice) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        bool extensionSupported = checkDeviceExtensionSupport(physicalDevice);

        bool swapChainAdequate = false;
        if (extensionSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        && deviceFeatures.geometryShader && indices.isComplete() && extensionSupported && swapChainAdequate;
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice) {
        QueueFamilyIndices indices;

        uint32 queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }
            VkBool32 presentSupport = false;

            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }
            if (indices.isComplete()) break;
            i++;
        }

        return indices;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice physicalDevice) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice physicalDevice) {
        SwapChainSupportDetails details = {};

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &details.capabilities);

        uint32 formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, details.formats.data());
        }

        uint32 presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    // Do we have the preferred surface format ? If not then just return the first one - could rank and get the second best
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB
                && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }
    /*
     * Available are:
     * VK_PRESENT_MODE_IMMEDIATE_KHR which displays instantly and cause tearing
     * VK_PRESENT_MODE_FIFO_KHR which holds images in a queue and is most similar to v-sync
     * VK_PRESENT_MODE_FIFO_RELAXED_KHR which is as above but won't wait for vertical blank if queue is empty
     * VK_PRESENT_MODE_MAILBOX_KHR as FIFO but doesn't block when queue full, instead replaces with newer commonly known as triple buffering
     */
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    /*
     * Swap extend is the resolution of the swap chain images. It is usually equal to the exact resolution of the window
     * but Vulkan will let us know if it can differ from this. In this case we set it to the resolution that best fits
     * the window
     */
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        // Vulkan tells us it isn't exact by setting to max uint32
        if (capabilities.currentExtent.width != std::numeric_limits<uint32>::max()) {
            return capabilities.currentExtent;
        }

        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        VkExtent2D actualExtent = {
            (uint32)width,
            (uint32)height,
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
};

// Entry point for SDL3 with header inclusion
int main(int argc, char **argv) {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
