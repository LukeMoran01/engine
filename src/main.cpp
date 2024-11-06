#include <iostream>
#include <array>
#include <vector>
#include <cstring>
#include <optional>
#include <set>
#include <algorithm>
#include <limits>
#include <fstream>

#include <SDL3/SDL_init.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>

#include <glm/glm.hpp>

#include <vulkan/vulkan.h>

typedef uint64_t uint64;
typedef uint32_t uint32;
typedef uint16_t uint16;
typedef uint8_t uint8;

#define global static

struct Vertex {
    glm::vec2 position;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        // Vertex data packed together into one array - so one binding and this is the index
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        // Vertex or Instance rendering - we use vertex
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        // Which location in the vertex shader, 0 = position
        attributeDescriptions[0].location = 0;
        // Confusingly uses colour format but really means vector of 2 32 bit floats
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        // Where to start reading from
        attributeDescriptions[0].offset = offsetof(Vertex, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);
        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{0.0f, -0.5f}, {1.0f, 1.0f, 1.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

const uint8 MAX_FRAMES_IN_FLIGHT = 2;

// Open the file at the end and seek to the start before reading so we know the exact size for buffer allocation
static std::vector<char> readFile(const std::string& fileName) {
    std::ifstream file(fileName, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + fileName);
    }

    uint64 fileSize = (uint64)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

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

    SDL_Window* window = SDL_CreateWindow("Engine", 1980, 1080, SDL_WINDOW_INPUT_FOCUS | SDL_WINDOW_ALWAYS_ON_TOP);
    RUNNING = true;
    while (RUNNING) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_EVENT_QUIT: RUNNING = false;
                    break;
            }
        }
    }

    SDL_Quit();
    return 0;
}

class FirstVulkanTriangleApplication {
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

        VkQueue graphicsQueue = nullptr;
        VkQueue presentQueue = nullptr;

        VkRenderPass renderPass = nullptr;
        VkPipelineLayout pipelineLayout = {};
        VkPipeline graphicsPipeline = nullptr;

        std::vector<VkImage> swapChainImages;
        std::vector<VkImageView> swapChainImageViews;
        std::vector<VkFramebuffer> swapChainFramebuffers;

        VkBuffer vertexBuffer = nullptr;
        VkDeviceMemory vertexBufferMemory = nullptr;

        VkCommandPool commandPool = nullptr;
        std::vector<VkCommandBuffer> commandBuffers;

        std::vector<VkSemaphore> imageAvailableSemaphores;
        std::vector<VkSemaphore> renderFinishedSemaphores;
        std::vector<VkFence> inFlightFences;

        uint32 currentFrame = 0;
        bool staleSwapChain = false;

        SDL_Window* window = nullptr;
        bool isMinimised = false;

        /*
         * We have to create an instance with enabled required extensions and optional validation layers
         * We then have to create a surface which as we using an SDL window, SDL provides a useful function for
         * We then have to find a suitable physical device (GPU) that supports what we need including queue families
         * We then we have to create a logical device which we use to interface with the physical device which cares about
         * and uses what we asked our physical device to support
         * We then have to create the swap chain which is responsible for managing the image buffers
         */
        void initVulkan() {
            createInstance();
            createSurface();
            pickPhysicalDevice();
            createLogicalDevice();
            createSwapChain();
            createImageViews();
            createRenderPass();
            createGraphicsPipeline();
            createFramebuffers();
            createCommandPool();
            createVertexBuffer();
            createCommandBuffers();
            createSyncObjects();
        }

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

            // How many images in the chain
            uint32 imageCount = std::clamp(swapChainSupport.capabilities.minImageCount + 1,
                                           swapChainSupport.capabilities.minImageCount,
                                           swapChainSupport.capabilities.maxImageCount);

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

        void createImageViews() {
            swapChainImageViews.resize(swapChainImages.size());
            for (uint32 i = 0; i < swapChainImages.size(); i++) {
                VkImageViewCreateInfo createInfo{};
                createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                createInfo.image = swapChainImages[i];

                // Specifies how the image data should be interpreted
                createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                createInfo.format = swapChainImageFormat;

                // Allows swizzling the colour channels around eg map all channels to the red channel
                createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

                // Describes the images purpose and which part we want to access
                createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                createInfo.subresourceRange.baseMipLevel = 0;
                createInfo.subresourceRange.levelCount = 1;
                createInfo.subresourceRange.baseArrayLayer = 0;
                createInfo.subresourceRange.layerCount = 1;

                if (vkCreateImageView(logicalDevice, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create image view");
                }
            }
        }

        void createRenderPass() {
            VkAttachmentDescription colorAttachment{};
            colorAttachment.format = swapChainImageFormat;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

            // What to do with colour and depth data
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            // What to do with the stencil buffer
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            VkAttachmentReference colorAttachmentReference{};
            colorAttachmentReference.attachment = 0;
            colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentReference;

            VkSubpassDependency subpassDependency{};
            subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            subpassDependency.dstSubpass = 0;
            subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            subpassDependency.srcAccessMask = 0;
            subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            subpassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            VkRenderPassCreateInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = 1;
            renderPassInfo.pAttachments = &colorAttachment;
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
            renderPassInfo.dependencyCount = 1;
            renderPassInfo.pDependencies = &subpassDependency;

            if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
                throw std::runtime_error("failed to create render pass");
            }

            SDL_Log("Render pass created");
        }

        void createGraphicsPipeline() {
            auto vertShaderCode = readFile("../shaders/compiled/vert.spv");
            auto fragShaderCode = readFile("../shaders/compiled/frag.spv");

            VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
            VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

            VkPipelineShaderStageCreateInfo vertShaderStageCreateInfo{};
            vertShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            vertShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
            vertShaderStageCreateInfo.module = vertShaderModule;
            vertShaderStageCreateInfo.pName = "main";
            // pSpecializationInfo allows for specifying constants at this time which is more performant than at runtime

            VkPipelineShaderStageCreateInfo fragShaderStageCreateInfo{};
            fragShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            fragShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            fragShaderStageCreateInfo.module = fragShaderModule;
            fragShaderStageCreateInfo.pName = "main";

            VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageCreateInfo, fragShaderStageCreateInfo};

            // Very few things can be changed without recreating the entire pipeline. So this means we must provide these
            // at draw time
            std::vector<VkDynamicState> dynamicStates = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR
            };

            VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo{};
            dynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamicStateCreateInfo.dynamicStateCount = dynamicStates.size();
            dynamicStateCreateInfo.pDynamicStates = dynamicStates.data();

            // This would be where we define how the vertex data looks and the attributes we are passing to the shader
            // but initially cheating by specifying vertex data in the shaders directly
            auto bindingDescription = Vertex::getBindingDescription();
            auto attributeDescriptions = Vertex::getAttributeDescriptions();

            VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo{};
            vertexInputCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputCreateInfo.vertexBindingDescriptionCount = 1;
            vertexInputCreateInfo.pVertexBindingDescriptions = &bindingDescription;
            vertexInputCreateInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
            vertexInputCreateInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

            /*
             * What kind of geometry will be drawn and if primitive restart should be enabled
             */
            VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo{};
            inputAssemblyCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssemblyCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssemblyCreateInfo.primitiveRestartEnable = VK_FALSE;

            // As we are defining them dynamically at drawtime, we only specify their count. The code above would be for
            // specifying them statically now but then to change them we would need to recreate the pipeline
            VkPipelineViewportStateCreateInfo viewportStateCreateInfo{};
            viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportStateCreateInfo.viewportCount = 1;
            viewportStateCreateInfo.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rasterizerCreateInfo{};
            rasterizerCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizerCreateInfo.depthClampEnable = VK_FALSE;

            // Setting this to true doesn't let anything through this stage
            rasterizerCreateInfo.rasterizerDiscardEnable = VK_FALSE;

            /*
             * FILL fills polygon with fragments
             * LINE edges are drawn as lines
             * POINT vertices are drawn as points
             */
            rasterizerCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizerCreateInfo.lineWidth = 1.0f;

            // Do we cull faces and which do we consider front facing
            rasterizerCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
            rasterizerCreateInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;

            // Can alter the depth values by adding a constant or biasing based on slope
            rasterizerCreateInfo.depthBiasEnable = VK_FALSE;
            rasterizerCreateInfo.depthBiasConstantFactor = 0.0f;
            rasterizerCreateInfo.depthBiasClamp = 0.0f;
            rasterizerCreateInfo.depthBiasSlopeFactor = 0.0f;

            // Disable multisampling for now
            VkPipelineMultisampleStateCreateInfo multisampleCreateInfo{};
            multisampleCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampleCreateInfo.sampleShadingEnable = VK_FALSE;
            multisampleCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            multisampleCreateInfo.minSampleShading = 1.0f;
            multisampleCreateInfo.pSampleMask = nullptr;
            multisampleCreateInfo.alphaToCoverageEnable = VK_FALSE;
            multisampleCreateInfo.alphaToOneEnable = VK_FALSE;

            // Depth and stencil testing createinfo would be here

            // Disable colour blending for now. Colour blending controls how the new color from the fragment shader
            // interacts with the colour currently in the framebuffer
            VkPipelineColorBlendAttachmentState colorBlendAttachmentState{};
            colorBlendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachmentState.blendEnable = VK_FALSE;
            colorBlendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
            colorBlendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;

            VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo{};
            colorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlendStateCreateInfo.logicOpEnable = VK_FALSE;
            colorBlendStateCreateInfo.logicOp = VK_LOGIC_OP_COPY;
            colorBlendStateCreateInfo.attachmentCount = 1;
            colorBlendStateCreateInfo.pAttachments = &colorBlendAttachmentState;
            colorBlendStateCreateInfo.blendConstants[0] = 0.0f;
            colorBlendStateCreateInfo.blendConstants[1] = 0.0f;
            colorBlendStateCreateInfo.blendConstants[2] = 0.0f;
            colorBlendStateCreateInfo.blendConstants[3] = 0.0f;

            // VK Pipeline is where we specify uniform shader values. Empty for now
            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
            pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutCreateInfo.setLayoutCount = 0;
            pipelineLayoutCreateInfo.pSetLayouts = nullptr;
            pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
            pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

            if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutCreateInfo, nullptr,
                                       &pipelineLayout) != VK_SUCCESS) {
                throw std::runtime_error("failed to create pipeline layout");
            }

            VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
            pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineCreateInfo.stageCount = 2;
            pipelineCreateInfo.pStages = shaderStages;

            pipelineCreateInfo.pVertexInputState = &vertexInputCreateInfo;
            pipelineCreateInfo.pInputAssemblyState = &inputAssemblyCreateInfo;
            pipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
            pipelineCreateInfo.pRasterizationState = &rasterizerCreateInfo;
            pipelineCreateInfo.pMultisampleState = &multisampleCreateInfo;
            pipelineCreateInfo.pDepthStencilState = nullptr;
            pipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
            pipelineCreateInfo.pDynamicState = &dynamicStateCreateInfo;

            pipelineCreateInfo.layout = pipelineLayout;

            pipelineCreateInfo.renderPass = renderPass;
            pipelineCreateInfo.subpass = 0;

            // We can create a new graphics pipeline by deriving it from an existing one. There is currently one pipeline
            // so specify invalid index. This would also require VK_PIPELINE_CREATE_DERIVATIVE_BIT flag
            pipelineCreateInfo.basePipelineHandle = nullptr;
            pipelineCreateInfo.basePipelineIndex = -1;

            if (vkCreateGraphicsPipelines(logicalDevice, nullptr, 1, &pipelineCreateInfo,
                                          nullptr, &graphicsPipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create graphics pipeline");
            }

            SDL_Log("Graphics pipeline created");

            vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
            vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
        }

        void createFramebuffers() {
            swapChainFramebuffers.resize(swapChainImageViews.size());
            for (uint64 i = 0; i < swapChainImageViews.size(); i++) {
                VkImageView attachments[] = {swapChainImageViews[i]};

                VkFramebufferCreateInfo framebufferCreateInfo{};
                framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferCreateInfo.renderPass = renderPass;
                framebufferCreateInfo.attachmentCount = 1;
                framebufferCreateInfo.pAttachments = attachments;
                framebufferCreateInfo.width = swapChainExtent.width;
                framebufferCreateInfo.height = swapChainExtent.height;
                framebufferCreateInfo.layers = 1;

                if (vkCreateFramebuffer(logicalDevice, &framebufferCreateInfo, nullptr,
                                        &swapChainFramebuffers[i]) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create framebuffer");
                }
            }

            SDL_Log("Framebuffers created");
        }

        /*
         * Command buffers are executed on one device queue like graphics or presentation
         * Each command pool can only allocate to a single type of queue - we have chosen graphics queue family here to draw
         */
        void createCommandPool() {
            QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

            VkCommandPoolCreateInfo commandPoolCreateInfo{};
            commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

            if (vkCreateCommandPool(logicalDevice, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS) {
                throw std::runtime_error("failed to create command pool");
            }

            SDL_Log("Command pool created");
        }

        void createVertexBuffer() {
            VkBufferCreateInfo bufferCreateInfo{};
            bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferCreateInfo.size = sizeof(vertices[0]) * vertices.size();
            bufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateBuffer(logicalDevice, &bufferCreateInfo, nullptr, &vertexBuffer) != VK_SUCCESS) {
                throw std::runtime_error("failed to create vertex buffer");
            }
            SDL_Log("Vertex buffer created");

            VkMemoryRequirements memoryRequirements;
            vkGetBufferMemoryRequirements(logicalDevice, vertexBuffer, &memoryRequirements);

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memoryRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate vertex buffer memory");
            }

            // The final argument is the offset in this block of memory, because this was created specifically for this
            // we start at the start
            vkBindBufferMemory(logicalDevice, vertexBuffer, vertexBufferMemory, 0);
            SDL_Log("Vertex buffer memory allocated and bound");

            // Now we have to copy the vertex data to the buffer
            void* mappedMemory;
            /*  This maps the buffer memory to CPU accessible memory, copy the data and unmap
             *  Of note, the data may not be copied immediately into the buffer memory and also the writes to the buffer
             *  may not be visible in the mapped memory yet. There are two solutions:
             *      Use VK_MEMORY_PROPERTY_HOST_COHERENT_BIT as we did or
             *      call a vk function to flush memory ranges after writing and a vk function to invalidate ranges
             *      before reading
             *  The bit way can may/maybe not affect performance?
             *  Also, this still does not guarantee the data is visible on the GPU yet. All we are guaranteed is that it
             *  will be completed before the next call to vkQueueSubmit
             */
            vkMapMemory(logicalDevice, vertexBufferMemory, 0, bufferCreateInfo.size, 0, &mappedMemory);
            memcpy(mappedMemory, vertices.data(), bufferCreateInfo.size);
            vkUnmapMemory(logicalDevice, vertexBufferMemory);
        }

        void createCommandBuffers() {
            commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
            commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandBufferAllocateInfo.commandPool = commandPool;
            commandBufferAllocateInfo.commandBufferCount = (uint32)commandBuffers.size();

            if (vkAllocateCommandBuffers(logicalDevice, &commandBufferAllocateInfo, commandBuffers.data()) !=
                VK_SUCCESS) {
                throw std::runtime_error("failed to create command buffers");
            }

            SDL_Log("Command buffers created");
        }

        void createSyncObjects() {
            imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

            VkSemaphoreCreateInfo semaphoreCreateInfo{};
            semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

            // We create the fence in a signaled state so we don't wait infinitely on the first frame
            VkFenceCreateInfo fenceCreateInfo{};
            fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                if ((vkCreateSemaphore(logicalDevice, &semaphoreCreateInfo, nullptr, &imageAvailableSemaphores[i]) !=
                        VK_SUCCESS)
                    || (vkCreateSemaphore(logicalDevice, &semaphoreCreateInfo, nullptr, &renderFinishedSemaphores[i]) !=
                        VK_SUCCESS)
                    || (vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)) {
                    throw std::runtime_error("failed to sync objects");
                }
            }

            SDL_Log("Sync objects created");
        }

        void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32 imageIndex) {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0;
            beginInfo.pInheritanceInfo = nullptr;

            if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to start recording command buffer");
            }

            VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];

            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = swapChainExtent;

            VkClearValue clearColorValue = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColorValue;

            vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            // Because we set viewport and scissor to be dynamic we have to specify them here
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)swapChainExtent.width;
            viewport.height = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            VkBuffer vertexBuffers[] = {vertexBuffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

            // ISSUE THE DRAW COMMAND!
            vkCmdDraw(commandBuffer, vertices.size(), 1, 0, 0);

            vkCmdEndRenderPass(commandBuffer);

            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer");
            }
        }

        void drawFrame() {
            vkWaitForFences(logicalDevice, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

            uint32 imageIndex = 0;
            VkResult result = vkAcquireNextImageKHR(logicalDevice, swapChain, UINT64_MAX,
                                                    imageAvailableSemaphores[currentFrame], nullptr,
                                                    &imageIndex);

            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapChain();
                return;
            } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
                throw std::runtime_error("failed to acquire swap chain image");
            }

            vkResetFences(logicalDevice, 1, &inFlightFences[currentFrame]);

            vkResetCommandBuffer(commandBuffers[currentFrame], 0);
            recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

            VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
            VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;

            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

            VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;

            if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
                throw std::runtime_error("failed to submit draw command buffer");
            }

            VkPresentInfoKHR presentInfo{};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores;

            VkSwapchainKHR swapChains[] = {swapChain};
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapChains;
            presentInfo.pImageIndices = &imageIndex;
            presentInfo.pResults = nullptr;

            // Present our triangle to the screen!
            result = vkQueuePresentKHR(presentQueue, &presentInfo);

            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || staleSwapChain) {
                staleSwapChain = false;
                recreateSwapChain();
            } else if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to acquire swap chain image");
            }

            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        /*
         * This is nasty. We would like to be creating our new swap chain whilst we are still processing images in the old
         * one. This is doable, you just need to pass the old swap chain in VkSwapchainCreateInfoKHR.oldSwapChain
         */
        void recreateSwapChain() {
            while (isMinimised) {
                SDL_Event event;
                while (SDL_PollEvent(&event)) {
                    switch (event.type) {
                        case SDL_EVENT_WINDOW_RESTORED: isMinimised = false;
                    }
                    SDL_Delay(100);
                }
            }

            vkDeviceWaitIdle(logicalDevice);

            cleanupSwapChain();

            createSwapChain();
            createImageViews();
            createFramebuffers();
        }

        void mainLoop() {
            RUNNING = true;
            while (RUNNING) {
                SDL_Event event;
                while (SDL_PollEvent(&event)) {
                    switch (event.type) {
                        case SDL_EVENT_QUIT: RUNNING = false;
                            break;
                        case SDL_EVENT_WINDOW_RESIZED: staleSwapChain = true;
                            break;
                        case SDL_EVENT_WINDOW_MINIMIZED: {
                            isMinimised = true;
                            staleSwapChain = true;
                            break;
                        }
                    }
                }
                drawFrame();
            }
            vkDeviceWaitIdle(logicalDevice);
        }

        void initWindow() {
            if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_EVENTS)) {
                SDL_GetError();
                return;
            }
            SDL_SetAppMetadata("Engine", "0.0.1", nullptr);
            // Hints are configuration variables, there are many more https://wiki.libsdl.org/SDL3/CategoryHints
            SDL_SetHint(SDL_HINT_EVENT_LOGGING, "1");
            window = SDL_CreateWindow("Engine", 1920, 1080,
                                      SDL_WINDOW_INPUT_FOCUS | SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
        }

        void cleanupSwapChain() {
            for (auto framebuffer : swapChainFramebuffers) {
                vkDestroyFramebuffer(logicalDevice, framebuffer, nullptr);
            }
            for (auto imageView : swapChainImageViews) {
                vkDestroyImageView(logicalDevice, imageView, nullptr);
            }
            vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);
        }

        /*
         * Feels largely pointless and think about removing in actual implementation
         * When we close the application we don't want the user to wait for us to needlessly clean up memory that the OS
         * will handle.
         */
        void cleanup() {
            cleanupSwapChain();

            vkDestroyBuffer(logicalDevice, vertexBuffer, nullptr);
            vkFreeMemory(logicalDevice, vertexBufferMemory, nullptr);

            vkDestroyPipeline(logicalDevice, graphicsPipeline, nullptr);
            vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);

            vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vkDestroySemaphore(logicalDevice, imageAvailableSemaphores[i], nullptr);
                vkDestroySemaphore(logicalDevice, renderFinishedSemaphores[i], nullptr);
                vkDestroyFence(logicalDevice, inFlightFences[i], nullptr);
            }

            vkDestroyCommandPool(logicalDevice, commandPool, nullptr);

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
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                                     queueFamilies.data());

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
            vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount,
                                                 availableExtensions.data());

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
                vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                                     details.formats.data());
            }

            uint32 presentModeCount;
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
            if (presentModeCount != 0) {
                details.presentModes.resize(presentModeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount,
                                                          details.presentModes.data());
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

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                                            capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                                             capabilities.maxImageExtent.height);

            return actualExtent;
        }

        VkShaderModule createShaderModule(const std::vector<char>& code) {
            VkShaderModuleCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32*>(code.data());

            VkShaderModule shaderModule = nullptr;
            if (vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
                throw std::runtime_error("Could not create shader module");
            }
            return shaderModule;
        }

        /*
        * This seems crazy. We need to get passed different types of memory available and check that one matches the
        * requirements we need for our operation...
        */
        uint32 findMemoryType(uint32 typeFilter, VkMemoryPropertyFlags properties) {
            VkPhysicalDeviceMemoryProperties memoryProperties;
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
            for (uint32 i = 0; i < memoryProperties.memoryTypeCount; i++) {
                if ((typeFilter & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                    SDL_Log("Found valid memory type");
                    return i;
                }
            }
            throw std::runtime_error("failed to find a suitable memory type");
        }
};

// Entry point for SDL3 with header inclusion
int main(int argc, char** argv) {
    FirstVulkanTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
