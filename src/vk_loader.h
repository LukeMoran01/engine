//
// Created by Luke Moran on 30/11/2024.
//

#pragma once

#include <vk_types.h>
#include <filesystem>

#include "vk_descriptors.h"

class VulkanRenderer;

struct GLTFMaterial {
    MaterialInstance data;
};

struct Bounds {
    glm::vec3 origin;
    float sphereRadius;
    glm::vec3 extents;
};

struct GeoSurface {
    uint32_t startIndex;
    uint32_t count;
    Bounds bounds;
    std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset {
    std::string name;

    std::vector<GeoSurface> surfaces;
    GPUMeshBuffers meshBuffers;
};

struct LoadedGLTF : IRenderable {
    std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
    std::unordered_map<std::string, AllocatedImage> images;
    std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;

    // Nodes without a parents
    std::vector<std::shared_ptr<Node>> topNodes;

    // TODO likely few of these and could maybe be stored globally but data separation for now
    std::vector<VkSampler> samplers;

    DescriptorAllocatorGrowable descriptorPool;

    AllocatedBuffer materialDataBuffer;

    // Could be singleton
    VulkanRenderer* creator;

    ~LoadedGLTF() { clearAll(); };

    void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;

private:
    void clearAll();
};


std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanRenderer* engine,
                                                                      const std::filesystem::path& filePath);

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanRenderer* engine, std::string_view filePath);


