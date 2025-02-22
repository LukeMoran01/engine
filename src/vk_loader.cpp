//
// Created by Luke Moran on 30/11/2024.
//

// ReSharper disable CppTooWideScopeInitStatement
#include <vk_loader.h>
#include <stb_image.h>
#include <iostream>

#include <vk_engine.h>
#include <vk_initializers.h>
#include <vk_types.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/core.hpp>

#include "vk_images.h"

// fastgltf samplers use openGL properties so we need to convert them
VkFilter extractFilter(fastgltf::Filter filter);
VkSamplerMipmapMode extractMipmapMode(fastgltf::Filter filter);

// Single mesh loading function
std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanRenderer* engine,
                                                                      const std::filesystem::path& filePath) {
    auto expData = fastgltf::GltfDataBuffer::FromPath(filePath);
    if (!expData) {
        spdlog::error("Failed to load gltf file: {} \n", to_underlying(expData.error()));
        return {};
    }
    auto data = std::move(expData.get());

    constexpr auto gltfOptions = fastgltf::Options::LoadExternalBuffers;

    fastgltf::Asset gltf;
    fastgltf::Parser parser{};

    auto load = parser.loadGltfBinary(data, filePath.parent_path(), gltfOptions);
    if (!load) {
        spdlog::error("Failed to load gltf: {} \n", to_underlying(load.error()));
        return {};
    }
    gltf = std::move(load.get());

    std::vector<std::shared_ptr<MeshAsset>> meshes;

    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;
    for (fastgltf::Mesh& mesh : gltf.meshes) {
        MeshAsset newMesh;
        newMesh.name = mesh.name;

        indices.clear();
        vertices.clear();

        for (auto&& p : mesh.primitives) {
            GeoSurface newSurface{};
            newSurface.startIndex = indices.size();
            newSurface.count      = gltf.accessors[p.indicesAccessor.value()].count;

            size_t initial_vtx = vertices.size();
            {
                fastgltf::Accessor& indexAccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexAccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexAccessor, [&](std::uint32_t index) {
                    indices.push_back(index + initial_vtx);
                });
            }

            {
                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->accessorIndex];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor, [&](glm::vec3 v, size_t index) {
                    Vertex newVertex{};
                    newVertex.position            = v;
                    newVertex.normal              = {1, 0, 0};
                    newVertex.color               = glm::vec4{1.0f};
                    newVertex.uv_x                = 0;
                    newVertex.uv_y                = 0;
                    vertices[initial_vtx + index] = newVertex;
                });
            }

            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[normals->accessorIndex],
                                                              [&](glm::vec3 v, size_t index) {
                                                                  vertices[initial_vtx + index].normal = v;
                                                              });
            }

            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[uv->accessorIndex],
                                                              [&](glm::vec2 v, size_t index) {
                                                                  vertices[initial_vtx + index].uv_x = v.x;
                                                                  vertices[initial_vtx + index].uv_y = v.y;
                                                              });
            }

            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[colors->accessorIndex],
                                                              [&](glm::vec4 v, size_t index) {
                                                                  vertices[initial_vtx + index].color = v;
                                                              });
            }
            newMesh.surfaces.push_back(newSurface);
        }

        constexpr bool OverrideColors = false;
        if (OverrideColors) {
            for (Vertex& v : vertices) {
                v.color = glm::vec4{v.normal, 1.0f};
            }
        }
        newMesh.meshBuffers = engine->uploadMesh(indices, vertices);

        meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newMesh)));
    }
    return meshes;
}

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanRenderer* engine, std::string_view filePath) {
    spdlog::info("Loading GLTF: {}", filePath);
    std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
    scene->creator                    = engine;
    LoadedGLTF& file                  = *scene;

    fastgltf::Parser parser{};

    constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble |
        fastgltf::Options::LoadExternalBuffers;

    auto expData = fastgltf::GltfDataBuffer::FromPath(filePath);
    if (!expData) {
        spdlog::error("Failed to load gltf file: {} \n", to_underlying(expData.error()));
        return {};
    }
    auto data = std::move(expData.get());

    fastgltf::Asset gltf;

    std::filesystem::path path = filePath;

    auto expLoad = parser.loadGltf(data, path.parent_path(), gltfOptions);
    if (!expLoad) {
        spdlog::error("Failed to parse gltf file: {} \n", to_underlying(expData.error()));
        return {};
    }
    gltf = std::move(expLoad.get());
    // TODO revisit understanding
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
    };
    file.descriptorPool.init(engine->device, gltf.materials.size(), sizes);

    for (fastgltf::Sampler& sampler : gltf.samplers) {
        VkSamplerCreateInfo samplerInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .pNext = nullptr};
        samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
        samplerInfo.minLod = 0;

        samplerInfo.magFilter = extractFilter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
        samplerInfo.minFilter = extractFilter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        samplerInfo.mipmapMode = extractMipmapMode(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        VkSampler newSampler;
        vkCreateSampler(engine->device, &samplerInfo, nullptr, &newSampler);

        file.samplers.push_back(newSampler);
    }

    // temporary arrays whilst creating the data
    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<AllocatedImage> images;
    std::vector<std::shared_ptr<GLTFMaterial>> materials;
    // Strict dependencies. MeshNodes dep Meshes dep Materials dep Textures
    for (fastgltf::Image& image : gltf.images) {
        std::optional<AllocatedImage> img = loadImage(engine, gltf, image);

        if (img.has_value()) {
            images.push_back(img.value());
            file.images[image.name.c_str()] = img.value();
        } else {
            images.push_back(engine->errorCheckerboardImage);
            std::cout << "gltf failed to load texture " << image.name << std::endl;
        }
    }

    file.materialDataBuffer = engine->createBuffer(
        sizeof(GLTFMetallic_Roughness::MaterialConstants) * gltf.materials.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    int dataIndex                = 0;
    auto* sceneMaterialConstants = static_cast<
        GLTFMetallic_Roughness::MaterialConstants*>(file.materialDataBuffer.allocationInfo.pMappedData);

    for (fastgltf::Material& material : gltf.materials) {
        std::shared_ptr<GLTFMaterial> newMat = std::make_shared<GLTFMaterial>();
        materials.push_back(newMat);
        file.materials[material.name.c_str()] = newMat;

        GLTFMetallic_Roughness::MaterialConstants constants{};
        constants.colorFactors.x = material.pbrData.baseColorFactor[0];
        constants.colorFactors.y = material.pbrData.baseColorFactor[1];
        constants.colorFactors.z = material.pbrData.baseColorFactor[2];
        constants.colorFactors.w = material.pbrData.baseColorFactor[3];

        constants.metal_rough_factors.x = material.pbrData.metallicFactor;
        constants.metal_rough_factors.y = material.pbrData.roughnessFactor;

        sceneMaterialConstants[dataIndex] = constants;

        MaterialPass passType = MaterialPass::MainColor;
        if (material.alphaMode == fastgltf::AlphaMode::Blend) {
            passType = MaterialPass::Transparent;
        }

        GLTFMetallic_Roughness::MaterialResources materialResources{};
        materialResources.colorImage        = engine->whiteImage;
        materialResources.colorSampler      = engine->defaultSamplerLinear;
        materialResources.metalRoughImage   = engine->whiteImage;
        materialResources.metalRoughSampler = engine->defaultSamplerLinear;

        // Set the uniform buffer for the mat data
        materialResources.dataBuffer       = file.materialDataBuffer.buffer;
        materialResources.dataBufferOffset = dataIndex * sizeof(GLTFMetallic_Roughness::MaterialConstants);

        if (material.pbrData.baseColorTexture.has_value()) {
            size_t img     = gltf.textures[material.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
            size_t sampler = gltf.textures[material.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();

            materialResources.colorImage   = images[img];
            materialResources.colorSampler = file.samplers[sampler];
        }

        newMat->data = engine->metalRoughMaterial.writeMaterial(engine->device, passType, materialResources,
                                                                file.descriptorPool);

        dataIndex++;
    }

    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh& mesh : gltf.meshes) {
        std::shared_ptr<MeshAsset> newMesh = std::make_shared<MeshAsset>();
        meshes.push_back(newMesh);
        file.meshes[mesh.name.c_str()] = newMesh;
        newMesh->name                  = mesh.name;

        indices.clear();
        vertices.clear();

        for (auto& p : mesh.primitives) {
            GeoSurface newSurface;
            newSurface.startIndex = indices.size();
            newSurface.count      = gltf.accessors[p.indicesAccessor.value()].count;

            size_t initialVtx = vertices.size();

            {
                fastgltf::Accessor& indexAccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexAccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexAccessor, [&](std::uint32_t idx) {
                    indices.push_back(idx + initialVtx);
                });
            }
            {
                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->accessorIndex];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor, [&](glm::vec3 v, size_t index) {
                    Vertex newVtx{};
                    newVtx.position              = v;
                    newVtx.normal                = {1, 0, 0};
                    newVtx.color                 = glm::vec4(1.f);
                    newVtx.uv_x                  = 0;
                    newVtx.uv_y                  = 0;
                    vertices[initialVtx + index] = newVtx;
                });
            }
            // TODO Change IDE formatting to not dump args to right side only
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[normals->accessorIndex],
                                                              [&](glm::vec3 v, size_t index) {
                                                                  vertices[initialVtx + index].normal = v;
                                                              });
            }

            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[uv->accessorIndex],
                                                              [&](glm::vec2 v, size_t index) {
                                                                  vertices[initialVtx + index].uv_x = v.x;
                                                                  vertices[initialVtx + index].uv_y = v.y;
                                                              });
            }

            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[colors->accessorIndex],
                                                              [&](glm::vec4 v, size_t index) {
                                                                  vertices[initialVtx + index].color = v;
                                                              });
            }

            if (p.materialIndex.has_value()) {
                newSurface.material = materials[p.materialIndex.value()];
            } else {
                newSurface.material = materials[0];
            }

            glm::vec3 minPos = vertices[initialVtx].position;
            glm::vec3 maxPos = vertices[initialVtx].position;
            for (int i = initialVtx; i < vertices.size(); i++) {
                minPos = glm::min(minPos, vertices[i].position);
                minPos = glm::max(minPos, vertices[i].position);
            }

            newSurface.bounds.origin       = (maxPos + minPos) / 2.f;
            newSurface.bounds.extents      = (maxPos - minPos) / 2.f;
            newSurface.bounds.sphereRadius = glm::length(newSurface.bounds.extents);

            newMesh->surfaces.push_back(newSurface);
        }
        newMesh->meshBuffers = engine->uploadMesh(indices, vertices);
    }

    for (fastgltf::Node& node : gltf.nodes) {
        std::shared_ptr<Node> newNode;

        if (node.meshIndex.has_value()) {
            newNode                                     = std::make_shared<MeshNode>();
            static_cast<MeshNode*>(newNode.get())->mesh = meshes[node.meshIndex.value()];
        } else {
            newNode = std::make_shared<Node>();
        }

        nodes.push_back(newNode);
        file.nodes[node.name.c_str()];

        std::visit(fastgltf::visitor{
                       [&](fastgltf::math::fmat4x4 matrix) {
                           memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix));
                       },
                       [&](fastgltf::TRS transform) {
                           glm::vec3 tl(transform.translation[0], transform.translation[1], transform.translation[2]);
                           glm::quat rot(transform.rotation[3], transform.rotation[0], transform.rotation[1],
                                         transform.rotation[2]);
                           glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

                           glm::mat4 tm = glm::translate(glm::mat4(1.0f), tl);
                           glm::mat4 rm = glm::toMat4(rot);
                           glm::mat4 sm = glm::scale(glm::mat4(1.0f), sc);

                           newNode->localTransform = tm * rm * sm;
                       }
                   },
                   node.transform);
    }

    for (int i = 0; i < gltf.nodes.size(); i++) {
        fastgltf::Node& node             = gltf.nodes[i];
        std::shared_ptr<Node>& sceneNode = nodes[i];

        for (auto& c : node.children) {
            sceneNode->children.push_back(nodes[c]);
            nodes[c]->parent = sceneNode;
        }
    }

    for (auto& node : nodes) {
        if (node->parent.lock() == nullptr) {
            file.topNodes.push_back(node);
            node->refreshTransform(glm::mat4{1.f});
        }
    }
    return scene;
}

void LoadedGLTF::Draw(const glm::mat4& topMatrix, DrawContext& ctx) {
    for (auto& n : topNodes) {
        n->Draw(topMatrix, ctx);
    }
}

void LoadedGLTF::clearAll() {
    VkDevice dv = creator->device;

    descriptorPool.destroyPools(dv);
    creator->destroyBuffer(materialDataBuffer);

    for (auto& v : meshes | std::views::values) {
        creator->destroyBuffer(v->meshBuffers.indexBuffer);
        creator->destroyBuffer(v->meshBuffers.vertexBuffer);
    }

    for (auto& v : images | std::views::values) {
        if (v.image == creator->errorCheckerboardImage.image) {
            continue;
        }
        creator->destroyImage(v);
    }

    for (const auto& sampler : samplers) {
        vkDestroySampler(dv, sampler, nullptr);
    }
}


VkFilter extractFilter(fastgltf::Filter filter) {
    switch (filter) {
        case fastgltf::Filter::Nearest:
        case fastgltf::Filter::NearestMipMapNearest:
        case fastgltf::Filter::NearestMipMapLinear:
            return VK_FILTER_NEAREST;

        case fastgltf::Filter::Linear:
        case fastgltf::Filter::LinearMipMapNearest:
        case fastgltf::Filter::LinearMipMapLinear:
        default:
            return VK_FILTER_LINEAR;
    }
}

VkSamplerMipmapMode extractMipmapMode(fastgltf::Filter filter) {
    switch (filter) {
        case fastgltf::Filter::NearestMipMapNearest:
        case fastgltf::Filter::LinearMipMapNearest:
            return VK_SAMPLER_MIPMAP_MODE_NEAREST;

        case fastgltf::Filter::NearestMipMapLinear:
        case fastgltf::Filter::LinearMipMapLinear:
        default:
            return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
}
