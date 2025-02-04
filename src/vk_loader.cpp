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

#include "fmt/printf.h"

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine,
                                                                      const std::filesystem::path& filePath) {
    auto expData = fastgltf::GltfDataBuffer::FromPath(filePath);
    if (!expData) {
        fmt::printf("Failed to load gltf file: {} \n", to_underlying(expData.error()));
        return {};
    }
    auto data = std::move(expData.get());

    constexpr auto gltfOptions = fastgltf::Options::LoadExternalBuffers;

    fastgltf::Asset gltf;
    fastgltf::Parser parser{};

    auto load = parser.loadGltfBinary(data, filePath.parent_path(), gltfOptions);
    if (!load) {
        fmt::print("Failed to load gltf: {} \n", to_underlying(load.error()));
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
