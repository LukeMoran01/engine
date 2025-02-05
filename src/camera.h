//
// Created by Luke Moran on 04/02/2025.
//

#pragma once

#include <vk_types.h>
#include <SDL3/SDL_events.h>

// TODO CHANGE CAMERA
// The camera is more of a gameplay layer object. We will add it into the VulkanEngine,
// but in a real architecture, you probably dont want to be doing input events and game logic within the engine itself,
// instead you would only store a camera struct that contains the render parameters, and when you update game logic,
// you refresh those matrices so they can be used with rendering.

class Camera {
public:
    glm::vec3 velocity;
    glm::vec3 position;
    float speed;

    // Vertical/Horizontal
    float pitch{0.f};
    float yaw{0.f};

    glm::mat4 getViewMatrix();
    glm::mat4 getRotationMatrix();

    void processSDLEvent(SDL_Event& e);

    void update();
};
