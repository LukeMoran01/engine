//
// Created by Luke Moran on 04/02/2025.
//

#include <camera.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

void Camera::update() {
    // Converts relative movement (forward in camera space) to world space
    glm::mat4 cameraRotation = getRotationMatrix();
    position += glm::vec3(cameraRotation * glm::vec4(velocity * 0.5f, 0.f));
}

// TODO currently doesnt take delta time into account which it should - our speed changes based on frame rate now
void Camera::processSDLEvent(SDL_Event& e) {
    switch (e.type) {
        case SDL_EVENT_KEY_DOWN:
            switch (e.key.key) {
                case SDLK_W: {
                    velocity.z = -1;
                    break;
                }
                case SDLK_S: {
                    velocity.z = 1;
                    break;
                }
                case SDLK_A: {
                    velocity.x = -1;
                    break;
                }
                case SDLK_D: {
                    velocity.x = 1;
                    break;
                }
                default:
                    break;
            }
            break;
        // TODO Awkward that on releasing one key after multiple pressed velocity hits 0
        case SDL_EVENT_KEY_UP:
            switch (e.key.key) {
                case SDLK_W: {
                    velocity.z = 0;
                    break;
                }
                case SDLK_S: {
                    velocity.z = 0;
                    break;
                }
                case SDLK_A: {
                    velocity.x = 0;
                    break;
                }
                case SDLK_D: {
                    velocity.x = 0;
                    break;
                }
                default:
                    break;
            }
            break;
        case SDL_EVENT_MOUSE_MOTION:
            yaw += e.motion.xrel / 200.f;
            pitch -= e.motion.yrel / 200.f;
            break;

        default: break;
    }
}

// To create a correct model view we move the world in opposite direction to the camera
glm::mat4 Camera::getViewMatrix() {
    glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.0f), position);
    glm::mat4 cameraRotation    = getRotationMatrix();
    return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::getRotationMatrix() {
    glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3{1.f, 0.f, 0.f});
    glm::quat yawRotation   = glm::angleAxis(yaw, glm::vec3{0.f, -1.f, 0.f});
    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}



