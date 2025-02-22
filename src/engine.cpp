//
// Created by Luke Moran on 22/02/2025.
//

#include "engine.h"

void Engine::init() {
    auto weq = new WindowEventQueue();

    // game->init();
    input    = new InputHandler(weq);
    renderer = new VulkanRenderer(weq);
    renderer->init();

    isRunning = true;
}


void Engine::update() {
    // auto start = std::chrono::steady_clock::now();
    // auto end   = std::chrono::steady_clock::now();
    // auto delta = std::chrono::duration<float>(end - start).count();
    input->update();
    renderer->run();
    // game->update(delta);
    // renderer->draw();
}

void Engine::cleanup() {
    // renderer->cleanup();
}

