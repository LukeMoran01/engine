//
// Created by Luke Moran on 22/02/2025.
//

#pragma once
#include "game.h"
#include "input_handler.h"
#include "vk_engine.h"

class Engine {
public:
    bool isRunning;

    InputHandler* input;
    Game* game;
    VulkanRenderer* renderer;


    void init();
    void update();
    void cleanup();

private:
};
