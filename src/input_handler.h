//
// Created by Luke Moran on 22/02/2025.
//

#pragma once
#include "event_queue.h"

class InputHandler {
public:
    WindowEventQueue* windowEventQueue;
    // EventQueue* userEventQueue;
    void update();
    bool quitEvent = false;
};
