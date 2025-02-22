//
// Created by Luke Moran on 22/02/2025.
//

#include "input_handler.h"

#include <imgui_impl_sdl3.h>
#include <SDL3/SDL_events.h>

#include "vk_types.h"

void InputHandler::update() {
    SDL_Event event;
    // Handle events on queue
    while (SDL_PollEvent(&event) != 0) {
        // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
        switch (event.type) {
            case SDL_EVENT_QUIT: {
                quitEvent = true;
                break;
            }
            case SDL_EVENT_WINDOW_MINIMIZED: {
                WindowEvent minimize = MINIMIZE;
                windowEventQueue->pushEvent(minimize);
                break;
            }
            case SDL_EVENT_WINDOW_RESTORED: {
                WindowEvent restore = RESTORE;
                windowEventQueue->pushEvent(restore);
                break;
            }
            // case SDL_EVENT_MOUSE_BUTTON_DOWN: {
            //     if (event.button.button == SDL_BUTTON_RIGHT) {
            //         SDL_SetWindowRelativeMouseMode(window, true);
            //         SDL_SetWindowMouseGrab(window, true);
            //     }
            //     break;
            // }
            // case SDL_EVENT_MOUSE_BUTTON_UP: {
            //     if (event.button.button == SDL_BUTTON_RIGHT) {
            //         SDL_SetWindowRelativeMouseMode(window, false);
            //         SDL_SetWindowMouseGrab(window, false);
            //     }
            //     break;
        }
    }
    // if (SDL_GetWindowRelativeMouseMode(window)) {
    //     mainCamera.processSDLEvent(event);
    // }
    ImGui_ImplSDL3_ProcessEvent(&event);
}

// }

