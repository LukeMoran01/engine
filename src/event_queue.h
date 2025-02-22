//
// Created by Luke Moran on 22/02/2025.
//

#pragma once
#include <optional>
#include <queue>

enum WindowEvent {
    MINIMIZE,
    RESTORE
};

class WindowEventQueue {
public:
    std::optional<WindowEvent> pollEvent();
    void pushEvent(WindowEvent e);

    WindowEventQueue() {
        queue = std::queue<WindowEvent>();
    }

    ~WindowEventQueue() {
        while (!queue.empty()) queue.pop();
    };

private:
    std::queue<WindowEvent> queue;
};

inline std::optional<WindowEvent> WindowEventQueue::pollEvent() {
    if (queue.empty()) return std::nullopt;
    WindowEvent e = queue.front();
    queue.pop();
    return e;
}

inline void WindowEventQueue::pushEvent(WindowEvent e) {
    queue.push(e);
}


