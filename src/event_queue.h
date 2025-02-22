//
// Created by Luke Moran on 22/02/2025.
//

#pragma once
#include <optional>
#include <queue>

#include "spdlog/spdlog.h"

// TODO Basic but will do the job for now

enum WindowEvent {
    MINIMIZE,
    RESTORE
};

template <>
struct fmt::formatter<WindowEvent> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(WindowEvent event, FormatContext& ctx) const {
        const char* name = "unknown";
        switch (event) {
            case MINIMIZE: name = "WindowEvent::Minimize";
                break;
            case RESTORE: name = "WindowEvent::Restore";
                break;
        }
        return fmt::format_to(ctx.out(), "{}", name);
    }
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


