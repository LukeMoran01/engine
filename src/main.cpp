#include <vk_engine.h>

#include "engine.h"

int main(int argc, char* argv[]) {
    Engine newEngine{};
    newEngine.init();
    while (!newEngine.input->quitEvent) {
        newEngine.update();
    }
    newEngine.cleanup();


    // VulkanRenderer engine;
    // engine.init();
    // engine.run();
    // engine.cleanup();
    return 0;
}
