glslc ./shaders/source/vertex/first.vert -o ./shaders/compiled/vert.spv
glslc ./shaders/source/fragment/first.frag -o ./shaders/compiled/frag.spv
glslc ./shaders/source/compute/gradient.comp -o ./shaders/compiled/gradient.comp.spv
glslc ./shaders/source/compute/gradient_color.comp -o ./shaders/compiled/gradient_color.comp.spv
glslc ./shaders/source/compute/sky.comp -o ./shaders/compiled/sky.comp.spv