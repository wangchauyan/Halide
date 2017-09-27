// Halide tutorial lesson 21: Auto-Scheduler

// Before reading this file, see lesson_21_auto_scheduler_generate.cpp

// This is the code that actually uses the Halide pipeline we've
// compiled. It does not depend on libHalide, so we won't be including
// Halide.h.
//
// Instead, it depends on the header files that lesson_21_auto_scheduler_generator produced.
#include "conv_layer.h"

// We'll use the Halide::Runtime::Buffer class for passing data into and out of
// the pipeline.
#include "HalideBuffer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "clock.h"

// A function to help us report runtimes of various things. Prints the
// time elapsed since the last time you called this function.
double tick(const char *name) {
    static double t_old = 0;
    double t_new = current_time();
    double dt = t_new - t_old;
    if (name) {
        printf("%s: %f\n", name, dt);
    }
    t_old = t_new;
    return dt;
}

int main(int argc, char **argv) {

    // Let's make some images stored with interleaved and planar
    // memory. Halide::Runtime::Buffer is planar by default.
    Halide::Runtime::Buffer<float> input(67, 67, 32, 4);
    Halide::Runtime::Buffer<float> filter(3, 3, 32, 32);
    Halide::Runtime::Buffer<float> bias(32);
    Halide::Runtime::Buffer<float> output(64, 64, 32, 4);

    // Start the clock
    tick(NULL);

    // Run the planar version of the code on the planar images and the
    // interleaved version of the code on the interleaved
    // images. We'll run each 1000 times for benchmarking.
    for (int i = 0; i < 1000; i++) {
        conv_layer(input, filter, bias, output);
    }
    double t = tick("conv_layer");

    printf("Auto-scheduled time: %f\n", t);

    return 0;
}
