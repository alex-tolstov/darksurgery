#ifndef C_CONVOLUTIONAL_LAYER_H
#define C_CONVOLUTIONAL_LAYER_H

//#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer c_convolutional_layer;

#ifdef GPU
void forward_convolutional_layer_gpu(c_convolutional_layer layer, network_state state);
void backward_convolutional_layer_gpu(c_convolutional_layer layer, network_state state);
void update_convolutional_layer_gpu(c_convolutional_layer layer, int batch, float learning_rate, float momentum, float decay);

void push_convolutional_layer(c_convolutional_layer layer);
void pull_convolutional_layer(c_convolutional_layer layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l);
#endif
#endif

c_convolutional_layer make_c_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
void denormalize_c_convolutional_layer(c_convolutional_layer l);
void resize_c_convolutional_layer(c_convolutional_layer *layer, int w, int h);
void forward_c_convolutional_layer(const c_convolutional_layer layer, network_state state);
void update_c_convolutional_layer(c_convolutional_layer layer, int batch, float learning_rate, float momentum, float decay);
image *visualize_c_convolutional_layer(c_convolutional_layer layer, char *window, image *prev_weights);
void c_binarize_weights(float *weights, int n, int size, float *binary);
void swap_binary(c_convolutional_layer *l);
void binarize_weights2(float *weights, int n, int size, char *binary, float *scales);

void backward_c_convolutional_layer(c_convolutional_layer layer, network_state state);

void c_add_bias(float *output, float *biases, int batch, int n, int size);
void c_backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

image get_c_convolutional_image(c_convolutional_layer layer);
image get_c_convolutional_delta(c_convolutional_layer layer);
image get_c_convolutional_weight(c_convolutional_layer layer, int i);

int c_convolutional_out_height(c_convolutional_layer layer);
int c_convolutional_out_width(c_convolutional_layer layer);
void c_rescale_weights(c_convolutional_layer l, float scale, float trans);
void c_rgbgr_weights(c_convolutional_layer l);

#endif

