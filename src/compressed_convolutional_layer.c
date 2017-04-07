#include "compressed_convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>
#include <math.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

#ifndef AI2
#define AI2 0
void forward_xnor_layer(layer l, network_state state);
#endif

void c_swap_binary(c_convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

void c_binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void c_binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void c_binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int c_convolutional_out_height(c_convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int c_convolutional_out_width(c_convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_c_convolutional_image(c_convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_c_convolutional_delta(c_convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &l->bf_algo);
}
#endif
#endif

c_convolutional_layer make_c_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    c_convolutional_layer l = {0};
    l.type = C_CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weight_updates = calloc(c*n*size*size, sizeof(float));
    l.weight_masks = calloc(c*n*size*size, sizeof(float));
    for (i = 0; i < c * n * size * size; i++) {
		l.weight_masks[i] = 1.0f;
	}

    l.weight_tmp = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));
    l.bias_masks = calloc(n, sizeof(float));
    for (i = 0; i < n; i++) {
		l.bias_masks[i] = 1.0f;
	}

	l.bias_tmp = calloc(n, sizeof(float));

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_c_convolutional_layer;
    l.backward = backward_c_convolutional_layer;
    l.update = update_c_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.cweights = calloc(c*n*size*size, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.adam = 1;
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, c*n*size*size);
            l.v_gpu = cuda_make_array(l.v, c*n*size*size);
        }

        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "cconv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

void denormalize_c_convolutional_layer(c_convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c*l.size*l.size; ++j){
            l.weights[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

void test_c_convolutional_layer()
{
    c_convolutional_layer l = make_c_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    network_state state = {0};
    state.input = data;
    forward_c_convolutional_layer(l, state);
}

void resize_c_convolutional_layer(c_convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = c_convolutional_out_width(*l);
    int out_h = c_convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

void c_add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void c_scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void c_backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void forward_c_convolutional_layer(c_convolutional_layer l, network_state state)
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input);
        state.input = l.binary_input;
    }

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;

    float *a = l.weights;
    float *b = state.workspace;
    float *c = l.output;

	float *weightMask = l.weight_masks;
	float *weight = l.weights;
	float *weightTmp = l.weight_tmp;

	float *biasMask = l.bias_masks;
	float *bias = l.biases;
	float *biasTmp = l.bias_tmp;

	const unsigned int sizeWeights = l.c * l.n * l.size * l.size;
	const unsigned int sizeBiases = l.n;
	const unsigned int sizeTotal = l.c * l.h * l.w;
	
    if (state.train != 0) {
		// the first entry only
		if (*l.surgery.surg_stdev == 0 && ((*state.net.iter_) == 0)) {
			*l.surgery.surg_mu = 0;
			*l.surgery.surg_stdev = 0;
			
			unsigned int ncount = 0;
			unsigned int maskFaults = 0;
			for (unsigned int k = 0; k < sizeWeights; ++k) {
				*l.surgery.surg_mu  += fabs(l.weight_masks[k] * l.weights[k]);       
				*l.surgery.surg_stdev += l.weight_masks[k] * l.weights[k] * l.weights[k];
				if (l.weight_masks[k] * l.weights[k] != 0) {
					ncount++;
				}
			}
			if (!l.batch_normalize) {
				for (unsigned int k = 0; k < sizeBiases; ++k) {
					*l.surgery.surg_mu  += fabs(l.bias_masks[k] * l.biases[k]);
					*l.surgery.surg_stdev += l.bias_masks[k] * l.biases[k] * l.biases[k];
					if (l.bias_masks[k] * l.biases[k] != 0) {
						ncount++;
					}
				}
			}
			*l.surgery.surg_mu /= ncount; 
			*l.surgery.surg_stdev -= ncount * (*l.surgery.surg_mu) * (*l.surgery.surg_mu); 
			*l.surgery.surg_stdev /= ncount; 
			*l.surgery.surg_stdev = sqrt(*l.surgery.surg_stdev);
		//	printf("mu %.2f std %.2f ncount %d, total = %d\r\n", *l.surgery.surg_mu, *l.surgery.surg_stdev, ncount, sizeWeights + sizeBiases);
		}
		
		// Calculate the weight mask and bias mask with probability
		const float r = (float)rand() / (float)RAND_MAX;
		const float crate = l.surgery.surg_c_rate;
		const float mu = *l.surgery.surg_mu;
		const float std = *l.surgery.surg_stdev;
		const float gamma = l.surgery.surg_gamma;
		const float power = l.surgery.surg_power;
		const float poww = pow(1.f + gamma * (*state.net.iter_), -power);
		
		if ((poww > r) && (*state.net.iter_ < l.surgery.stop_iter_)) {
			int changes = 0;
			static int zeroFilling = 0;
			static int repairing = 0;
			float sumWeights = 0.0f;
			for (unsigned int k = 0; k < sizeWeights; ++k) {
				if (weightMask[k] == 1 && fabs(weight[k]) <= 0.9 * fmaxf(mu + crate * std, 0)) {
					weightMask[k] = 0;
					changes++;
					zeroFilling++;
				} else if (weightMask[k]==0 && fabs(weight[k]) > 1.1 * fmaxf(mu + crate * std, 0)) {
					weightMask[k] = 1;
					changes++;
					repairing++;
				}
				sumWeights += weight[k];
			}
			if ((*state.net.iter_) % 10 == 0) {
				printf("iter = %d, Changes = %d, sumWeights=%.4f, zeroF=%d, repair=%d\n", *state.net.iter_, changes, sumWeights, zeroFilling, repairing);
			}
			// equivalent to l.add_bias
			if (!l.batch_normalize) {
				for (unsigned int k = 0; k < sizeBiases; ++k) {
					if (biasMask[k] == 1 && fabs(bias[k]) <= 0.9 * fmaxf(mu + crate * std, 0)) {
						biasMask[k] = 0;
					} else if (biasMask[k] == 0 && fabs(bias[k]) > 1.1 * fmaxf(mu + crate * std, 0)) {
						biasMask[k] = 1;
					}
				}
			}
		}
		
		if ((*state.net.iter_) % 10 == 0) {
			int weightZeroes = 0;
			for (i = 0; i < sizeWeights; ++i) {
				if (fabs(weight[i] * weightMask[i]) <= 1e-5f) {
					weightZeroes++;
				}
			}
			float perc = 100.f * weightZeroes / sizeWeights;
			printf("self-defined iter = %d, zeroes weights = %d, percentage = %.4f\r\n", *state.net.iter_, weightZeroes, perc); 
		}
	}
	
	for (i = 0; i < sizeWeights; ++i) {
		weightTmp[i] = weight[i] * weightMask[i];
	}
	
	if (!l.batch_normalize) {
		for (i = 0; i < sizeBiases; i++) {
			biasTmp[i] = bias[i] * biasMask[i];
		}
	}

    for(i = 0; i < l.batch; ++i) {
        im2col_cpu(state.input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0, 0, m, n, k, 1, weightTmp, k, b, n, 1, c, n);
        c += n * m;
        state.input += sizeTotal;
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, state);
        
    } else {
        add_bias(l.output, biasTmp, l.batch, l.n, out_h*out_w);
    }

    activate_array(l.output, m*n*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

void backward_c_convolutional_layer(c_convolutional_layer l, network_state state)
{
    int i;
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, state);
    } else {
		const unsigned int sizeBiases = l.n;
		for (unsigned int k = 0; k < sizeBiases; ++k) {
			l.bias_updates[k] *= l.bias_masks[k];
		}
		backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

	const unsigned int sizeWeights = l.c * l.n * l.size * l.size;
	for (unsigned int k = 0; k < sizeWeights; ++k) {
		l.weight_updates[k] *= l.weight_masks[k];
	}

    for (i = 0; i < l.batch; ++i) {
        float *a = l.delta + i * m * k;
        float *b = state.workspace;
        float *c = l.weight_updates;
		
        float *im = state.input + i * l.c * l.h * l.w;

		// w.r.t. weight_diff, accumulating it!!!
        im2col_cpu(im, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

        if (state.delta) {
			// changed to weight_tmp because of pruned weights.
            a = l.weight_tmp;
            b = l.delta + i * m * k;
            c = state.workspace;

			// w.r.t. bottom data, if necessary
            gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

            col2im_cpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta + i * l.c * l.h * l.w);
        }
    }
}

void update_c_convolutional_layer(c_convolutional_layer l, int batch, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}


image get_c_convolutional_weight(c_convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void c_rgbgr_weights(c_convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_c_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void c_rescale_weights(c_convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_c_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *c_get_weights(c_convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_c_convolutional_weight(l, i));
        //normalize_image(weights[i]);
    }
    return weights;
}

image *visualize_c_convolutional_layer(c_convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_c_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

