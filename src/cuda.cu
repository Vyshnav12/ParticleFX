#include "cuda.cuh"
#include "helper.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <cstring>
#include <cmath>

///
/// Algorithm storage
///
unsigned int cuda_particles_count;
Particle* d_particles;
unsigned int* d_pixel_contribs;
unsigned int* d_pixel_index;
unsigned char* d_pixel_contrib_colours;
float* d_pixel_contrib_depth;
unsigned int cuda_pixel_contrib_count;
int cuda_output_image_width;
int cuda_output_image_height;
__constant__ int D_OUTPUT_IMAGE_WIDTH;
__constant__ int D_OUTPUT_IMAGE_HEIGHT;
unsigned char* d_output_image_data;

void cpu_sort_pairs(float* keys_start, unsigned char* colours_start, int first, int last);

void cuda_begin(const Particle* init_particles, const unsigned int init_particles_count,
    const unsigned int out_image_width, const unsigned int out_image_height) {
    cuda_particles_count = init_particles_count;
    CUDA_CALL(cudaMalloc(&d_particles, init_particles_count * sizeof(Particle)));
    CUDA_CALL(cudaMemcpy(d_particles, init_particles, init_particles_count * sizeof(Particle), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_pixel_contribs, out_image_width * out_image_height * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc(&d_pixel_index, (out_image_width * out_image_height + 1) * sizeof(unsigned int)));
    d_pixel_contrib_colours = 0;
    d_pixel_contrib_depth = 0;
    cuda_pixel_contrib_count = 0;

    cuda_output_image_width = (int)out_image_width;
    cuda_output_image_height = (int)out_image_height;
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_WIDTH, &cuda_output_image_width, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_HEIGHT, &cuda_output_image_height, sizeof(int)));
    const int CHANNELS = 3;
    CUDA_CALL(cudaMalloc(&d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char)));
}

__global__ void cuda_stage1_kernel(const Particle* particles, const unsigned int particles_count,
    unsigned int* pixel_contribs, const int output_image_width, const int output_image_height) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < particles_count; i += stride) {
        Particle particle = particles[i];

        int x_min = (int)roundf(particle.location[0] - particle.radius);
        int y_min = (int)roundf(particle.location[1] - particle.radius);
        int x_max = (int)roundf(particle.location[0] + particle.radius);
        int y_max = (int)roundf(particle.location[1] + particle.radius);
        x_min = fmaxf(x_min, 0);
        y_min = fmaxf(y_min, 0);
        x_max = fminf(x_max, output_image_width - 1);
        y_max = fminf(y_max, output_image_height - 1);

        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - particle.location[0];
                const float y_ab = (float)y + 0.5f - particle.location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= particle.radius) {
                    const unsigned int pixel_offset = y * output_image_width + x;
                    atomicAdd(&pixel_contribs[pixel_offset], 1);
                }
            }
        }
    }
}


void cuda_stage1() {
    const int block_size = 32;
    const int count_blocks = (cuda_particles_count + block_size - 1) / block_size;
    cuda_stage1_kernel << <count_blocks, block_size >> > (d_particles, cuda_particles_count, d_pixel_contribs, cuda_output_image_width, cuda_output_image_height);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());



#ifdef VALIDATION
    unsigned int* h_pixel_contribs = new unsigned int[cuda_output_image_width * cuda_output_image_height];
    CUDA_CALL(cudaMemcpy(h_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    Particle* h_particles = new Particle[cuda_particles_count];
    CUDA_CALL(cudaMemcpy(h_particles, d_particles, cuda_particles_count * sizeof(Particle), cudaMemcpyDeviceToHost));

    // Copy additional arguments for validate_pixel_contribs
    int out_image_width = cuda_output_image_width;
    int out_image_height = cuda_output_image_height;
    int particles_count = cuda_particles_count;
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    validate_pixel_contribs(h_particles, particles_count, h_pixel_contribs, out_image_width, out_image_height);
#endif
}

//Quick sort function created on device so that it can be called by kernel
__device__ void gpu_sort_pairs(float* keys_start, unsigned char* colours_start, const int first, const int last)
{
    if (first < last)
    {
        int pivot = first;
        int i = first;
        int j = last;
        float pivot_value = keys_start[pivot];
        unsigned char pivot_color[4];
        for (int c = 0; c < 4; ++c)
        {
            pivot_color[c] = colours_start[4 * pivot + c];
        }

        while (i < j)
        {
            while (keys_start[i] <= pivot_value && i < last)
                i++;
            while (keys_start[j] > pivot_value)
                j--;
            if (i < j)
            {
                // Swap key
                float temp_key = keys_start[i];
                keys_start[i] = keys_start[j];
                keys_start[j] = temp_key;
                // Swap color
                for (int c = 0; c < 4; ++c)
                {
                    unsigned char temp_color = colours_start[4 * i + c];
                    colours_start[4 * i + c] = colours_start[4 * j + c];
                    colours_start[4 * j + c] = temp_color;
                }
            }
        }

        // Swap key
        float temp_key = keys_start[pivot];
        keys_start[pivot] = keys_start[j];
        keys_start[j] = temp_key;
        // Swap color
        for (int c = 0; c < 4; ++c)
        {
            unsigned char temp_color = colours_start[4 * pivot + c];
            colours_start[4 * pivot + c] = colours_start[4 * j + c];
            colours_start[4 * j + c] = temp_color;
        }

        // Recurse
        gpu_sort_pairs(keys_start, colours_start, first, j - 1);
        gpu_sort_pairs(keys_start, colours_start, j + 1, last);
    }
}


//Stage 2 Kernel for computing color/depth contibution and bounding box
__global__ void cuda_stage2_kernel(
    const Particle* particles, int particles_count, unsigned int* pixel_contribs, unsigned int* pixel_index,
    unsigned char* pixel_contrib_colours, float* pixel_contrib_depth,
    int output_image_width, int output_image_height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    // For each particle, store a copy of the colour/depth in d_pixel_contribs for each contributed pixel
    for (int i = idx; i < particles_count; i += stride)
    {
        // Compute bounding box [inclusive-inclusive]
        Particle particle = particles[i];
        int x_min = (int)roundf(particle.location[0] - particle.radius);
        int y_min = (int)roundf(particle.location[1] - particle.radius);
        int x_max = (int)roundf(particle.location[0] + particle.radius);
        int y_max = (int)roundf(particle.location[1] + particle.radius);
        // Clamp bounding box to image bounds
        x_min = fmaxf(x_min, 0);
        y_min = fmaxf(y_min, 0);
        x_max = fminf(x_max, output_image_width - 1);
        y_max = fminf(y_max, output_image_height - 1);
        // Store data for every pixel within the bounding box that falls within the radius
        for (int x = x_min; x <= x_max; ++x)
        {
            for (int y = y_min; y <= y_max; ++y)
            {
                const float x_ab = (float)x + 0.5f - particle.location[0];
                const float y_ab = (float)y + 0.5f - particle.location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);

                if (pixel_distance <= particle.radius)
                {
                    const unsigned int pixel_offset = y * output_image_width + x;
                    const unsigned int storage_offset = atomicAdd(&pixel_contribs[pixel_offset], 1);
                    const unsigned int tot_offset = pixel_index[pixel_offset] + storage_offset;

                    // Copy data to d_pixel_contrib buffers
                    pixel_contrib_colours[4 * tot_offset + 0] = particle.color[0];
                    pixel_contrib_colours[4 * tot_offset + 1] = particle.color[1];
                    pixel_contrib_colours[4 * tot_offset + 2] = particle.color[2];
                    pixel_contrib_colours[4 * tot_offset + 3] = particle.color[3];

                    pixel_contrib_depth[tot_offset] = particle.location[2];
                }
            }
        }
    }
}

//Kernel that calls Quick Sort Function
__global__ void cuda_stage2_sortColorsKernel(
    float* pixel_contrib_depth,
    unsigned char* pixel_contrib_colours,
    unsigned int* pixel_index,
    int output_image_width,
    int output_image_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < output_image_width * output_image_height; i += stride) {
        // Pair sort the colors which contribute to a single pixel
        float* depth = pixel_contrib_depth + pixel_index[i];
        unsigned char* color = pixel_contrib_colours + 4 * pixel_index[i];
        int first = 0;
        int last = pixel_index[i + 1] - pixel_index[i] - 1;

        gpu_sort_pairs(depth, color, first, last);
    }
}



void cuda_stage2()
{
    // Thrust calls to update d_pixel_index
    thrust::device_ptr<unsigned int> d_pixel_contribs_ptr(d_pixel_contribs);
    thrust::device_ptr<unsigned int> d_pixel_index_ptr(d_pixel_index);
    unsigned int* raw_pixel_contribs_ptr = d_pixel_contribs_ptr.get();
    unsigned int* raw_pixel_index_ptr = d_pixel_index_ptr.get();
    thrust::exclusive_scan(raw_pixel_contribs_ptr, raw_pixel_contribs_ptr + cuda_output_image_width * cuda_output_image_height + 1, d_pixel_index_ptr);

    // Recover the total from the index
    unsigned int TOTAL_CONTRIBS;
    cudaMemcpy(&TOTAL_CONTRIBS, &d_pixel_index[cuda_output_image_width * cuda_output_image_height], sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (TOTAL_CONTRIBS > cuda_pixel_contrib_count)
    {
        // (Re)Allocate colour storage
        if (d_pixel_contrib_colours) cudaFree(d_pixel_contrib_colours);
        if (d_pixel_contrib_depth) cudaFree(d_pixel_contrib_depth);
        CUDA_CALL(cudaMalloc(&d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char)));
        CUDA_CALL(cudaMalloc(&d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float)));
        cuda_pixel_contrib_count = TOTAL_CONTRIBS;
    }

    // Reset the pixel contributions histogram
    cudaMemset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int));

    // Store colours according to index
    int block_size = 32;
    int count_blocks = (cuda_particles_count + block_size - 1) / block_size;
    cuda_stage2_kernel << <count_blocks, block_size >> > (
        d_particles, cuda_particles_count, d_pixel_contribs, d_pixel_index,
        d_pixel_contrib_colours, d_pixel_contrib_depth,
        cuda_output_image_width, cuda_output_image_height);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    cuda_stage2_sortColorsKernel << <count_blocks, block_size >> > (
        d_pixel_contrib_depth, d_pixel_contrib_colours,
        d_pixel_index, cuda_output_image_width, cuda_output_image_height);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

#ifdef VALIDATION
    // Copy the data back to host for validation
    std::vector<unsigned int> h_pixel_index(cuda_output_image_width * cuda_output_image_height + 1);
    cudaMemcpy(h_pixel_index.data(), d_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    std::vector<unsigned int> h_pixel_contribs(cuda_output_image_width * cuda_output_image_height);
    cudaMemcpy(h_pixel_contribs.data(), d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pixel_index.data(), d_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    CUDA_CALL(cudaMemcpy(d_pixel_contrib_colours, d_pixel_contrib_colours, cuda_pixel_contrib_count * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(d_pixel_contrib_depth, d_pixel_contrib_depth, cuda_pixel_contrib_count * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::vector<float> h_pixel_contrib_depth(TOTAL_CONTRIBS);
    std::vector<unsigned char> h_pixel_contrib_colours(TOTAL_CONTRIBS * 4);
    cudaMemcpy(h_pixel_contrib_depth.data(), d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pixel_contrib_colours.data(), d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    std::vector<Particle> h_particles(cuda_particles_count);
    cudaMemcpy(h_particles.data(), d_particles, cuda_particles_count * sizeof(Particle), cudaMemcpyDeviceToHost);

    // Call the validation functions
    validate_pixel_index(h_pixel_contribs.data(), h_pixel_index.data(), cuda_output_image_width, cuda_output_image_height);
    validate_sorted_pairs(h_particles.data(), cuda_particles_count, h_pixel_index.data(), cuda_output_image_width, cuda_output_image_height, h_pixel_contrib_colours.data(), h_pixel_contrib_depth.data());    
#endif
}


__global__ void cuda_stage3_kernel(unsigned char* output_image_data, unsigned char* pixel_contrib_colours, unsigned int* pixel_index, int output_image_width, int output_image_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * output_image_width + x;
    int channels = 3;

    if (x < output_image_width && y < output_image_height) {
        for (unsigned int j = 0; j < pixel_index[index + 1] - pixel_index[index]; ++j) {
            const float opacity = (float)pixel_contrib_colours[(pixel_index[index] + j) * 4 + 3] / (float)255;
            output_image_data[(index * channels) + 0] = (unsigned char)((float)pixel_contrib_colours[(pixel_index[index] + j) * 4 + 0] * opacity + (float)output_image_data[(index * channels) + 0] * (1 - opacity));
            output_image_data[(index * channels) + 1] = (unsigned char)((float)pixel_contrib_colours[(pixel_index[index] + j) * 4 + 1] * opacity + (float)output_image_data[(index * channels) + 1] * (1 - opacity));
            output_image_data[(index * channels) + 2] = (unsigned char)((float)pixel_contrib_colours[(pixel_index[index] + j) * 4 + 2] * opacity + (float)output_image_data[(index * channels) + 2] * (1 - opacity));
        }
    }
}


void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_blend(pixel_index, pixel_contrib_colours, return_output_image);
    dim3 blockDim(16, 16);
    dim3 gridDim((cuda_output_image_width + blockDim.x - 1) / blockDim.x, (cuda_output_image_height + blockDim.y - 1) / blockDim.y);
    int channels = 3;
    cudaMemset(d_output_image_data, 255, cuda_output_image_width * cuda_output_image_height * channels * sizeof(unsigned char));
    // Invoke CUDA kernel
    cuda_stage3_kernel << <gridDim, blockDim >> > (d_output_image_data, d_pixel_contrib_colours, d_pixel_index,
        cuda_output_image_width, cuda_output_image_height);


#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    unsigned int* h_pixel_index = new unsigned int[(cuda_output_image_width * cuda_output_image_height) + 1];
    cudaMemcpy(h_pixel_index, d_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned char* h_pixel_contrib_colours = new unsigned char[(h_pixel_index[cuda_output_image_width * cuda_output_image_height]) * 4];
    cudaMemcpy(h_pixel_contrib_colours, d_pixel_contrib_colours, (h_pixel_index[cuda_output_image_width * cuda_output_image_height]) * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
   
    CImage* h_output_image = new CImage();
    unsigned char* h_output_image_data = new unsigned char[cuda_output_image_width * cuda_output_image_height * channels];
    cudaMemcpy(h_output_image_data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Set the fields of the CImage structure
    h_output_image->data = h_output_image_data;
    h_output_image->width = cuda_output_image_width;
    h_output_image->height = cuda_output_image_height;
    h_output_image->channels = channels;

    validate_blend(h_pixel_index, h_pixel_contrib_colours, h_output_image);
#endif    
}
void cuda_end(CImage *output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    const int CHANNELS = 3;
    output_image->width = cuda_output_image_width;
    output_image->height = cuda_output_image_height;
    output_image->channels = CHANNELS;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    CUDA_CALL(cudaFree(d_pixel_contrib_depth));
    CUDA_CALL(cudaFree(d_pixel_contrib_colours));
    CUDA_CALL(cudaFree(d_output_image_data));
    CUDA_CALL(cudaFree(d_pixel_index));
    CUDA_CALL(cudaFree(d_pixel_contribs));
    CUDA_CALL(cudaFree(d_particles));
    // Return ptrs to nullptr
    d_pixel_contrib_depth = 0;
    d_pixel_contrib_colours = 0;
    d_output_image_data = 0;
    d_pixel_index = 0;
    d_pixel_contribs = 0;
    d_particles = 0;
}




