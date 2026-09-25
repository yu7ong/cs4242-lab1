# CS4243 Lab 1 reference implementation report

This compact report records how every `YOUR CODE HERE` section is exercised. This is where you elaborate and explain your implementation and results you observed. Try to add visual/graphs and grounded reasoning to highlight your understanding.

## Gabor Implementation
## Edge Implementation
### Functions Explanation 
- bilinear_sample(image, y, x)
    - Find the pixel value at a coordinate that fall between pixels based on the four nearest pixels, through weighing how close the point is to each of the four pixels. 
- nms_interpolated(magnitude, direction)
    - Makes the edges thinner. So for every pixel, check the pixel just ahead and just behind it using the direction the edge is pointing across. bilinear_sample is used to compute these pixels. If pixel if stronger than those two neighbours, we keep it, otherwise we set it to 0. 
- adaptive_thresholds(nms, config)
    - Decides what count as strong edge and what counts as week edge. Instead of picking a fixed number, looks at the actual pixel values in the image and calculates good cutoff points, using either percentile or median-based method
- hysteresis(strong, weak, connectivity)
    - Connects broken edge lines. It starts at every strong edge pixel, then picks the neighbouring weak pixels, by checking either 4 or 8 neighbouring directions. If a weak pixel is connected to a strong one, keep it. Any weak pixel that isn't connected to a strong pixel gets thrown away.
## Features and Representations

## Normality Model

## Results and Analysis Task A-C
### Question 1
Frequency determines the how tightly spaced the sine wave inside the kernel, higher frequency is used to detect finer features. The orientation rotates the direction of the sin curve. A response map having an orientation matching the direction of the texture will give higher response. Phase shifts where along the stripe the peak of the sine wave sits. Different phases can detect the same stripe at slightly different positions. One phase might detect the middle of the stripe, while another might detect its edge. 

Case where large pooling increases stability: Carpet texture, since no two adjacent tufts will be identical even though the overall texture will be uniform. A large pooling window averages the noise away, giving a smooth and stable energy value. 

Case when large pooling erases small features: A small scratch or defect on wood sample. If the pooling window is much larger than the scratch, then the scratch's energy will be diluted with surrounding pixels. 

### Question 2
Largest values are the color channels (channels 0-2), where RGB is stored on 0 - 1 scale. Smallest values includes gabor channels (channels 11-14), such as those in the higher frequencies which might be due to the relatively smooth and low-frequency texture of wood and the gradient energy (channel 15), which is computed as a pooled energy. Scaling is required as a linear classifier will penalize cofficient magnitude uniformly across all features. Scaling helps to prevent cases where classifier underuses informative but small scale channels because of their units not their actual predictive value. 

### Question 3

## Results and Analysis Task D

## Results and Analysis Custom Photos

## AI-use disclosure table
[AI-CODE] [AI-DESIGN] [HUMAN-CHECK]
