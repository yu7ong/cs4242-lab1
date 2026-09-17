# CS4243 Lab 1 reference implementation report

This compact report records how every `YOUR CODE HERE` section is exercised. This is where you elaborate and explain your implementation and results you observed. Try to add visual/graphs and grounded reasoning to highlight your understanding.

## Gabor Implementation
# Question 1
Frequency determines the how tightly spaced the sine wave inside the kernel, higher frequency is used to detect finer features. The orientation rotates the direction of the sin curve. A response map having an orientation matching the direction of the texture will give higher response. Phase shifts where along the stripe the peak of the sine wave sits. Different phases can detect the same stripe at slightly different positions. One phase might detect the middle of the stripe, while another might detect its edge. 

Case where large pooling increases stability: Carpet texture, since no two adjacent tufts will be identical even though the overall texture will be uniform. A large pooling window averages the noise away, giving a smooth and stable energy value. 

Case when large pooling erases small features: A small scratch or defect on wood sample. If the pooling window is much larger than the scratch, then the scratch's energy will be diluted with surrounding pixels. 
## Edge Implementation

## Features and Representations

## Normality Model

## Results and Analysis Task A-C

## Results and Analysis Task D

## Results and Analysis Custom Photos

## AI-use disclosure table
[AI-CODE] [AI-DESIGN] [HUMAN-CHECK]
