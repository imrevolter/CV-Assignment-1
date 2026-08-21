# Computer Vision - Assignment 1

Four questions on image resizing/interpolation, smoothing, hybrid images, and edge detection.
Everything is done with just NumPy, PIL and matplotlib (no OpenCV/scipy) — for each question at
least one core operation (resize, filter, derivative) is implemented from scratch instead of
using a library function, as required by the assignment.

## Files

- `Assignment1_q1.py` - downsampling + nearest/bilinear/bicubic reconstruction, MSE/PSNR, error maps
- `Assignment1_q2.py` - box/weighted/gaussian/median smoothing on salt&pepper and gaussian noise
- `Assignment1_q3.py` - hybrid images (spatial + frequency domain), gaussian/laplacian pyramids, bilateral pyramid blending
- `Assignment1_q4.py` - first/second order derivatives, thresholded edge maps, LoG, Canny (all manual)

Sample images used: `cameraman.png`, `moon.png`, `coins.png` (standard grayscale test images).

Each script writes its plots/results into its own `Q<n>_outputs/` folder and prints a results
table + a short discussion to the console.

## Running

```
pip install numpy pillow matplotlib
python Assignment1_q1.py
python Assignment1_q2.py
python Assignment1_q3.py
python Assignment1_q4.py
```

Run from inside this folder, since the scripts load the sample images by relative path.

## Notes

- Q1: nearest-neighbour resize is implemented manually; bilinear/bicubic use PIL.
- Q2: all four filters (box, weighted-average, gaussian, median) are implemented manually via convolution/sliding windows.
- Q3: hybrid images follow H = alpha*LPF(I1) + beta*HPF(I2). Includes a spatial vs FFT-based comparison and a bilateral-filter variant of the Laplacian pyramid blend.
- Q4: Sobel gradients, Laplacian zero-crossing, LoG and Canny are all implemented from scratch (no cv2).
