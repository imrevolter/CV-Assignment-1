import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Q4 - Edge Detection and Analysis
# coins_img has clean, well defined object boundaries (coins on a plain background)
# fine_img has a lot of fine detail/texture (grass, tripod legs etc)
coins_img = np.array(Image.open("coins.png").convert("L"), dtype=np.float64)
fine_img = np.array(Image.open("cameraman.png").convert("L"), dtype=np.float64)
print("coins_img:", coins_img.shape, " fine_img:", fine_img.shape)

if not os.path.exists("Q4_outputs"):
    os.mkdir("Q4_outputs")


def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return out


# ---------------- first order derivatives (Sobel) ----------------
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def first_order_grad(img):
    gx = convolve2d(img, SOBEL_X)
    gy = convolve2d(img, SOBEL_Y)
    return gx, gy


def grad_mag_dir(gx, gy):
    mag = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.degrees(np.arctan2(gy, gx))
    return mag, direction


# ---------------- second order derivative (Laplacian) ----------------
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def laplacian_edges(img, thresh=4.0):
    lap = convolve2d(img, LAPLACIAN)
    edges = np.zeros_like(lap)
    # zero crossing: neighbouring pixels have opposite sign and a big enough jump
    h, w = lap.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch = lap[i - 1:i + 2, j - 1:j + 2]
            if patch.max() - patch.min() > thresh:
                if (lap[i, j - 1] * lap[i, j + 1] < 0) or (lap[i - 1, j] * lap[i + 1, j] < 0):
                    edges[i, j] = 255
    return lap, edges


# ---------------- gaussian blur (reused for LoG and Canny) ----------------
def gaussian_kernel(k, sigma):
    c = (k - 1) / 2
    kernel = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            kernel[i, j] = np.exp(-((i - c) ** 2 + (j - c) ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur(img, k=5, sigma=1.4):
    return convolve2d(img, gaussian_kernel(k, sigma))


def log_edges(img, k=15, sigma=2.5, thresh=6.0):
    blurred = gaussian_blur(img, k, sigma)
    return laplacian_edges(blurred, thresh)


# ---------------- canny, done manually ----------------
def non_max_suppression(mag, direction):
    h, w = mag.shape
    out = np.zeros_like(mag)
    ang = direction % 180
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = ang[i, j]
            if a < 22.5 or a >= 157.5:
                n1, n2 = mag[i, j - 1], mag[i, j + 1]
            elif a < 67.5:
                n1, n2 = mag[i - 1, j + 1], mag[i + 1, j - 1]
            elif a < 112.5:
                n1, n2 = mag[i - 1, j], mag[i + 1, j]
            else:
                n1, n2 = mag[i - 1, j - 1], mag[i + 1, j + 1]
            if mag[i, j] >= n1 and mag[i, j] >= n2:
                out[i, j] = mag[i, j]
    return out


def hysteresis(thin, low, high):
    strong = thin >= high
    weak = (thin >= low) & (thin < high)
    out = strong.copy()
    stack = list(zip(*np.where(strong)))
    h, w = thin.shape
    while stack:
        i, j = stack.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and weak[ni, nj] and not out[ni, nj]:
                    out[ni, nj] = True
                    stack.append((ni, nj))
    return (out * 255).astype(np.uint8)


def canny_edges(img, low_ratio=0.08, high_ratio=0.2, k=5, sigma=1.4):
    blurred = gaussian_blur(img, k, sigma)
    gx, gy = first_order_grad(blurred)
    mag, direction = grad_mag_dir(gx, gy)
    thin = non_max_suppression(mag, direction)
    high = thin.max() * high_ratio
    low = thin.max() * low_ratio
    return hysteresis(thin, low, high)


def calc_mse(a, b):
    return np.mean((a - b) ** 2)


# ---------------- run everything on both images ----------------
images = {"coins (boundaries)": coins_img, "cameraman (fine detail)": fine_img}

for label, img in images.items():
    tag = label.split(" ")[0]

    gx, gy = first_order_grad(img)
    mag, direction = grad_mag_dir(gx, gy)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(gx, cmap="gray")
    axes[1].set_title("Gx")
    axes[2].imshow(gy, cmap="gray")
    axes[2].set_title("Gy")
    im = axes[3].imshow(mag, cmap="gray")
    axes[3].set_title("Gradient magnitude")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("Q4_outputs/" + tag + "_first_order.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(direction, cmap="hsv")
    ax.set_title("Gradient direction (deg) - " + label)
    ax.axis("off")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("Q4_outputs/" + tag + "_direction.png")
    plt.close()

    # three thresholds on gradient magnitude, based on percentiles
    thresholds = [np.percentile(mag, p) for p in (70, 85, 95)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, t in zip(axes, thresholds):
        binary = (mag > t).astype(np.uint8) * 255
        ax.imshow(binary, cmap="gray")
        ax.set_title("threshold=%.1f" % t)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("Q4_outputs/" + tag + "_thresholds.png")
    plt.close()

    lap, lap_edges = laplacian_edges(img)
    log_lap, log_edge_map = log_edges(img)
    canny_map = canny_edges(img)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow((mag > thresholds[1]).astype(np.uint8), cmap="gray")
    axes[0].set_title("First-order (Sobel)")
    axes[1].imshow(lap_edges, cmap="gray")
    axes[1].set_title("Second-order (Laplacian)")
    axes[2].imshow(log_edge_map, cmap="gray")
    axes[2].set_title("LoG")
    axes[3].imshow(canny_map, cmap="gray")
    axes[3].set_title("Canny")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Edge detector comparison - " + label)
    plt.tight_layout()
    plt.savefig("Q4_outputs/" + tag + "_comparison.png")
    plt.close()

    # rough numbers to back up the discussion: how many edge pixels each
    # method fires on, as a proxy for edge thickness/density
    sobel_binary = (mag > thresholds[1]).astype(np.uint8)
    print("\n%s edge pixel counts:" % label)
    print("  Sobel (mid threshold): %d" % sobel_binary.sum())
    print("  Laplacian zero-cross : %d" % (lap_edges > 0).sum())
    print("  LoG                  : %d" % (log_edge_map > 0).sum())
    print("  Canny                : %d" % (canny_map > 0).sum())

# - lower thresholds on the gradient magnitude let more edges through, but a
#   lot of them are just noise/texture (unwanted edges). higher thresholds
#   clean that up but start dropping faint, real edges too (missing edges).
#   there's no single threshold that's perfect for both images.
# - first-order (Sobel) edges are thick and respond a lot to texture, so
#   they're noisy on the cameraman image (grass etc) but fine on coins.
# - the plain Laplacian is very sensitive to noise since it's a second
#   derivative, so it picks up a lot of spurious zero-crossings, especially
#   on the textured image.
# - LoG (gaussian blur before the laplacian) fixes a lot of that noise
#   sensitivity and gives thin, well-localised edges, at the cost of
#   rounding off very fine detail.
# - Canny gives the cleanest, thinnest, most continuous edges thanks to
#   non-max suppression + hysteresis, and is the best overall for the coins
#   image with clean boundaries. On the cameraman image it still does
#   reasonably well but can miss some very fine grass texture that a raw
#   Sobel edge map would still show.
# - overall: Canny is the best default choice for the coins image (clean
#   boundaries, needs continuity), while a lower-threshold Sobel or LoG is
#   more useful than Canny if the goal is to catch fine texture detail.
print("\nSee comments at the bottom for the edge-detector comparison/discussion.")
print("All outputs saved in the Q4_outputs folder.")
