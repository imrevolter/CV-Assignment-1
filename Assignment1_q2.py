import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Q2 - Image Smoothing
# fine_img has lots of texture/detail, smooth_img has large flat regions
fine_img = np.array(Image.open("cameraman.png").convert("L"), dtype=np.float64)
smooth_img = np.array(Image.open("moon.png").convert("L"), dtype=np.float64)
print("fine_img:", fine_img.shape, " smooth_img:", smooth_img.shape)

if not os.path.exists("Q2_outputs"):
    os.mkdir("Q2_outputs")

np.random.seed(0)


def add_salt_pepper(img, amount=0.05):
    out = img.copy()
    n = out.size
    n_salt = int(n * amount / 2)
    n_pepper = int(n * amount / 2)

    coords = [np.random.randint(0, d, n_salt) for d in out.shape]
    out[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, d, n_pepper) for d in out.shape]
    out[coords[0], coords[1]] = 0
    return out


def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255)


fine_noisy = add_salt_pepper(fine_img, amount=0.05)
smooth_noisy = add_gaussian_noise(smooth_img, sigma=25)

Image.fromarray(fine_noisy.astype(np.uint8)).save("Q2_outputs/fine_noisy.png")
Image.fromarray(smooth_noisy.astype(np.uint8)).save("Q2_outputs/smooth_noisy.png")


def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return out


def box_filter(img, k):
    kernel = np.ones((k, k)) / (k * k)
    return convolve2d(img, kernel)


def weighted_avg_filter(img, k):
    # weight falls off with distance from the center pixel, unlike a flat box
    c = (k - 1) / 2
    kernel = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            dist = np.sqrt((i - c) ** 2 + (j - c) ** 2)
            kernel[i, j] = 1.0 / (1.0 + dist)
    kernel /= kernel.sum()
    return convolve2d(img, kernel)


def gaussian_filter(img, k, sigma=None):
    if sigma is None:
        sigma = k / 3.0
    c = (k - 1) / 2
    kernel = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            kernel[i, j] = np.exp(-((i - c) ** 2 + (j - c) ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return convolve2d(img, kernel)


def median_filter(img, k):
    p = k // 2
    padded = np.pad(img, p, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    return np.median(windows, axis=(-2, -1))


def calc_mse(a, b):
    return np.mean((a - b) ** 2)


def calc_psnr(a, b):
    m = calc_mse(a, b)
    if m == 0:
        return 100.0
    return 10 * np.log10((255.0 ** 2) / m)


filters = {
    "Box": box_filter,
    "Weighted": weighted_avg_filter,
    "Gaussian": gaussian_filter,
    "Median": median_filter,
}
kernel_sizes = [3, 5, 7]

images = {
    "fine (salt&pepper)": (fine_img, fine_noisy),
    "smooth (gaussian)": (smooth_img, smooth_noisy),
}

table = []
filtered_cache = {}
for img_label, (clean, noisy) in images.items():
    for fname, ffunc in filters.items():
        for k in kernel_sizes:
            out = ffunc(noisy, k)
            filtered_cache[(img_label, fname, k)] = out
            mse_val = calc_mse(clean, out)
            psnr_val = calc_psnr(clean, out)
            table.append((img_label, fname, k, mse_val, psnr_val))

print("\n%-20s %-10s %-6s %-10s %-10s" % ("Image", "Filter", "K", "MSE", "PSNR"))
for row in table:
    print("%-20s %-10s %-6d %-10.3f %-10.3f" % row)

# best filter+kernel for each noise type, based on PSNR
for img_label in images:
    rows = [r for r in table if r[0] == img_label]
    best = max(rows, key=lambda r: r[4])
    print("Best for %-20s -> %-10s k=%d  PSNR=%.3f" % (img_label, best[1], best[2], best[4]))


# ---------------- abs-difference maps D = |noisy - filtered| ----------------
for img_label, (clean, noisy) in images.items():
    fig, axes = plt.subplots(len(filters), len(kernel_sizes), figsize=(11, 12))
    fig.suptitle("D(x,y) = |noisy - filtered|  --  " + img_label)
    for r, fname in enumerate(filters):
        for c, k in enumerate(kernel_sizes):
            out = filtered_cache[(img_label, fname, k)]
            diff = np.abs(noisy - out)
            axes[r, c].imshow(diff, cmap="hot")
            axes[r, c].set_title("%s k=%d" % (fname, k))
            axes[r, c].axis("off")
    plt.tight_layout()
    safe_name = img_label.split(" ")[0]
    plt.savefig("Q2_outputs/diff_" + safe_name + ".png")
    plt.close()


# ---------------- visual comparison grid ----------------
for img_label, (clean, noisy) in images.items():
    fig, axes = plt.subplots(len(filters), len(kernel_sizes) + 1, figsize=(14, 12))
    fig.suptitle("Filtered outputs -- " + img_label)
    for r, fname in enumerate(filters):
        axes[r, 0].imshow(noisy, cmap="gray")
        axes[r, 0].set_title("noisy")
        axes[r, 0].axis("off")
        for c, k in enumerate(kernel_sizes):
            out = filtered_cache[(img_label, fname, k)]
            axes[r, c + 1].imshow(out, cmap="gray")
            axes[r, c + 1].set_title("%s k=%d" % (fname, k))
            axes[r, c + 1].axis("off")
    plt.tight_layout()
    safe_name = img_label.split(" ")[0]
    plt.savefig("Q2_outputs/filtered_" + safe_name + ".png")
    plt.close()


# ---------------- mixed noise image (left = salt&pepper, right = gaussian) ----------------
h, w = fine_img.shape
mixed_clean = fine_img.copy()
mixed_noisy = mixed_clean.copy()
mixed_noisy[:, : w // 2] = add_salt_pepper(mixed_clean[:, : w // 2], amount=0.05)
mixed_noisy[:, w // 2 :] = add_gaussian_noise(mixed_clean[:, w // 2 :], sigma=25)
Image.fromarray(mixed_noisy.astype(np.uint8)).save("Q2_outputs/mixed_noisy.png")

k_mix = 5
print("\nMixed image, single global filter (k=%d) applied to the whole thing:" % k_mix)
mixed_results = {}
for fname, ffunc in filters.items():
    out = ffunc(mixed_noisy, k_mix)
    mixed_results[fname] = out
    mse_val = calc_mse(mixed_clean, out)
    psnr_val = calc_psnr(mixed_clean, out)
    print("%-10s MSE=%-10.3f PSNR=%-10.3f" % (fname, mse_val, psnr_val))

# region-wise: median on the salt&pepper half, gaussian on the gaussian half
region_wise = mixed_noisy.copy()
region_wise[:, : w // 2] = median_filter(mixed_noisy[:, : w // 2], k_mix)
region_wise[:, w // 2 :] = gaussian_filter(mixed_noisy[:, w // 2 :], k_mix)
Image.fromarray(np.clip(region_wise, 0, 255).astype(np.uint8)).save("Q2_outputs/region_wise.png")

rw_mse = calc_mse(mixed_clean, region_wise)
rw_psnr = calc_psnr(mixed_clean, region_wise)
print("\nRegion-wise (median left half + gaussian right half): MSE=%.3f PSNR=%.3f" % (rw_mse, rw_psnr))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(mixed_clean, cmap="gray")
axes[0].set_title("Clean")
axes[1].imshow(mixed_noisy, cmap="gray")
axes[1].set_title("Mixed noise (S&P left / Gaussian right)")
axes[2].imshow(region_wise, cmap="gray")
axes[2].set_title("Region-wise filtered")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("Q2_outputs/region_wise_comparison.png")
plt.close()

# kernel size 3,5,7 chosen as small/medium/large, gaussian sigma = k/3 so the
# blur radius scales with kernel size
# - box filter is fast but blurs edges the same way in every direction and
#   does nothing special about outliers, so it doesn't clean salt & pepper
#   noise very well
# - weighted-average is a softer version of box, a bit better at keeping
#   edges but still gets dragged around by extreme salt & pepper values
# - gaussian filter is good for gaussian noise since it's a smooth weighted
#   average, but again doesn't handle salt & pepper impulses well
# - median filter is the clear winner for salt & pepper since it just
#   throws out extreme outlier values instead of averaging them in, but it
#   is a bit worse than gaussian for pure gaussian noise
# - increasing kernel size removes more noise but blurs more detail/edges
#   in every filter, so there's a trade-off, k=5 is usually a decent middle
#   ground for these images
# - on the mixed image no single filter is best everywhere, which is why
#   the region-wise version above beats every single global filter
print("\nSee comments at the bottom for filter comparison / kernel size discussion.")
print("All outputs saved in the Q2_outputs folder.")
