import os
import time
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

if not os.path.exists("Q3_outputs"):
    os.mkdir("Q3_outputs")


def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return out


def gaussian_kernel(k, sigma):
    c = (k - 1) / 2
    kernel = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            kernel[i, j] = np.exp(-((i - c) ** 2 + (j - c) ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur(img, k, sigma):
    return convolve2d(img, gaussian_kernel(k, sigma))


def resize_float(img, shape):
    pil = Image.fromarray(img.astype(np.float32), mode="F")
    resized = pil.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.float64)


def draw_face(happy, size=320):
    img = Image.new("L", (size, size), color=210)
    d = ImageDraw.Draw(img)
    d.ellipse([40, 40, size - 40, size - 40], outline=0, fill=225, width=4)
    d.ellipse([size * 0.30 - 20, size * 0.40 - 15, size * 0.30 + 20, size * 0.40 + 15], fill=0)
    d.ellipse([size * 0.70 - 20, size * 0.40 - 15, size * 0.70 + 20, size * 0.40 + 15], fill=0)
    mouth_box = [size * 0.28, size * 0.55, size * 0.72, size * 0.85]
    if happy:
        d.line([size * 0.20, size * 0.30, size * 0.40, size * 0.20], fill=0, width=14)
        d.line([size * 0.60, size * 0.20, size * 0.80, size * 0.30], fill=0, width=14)
        d.arc(mouth_box, start=20, end=160, fill=0, width=18)
    else:
        d.line([size * 0.20, size * 0.20, size * 0.40, size * 0.30], fill=0, width=14)
        d.line([size * 0.60, size * 0.30, size * 0.80, size * 0.20], fill=0, width=14)
        d.arc(mouth_box, start=200, end=340, fill=0, width=18)
    return np.array(img, dtype=np.float64)


face_a = draw_face(happy=True)
face_b_true = draw_face(happy=False)
face_b_shifted = np.roll(np.roll(face_b_true, 8, axis=0), -6, axis=1)

Image.fromarray(face_a.astype(np.uint8)).save("Q3_outputs/face_a.png")
Image.fromarray(face_b_shifted.astype(np.uint8)).save("Q3_outputs/face_b_shifted.png")


def align_translation(ref, moving, max_shift=15):
    best_score = None
    best_shift = (0, 0)
    h, w = ref.shape
    c = max_shift
    ref_crop = ref[c:h - c, c:w - c]
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            cand = moving[c + dy:h - c + dy, c + dx:w - c + dx]
            score = np.sum((ref_crop - cand) ** 2)
            if best_score is None or score < best_score:
                best_score = score
                best_shift = (dy, dx)
    dy, dx = best_shift
    aligned = np.roll(np.roll(moving, -dy, axis=0), -dx, axis=1)
    return aligned, best_shift


face_b_aligned, shift_found = align_translation(face_a, face_b_shifted)
print("Pair 1 (faces): recovered shift =", shift_found, " (true shift was (8, -6))")
Image.fromarray(face_b_aligned.astype(np.uint8)).save("Q3_outputs/face_b_aligned.png")

cam = np.array(Image.open("cameraman.png").convert("L"), dtype=np.float64)
moon = np.array(Image.open("moon.png").convert("L"), dtype=np.float64)
print("Pair 2 (cameraman/moon): already same size and framing, no shift needed.")

pairs = {
    "faces": (face_a, face_b_aligned),
    "cam_moon": (cam, moon),
}


def hybrid_spatial(img1, img2, k, sigma, alpha=1.0, beta=1.0):
    low1 = gaussian_blur(img1, k, sigma)
    low2 = gaussian_blur(img2, k, sigma)
    high2 = img2 - low2
    hybrid = alpha * low1 + beta * high2
    return hybrid, low1, high2


for label, (img1, img2) in pairs.items():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, sigma in zip(axes, [3, 6, 10]):
        k = int(sigma * 4) | 1
        hyb, _, _ = hybrid_spatial(img1, img2, k, sigma)
        ax.imshow(np.clip(hyb, 0, 255), cmap="gray")
        ax.set_title("sigma=%d, k=%d" % (sigma, k))
        ax.axis("off")
    fig.suptitle("Hybrid image, different LPF/HPF cutoffs - " + label)
    plt.tight_layout()
    plt.savefig("Q3_outputs/hybrid_sigmas_" + label + ".png")
    plt.close()

BEST_K, BEST_SIGMA = 25, 6
print("\nChosen params: k=%d, sigma=%d" % (BEST_K, BEST_SIGMA))
print("Reasoning: sigma=6 looked like the best balance for both pairs (sigma=3")
print("barely changes with distance, sigma=10 loses too much of the low-freq image),")
print("so that's the one used from here on.")

hybrid_cache = {}
for label, (img1, img2) in pairs.items():
    hyb, low1, high2 = hybrid_spatial(img1, img2, BEST_K, BEST_SIGMA)
    hybrid_cache[label] = hyb

    small = Image.fromarray(np.clip(hyb, 0, 255).astype(np.uint8))
    small = small.resize((max(1, small.width // 18), max(1, small.height // 18)), Image.Resampling.BILINEAR)
    small = small.resize((hyb.shape[1], hyb.shape[0]), Image.Resampling.BILINEAR)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title("Image 1 (low-freq source)")
    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title("Image 2 (high-freq source)")
    axes[2].imshow(np.clip(hyb, 0, 255), cmap="gray")
    axes[2].set_title("Hybrid - close up")
    axes[3].imshow(small, cmap="gray")
    axes[3].set_title("Hybrid - blurred/far away")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("Q3_outputs/hybrid_final_" + label + ".png")
    plt.close()
    Image.fromarray(np.clip(hyb, 0, 255).astype(np.uint8)).save("Q3_outputs/hybrid_" + label + ".png")


def fft_lowpass(img, sigma):
    h, w = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    sigma_fy = h / (2 * np.pi * sigma)
    sigma_fx = w / (2 * np.pi * sigma)
    mask = np.exp(-((x - cx) ** 2 / (2 * sigma_fx ** 2) + (y - cy) ** 2 / (2 * sigma_fy ** 2)))
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))


def hybrid_freq(img1, img2, sigma):
    low1 = fft_lowpass(img1, sigma)
    low2 = fft_lowpass(img2, sigma)
    high2 = img2 - low2
    return low1 + high2


for label, (img1, img2) in pairs.items():
    t0 = time.time()
    hyb_spatial, _, _ = hybrid_spatial(img1, img2, BEST_K, BEST_SIGMA)
    t_spatial = time.time() - t0

    t0 = time.time()
    hyb_freq = hybrid_freq(img1, img2, BEST_SIGMA)
    t_freq = time.time() - t0

    print("%-10s spatial: %.4fs   frequency: %.4fs" % (label, t_spatial, t_freq))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(np.clip(hyb_spatial, 0, 255), cmap="gray")
    axes[0].set_title("Spatial domain (%.3fs)" % t_spatial)
    axes[1].imshow(np.clip(hyb_freq, 0, 255), cmap="gray")
    axes[1].set_title("Frequency domain / FFT (%.3fs)" % t_freq)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("Q3_outputs/spatial_vs_freq_" + label + ".png")
    plt.close()


def build_gaussian_pyramid(img, levels, blur_fn):
    pyr = [img]
    cur = img
    for _ in range(levels - 1):
        cur = blur_fn(cur)[::2, ::2]
        pyr.append(cur)
    return pyr


def build_laplacian_pyramid(gauss_pyr):
    lap_pyr = []
    for i in range(len(gauss_pyr) - 1):
        up = resize_float(gauss_pyr[i + 1], gauss_pyr[i].shape)
        lap_pyr.append(gauss_pyr[i] - up)
    lap_pyr.append(gauss_pyr[-1])
    return lap_pyr


def reconstruct(lap_pyr):
    img = lap_pyr[-1]
    for lvl in reversed(lap_pyr[:-1]):
        img = resize_float(img, lvl.shape) + lvl
    return img


def blend_pyramids(lapA, lapB, mask_pyr):
    return [la * m + lb * (1 - m) for la, lb, m in zip(lapA, lapB, mask_pyr)]


def bilateral_filter(img, k=5, sigma_spatial=2.0, sigma_range=25.0):
    pad = k // 2
    padded = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img)
    weight_sum = np.zeros_like(img)
    for di in range(-pad, pad + 1):
        for dj in range(-pad, pad + 1):
            shifted = padded[pad + di:pad + di + img.shape[0], pad + dj:pad + dj + img.shape[1]]
            spatial_w = np.exp(-(di ** 2 + dj ** 2) / (2 * sigma_spatial ** 2))
            range_w = np.exp(-((shifted - img) ** 2) / (2 * sigma_range ** 2))
            w = spatial_w * range_w
            out += shifted * w
            weight_sum += w
    return out / weight_sum


LEVELS = 4
plain_blur = lambda im: gaussian_blur(im, 5, 1.2)
bilateral_blur = lambda im: bilateral_filter(im, 5, 2.0, 25.0)

for label, (img1, img2) in pairs.items():
    h, w = img1.shape
    mask = np.zeros((h, w))
    mask[:, : w // 2] = 1.0
    mask = gaussian_blur(mask, 31, 10)

    gpA = build_gaussian_pyramid(img1, LEVELS, plain_blur)
    gpB = build_gaussian_pyramid(img2, LEVELS, plain_blur)
    gpM = build_gaussian_pyramid(mask, LEVELS, plain_blur)
    lapA = build_laplacian_pyramid(gpA)
    lapB = build_laplacian_pyramid(gpB)
    blended_lap = blend_pyramids(lapA, lapB, gpM)
    blended_gaussian = np.clip(reconstruct(blended_lap), 0, 255)

    check = reconstruct(lapA)
    recon_error = np.mean((check - img1) ** 2)
    print("%-10s laplacian-pyramid reconstruction MSE (sanity check): %.5f" % (label, recon_error))

    gpA_bi = build_gaussian_pyramid(img1, LEVELS, bilateral_blur)
    gpB_bi = build_gaussian_pyramid(img2, LEVELS, bilateral_blur)
    lapA_bi = build_laplacian_pyramid(gpA_bi)
    lapB_bi = build_laplacian_pyramid(gpB_bi)
    blended_lap_bi = blend_pyramids(lapA_bi, lapB_bi, gpM)
    blended_bilateral = np.clip(reconstruct(blended_lap_bi), 0, 255)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title("Image 1")
    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title("Image 2")
    axes[2].imshow(blended_gaussian, cmap="gray")
    axes[2].set_title("Pyramid blend (gaussian)")
    axes[3].imshow(blended_bilateral, cmap="gray")
    axes[3].set_title("Pyramid blend (bilateral)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Multi-resolution blending - " + label)
    plt.tight_layout()
    plt.savefig("Q3_outputs/pyramid_blend_" + label + ".png")
    plt.close()

    Image.fromarray(blended_gaussian.astype(np.uint8)).save("Q3_outputs/blend_gaussian_" + label + ".png")
    Image.fromarray(blended_bilateral.astype(np.uint8)).save("Q3_outputs/blend_bilateral_" + label + ".png")


print("\nDiscussion:")
print("Alignment matters a lot: the face pair only reads as one coherent face")
print("because the eyes/mouth line up after correcting the (8,-6) shift. A")
print("misaligned high-frequency layer just looks like a ghosting artefact on top")
print("of the low-frequency image instead of a clean hybrid.")
print("The cutoff (sigma) controls how the two images trade off: too small a sigma")
print("and the low-frequency image still has sharp detail fighting with the")
print("high-frequency layer; too large and the low-frequency image turns into a")
print("featureless blob and the high-frequency layer dominates at every viewing")
print("distance.")
print("Spatial and frequency domain filtering give visually near-identical results")
print("once the cutoffs are matched, which makes sense since gaussian convolution")
print("is just multiplication by a gaussian in the frequency domain. Frequency")
print("domain is faster for large kernels since FFT-based filtering doesn't scale")
print("with kernel size the way spatial convolution does.")
print("Plain gaussian-pyramid blending gives a smooth transition but can blur edges")
print("that cross the seam. Building the pyramid with a bilateral filter instead")
print("keeps edges sharper going into the blend since it avoids smoothing across")
print("strong intensity boundaries, at the cost of being slower to compute.")

print("\nAll outputs saved in the Q3_outputs folder.")
