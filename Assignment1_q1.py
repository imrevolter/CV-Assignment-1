import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("cameraman.png").convert("L")
orig = np.array(img, dtype=np.float64)
print("Original image shape:", orig.shape)

if not os.path.exists("Q1_outputs"):
    os.mkdir("Q1_outputs")


def downsample(image, factor):
    return image[::factor, ::factor]


img_256 = downsample(orig, 2)
img_128 = downsample(orig, 4)
print("Downsampled sizes:", img_256.shape, img_128.shape)

Image.fromarray(img_256.astype(np.uint8)).save("Q1_outputs/down_256.png")
Image.fromarray(img_128.astype(np.uint8)).save("Q1_outputs/down_128.png")


def nn_resize(small_img, new_h, new_w):
    old_h, old_w = small_img.shape
    out = np.zeros((new_h, new_w))
    for i in range(new_h):
        src_i = int(i * old_h / new_h)
        if src_i >= old_h:
            src_i = old_h - 1
        for j in range(new_w):
            src_j = int(j * old_w / new_w)
            if src_j >= old_w:
                src_j = old_w - 1
            out[i, j] = small_img[src_i, src_j]
    return out


def resize_builtin(small_img, size, mode):
    pil_img = Image.fromarray(small_img.astype(np.uint8))
    resized = pil_img.resize(size, mode)
    return np.array(resized, dtype=np.float64)


nn_256 = nn_resize(img_256, 512, 512)
bilinear_256 = resize_builtin(img_256, (512, 512), Image.Resampling.BILINEAR)
bicubic_256 = resize_builtin(img_256, (512, 512), Image.Resampling.BICUBIC)

nn_128 = nn_resize(img_128, 512, 512)
bilinear_128 = resize_builtin(img_128, (512, 512), Image.Resampling.BILINEAR)
bicubic_128 = resize_builtin(img_128, (512, 512), Image.Resampling.BICUBIC)


def calc_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def calc_psnr(img1, img2):
    mse_val = calc_mse(img1, img2)
    if mse_val == 0:
        return 100.0
    return 10 * np.log10((255.0 ** 2) / mse_val)


def edge_map(image):
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[:, :-1] = image[:, 1:] - image[:, :-1]
    gy[:-1, :] = image[1:, :] - image[:-1, :]
    return np.sqrt(gx ** 2 + gy ** 2)


orig_edges = edge_map(orig)

results = [
    ("256_NN", nn_256),
    ("256_Bilinear", bilinear_256),
    ("256_Bicubic", bicubic_256),
    ("128_NN", nn_128),
    ("128_Bilinear", bilinear_128),
    ("128_Bicubic", bicubic_128),
]

table = []
for name, rec in results:
    m = calc_mse(orig, rec)
    p = calc_psnr(orig, rec)
    e = calc_mse(orig_edges, edge_map(rec))
    table.append((name, m, p, e))

print("\n%-15s %-10s %-10s %-10s" % ("Method", "MSE", "PSNR", "EdgeErr"))
for name, m, p, e in table:
    print("%-15s %-10.3f %-10.3f %-10.3f" % (name, m, p, e))

best = max(table, key=lambda row: row[2])
print("\nBest performing method based on PSNR:", best[0], "-> PSNR = %.3f dB" % best[2])


for name, rec in results:
    abs_err = np.abs(orig - rec)
    sq_err = (orig - rec) ** 2

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(abs_err, cmap="hot")
    plt.title(name + " - Absolute Error")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(sq_err, cmap="hot")
    plt.title(name + " - Squared Error")
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("Q1_outputs/error_" + name + ".png")
    plt.close()


plt.figure(figsize=(16, 8))

plt.subplot(2, 4, 1)
plt.imshow(orig, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(nn_256, cmap="gray")
plt.title("NN (256->512)")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(bilinear_256, cmap="gray")
plt.title("Bilinear (256->512)")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(bicubic_256, cmap="gray")
plt.title("Bicubic (256->512)")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(orig, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(nn_128, cmap="gray")
plt.title("NN (128->512)")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(bilinear_128, cmap="gray")
plt.title("Bilinear (128->512)")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(bicubic_128, cmap="gray")
plt.title("Bicubic (128->512)")
plt.axis("off")

plt.tight_layout()
plt.savefig("Q1_outputs/comparison.png")
plt.close()

for name, rec in results:
    Image.fromarray(np.clip(rec, 0, 255).astype(np.uint8)).save("Q1_outputs/recon_" + name + ".png")

print("\nDiscussion:")
print("Going from 512 -> 128 loses a lot more detail than 512 -> 256, so no matter")
print("which interpolation method is used, MSE goes up and PSNR goes down when")
print("starting from the smaller image. Edge error also increases a lot more, since")
print("fine edges just aren't there anymore after that much downsampling.")
print("Nearest neighbour gives blocky/jagged edges, it's the simplest but usually the")
print("worst in terms of MSE/PSNR. Bilinear blurs things out a bit more, so edges are")
print("softer but overall error is usually a bit better than NN. Bicubic generally")
print("gives the best PSNR and keeps edges looking closer to the original, since it")
print("uses a bigger neighbourhood of pixels.")

print("\nAll outputs saved in the Q1_outputs folder.")
