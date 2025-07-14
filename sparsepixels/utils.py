import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def pool_pad_noise_inflate(img, pool_size, pool_type, target_size=None, noise_type=None, noise_level=0, inflate_factor=1):
    x = tf.convert_to_tensor(img, dtype=tf.float32)
    #x = tf.where(x<threshold, tf.zeros_like(x), x)

    k_h = k_w = pool_size

    if pool_type == 'max':
        pooled = tf.nn.max_pool2d(x, ksize=[1, k_h, k_w, 1], strides=[1, k_h, k_w, 1], padding='VALID')
    elif pool_type in ('avg'):
        pooled = tf.nn.avg_pool2d(x, ksize=[1, k_h, k_w, 1], strides=[1, k_h, k_w, 1], padding='VALID')

    old_h = tf.shape(x)[1]
    old_w = tf.shape(x)[2]
    new_h  = tf.shape(pooled)[1]
    new_w  = tf.shape(pooled)[2]

    if target_size is None:
        target_h, target_w = old_h, old_w
    else:
        target_h, target_w = target_size

    pad_y = target_h - new_h
    pad_x = target_w - new_w
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left

    padded = tf.pad(
        pooled,
        [[0, 0], # batch
         [pad_top, pad_bottom], # h
         [pad_left, pad_right], # w
         [0, 0]], # c
        mode='CONSTANT',
        constant_values=0
    )

    max_per_img = tf.reduce_max(padded, axis=[1,2,3], keepdims=True)
    padded = tf.math.divide_no_nan(padded, max_per_img)

    if inflate_factor != 1.0:
        h, w = tf.shape(padded)[1], tf.shape(padded)[2]
        
        center_h = tf.cast(h-1, tf.float32) / 2
        center_w = tf.cast(w-1, tf.float32) / 2
        
        coords = tf.where(padded>0)
        vals = tf.gather_nd(padded, coords)
        b_idx = coords[:,0]
        y0 = tf.cast(coords[:,1], tf.float32)
        x0 = tf.cast(coords[:,2], tf.float32)
        c_idx = coords[:,3]
        
        y_new = tf.round((y0 - center_h) * inflate_factor + center_h)
        x_new = tf.round((x0 - center_w) * inflate_factor + center_w)
        
        y_new = tf.clip_by_value(y_new, 0, tf.cast(h-1, tf.float32))
        x_new = tf.clip_by_value(x_new, 0, tf.cast(w-1, tf.float32))
        
        new_idx = tf.stack([tf.cast(b_idx, tf.int32),
                            tf.cast(y_new, tf.int32),
                            tf.cast(x_new, tf.int32),
                            tf.cast(c_idx, tf.int32)], axis=1)
        
        inflated = tf.zeros_like(padded)
        padded = tf.tensor_scatter_nd_update(inflated, new_idx, vals)
        
    if noise_level > 0:
        noise_mask = tf.cast(tf.equal(padded, 0), tf.float32)
        if noise_type=='uniform':
            noise = tf.random.uniform(tf.shape(padded), minval=0, maxval=noise_level, dtype=tf.float32)
        elif noise_type=='poisson':
            noise = tf.random.poisson(shape=tf.shape(padded), lam=noise_level, dtype=tf.float32)
            noise = tf.clip_by_value(noise, 0, noise_level * 3)
        padded = padded + noise * noise_mask

        #max_per_img = tf.reduce_max(padded, axis=[1,2,3], keepdims=True)
        #padded = tf.math.divide_no_nan(padded, max_per_img)

    return padded.numpy()

def plot_sparsemnist(x_original, x_modified1, x_modified2, x_modified3, n_example, threshold):
    img1 = x_original[n_example+1011]
    img2 = x_modified1[n_example+1011]
    img3 = x_modified2[n_example+1011]
    img4 = x_modified3[n_example+1011]
    img5 = np.where(img4 > threshold, img4, 0)

    print('no. of active pixels [0]: ' + str(np.count_nonzero(img1)) + ' / ' + str(img1.size) + ' = ' + str(np.count_nonzero(img1)/img1.size))
    print('no. of active pixels [1]: ' + str(np.count_nonzero(img2)) + ' / ' + str(img2.size) + ' = ' + str(np.count_nonzero(img2)/img2.size))
    print('no. of active pixels [2]: ' + str(np.count_nonzero(img3)) + ' / ' + str(img3.size) + ' = ' + str(np.count_nonzero(img3)/img3.size))
    print('no. of active pixels [3]: ' + str(np.count_nonzero(img4)) + ' / ' + str(img4.size) + ' = ' + str(np.count_nonzero(img4)/img4.size))
    print('no. of active pixels [4]: ' + str(np.count_nonzero(img5)) + ' / ' + str(img5.size) + ' = ' + str(np.count_nonzero(img5)/img5.size))

    fontsize=18
    fig, axes = plt.subplots(1, 5, figsize=(25,5))
    axes[0].imshow(img1)
    axes[0].set_title('[0] original', fontsize=fontsize)
    axes[1].imshow(img2)
    axes[1].set_title('[1] pooled+padded', fontsize=fontsize)
    axes[2].imshow(img3)
    axes[2].set_title('[2] inflated', fontsize=fontsize)
    axes[3].imshow(img4)
    axes[3].set_title('[3] noised', fontsize=fontsize)
    axes[4].imshow(img5)
    axes[4].set_title(f'[4] noised (threshold>{threshold})', fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def plot_jetimage(x, y, n_examples, threshold=0, normalized=False):
    classes = ['g','q','W','Z','t']
    class_indices = []
    for i in range(5):
        idx = np.where(y[:, i]==1)[0][:n_examples]
        class_indices.append(idx)
    class_indices = np.array(class_indices).T.flatten()

    fig, axes = plt.subplots(n_examples, 5, figsize=(25, n_examples*5), constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        img = x[class_indices[i]]
        img = np.where(img>threshold, img, 0)
        nonzero_count = np.count_nonzero(img)

        if normalized is False:
            im = ax.imshow(
                img,
                cmap='viridis',
                norm=colors.LogNorm(vmin=(threshold if threshold>0 else 1e-2), vmax=5e2),
                origin='lower',
                extent=[0, img.shape[0], 0, img.shape[1]]
            )
        else:
            im = ax.imshow(
                img,
                cmap='viridis',
                norm=colors.LogNorm(vmin=(threshold if threshold>0 else 1e-5), vmax=1),
                origin='lower',
                extent=[0, img.shape[0], 0, img.shape[1]]
            )
        ax.set_title(f'{classes[i % 5]} [active={nonzero_count}/({img.shape[0]}*{img.shape[1]})]', fontsize=16)
        #ax.set_xlabel("delta eta", fontsize=16)
        #ax.set_ylabel("delta phi", fontsize=16)

        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01)
        cbar.ax.tick_params(labelsize=16)
    plt.show()


