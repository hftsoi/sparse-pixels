import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
from pathlib import Path
from glob import glob


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



BASE_DIR = "datasets/neutrino"
PAT_INCL = f"{BASE_DIR}/bnb_WithWire_*.h5"
PAT_NUE = f"{BASE_DIR}/nue_WithWire_*.h5"

TIME_DS = 6
NRMS = 2.0
SIG_HIT_THRESHOLD = 50  # only when store_all=False

W_PER_PLANE = {0:2400, 1:2400, 2:3456}
T_RAW = 6400
T_DS = T_RAW // TIME_DS

OUTS_INCL = {p: f"{BASE_DIR}/data_inclusive_plane{p}.h5" for p in (0,1,2)}
OUTS_NUE = {p: f"{BASE_DIR}/data_nue_plane{p}.h5" for p in (0,1,2)}
OVERWRITE_OUTPUTS = True

def changepoint_index(eid_array):
    change = np.any(eid_array[1:] != eid_array[:-1], axis=1)
    starts = np.concatenate(([0], np.where(change)[0] + 1)).astype(np.int64)
    lengths = np.diff(np.concatenate((starts, [len(eid_array)]))).astype(np.int64)
    eids = [tuple(eid_array[s]) for s in starts]
    return starts, lengths, eids

def build_table_index_by_changes(f, table):
    eid_all = f[f"{table}/event_id"][:]
    s, c, e = changepoint_index(eid_all)
    return s, c, e, {eid:(int(ss), int(cc)) for eid, ss, cc in zip(e, s, c)}

def downsample_time_sum(img_WT, factor=TIME_DS, apply_nb_clip=True):
    trim = img_WT.shape[1] % factor
    if trim: img_WT = img_WT[:, :-trim]
    out = img_WT.reshape(img_WT.shape[0], -1, factor).sum(axis=2)
    if apply_nb_clip:
        adccutoff = 10 * factor / 6
        adcsaturation = 100 * factor / 6
        out = np.where(out < adccutoff, 0.0, out)
        out = np.minimum(out, adcsaturation)
    return out

def create_plane_out(path, plane, t_bins=T_DS, w_bins=None, compression="gzip", overwrite=False):
    if w_bins is None: w_bins = W_PER_PLANE[plane]
    p = Path(path)
    if p.exists():
        if overwrite: p.unlink()
        else:
            with h5py.File(path, "a") as g:
                g.attrs.setdefault("plane", int(plane))
                g.attrs.setdefault("time_downsample", int(TIME_DS))
                g.attrs.setdefault("wire_downsample", 1)
                g.attrs.setdefault("nb_clip", True)
            return
        
    with h5py.File(path, "w") as g:
        g.attrs["plane"] = int(plane)
        g.attrs["time_downsample"] = int(TIME_DS)
        g.attrs["wire_downsample"] = 1
        g.attrs["nb_clip"] = True
        maxshape = (None, t_bins, w_bins)
        chunks = (1, t_bins, w_bins)
        g.create_dataset("image", shape=(0, t_bins, w_bins), maxshape=maxshape, chunks=chunks, dtype="float32", compression=compression)
        g.create_dataset("sigmask", shape=(0, t_bins, w_bins), maxshape=maxshape, chunks=chunks, dtype="uint8", compression=compression)
        g.create_dataset("bkgmask", shape=(0, t_bins, w_bins), maxshape=maxshape, chunks=chunks, dtype="uint8", compression=compression)
        g.create_dataset("event_id", shape=(0, 3), maxshape=(None, 3), chunks=(1024,3), dtype="int32", compression=compression)
        g.create_dataset("event_idx_in_file", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="int64", compression=compression)
        g.create_dataset("source_file", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=h5py.string_dtype(encoding="utf-8"), compression=compression)

def append_plane_record(path, img_TW, sigmask_TW, bkgmask_TW, eid, evt_idx, src_file):
    with h5py.File(path, "a") as g:
        n = g["image"].shape[0]
        for k, arr in (("image", img_TW.astype(np.float32)),
                       ("sigmask", sigmask_TW.astype(np.uint8)),
                       ("bkgmask", bkgmask_TW.astype(np.uint8))):
            g[k].resize((n+1,) + g[k].shape[1:])
            g[k][n] = arr
        for name, val in (("event_id", np.asarray(eid, dtype=np.int32)),
                          ("event_idx_in_file", np.int64(evt_idx)),
                          ("source_file", str(src_file))):
            g[name].resize((n+1,) + g[name].shape[1:])
            g[name][n] = val

def dedupe_edep_max_energyfraction(hit_id_e, ef_e, g4_e):
    idx = np.argsort(-ef_e, kind="mergesort")
    hi = hit_id_e[idx]
    g4 = g4_e[idx]
    _, first = np.unique(hi, return_index=True)
    return {int(hi[i]): int(g4[i]) for i in first}

def build_label_notebook_exact(plane_wire, h_wire, h_time, h_rms, h_g4, T_raw=T_RAW, time_ds=TIME_DS, nrms=NRMS):
    W = len(plane_wire)
    lab_full = np.zeros((W, T_raw), dtype=np.int8)
    wire_to_row = {int(w): i for i, w in enumerate(plane_wire)}
    x = np.fromiter((wire_to_row[int(w)] for w in h_wire), dtype=np.int64, count=h_wire.size)
    t0 = np.floor(h_time - nrms*h_rms).astype(np.int64)
    t1 = np.ceil (h_time + nrms*h_rms).astype(np.int64)
    t0 = np.clip(t0, 0, T_raw-1); t1 = np.clip(t1, 0, T_raw-1)
    # neutrino first (+1)
    for xi, lo, hi, is_nu in zip(x, t0, t1, (h_g4 >= 0)):
        if is_nu: lab_full[int(xi), int(lo):int(hi)+1] = 1
    # cosmic (−1) overrides
    for xi, lo, hi, is_cos in zip(x, t0, t1, (h_g4 < 0)):
        if is_cos: lab_full[int(xi), int(lo):int(hi)+1] = -1
    trim = T_raw % time_ds
    if trim: lab_full = lab_full[:, :-trim]
    lab_ds_WT = lab_full.reshape(W, -1, time_ds).sum(axis=2)
    lab_ds_WT = np.sign(lab_ds_WT).astype(np.int8)
    return lab_ds_WT.T  # shape (T_ds, W)

def process_raw_file_wirewise(raw_path, out_paths, sig_hit_threshold=SIG_HIT_THRESHOLD, nrms=NRMS, store_all=False):
    print(f"\n>> Processing: {raw_path}")

    with h5py.File(raw_path, "r") as f:
        w_starts, w_counts, w_eids, _ = build_table_index_by_changes(f, "wire_table")
        _, _, _, h_map = build_table_index_by_changes(f, "hit_table")
        if "edep_table/event_id" in f:
            _, _, _, e_map = build_table_index_by_changes(f, "edep_table")
        else:
            e_map = {}

        kept = {0:0, 1:0, 2:0}
        for evt_idx, eid in enumerate(w_eids):
            ws, wc = int(w_starts[evt_idx]), int(w_counts[evt_idx])
            lp = f["wire_table/local_plane"][ws:ws+wc, 0]
            lw = f["wire_table/local_wire"][ws:ws+wc, 0].astype(np.int64)
            adc = f["wire_table/adc"][ws:ws+wc, :]

            if eid in h_map:
                hs, hc = h_map[eid]
                hit_id = f["hit_table/hit_id"][hs:hs+hc, 0]
                hit_lp = f["hit_table/local_plane"][hs:hs+hc, 0]
                hit_lw = f["hit_table/local_wire"][hs:hs+hc, 0].astype(np.int64)
                hit_time = f["hit_table/local_time"][hs:hs+hc, 0]
                hit_rms = f["hit_table/rms"][hs:hs+hc, 0]
            else:
                hit_id = hit_lp = hit_lw = hit_time = hit_rms = np.zeros((0,), dtype=np.float32)

            g4_by_hit = {}
            if eid in e_map:
                es, ec = e_map[eid]
                if ec > 0:
                    g4_by_hit = dedupe_edep_max_energyfraction(
                        f["edep_table/hit_id"][es:es+ec, 0],
                        f["edep_table/energy_fraction"][es:es+ec, 0],
                        f["edep_table/g4_id"][es:es+ec, 0],)
            if hit_id.size:
                g4 = np.array([g4_by_hit.get(int(h), -1) for h in hit_id], dtype=np.int32)
            else:
                g4 = np.zeros((0,), dtype=np.int32)

            for p in (0,1,2):
                m_rows = (lp == p)
                if not np.any(m_rows):
                    if store_all:
                        Wp = W_PER_PLANE[p]
                        img_TW = np.zeros((T_DS, Wp), dtype=np.float32)
                        sigmask_TW = np.zeros((T_DS, Wp), dtype=np.uint8)
                        bkgmask_TW = np.zeros((T_DS, Wp), dtype=np.uint8)
                        append_plane_record(out_paths[p], img_TW, sigmask_TW, bkgmask_TW, eid=eid, evt_idx=evt_idx, src_file=raw_path)
                        kept[p] += 1
                    continue

                plane_adc = adc[m_rows]
                plane_wire = lw[m_rows]
                order = np.argsort(plane_wire)
                plane_wire = plane_wire[order]
                plane_adc = plane_adc[order, :]

                img_W_Tds = downsample_time_sum(plane_adc, factor=TIME_DS, apply_nb_clip=True)
                img_TW = img_W_Tds.T

                if hit_id.size:
                    m_hits = (hit_lp == p) & np.isin(hit_lw, plane_wire)
                else:
                    m_hits = np.zeros((0,), dtype=bool)

                if np.any(m_hits):
                    h_wire = hit_lw[m_hits]
                    h_time = hit_time[m_hits]
                    h_rms_ = hit_rms[m_hits]
                    h_g4 = g4[m_hits]
                    nu_count = int(np.count_nonzero(h_g4 >= 0))
                else:
                    h_wire = np.zeros((0,), dtype=np.int64)
                    h_time = np.zeros((0,), dtype=np.float32)
                    h_rms_ = np.zeros((0,), dtype=np.float32)
                    h_g4 = np.zeros((0,), dtype=np.int32)
                    nu_count = 0

                keep = True if store_all else (nu_count >= int(sig_hit_threshold))
                if not keep:
                    continue

                label_TW  = build_label_notebook_exact(plane_wire, h_wire, h_time, h_rms_, h_g4, T_raw=T_RAW, time_ds=TIME_DS, nrms=nrms)
                sigmask_TW = (label_TW > 0).astype(np.uint8)
                bkgmask_TW = (label_TW < 0).astype(np.uint8)

                append_plane_record(out_paths[p], img_TW, sigmask_TW, bkgmask_TW, eid=eid, evt_idx=evt_idx, src_file=raw_path)
                kept[p] += 1

        print(f"Kept per plane: {kept}")
    return True

def run_batch(which="both", overwrite_outputs=OVERWRITE_OUTPUTS, store_all=False, sig_hit_threshold=SIG_HIT_THRESHOLD):
    if which in ("inclusive", "both"):
        for p in (0,1,2):
            create_plane_out(OUTS_INCL[p], plane=p, overwrite=overwrite_outputs)
    if which in ("nue", "both"):
        for p in (0,1,2):
            create_plane_out(OUTS_NUE[p], plane=p, overwrite=overwrite_outputs)

    incl_files = sorted(glob(PAT_INCL)) if which in ("inclusive", "both") else []
    nue_files = sorted(glob(PAT_NUE)) if which in ("nue", "both") else []
    print(f"there are {len(incl_files)} inclusive files and {len(nue_files)} nue files (which='{which}').")

    for raw_path in incl_files:
        process_raw_file_wirewise(raw_path, OUTS_INCL, sig_hit_threshold=sig_hit_threshold, nrms=NRMS, store_all=store_all)

    for raw_path in nue_files:
        process_raw_file_wirewise(raw_path, OUTS_NUE, sig_hit_threshold=sig_hit_threshold, nrms=NRMS, store_all=store_all)

    print("\nall done")
    if which in ("inclusive", "both"):
        print("inclusive outputs:", OUTS_INCL)
    if which in ("nue", "both"):
        print("nue outputs      :", OUTS_NUE)



def plot_plane_sample(plane_file, idx=0, p='raw', cmap='viridis',
                      jet_vmin=0.0, jet_vmax=100.0,
                      point_size=6, alpha=0.6,
                      figsize=(15,6), show_colorbar=True):
    with h5py.File(plane_file, "r") as g:
        N = g["image"].shape[0]
        if not (0 <= idx < N):
            raise IndexError(f"idx out of [0,{N-1}]")
        
        img = g["image"][idx]
        sigm = g["sigmask"][idx]
        bkgm = g["bkgmask"][idx]
        eid = tuple(g["event_id"][idx])
        src = g["source_file"][idx]

        if isinstance(src, (bytes, bytearray)): src = src.decode("utf-8", errors="ignore")

        evt_ord = int(g["event_idx_in_file"][idx])
        plane = int(g.attrs["plane"])
        time_ds = int(g.attrs["time_downsample"])

    T, W = img.shape
    print(f"Loaded plane={plane}, idx={idx} | eid={eid} | src={src} | event_idx_in_file={evt_ord}")
    print(f"  labeled pixels: sig={int(sigm.sum())}, bkg={int(bkgm.sum())}")

    fig, ax = plt.subplots(figsize=figsize)

    if p == 'raw':
        im = ax.imshow(img, origin='lower', aspect='auto', cmap=cmap, vmin=jet_vmin, vmax=jet_vmax)
        #if show_colorbar:
            #plt.colorbar(im, ax=ax).set_label("ADC sum (downsampled)")
    else:
        # keep axes identical to raw
        ax.set_xlim(0, W-1)
        ax.set_ylim(0, T-1)
        if p in ('sig', 'sigbkg'):
            y_sig, x_sig = np.where(sigm > 0)
            if x_sig.size:
                ax.scatter(x_sig, y_sig, s=point_size, marker='.', linewidths=0, alpha=alpha, color='tab:red', label='signal')
        if p in ('bkg', 'sigbkg'):
            y_bkg, x_bkg = np.where(bkgm > 0)
            if x_bkg.size:
                ax.scatter(x_bkg, y_bkg, s=point_size, marker='.', linewidths=0, alpha=alpha, color='tab:blue', label='background')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right')

    ax.set_title(f"plane={plane} eid={eid} (time downsample x{time_ds})")
    ax.set_xlabel("wire")
    ax.set_ylabel("time (downsampled ticks)")
    plt.tight_layout(); plt.show()