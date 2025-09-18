import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
from pathlib import Path
from glob import glob
from matplotlib.patches import Rectangle

def pool_pad_noise_inflate(img, pool_size, pool_type, target_size=None, noise_type=None, noise_level=0, inflate_factor=1):
    x = tf.convert_to_tensor(img, dtype=tf.float32)
    #x = tf.where(x<threshold, tf.zeros_like(x), x)

    k_h = k_w = pool_size

    if pool_size is not None:
        if pool_type == 'max':
            pooled = tf.nn.max_pool2d(x, ksize=[1, k_h, k_w, 1], strides=[1, k_h, k_w, 1], padding='VALID')
        elif pool_type in ('avg'):
            pooled = tf.nn.avg_pool2d(x, ksize=[1, k_h, k_w, 1], strides=[1, k_h, k_w, 1], padding='VALID')
    else:
        pooled = x

    old_h = tf.shape(x)[1]
    old_w = tf.shape(x)[2]
    new_h = tf.shape(pooled)[1]
    new_w = tf.shape(pooled)[2]

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

    if target_size is not None:
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


def plot_jetimage(x, y, n_examples, threshold=0, normalized=False, figname=None):
    classes = ['g','q','W','Z','t']
    class_indices = []
    for i in range(5):
        idx = np.where(y[:, i]==1)[0][n_examples:n_examples+1]#[:n_examples]
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
        #ax.set_title(f'{classes[i % 5]} [active={nonzero_count}/({img.shape[0]}*{img.shape[1]})]', fontsize=16)
        ax.set_title(f'{classes[i % 5]}', fontsize=16)
        #ax.set_xlabel("delta eta", fontsize=16)
        #ax.set_ylabel("delta phi", fontsize=16)

        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01)
        cbar.ax.tick_params(labelsize=16)
    #plt.show()
    if figname is not None:
        plt.savefig(f'plots/{figname}')



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
    t0 = np.clip(t0, 0, T_raw-1)
    t1 = np.clip(t1, 0, T_raw-1)
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
    plt.tight_layout()
    plt.show()




def _ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def _signal_counts(plane_file, chunk=1024):
    with h5py.File(plane_file, "r") as g:
        N = g["sigmask"].shape[0]
        counts = np.zeros(N, dtype=np.int64)
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            sm = g["sigmask"][s:e]
            counts[s:e] = sm.reshape(e - s, -1).sum(axis=1)
    return counts

def _create_empty_like(in_path, out_path, overwrite=True):
    _ensure_parent_dir(out_path)
    p = Path(out_path)
    if p.exists():
        if not overwrite:
            raise FileExistsError(f"{out_path} exists..")
        p.unlink()

    with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        def mk(name, src, first_dim=0):
            maxshape = (None,) + src.shape[1:]
            chunks = (1,) + src.shape[1:]
            comp = src.compression or "gzip"
            fout.create_dataset(
                name,
                shape=(first_dim,) + src.shape[1:],
                maxshape=maxshape,
                chunks=chunks,
                dtype=src.dtype,
                compression=comp, )

        mk("image", fin["image"])
        mk("sigmask", fin["sigmask"])
        mk("bkgmask", fin["bkgmask"])

        fout.create_dataset("event_id", shape=(0, 3), maxshape=(None, 3), chunks=(1024, 3), dtype=fin["event_id"].dtype, compression="gzip")
        fout.create_dataset("event_idx_in_file", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=fin["event_idx_in_file"].dtype, compression="gzip")
        fout.create_dataset("source_file", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=fin["source_file"].dtype, compression="gzip")

def _append_batch(out_path, batch):
    with h5py.File(out_path, "a") as g:
        n = g["image"].shape[0]
        m = batch["image"].shape[0]
        for name in ("image", "sigmask", "bkgmask"):
            g[name].resize((n + m,) + g[name].shape[1:])
            g[name][n:n + m] = batch[name]
        for name in ("event_id", "event_idx_in_file", "source_file"):
            g[name].resize((n + m,) + g[name].shape[1:])
            g[name][n:n + m] = batch[name]

def write_slim_plane_ge_threshold(in_plane_file,
                                  out_plane_file,
                                  threshold,
                                  count_chunk=512,
                                  copy_batch=128,
                                  overwrite=True,
                                  verbose=True):
    counts = _signal_counts(in_plane_file, chunk=count_chunk)
    keep_idx = np.where(counts >= threshold)[0]

    if verbose:
        N = counts.size
        pct = (keep_idx.size / N) if N else 0.0
        print(f"{Path(in_plane_file).name}: N={N}, threshold >= {threshold} -> keep {keep_idx.size} ({pct:.1%})")

    _create_empty_like(in_plane_file, out_plane_file, overwrite=overwrite)

    if keep_idx.size == 0:
        if verbose:
            print(f"nothing to keep, wrote empty file to {out_plane_file}")
        return

    with h5py.File(in_plane_file, "r") as fin:
        Nsrc = fin["image"].shape[0]
        if not np.all((0 <= keep_idx) & (keep_idx < Nsrc)):
            raise IndexError("keep_idx out of range.")

        for s in range(0, keep_idx.size, copy_batch):
            idx = keep_idx[s:s + copy_batch]
            batch = dict(
                image = fin["image"][idx],
                sigmask = fin["sigmask"][idx],
                bkgmask = fin["bkgmask"][idx],
                event_id = fin["event_id"][idx],
                event_idx_in_file = fin["event_idx_in_file"][idx],
                source_file = fin["source_file"][idx],
            )
            _append_batch(out_plane_file, batch)

            if verbose and ((s // copy_batch) % 50 == 0 or s + copy_batch >= keep_idx.size):
                print(f"  copied {min(s + copy_batch, keep_idx.size)}/{keep_idx.size}")
    if verbose:
        print(f"done: wrote {keep_idx.size} rows -> {out_plane_file}")





def run_window_demo(
    plane_file,
    indices=None, first_N=5,
    win_t=256, win_w=512,
    num_bkg=5,
    prefer_zero_signal_bkg=True,
    cmap='gist_ncar', vmin=0, vmax=100,
    rng_seed=123,
):
    rng = np.random.default_rng(rng_seed)

    def integral_image(mask_TW: np.ndarray) -> np.ndarray:
        sat = np.cumsum(np.cumsum(mask_TW.astype(np.int32), axis=0), axis=1)
        return np.pad(sat, ((1,0),(1,0)), mode='constant')

    def rect_sum(sat: np.ndarray, t0: int, w0: int, h: int, w: int) -> int:
        t1, w1 = t0 + h, w0 + w
        return int(sat[t1, w1] - sat[t0, w1] - sat[t1, w0] + sat[t0, w0])

    def find_best_signal_window(sigm_TW: np.ndarray, win_t: int, win_w: int):
        T, W = sigm_TW.shape
        sat = integral_image(sigm_TW)
        sums = (sat[win_t:, win_w:] - sat[:-win_t, win_w:] - sat[win_t:, :-win_w] + sat[:-win_t, :-win_w])
        r, c = np.unravel_index(int(np.argmax(sums)), sums.shape)
        return int(r), int(c), int(sums[r, c])

    def rects_overlap(a, b):
        t0a, w0a, ha, wa = a
        t0b, w0b, hb, wb = b
        if t0a + ha <= t0b or t0b + hb <= t0a: return False
        if w0a + wa <= w0b or w0b + wb <= w0a: return False
        return True

    def sample_background_windows(sat_sig, T, W, win_t, win_w, sig_rect, K=5, zero_sig_only=True, max_trials_per_window=4000):
        candidates, used, tries = [], [sig_rect], 0
        def rand_pos():
            return int(rng.integers(0, T - win_t + 1)), int(rng.integers(0, W - win_w + 1))
        
        # random zero-signal
        while len(candidates) < K and tries < max_trials_per_window * K:
            t0, w0 = rand_pos()
            rect = (t0, w0, win_t, win_w)
            if any(rects_overlap(rect, u) for u in used):
                tries += 1
                continue
            c = rect_sum(sat_sig, t0, w0, win_t, win_w)
            if (zero_sig_only and c == 0) or (not zero_sig_only):
                candidates.append((t0, w0, int(c)))
                used.append(rect)
            tries += 1

        # fallback minimal-signal
        if zero_sig_only and len(candidates) < K:
            need = K - len(candidates)
            step_t, step_w = max(1, win_t//2), max(1, win_w//2)
            mins = []
            for t0 in range(0, T - win_t + 1, step_t):
                for w0 in range(0, W - win_w + 1, step_w):
                    rect = (t0, w0, win_t, win_w)
                    if any(rects_overlap(rect, u) for u in used):
                        continue
                    c = rect_sum(sat_sig, t0, w0, win_t, win_w)
                    mins.append((c, t0, w0))
            if mins:
                mins.sort(key=lambda x: x[0])
                for c, t0, w0 in mins[:need]:
                    candidates.append((t0, w0, int(c)))
                    used.append((t0, w0, win_t, win_w))
        return candidates[:K]

    def overlay_rect(ax, t0, w0, h, w, color, lw=2, label=None):
        ax.add_patch(Rectangle((w0, t0), w, h, fill=False, ec=color, lw=lw, label=label))

    def mask_rgba(sig, bkg, alpha_sig=0.9, alpha_bkg=0.9):
        T, W = sig.shape
        rgba = np.zeros((T, W, 4), dtype=np.float32)
        rgba[..., 0] = (sig > 0).astype(np.float32)
        rgba[..., 2] = (bkg > 0).astype(np.float32)
        rgba[..., 3] = np.clip(alpha_sig*(sig>0) + alpha_bkg*(bkg>0), 0.0, 1.0)
        return rgba

    with h5py.File(plane_file, "r") as g:
        N = g["image"].shape[0]
        use_idx = indices if indices is not None else list(range(min(first_N, N)))
        plane = int(g.attrs["plane"])
        tds = int(g.attrs["time_downsample"])

        for idx in use_idx:
            img = g["image"][idx]
            sigm = g["sigmask"][idx]
            bkgm = g["bkgmask"][idx]
            eid = tuple(g["event_id"][idx])
            src = g["source_file"][idx]
            if isinstance(src, (bytes, bytearray)):
                src = src.decode("utf-8","ignore")

            # find signal window + sample bkg
            T, W = img.shape
            t0, w0, best = find_best_signal_window(sigm, win_t, win_w)
            sig_rect = (t0, w0, win_t, win_w)
            sat_sig = integral_image(sigm)
            bkg_list = sample_background_windows(sat_sig, T, W, win_t, win_w, sig_rect, K=num_bkg, zero_sig_only=prefer_zero_signal_bkg)

            # crops
            def crop(t, w): 
                return (img[t:t+win_t, w:w+win_w],
                        sigm[t:t+win_t, w:w+win_w],
                        bkgm[t:t+win_t, w:w+win_w])
            sig_img, sig_sig, sig_bkg = crop(t0, w0)
            bkg_tiles = []
            for tb, wb, cb in bkg_list:
                im, sg, bk = crop(tb, wb)
                bkg_tiles.append((im, sg, bk, tb, wb, cb))

            total_sig = int(sigm.sum())
            cap_frac = float(best) / max(1, total_sig)

            print(f"[{Path(plane_file).name}] idx={idx} eid={eid} "
                  f"total_sig={total_sig}, in_win={best} ({cap_frac:.1%}), "
                  f"bkg_found={len(bkg_tiles)}")

            # plot
            fig, ax = plt.subplots(figsize=(14,6))
            imshow = ax.imshow(img, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            #plt.colorbar(imshow, ax=ax).set_label("ADC (downsampled)")
            overlay_rect(ax, t0, w0, win_t, win_w, 'red', 2, 'sig window')
            for i, (_,_,_, tb, wb, _) in enumerate(bkg_tiles):
                overlay_rect(ax, tb, wb, win_t, win_w, 'blue', 1.5, 'bkg window' if i==0 else None)
            #ax.set_title(f"plane={plane} eid={eid} (time x{tds})")
            ax.set_xlabel("wire")
            ax.set_ylabel("time")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc='upper right')
            plt.tight_layout()
            plt.show()

            tiles = [("signal", (sig_img, sig_sig, sig_bkg, t0, w0, best))]
            tiles += [(f"bkg {i+1}", bt) for i, bt in enumerate(bkg_tiles)]
            rows, cols = 2, 3
            fig, axes = plt.subplots(rows, cols, figsize=(15,8), constrained_layout=True)
            for k, ax in enumerate(axes.flatten()):
                if k >= len(tiles):
                    ax.axis('off')
                    continue
                name, (im, sg, bk, tb, wb, cb) = tiles[k]
                ax.imshow(im, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{name} (sigpix={int(sg.sum())})")
                ax.set_xlabel("wire")
                ax.set_ylabel("time")
            plt.show()

            fig, axes = plt.subplots(rows, cols, figsize=(15,8), constrained_layout=True)
            for k, ax in enumerate(axes.flatten()):
                if k >= len(tiles):
                    ax.axis('off')
                    continue
                name, (im, sg, bk, tb, wb, cb) = tiles[k]
                ax.set_facecolor('white')
                ax.imshow(mask_rgba(sg, bk, 0.9, 0.9), origin='lower', aspect='auto', interpolation='nearest')
                ax.set_title(f"{name} (sigpix={int(sg.sum())})")
                ax.set_xlabel("wire")
                ax.set_ylabel("time")
            plt.show()



def write_patches_from_slim_file(
    in_plane_file,
    out_sig_file=None,
    out_bkg_file=None,
    win_t=256, win_w=512,
    num_bkg=5,
    prefer_zero_signal_bkg=True,
    copy_batch=128,
    overwrite=True,
    verbose=True,
    rng_seed=123,
):
    rng = np.random.default_rng(rng_seed)
    in_p = Path(in_plane_file)
    if out_sig_file is None:
        out_sig_file = in_p.with_suffix("").as_posix() + f"_patches_sig_t{win_t}w{win_w}.h5"
    if out_bkg_file is None:
        out_bkg_file = in_p.with_suffix("").as_posix() + f"_patches_bkg_t{win_t}w{win_w}.h5"

    def integral_image(mask_TW):
        sat = np.cumsum(np.cumsum(mask_TW.astype(np.int32), axis=0), axis=1)
        return np.pad(sat, ((1,0),(1,0)), mode='constant')

    def rect_sum(sat, t0, w0, h, w):
        t1, w1 = t0 + h, w0 + w
        return int(sat[t1, w1] - sat[t0, w1] - sat[t1, w0] + sat[t0, w0])

    def find_best_signal_window(sigm_TW, win_t, win_w):
        T, W = sigm_TW.shape
        sat = integral_image(sigm_TW)
        sums = (sat[win_t:, win_w:] - sat[:-win_t, win_w:] - sat[win_t:, :-win_w] + sat[:-win_t, :-win_w])
        r, c = np.unravel_index(int(np.argmax(sums)), sums.shape)
        return int(r), int(c), int(sums[r, c])

    def rects_overlap(a, b):
        t0a, w0a, ha, wa = a
        t0b, w0b, hb, wb = b
        if t0a + ha <= t0b or t0b + hb <= t0a: return False
        if w0a + wa <= w0b or w0b + wb <= w0a: return False
        return True

    def sample_background_windows(sat_sig, T, W, win_t, win_w, sig_rect, K=5, zero_sig_only=True, max_trials_per_window=4000):
        candidates, used, tries = [], [sig_rect], 0
        def rand_pos():
            return int(rng.integers(0, T - win_t + 1)), int(rng.integers(0, W - win_w + 1))
        while len(candidates) < K and tries < max_trials_per_window * K:
            t0, w0 = rand_pos()
            rect = (t0, w0, win_t, win_w)
            if any(rects_overlap(rect, u) for u in used):
                tries += 1
                continue
            c = rect_sum(sat_sig, t0, w0, win_t, win_w)
            if (zero_sig_only and c == 0) or (not zero_sig_only):
                candidates.append((t0, w0, int(c)))
                used.append(rect)
            tries += 1
        if zero_sig_only and len(candidates) < K:
            need = K - len(candidates)
            step_t, step_w = max(1, win_t//2), max(1, win_w//2)
            mins = []
            for t0 in range(0, T - win_t + 1, step_t):
                for w0 in range(0, W - win_w + 1, step_w):
                    rect = (t0, w0, win_t, win_w)
                    if any(rects_overlap(rect, u) for u in used):
                        continue
                    c = rect_sum(sat_sig, t0, w0, win_t, win_w)
                    mins.append((c, t0, w0))
            if mins:
                mins.sort(key=lambda x: x[0])
                for c, t0, w0 in mins[:need]:
                    candidates.append((t0, w0, int(c)))
                    used.append((t0, w0, win_t, win_w))
        return candidates[:K]

    def create_patch_file(out_path, plane_attr, time_ds_attr, win_t, win_w):
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            if not overwrite:
                raise FileExistsError(f"{out_path} exists...")
            p.unlink()
        with h5py.File(out_path, "w") as g:
            g.attrs["plane"] = int(plane_attr)
            g.attrs["time_downsample"] = int(time_ds_attr)
            g.attrs["win_t"] = int(win_t)
            g.attrs["win_w"] = int(win_w)
            maxshape = (None, win_t, win_w)
            chunks = (1, win_t, win_w)
            g.create_dataset("image", shape=(0, win_t, win_w), maxshape=maxshape, chunks=chunks, dtype="float32", compression="gzip")
            g.create_dataset("sigmask", shape=(0, win_t, win_w), maxshape=maxshape, chunks=chunks, dtype="uint8", compression="gzip")
            g.create_dataset("bkgmask", shape=(0, win_t, win_w), maxshape=maxshape, chunks=chunks, dtype="uint8", compression="gzip")
            g.create_dataset("event_id", shape=(0,3), maxshape=(None,3), chunks=(1024,3), dtype="int32", compression="gzip")
            g.create_dataset("event_idx_in_file", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="int64", compression="gzip")
            g.create_dataset("source_file", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=h5py.string_dtype(encoding="utf-8"), compression="gzip")
            g.create_dataset("win_origin", shape=(0,2), maxshape=(None,2), chunks=(1024,2), dtype="int32", compression="gzip")
            g.create_dataset("sigpix_in_win", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="int32", compression="gzip")
            g.create_dataset("total_sigpix_evt", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="int32", compression="gzip")
            g.create_dataset("capture_fraction", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="float32", compression="gzip")

    def append_patch_batch(out_path, batch):
        with h5py.File(out_path, "a") as g:
            n = g["image"].shape[0]
            m = batch["image"].shape[0]
            for k in ("image","sigmask","bkgmask"):
                g[k].resize((n+m,)+g[k].shape[1:])
                g[k][n:n+m] = batch[k]
            for k in ("event_id","event_idx_in_file","source_file", "win_origin", "sigpix_in_win", "total_sigpix_evt", "capture_fraction"):
                g[k].resize((n+m,)+g[k].shape[1:])
                g[k][n:n+m] = batch[k]

    if verbose:
        print(f"\n>> patching {in_plane_file}\n  sig -> {out_sig_file}\n  bkg -> {out_bkg_file}")

    with h5py.File(in_plane_file, "r") as fin:
        N = fin["image"].shape[0]
        plane = int(fin.attrs["plane"])
        time_ds = int(fin.attrs["time_downsample"])

        create_patch_file(out_sig_file, plane, time_ds, win_t, win_w)
        create_patch_file(out_bkg_file, plane, time_ds, win_t, win_w)

        sig_buf = {k: [] for k in ("image","sigmask","bkgmask","event_id","event_idx_in_file",
                                   "source_file","win_origin","sigpix_in_win","total_sigpix_evt","capture_fraction")}
        bkg_buf = {k: [] for k in ("image","sigmask","bkgmask","event_id","event_idx_in_file",
                                   "source_file","win_origin","sigpix_in_win","total_sigpix_evt","capture_fraction")}

        def flush(force=False):
            if sig_buf["image"] and (force or len(sig_buf["image"]) >= copy_batch):
                batch = {k: (np.asarray(v, dtype=object) if k=="source_file" else np.asarray(v)) for k,v in sig_buf.items()}
                append_patch_batch(out_sig_file, batch)
                for k in sig_buf:
                    sig_buf[k].clear()
            if bkg_buf["image"] and (force or len(bkg_buf["image"]) >= copy_batch):
                batch = {k: (np.asarray(v, dtype=object) if k=="source_file" else np.asarray(v)) for k,v in bkg_buf.items()}
                append_patch_batch(out_bkg_file, batch)
                for k in bkg_buf:
                    bkg_buf[k].clear()

        for idx in range(N):
            img = fin["image"][idx]
            sigm = fin["sigmask"][idx]
            bkgm = fin["bkgmask"][idx]
            eid = tuple(fin["event_id"][idx])
            src = fin["source_file"][idx]
            if isinstance(src, (bytes, bytearray)):
                src = src.decode("utf-8","ignore")
            evt_i = int(fin["event_idx_in_file"][idx])

            T, W = img.shape
            t0, w0, sig_in_win = find_best_signal_window(sigm, win_t, win_w)
            sig_rect = (t0, w0, win_t, win_w)
            sat_sig = integral_image(sigm)

            # signal crop
            sig_img = img [t0:t0+win_t, w0:w0+win_w]
            sig_sigm = sigm[t0:t0+win_t, w0:w0+win_w]
            sig_bkgm = bkgm[t0:t0+win_t, w0:w0+win_w]
            total_sig = int(sigm.sum())
            cap_frac = float(sig_in_win)/max(1,total_sig)

            sig_buf["image"].append(sig_img.astype(np.float32))
            sig_buf["sigmask"].append(sig_sigm.astype(np.uint8))
            sig_buf["bkgmask"].append(sig_bkgm.astype(np.uint8))
            sig_buf["event_id"].append(np.asarray(eid, dtype=np.int32))
            sig_buf["event_idx_in_file"].append(np.int64(evt_i))
            sig_buf["source_file"].append(str(src))
            sig_buf["win_origin"].append(np.asarray([t0, w0], dtype=np.int32))
            sig_buf["sigpix_in_win"].append(np.int32(sig_in_win))
            sig_buf["total_sigpix_evt"].append(np.int32(total_sig))
            sig_buf["capture_fraction"].append(np.float32(cap_frac))

            # background crop
            for (tb, wb, c) in sample_background_windows(sat_sig, T, W, win_t, win_w, sig_rect, K=num_bkg, zero_sig_only=prefer_zero_signal_bkg):
                bkg_img = img [tb:tb+win_t, wb:wb+win_w]
                bkg_sigm = sigm[tb:tb+win_t, wb:wb+win_w]
                bkg_bkgm = bkgm[tb:tb+win_t, wb:wb+win_w]
                bkg_buf["image"].append(bkg_img.astype(np.float32))
                bkg_buf["sigmask"].append(bkg_sigm.astype(np.uint8))
                bkg_buf["bkgmask"].append(bkg_bkgm.astype(np.uint8))
                bkg_buf["event_id"].append(np.asarray(eid, dtype=np.int32))
                bkg_buf["event_idx_in_file"].append(np.int64(evt_i))
                bkg_buf["source_file"].append(str(src))
                bkg_buf["win_origin"].append(np.asarray([tb, wb], dtype=np.int32))
                bkg_buf["sigpix_in_win"].append(np.int32(int(bkg_sigm.sum())))
                bkg_buf["total_sigpix_evt"].append(np.int32(total_sig))
                bkg_buf["capture_fraction"].append(np.float32(float(bkg_sigm.sum())/max(1,total_sig)))

            flush(force=False)
            if verbose and ((idx+1) % 200 == 0 or idx+1 == N):
                print(f"  processed {idx+1}/{N}")
        flush(force=True)
    if verbose:
        print("done!\n")





def preview_patches(
    patch_file,
    max_examples=6,
    start_idx=0,
    cmap='gist_ncar',
    vmin=0, vmax=100,
    alpha_sig=0.9, alpha_bkg=0.9,
):
    def _mask_rgba(sig, bkg, a_s=0.9, a_b=0.9):
        T, W = sig.shape
        rgba = np.zeros((T, W, 4), dtype=np.float32)
        rgba[..., 0] = (sig > 0).astype(np.float32) # red
        rgba[..., 2] = (bkg > 0).astype(np.float32) # blue
        rgba[..., 3] = np.clip(a_s*(sig>0) + a_b*(bkg>0), 0.0, 1.0)
        return rgba

    patch_file = str(patch_file)
    with h5py.File(patch_file, "r") as g:
        N = g["image"].shape[0]
        plane = int(g.attrs.get("plane", -1))
        time_ds = int(g.attrs.get("time_downsample", 6))
        win_t = int(g.attrs.get("win_t", g["image"].shape[1]))
        win_w = int(g.attrs.get("win_w", g["image"].shape[2]))

        end = min(start_idx + max_examples, N)
        if start_idx >= N:
            print(f"{Path(patch_file).name}: start_idx={start_idx} >= N={N}, nothing there..")
            return

        for idx in range(start_idx, end):
            img = g["image"][idx]
            sigm = g["sigmask"][idx]
            bkgm = g["bkgmask"][idx]

            eid = tuple(g["event_id"][idx])
            src = g["source_file"][idx]
            if isinstance(src, (bytes, bytearray)):
                src = src.decode("utf-8", "ignore")
            evt_i = int(g["event_idx_in_file"][idx])

            win_origin = g["win_origin"][idx] if "win_origin" in g else np.array([-1,-1], dtype=np.int32)
            sig_in_win = int(g["sigpix_in_win"][idx]) if "sigpix_in_win" in g else int(sigm.sum())
            tot_sig = int(g["total_sigpix_evt"][idx]) if "total_sigpix_evt" in g else sig_in_win
            cap_frac = float(g["capture_fraction"][idx]) if "capture_fraction" in g else (float(sig_in_win)/max(1,tot_sig))

            bkg_in_win = int(bkgm.sum())
            total_in_win = int(sigm.sum() + bkgm.sum())

            print(#f"idx={idx} | eid={eid} | src={src} | evt_idx={evt_i} | "
                  #f"win_origin={tuple(map(int,win_origin))} "
                  f"sig_in_window / sig_in_evt = {sig_in_win} / {tot_sig} | "
                  f"bkg_in_window = {bkg_in_win} | total_in_window = {total_in_win}")

            fig, axes = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True, sharex=True, sharey=True)

            # raw
            ax = axes[0]
            im = ax.imshow(img, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            #ax.set_title("raw intensity")
            ax.set_xlabel("wire", fontsize=15)
            ax.set_ylabel("time", fontsize=15)
            #cb = plt.colorbar(im, ax=ax)
            #cb.set_label("ADC (downsampled)")

            # mask overlay on white
            ax = axes[1]
            ax.set_facecolor('white')
            rgba = _mask_rgba(sigm, bkgm, alpha_sig, alpha_bkg)
            ax.imshow(rgba, origin='lower', aspect='auto', interpolation='nearest')
            #ax.set_title("mask overlay (red=sig, blue=bkg)")
            ax.set_xlabel("wire", fontsize=15)

            #supt = (f"{Path(patch_file).name} plane={plane} eid={eid} window={win_t}x{win_w}")
            #fig.suptitle(supt, y=1.02, fontsize=10)
            plt.show()





def preview_patches_with_pooling(
    patch_file, max_examples=5, start_idx=0,
    pool_t=1, pool_w=1,
    cmap='gist_ncar',
    vmin_raw=0, vmax_raw=100,
    vmin_pool=0, vmax_pool=100,
    thr=0, vmin_thr=0, vmax_thr=100,
    alpha_sig=0.9, alpha_bkg=0.9,
):
    def _mask_rgba(sig, bkg, a_s=0.9, a_b=0.9):
        T, W = sig.shape
        rgba = np.zeros((T, W, 4), dtype=np.float32)
        rgba[..., 0] = (sig > 0).astype(np.float32) # red
        rgba[..., 2] = (bkg > 0).astype(np.float32) # blue
        rgba[..., 3] = np.clip(a_s*(sig>0) + a_b*(bkg>0), 0.0, 1.0)
        return rgba

    def _mask_rgba_with_keep(sig, bkg, keep, a_s=0.9, a_b=0.9):
        T, W = sig.shape
        rgba = np.zeros((T, W, 4), dtype=np.float32)
        rgba[..., 0] = (sig > 0).astype(np.float32)
        rgba[..., 2] = (bkg > 0).astype(np.float32)
        alpha = np.clip(a_s*(sig>0) + a_b*(bkg>0), 0.0, 1.0)
        rgba[..., 3] = alpha * (keep.astype(np.float32))
        return rgba

    def _pool_sum_2d(a_TW, pt, pw):
        if pt == 1 and pw == 1:
            return a_TW
        T, W = a_TW.shape
        Tt, Wt = (T // pt) * pt, (W // pw) * pw
        a = a_TW[:Tt, :Wt]
        return a.reshape(Tt//pt, pt, Wt//pw, pw).sum(axis=(1,3))

    def _pool_or_2d(m_TW, pt, pw):
        if pt == 1 and pw == 1:
            return (m_TW > 0).astype(np.uint8)
        T, W = m_TW.shape
        Tt, Wt = (T // pt) * pt, (W // pw) * pw
        m = (m_TW[:Tt, :Wt] > 0).astype(np.uint8)
        return m.reshape(Tt//pt, pt, Wt//pw, pw).max(axis=(1,3)).astype(np.uint8)

    patch_file = str(patch_file)
    with h5py.File(patch_file, "r") as g:
        N = g["image"].shape[0]
        end = min(start_idx + max_examples, N)
        if start_idx >= N:
            print(f"{Path(patch_file).name}: start_idx={start_idx} >= N={N}, nothing there..")
            return

        plane = int(g.attrs.get("plane", -1))
        time_ds = int(g.attrs.get("time_downsample", 6))
        win_t = int(g.attrs.get("win_t", g["image"].shape[1]))
        win_w = int(g.attrs.get("win_w", g["image"].shape[2]))

        for idx in range(start_idx, end):
            img = g["image"][idx]
            sigm = g["sigmask"][idx]
            bkgm = g["bkgmask"][idx]

            eid = tuple(g["event_id"][idx])
            src = g["source_file"][idx]
            if isinstance(src, (bytes, bytearray)):
                src = src.decode("utf-8", "ignore")
            evt_i = int(g["event_idx_in_file"][idx])

            sig_in_win = int(g["sigpix_in_win"][idx]) if "sigpix_in_win" in g else int(sigm.sum())
            tot_sig_evt = int(g["total_sigpix_evt"][idx]) if "total_sigpix_evt" in g else sig_in_win
            bkg_in_win = int(bkgm.sum())
            total_in_win = int(sigm.sum() + bkgm.sum())

            #print(f"idx={idx} | eid={eid} | src={src} | evt_idx={evt_i} | "
            #      f"sig_in_window / sig_in_evt = {sig_in_win} / {tot_sig_evt} | "
            #      f"bkg_in_window = {bkg_in_win} | total_in_window = {total_in_win}")

            # pooled
            img_p = _pool_sum_2d(img, pool_t, pool_w)
            sig_p = _pool_or_2d(sigm, pool_t, pool_w)
            bkg_p = _pool_or_2d(bkgm, pool_t, pool_w)

            total_hits_raw = int((sigm > 0).sum() + (bkgm > 0).sum())
            total_hits_pooled = int((sig_p > 0).sum() + (bkg_p > 0).sum())

            # threshold
            has_thr = thr is not None
            if has_thr:
                img_thr = img_p.copy()
                img_thr[img_thr < thr] = 0
                keep = (img_p >= thr)
                total_hits_thr = int((sig_p[keep] > 0).sum() + (bkg_p[keep] > 0).sum())

            # rows: raw, pooled, thresholded pooled
            nrows = 3 if has_thr else 2
            fig, axes = plt.subplots(nrows, 2, figsize=(16, 12 if has_thr else 8), constrained_layout=True, sharex=False, sharey=False)

            # row 0: raw intensity
            ax = axes[0,0]
            im = ax.imshow(img, origin='lower', aspect='auto', cmap=cmap, vmin=vmin_raw, vmax=vmax_raw)
            ax.set_xlabel("wire", fontsize=13)
            ax.set_ylabel("time", fontsize=13)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            #cb.set_label("ADC", fontsize=11)

            # row 0: raw mask
            ax = axes[0,1]
            ax.set_facecolor('white')
            ax.imshow(_mask_rgba(sigm, bkgm, alpha_sig, alpha_bkg), origin='lower', aspect='auto', interpolation='nearest')
            #ax.set_xlabel("wire", fontsize=13)
            #ax.set_ylabel("time", fontsize=13)
            ax.set_title(f"total hits={total_hits_raw} / ({img.shape[0]}x{img.shape[1]})", fontsize=12)

            # row 1: pooled intensity
            ax = axes[1,0]
            im2 = ax.imshow(img_p, origin='lower', aspect='auto', cmap=cmap, vmin=vmin_pool, vmax=vmax_pool)
            ax.set_xlabel("wire", fontsize=13)
            ax.set_ylabel("time", fontsize=13)
            cb2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)
            #cb2.set_label(f"ADC", fontsize=11)

            # row 1: pooled mask
            ax = axes[1,1]
            ax.set_facecolor('white')
            ax.imshow(_mask_rgba(sig_p, bkg_p, alpha_sig, alpha_bkg), origin='lower', aspect='auto', interpolation='nearest')
            #ax.set_xlabel("wire", fontsize=13)
            #ax.set_ylabel("time", fontsize=13)
            ax.set_title(f"total hits = {total_hits_pooled} / ({img_p.shape[0]}x{img_p.shape[1]})  [pool={pool_t}x{pool_w}]", fontsize=12)

            # row 2: thresholded intensity + mask
            if has_thr:
                ax = axes[2,0]
                im3 = ax.imshow(img_thr, origin='lower', aspect='auto', cmap=cmap, vmin=vmin_thr, vmax=vmax_thr)
                ax.set_xlabel("wire", fontsize=13)
                ax.set_ylabel("time", fontsize=13)
                cb3 = plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.02)
                #cb3.set_label(f"ADC, thr={thr}", fontsize=11)

                ax = axes[2,1]
                ax.set_facecolor('white')
                ax.imshow(_mask_rgba_with_keep(sig_p, bkg_p, keep, alpha_sig, alpha_bkg), origin='lower', aspect='auto', interpolation='nearest')
                #ax.set_xlabel("wire", fontsize=13)
                #ax.set_ylabel("time", fontsize=13)
                ax.set_title(f"total hits = {total_hits_thr} / ({img_p.shape[0]}x{img_p.shape[1]})  [pool={pool_t}x{pool_w}, threshold={thr}]", fontsize=12)
            plt.show()


