import time, torch, metasmooth as ms
dev = torch.device('cuda')
sub = ms.load_cifar_subset('/workspace/tmp/data', n_train=6000, n_val=2000, seed=0, device=dev)
mp = ms.make_metaparam('per_cluster', sub); z = torch.zeros(mp.dim, device=dev)
cfg = ms.TrainConfig(epochs=18, batch_size=500, lr=0.08, amp='off')
for r in (ms.BASELINE_ROUTINE, ms.SMOOTH_ROUTINE, ms.SMOOTH_WIDE_ROUTINE):
    ms.run_algorithm(sub, mp, z, r, cfg, device=dev); torch.cuda.synchronize()
    t = time.time(); res = ms.run_algorithm(sub, mp, z, r, cfg, device=dev); torch.cuda.synchronize()
    print(f'{r.name:12s} {time.time()-t:6.2f}s/run  f0={res.val_loss:.3f} acc={res.val_acc:.3f}', flush=True)
