import json

with open(r'D:\satellite-interp\outputs\train_log.json') as f:
    log = json.load(f)

print('Best epochs by SSIM:')
sorted_log = sorted(log, key=lambda x: x['val_ssim'], reverse=True)
for e in sorted_log[:5]:
    print(f"  Epoch {e['epoch']:03d} | SSIM={e['val_ssim']:.4f} | val_loss={e['val_loss']:.6f}")

print()
print('Full training progression:')
for e in log:
    print(f"  Epoch {e['epoch']:03d} | SSIM={e['val_ssim']:.4f} | train={e['train_loss']:.6f}")