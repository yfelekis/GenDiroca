#!/usr/bin/env python3
import argparse, os, sys, types, torch

# -----------------------
# 0) CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Load Xia et al. ColorMNIST encoder and export a standalone adapter.")
    p.add_argument("--repo", type=str, default="third_party/xia_nca/NeuralCausalAbstractions",
                   help="Path to Xia's NeuralCausalAbstractions repo (must contain src/ and dat/mnist/).")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to Lightning checkpoint (epoch=...step=....ckpt).")
    p.add_argument("--save-adapter", type=str, default="checkpoints/xia_color_mnist_rep/rep_encoder_only.pt",
                   help="Where to save the plain PyTorch encoder adapter.")
    p.add_argument("--save-torchscript", type=str, default="checkpoints/xia_color_mnist_rep/rep_encoder_only_traced.pt",
                   help="Where to save the TorchScript-traced adapter.")
    p.add_argument("--img-size", type=int, default=32, help="Expected image size for DG/encoder.")
    return p.parse_args()

# -----------------------
# 1) Make Xia's repo importable
# -----------------------
def make_repo_importable(repo_path: str):
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"Repo path not found: {repo_path}")

    # change CWD to the repo so 'dat/mnist/' resolves correctly for mnist.MNIST loader
    os.chdir(repo_path)

    # ensure the repo is on sys.path
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    # ensure 'src' behaves like a proper package
    src_root = os.path.join(repo_path, "src")
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Expected 'src' folder under repo: {src_root}")

    for root, _, _ in os.walk(src_root):
        ip = os.path.join(root, "__init__.py")
        if not os.path.exists(ip):
            open(ip, "a").close()

    # register a module alias 'src' pointing at repo/src
    sys.modules.pop("src", None)
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [src_root]
    sys.modules["src"] = src_pkg

    # sanity: MNIST files present?
    mnist_dir = os.path.join(repo_path, "dat", "mnist")
    needed = [
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"
    ]
    missing = [f for f in needed if not os.path.exists(os.path.join(mnist_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing MNIST files in {mnist_dir}: {missing}\n"
            f"Put the four IDX files there (uncompressed)."
        )

# -----------------------
# 2) Build model + DG
# -----------------------
def build_representation_model(img_size: int):
    # now that the repo is importable, we can import
    from src.datagen.color_mnist import ColorMNISTDataGenerator
    from src.datagen.scm_datagen import SCMDataTypes as sdt
    from src.scm.repr_nn.representation_nn import RepresentationalNN

    DG = ColorMNISTDataGenerator(img_size, "sampling")
    v_size, v_type = DG.v_size, DG.v_type

    class _CG:
        def __init__(self, keys): self.pa = {k: [] for k in keys}
    cg = _CG(v_type.keys())

    # Hyperparams must match training run (as in your Colab)
    hyper = {
        'pipeline': 'gan', 'mode': 'sampling', 'transform': 'mnist',
        'lr': 0.0001, 'alpha': 0.99, 'data-bs': 128, 'ncm-bs': 128, 'grad-acc': 1,
        'max-epochs': 1000, 'patience': 1000,
        'h-layers': 3, 'h-size': 2, 'scale-h-size': True, 'feature-maps': 64,
        'u-size': 1, 'scale-u-size': True, 'neural-pu': False,
        'batch-norm': True, 'gan-mode': 'wgan', 'gan-arch': 'biggan',
        'disc-type': 'biggan', 'disc-lr': 0.0002, 'disc-h-layers': 2, 'disc-h-size': -1,
        'd-iters': 1, 'grad-clamp': 0.01, 'gp-weight': 10.0, 'gp-one-side': False,
        'img-size': img_size,
        'repr': 'auto_enc_conditional', 'rep-size': 64, 'rep-type': 'real',
        'rep-image-only': True, 'rep-lr': 0.0001, 'rep-bs': 128, 'rep-grad-acc': 1,
        'rep-h-layers': 3, 'rep-h-size': 128, 'rep-feature-maps': 64,
        'rep-class-lambda': 0.1, 'rep-temperature': 1.0, 'rep-contrast-lambda': 0.1,
        'identify': False, 'normalize': True, 'id-reruns': 1,
        'max-lambda': 0.01, 'min-lambda': 0.0001, 'use-tau': False,
        'img-query': True, 'verbose': False,
    }

    rep = RepresentationalNN(cg=cg, v_size=v_size, v_type=v_type, default_v_size=1, hyperparams=hyper)
    image_key = next(k for k, t in v_type.items() if t == sdt.IMAGE)
    return rep, DG, image_key

# -----------------------
# 3) Load weights from .ckpt
# -----------------------
def load_checkpoint_into_rep(rep, ckpt_path: str):
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    prefix = "model." if any(k.startswith("model.") for k in sd) else ""
    # drop parent_heads.* to avoid size mismatch (we don't need them)
    rep_sd = {
        k[len(prefix):]: v
        for k, v in sd.items()
        if k.startswith(prefix) and not k.startswith(prefix + "parent_heads.")
    }
    missing, unexpected = rep.load_state_dict(rep_sd, strict=False)
    print(f"Loaded (strict=False). Missing={len(missing)} Unexpected={len(unexpected)}")
    if missing:
        print("Examples missing:", [m for m in missing[:3]], "..." if len(missing) > 3 else "")
    if unexpected:
        print("Examples unexpected:", [u for u in unexpected[:3]], "..." if len(unexpected) > 3 else "")
    rep.eval()
    return rep

# -----------------------
# 4) Adapter module
# -----------------------
class ImageEncoderAdapter(torch.nn.Module):
    def __init__(self, rep, image_key: str):
        super().__init__()
        self.rep = rep
        self.image_key = image_key
    def forward(self, x):
        out = self.rep.encode({self.image_key: x})
        return out[self.image_key]  # [B,64]

# -----------------------
# 5) Main
# -----------------------
def main():
    args = parse_args()

    # make sure output dirs exist
    for p in [args.save_adapter, args.save_torchscript]:
        out_dir = os.path.dirname(os.path.abspath(p)) or "."
        os.makedirs(out_dir, exist_ok=True)

    # repo -> importable + MNIST path
    make_repo_importable(args.repo)

    # build + load
    rep, DG, image_key = build_representation_model(args.img_size)
    load_checkpoint_into_rep(rep, args.ckpt)

    adapter = ImageEncoderAdapter(rep, image_key).eval()
    torch.save(adapter.state_dict(), args.save_adapter)
    print(f"✅ Saved encoder adapter → {args.save_adapter}")

    # TorchScript
    example = torch.randn(1, 3, args.img_size, args.img_size)
    traced = torch.jit.trace(adapter, example)
    traced.save(args.save_torchscript)
    print(f"✅ Saved TorchScript adapter → {args.save_torchscript}")

if __name__ == "__main__":
    main()
