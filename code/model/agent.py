from header import *

class DeepSpeedAgent:
    """
    Wrapper that:
      • freezes everything by default,
      • re-enables only explicitly-listed sub-modules,
      • trains with DeepSpeed,
      • works with the current CLIP/DINO-based PaLaDiN model.
    """

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model

        # -------- 3) DeepSpeed engine --------------------------------
        self._init_deepspeed_engine()

    def _init_deepspeed_engine(self):
        ds_cfg = json.load(open(self.args["ds_config_path"]))
        ds_cfg["scheduler"]["params"]["total_num_steps"] = self.args["total_steps"]
        ds_cfg["scheduler"]["params"]["warmup_num_steps"] = max(
            10, int(self.args["total_steps"] * self.args["warmup_rate"])
        )

        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=[p for p in self.model.parameters() if p.requires_grad],
            config_params=ds_cfg,
            dist_init_required=True,
            args=types.SimpleNamespace(**self.args),
        )


    # ────────────────────────────────────────────────────────────────
    # inference
    # ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        return self.model.generate_one_sample(batch)

    # ────────────────────────────────────────────────────────────────
    # training step
    # ────────────────────────────────────────────────────────────────
    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss, info = self.ds_engine(batch)
        
        # -------- print trainable param names once (e.g., at step 0) --------
        if current_step == 0 and self.args.get("verbose", True):
            trainable_names = [
                name for name, p in self.ds_engine.module.named_parameters() if p.requires_grad
            ]
            total = sum(
                p.numel()
                for _, p in self.ds_engine.module.named_parameters()
                if p.requires_grad
            )
            print("Trainable parameter names:")
            for n in trainable_names:
                print(" -", n)
            print(f"Total trainable params: {total:,}")

        # -------- backward + step --------------------------------
        self.ds_engine.backward(loss)
        self.ds_engine.step()

        if pbar is not None:
            pbar.update(1)

            if current_step % 1000 == 0:
                # Safely get train_stage if present
                if hasattr(self.model, "train_stage"):
                    stage_tag = f"[{self.model.train_stage}] "
                else:
                    stage_tag = ""

                pbar.write(
                    f"{stage_tag}step {current_step} loss: {loss.item():.4f}")
        return loss.item()


    # ────────────────────────────────────────────────────────────────
    # save utilities
    # ────────────────────────────────────────────────────────────────
    def save_model(self, path):
        # Only parameters with requires_grad = True
        to_save = {
            k: v.data.cpu().clone()
            for k, v in self.ds_engine.module.named_parameters()
            if v.requires_grad
        }

        if self.args.get("verbose_save_params", False):
            print("\n Trainable parameters going into pytorch_model.pt")
            print("(name → shape • dtype)")
            for name, tensor in to_save.items():
                print(f"🔸 {name:<55} {tuple(tensor.shape)} • {tensor.dtype}")

        # Save model parameters with step in the filename
        save_path = f"{path}/pytorch_model.pt"
        torch.save(to_save, save_path)

        print(f"[!] model saved in {save_path}")
