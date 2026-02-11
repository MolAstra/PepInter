import litserve as ls
import lightning as L

from src.lightning import KPGTLit
from src.models.tokenization_kpgt import (
    KPGTInferFeaturizer,
    InferDataset,
    collate_fn_infer,
)
from torch.utils.data import DataLoader
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import Depends, HTTPException


class KPGTAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.model = KPGTLit.load_from_checkpoint(
            "./last.ckpt",
            map_location=device,
            load_from_kpgt=False,
        )
        self.model.eval()

        self.trainer = L.Trainer(
            accelerator="gpu" if "cuda" in device else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )

        self.featurizer = KPGTInferFeaturizer()

    def encode_response(self, outputs):
        # outputs: List[Dict]  (one per batch)
        cids_all = []
        abs_all = []
        emi_all = []
        qy_all = []
        loge_all = []

        for o in outputs:
            # o: dict returned by predict_step
            cids_all.extend(o["cid"])  # cid 是 list[str]
            abs_all.extend(o["abs"].view(-1).tolist())
            emi_all.extend(o["emi"].view(-1).tolist())
            qy_all.extend(o["qy"].view(-1).tolist())
            loge_all.extend(o["loge"].view(-1).tolist())

        return [
            {
                "cid": cids_all[i],
                "absorption": abs_all[i],
                "emission": emi_all[i],
                "quantum_yield": qy_all[i],
                "log_molar_absorptivity": loge_all[i],
            }
            for i in range(len(cids_all))
        ]

    def decode_request(self, request):
        return request["inputs"]

    def predict(self, inputs):
        samples = self.featurizer(inputs)  # ✅ 不叫 batch，只是样本列表
        ds = InferDataset(samples)
        dl = DataLoader(
            ds, batch_size=64, shuffle=False, collate_fn=collate_fn_infer, num_workers=1
        )

        preds = self.trainer.predict(self.model, dataloaders=dl)
        return preds

    def authorize(self, auth: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if auth.scheme != "Bearer" or auth.credentials != "88688844":
            raise HTTPException(status_code=401, detail="Bad token")


if __name__ == "__main__":
    api = KPGTAPI()
    # Only run on GPU:0
    server = ls.LitServer(api, accelerator="gpu", devices=[0])
    server.run(port=4000, host="127.0.0.1")
