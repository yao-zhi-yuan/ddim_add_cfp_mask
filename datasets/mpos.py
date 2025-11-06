# datasets/mpos.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def _pad_to_square(img, fill=0):
    w, h = img.size
    if w == h: return img
    if w > h:
        pad = (0, (w-h)//2, 0, w-h-(w-h)//2)
    else:
        pad = ((h-w)//2, 0, h-w-(h-w)//2, 0)
    return TF.pad(img, pad, fill=fill)


class MPOSPairs(Dataset):
    """
    期望目录：
      root/
        CFP/  xxx.png|jpg
        FFA/  xxx.png|jpg
        CFP_Vessel_Seg/ xxx.png|jpg  <--- (新增) 假设您的分割图在这里
        splits/
          train.txt
          val.txt
    """

    def __init__(self, root, split='train', size=256, ffa_gray=True, random_flip=False):
        self.root = Path(root)
        self.size = size
        self.ffa_gray = ffa_gray
        self.random_flip = random_flip

        with open(self.root / 'splits' / f'{split}.txt') as f:
            self.ids = [ln.strip() for ln in f if ln.strip()]

        self.to_tensor_rgb = T.Compose([
            T.Lambda(_pad_to_square),
            T.Resize(size),
            T.ToTensor(),  # [0,1] (3,H,W)
        ])
        self.to_tensor_gray = T.Compose([
            T.Lambda(_pad_to_square),
            T.Resize(size),
            T.ToTensor(),  # [0,1], shape(1,H,W)
        ])

        # (新增) 我们可以复用 to_tensor_gray 来处理分割图
        self.to_tensor_seg = self.to_tensor_gray

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]  # id_ 示例: '1_DR/11.png'

        cfp_path = self.root / 'CFP' / id_
        ffa_path = self.root / 'FFA' / id_
        # (新增) 定义分割图路径
        seg_path = self.root / 'CFP_Vessel_Seg' / id_

        # --- (新增) 检查分割图是否存在 ---
        if not cfp_path.exists():
            raise FileNotFoundError(f"CFP File not found. Looked for: {cfp_path}")
        if not ffa_path.exists():
            raise FileNotFoundError(f"FFA File not found. Looked for: {ffa_path}")
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation File not found. Looked for: {seg_path}")
        # --- 检查结束 ---

        # 读图
        cfp = Image.open(cfp_path).convert('RGB')
        ffa = Image.open(ffa_path)
        ffa = ffa.convert('L') if self.ffa_gray else ffa.convert('RGB')
        # (新增) 读取分割图 (假设为单通道L)
        seg = Image.open(seg_path).convert('L')

        if self.random_flip:
            if torch.rand(1) < 0.5:
                cfp = TF.hflip(cfp)
                ffa = TF.hflip(ffa)
                seg = TF.hflip(seg)  # (新增) 分割图也要一起翻转

        cfp_t = self.to_tensor_rgb(cfp)  # (3,H,W)
        ffa_t = self.to_tensor_gray(ffa) if self.ffa_gray else self.to_tensor_rgb(ffa)  # (1,H,W)
        seg_t = self.to_tensor_seg(seg)  # (1,H,W)

        # --- (核心修改) ---
        # 将 CFP (3通道) 和 分割图 (1通道) 拼接为新的条件
        cond_combined_t = torch.cat([cfp_t, seg_t], dim=0)  # 拼接后为 (4,H,W)
        # --- (修改结束) ---

        # 多返回原图路径，供可视化用
        meta = {"cond_path": str(cfp_path), "gt_path": str(ffa_path), "id": str(id_)}

        # (修改) 返回合并后的条件
        return cond_combined_t, ffa_t, meta