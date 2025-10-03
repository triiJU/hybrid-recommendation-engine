import argparse, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .config import Config
from .utils import ensure_dir

class RatingDataset(Dataset):
    def __init__(self, df, user_map, item_map):
        self.u = df.user_id.map(user_map).values
        self.i = df.item_id.map(item_map).values
        self.r = df.rating.values.astype("float32")
    def __len__(self): return len(self.r)
    def __getitem__(self, idx): return int(self.u[idx]), int(self.i[idx]), self.r[idx]

class NeuralCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*2, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        self.user_map = {}
        self.item_map = {}
        
    def forward(self, users, items):
        ue = self.user_emb(users)
        ie = self.item_emb(items)
        x = torch.cat([ue, ie], dim=-1)
        return self.mlp(x).squeeze(-1)
    
    @classmethod
    def load_from_file(cls, path):
        """
        Load a trained NeuralCF model from file.
        
        Args:
            path: Path to the saved model file
            
        Returns:
            Loaded NeuralCF model instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        user_map = checkpoint['user_map']
        item_map = checkpoint['item_map']
        
        model = cls(len(user_map), len(item_map))
        model.load_state_dict(checkpoint['state_dict'])
        model.user_map = user_map
        model.item_map = item_map
        model.eval()
        
        return model
    
    def predict_single(self, user_id, item_id):
        """
        Predict rating for a single user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Float score (0.0 if user or item is unknown)
        """
        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0
        
        u_idx = self.user_map[user_id]
        i_idx = self.item_map[item_id]
        
        with torch.no_grad():
            users_tensor = torch.tensor([u_idx], dtype=torch.long)
            items_tensor = torch.tensor([i_idx], dtype=torch.long)
            score = self.forward(users_tensor, items_tensor).item()
        
        return float(score)

def train_loop(model, loader, val_loader, epochs, device, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best = 1e9
    for ep in range(epochs):
        model.train(); tr_loss=0
        for u,i,r in loader:
            u,i,r = u.to(device), i.to(device), r.to(device)
            opt.zero_grad()
            pred = model(u,i)
            loss = loss_fn(pred, r)
            loss.backward()
            opt.step()
            tr_loss += loss.item()*len(r)
        tr_loss /= len(loader.dataset)

        model.eval(); vl=0
        with torch.no_grad():
            for u,i,r in val_loader:
                u,i,r = u.to(device), i.to(device), r.to(device)
                pred = model(u,i)
                vl += loss_fn(pred,r).item()*len(r)
        vl /= len(val_loader.dataset)
        print(f"[Epoch {ep+1}] train={tr_loss:.4f} val={vl:.4f}")
        if vl < best:
            best=vl
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/processed/train.csv")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    args = parser.parse_args()

    df = pd.read_csv(args.train_path)
    users = sorted(df.user_id.unique()); items = sorted(df.item_id.unique())
    user_map = {u:i for i,u in enumerate(users)}
    item_map = {it:i for i,it in enumerate(items)}

    val = df.sample(frac=0.1, random_state=Config.SEED)
    train_df = df.drop(val.index)

    train_ds = RatingDataset(train_df, user_map, item_map)
    val_ds = RatingDataset(val, user_map, item_map)

    device = Config.DEVICE if torch.cuda.is_available() else "cpu"
    model = NeuralCF(len(user_map), len(item_map), emb_dim=Config.EMBEDDING_DIM).to(device)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE)

    ensure_dir(Config.SAVE_DIR)
    model = train_loop(model, train_loader, val_loader, args.epochs, device, Config.LR)
    torch.save({"state_dict": model.state_dict(),
                "user_map": user_map,
                "item_map": item_map},
               f"{Config.SAVE_DIR}/neural_cf.pt")
    print("[INFO] Saved neural CF model.")