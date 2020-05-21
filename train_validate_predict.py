import numpy as np
import torch
import tqdm

from dataset import CROP_SIZE, NUM_PTS, restore_landmarks_batch

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model, loader, loss_fn, optimizer, device):
    model.train()

    train_loss = []
    B = loader.batch_size

    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)
        assert images.shape == (B, 3, CROP_SIZE, CROP_SIZE)

        landmarks = batch["landmarks"]
        assert landmarks.shape == (B, NUM_PTS * 2)

        pred_landmarks = model(images).cpu()
        assert pred_landmarks.shape == landmarks.shape

        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()

    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        cur_B = images.shape[0]
        assert images.shape == (cur_B, 3, CROP_SIZE, CROP_SIZE)

        landmarks = batch["landmarks"]
        assert landmarks.shape == (cur_B, NUM_PTS * 2)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
            assert pred_landmarks.shape == landmarks.shape

        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

    return np.mean(val_loss)


def predict(model, loader, device):
    model.eval()

    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(
        tqdm.tqdm(loader, total=len(loader), desc="test prediction...")
    ):
        images = batch["image"].to(device)
        cur_B = images.shape[0]
        assert images.shape == (cur_B, 3, CROP_SIZE, CROP_SIZE)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
            assert pred_landmarks.shape == (cur_B, NUM_PTS * 2)

        pred_shape = (cur_B, NUM_PTS, 2)
        pred_landmarks = pred_landmarks.numpy().reshape(pred_shape)

        fs = batch["scale_coef"].numpy()
        margins_x = batch["crop_margin_x"].numpy()
        margins_y = batch["crop_margin_y"].numpy()
        assert fs.shape == margins_x.shape == margins_y.shape == (cur_B,)

        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)
        assert prediction.shape == pred_shape
        predictions[i * loader.batch_size : (i + 1) * loader.batch_size] = prediction

    return predictions
