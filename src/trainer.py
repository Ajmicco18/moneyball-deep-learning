import torch


def train_model(model, criterion, optimizer, train_loader, val_loader, n_epochs, metric):

    history = {"train_losses": [],
               "train_metrics": [], "validation_metrics": []}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        metric.reset()

        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)

        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["validation_metrics"].append(
            evaluate_model(model, val_loader, metric).item())

        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['validation_metrics'][-1]:.4f}"
              )

    return history


def evaluate_model(model, val_loader, metric):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)

    return metric.compute()
