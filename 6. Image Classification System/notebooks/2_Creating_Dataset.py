from image_classification_system.models.alex_net import AlexNet
import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from root_folders import CACHE_DIR, RICE_TRAIN_DIR, RICE_TEST_DIR,RICE_VAL_DIR

    import torch

    from image_classification_system.images_processing import (
        create_train_dataloader_pipeline,
        create_test_dataloader_pipeline,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    train_path = RICE_TRAIN_DIR
    test_path = RICE_TEST_DIR
    val_path = RICE_VAL_DIR
    return (
        create_test_dataloader_pipeline,
        create_train_dataloader_pipeline,
        device,
        test_path,
        torch,
        train_path,
        val_path,
    )


@app.cell
def _(
    create_test_dataloader_pipeline,
    create_train_dataloader_pipeline,
    test_path,
    train_path,
    val_path,
):
    dataloader, mean, std = create_train_dataloader_pipeline(
        train_path, batch_size=128, cache_dir=CACHE_DIR
    )

    test_dataloader = create_test_dataloader_pipeline(test_path, mean, std=std)
    val_dataloader = create_test_dataloader_pipeline(val_path, mean, std=std)

    return dataloader, test_dataloader, val_dataloader


@app.cell
def _(dataloader, device, test_dataloader, torch, val_dataloader):
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    N_EPOCHS = 1000
    VAL_EVERY = 1   # run validation every N epochs
    TEST_EVERY = 10  # print test results every N epochs
    PATIENCE = 100  # validation rounds without improvement → stop (= PATIENCE * VAL_EVERY epochs)

    model = AlexNet(num_classes=5)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    scaler = torch.amp.GradScaler(device) if device == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    stopped_epoch = N_EPOCHS

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[metrics]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        epoch_task = progress.add_task("[bold green]Epochs", total=N_EPOCHS, metrics="")

        for epoch in range(N_EPOCHS):
            # ── Train ──────────────────────────────────────────────────────────
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            train_task = progress.add_task(
                f"[yellow]Epoch {epoch + 1} — Train", total=len(dataloader), metrics=""
            )
            for image, label in dataloader:
                image, label = image.to(device), label.to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, enabled=device in ("cuda", "mps")):
                    output = model(image)
                    loss = criterion(output, label)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item() * image.size(0)
                train_correct += (output.argmax(1) == label).sum().item()
                train_total += image.size(0)
                progress.advance(train_task)
            progress.remove_task(train_task)

            # ── Validation ─────────────────────────────────────────────────────
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            if (epoch + 1) % VAL_EVERY == 0:
                val_task = progress.add_task(
                    f"[blue]Epoch {epoch + 1} — Val", total=len(val_dataloader), metrics=""
                )
                with torch.no_grad():
                    for image, label in val_dataloader:
                        image, label = image.to(device), label.to(device)
                        with torch.autocast(
                            device_type=device, enabled=device in ("cuda", "mps")
                        ):
                            output = model(image)
                            loss = criterion(output, label)

                        val_loss += loss.item() * image.size(0)
                        val_correct += (output.argmax(1) == label).sum().item()
                        val_total += image.size(0)
                        progress.advance(val_task)
                progress.remove_task(val_task)

                # ── Metrics ────────────────────────────────────────────────────
                epoch_train_loss = train_loss / train_total
                epoch_val_loss = val_loss / val_total
                epoch_train_acc = train_correct / train_total
                epoch_val_acc = val_correct / val_total

                history["train_loss"].append(epoch_train_loss)
                history["val_loss"].append(epoch_val_loss)
                history["train_acc"].append(epoch_train_acc)
                history["val_acc"].append(epoch_val_acc)

                scheduler.step(epoch_val_loss)

                progress.update(
                    epoch_task,
                    metrics=f"loss={epoch_val_loss:.4f} acc={epoch_val_acc:.3f}",
                )

                # ── Early stopping ─────────────────────────────────────────────
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        stopped_epoch = epoch + 1
                        break

            # ── Test snapshot ───────────────────────────────────────────────
            if (epoch + 1) % TEST_EVERY == 0:
                model.eval()
                _class_correct = [0, 0]
                _class_total = [0, 0]
                _test_correct = _test_total = 0

                with torch.no_grad():
                    for _img, _lbl in test_dataloader:
                        _img, _lbl = _img.to(device), _lbl.to(device)
                        with torch.autocast(device_type=device, enabled=device in ("cuda", "mps")):
                            _out = model(_img)
                        _preds = _out.argmax(1)
                        for _p, _g in zip(_preds, _lbl):
                            _class_correct[_g] += (_p == _g).item()
                            _class_total[_g] += 1
                        _test_correct += (_preds == _lbl).sum().item()
                        _test_total += _lbl.size(0)

                _table = Table(title=f"Test — Epoch {epoch + 1}", show_header=True)
                _table.add_column("Class", style="bold")
                _table.add_column("Correct", justify="right")
                _table.add_column("Total", justify="right")
                _table.add_column("Accuracy", justify="right", style="cyan")
                for _i, _name in enumerate(["NORMAL", "PNEUMONIA"]):
                    _acc = _class_correct[_i] / _class_total[_i] if _class_total[_i] else 0.0
                    _table.add_row(_name, str(_class_correct[_i]), str(_class_total[_i]), f"{_acc:.3f}")
                _overall = _test_correct / _test_total if _test_total else 0.0
                _table.add_section()
                _table.add_row(
                    "[bold green]Overall", str(_test_correct), str(_test_total),
                    f"[bold green]{_overall:.3f}",
                )
                progress.console.print(_table)

            progress.advance(epoch_task)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history, stopped_epoch


@app.cell
def _(device, model, test_dataloader, torch):
    from rich.console import Console
    from rich.table import Table as _Table

    console = Console()

    model.eval()
    test_correct, test_total = 0, 0
    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():
        for _image, _label in test_dataloader:
            _image, _label = _image.to(device), _label.to(device)
            with torch.autocast(device_type=device, enabled=device in ("cuda", "mps")):
                _output = model(_image)
            _preds = _output.argmax(1)
            for _pred, _gt in zip(_preds, _label):
                class_correct[_gt] += (_pred == _gt).item()
                class_total[_gt] += 1
            test_correct += (_preds == _label).sum().item()
            test_total += _label.size(0)

    class_names = ["NORMAL", "PNEUMONIA"]

    table = _Table(title="Test Set Results", show_header=True)
    table.add_column("Class", style="bold")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right", style="cyan")

    for i, name in enumerate(class_names):
        acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        table.add_row(name, str(class_correct[i]), str(class_total[i]), f"{acc:.3f}")

    overall = test_correct / test_total if test_total > 0 else 0.0
    table.add_section()
    table.add_row(
        "[bold green]Overall", str(test_correct), str(test_total), f"[bold green]{overall:.3f}"
    )

    console.print(table)
    return


if __name__ == "__main__":
    app.run()
