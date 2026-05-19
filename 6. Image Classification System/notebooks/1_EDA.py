import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    from root_folders import CHEST_XRAY_DIR

    sub_dirs = os.listdir(CHEST_XRAY_DIR)

    def total_images(dir):
        cases = ["NORMAL", "PNEUMONIA"]
        return sum(len(os.listdir(dir / case)) for case in cases)

    def image_per_type(dir):
        cases = ["NORMAL", "PNEUMONIA"]
        return {case: len(os.listdir(dir / case)) for case in cases}

    size_dataset = {s_d: total_images(CHEST_XRAY_DIR / s_d) for s_d in sub_dirs}
    size_for_image_type = {
        s_d: image_per_type(CHEST_XRAY_DIR / s_d) for s_d in sub_dirs
    }
    return (size_for_image_type,)


@app.cell
def _(size_for_image_type):
    size_for_image_type
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
