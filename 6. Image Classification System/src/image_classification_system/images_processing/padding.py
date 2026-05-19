from PIL import Image, ImageOps


class PadToSquare:
    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        max_side = max(width, height)

        pad_left = (max_side - width) // 2
        pad_top = (max_side - height) // 2
        pad_right = max_side - width - pad_left
        pad_bottom = max_side - height - pad_top

        return ImageOps.expand(
            img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0
        )
