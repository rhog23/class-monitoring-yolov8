import tensorflow as tf
import PIL
import matplotlib.pyplot as plt
import numpy as np

upscale_factor = 1

HR_image = PIL.Image.open(
    "data/20240321213000-20240321215500/Camera14_MESIN-04_MESIN-04_20240321213000_20240321215500_2937239.jpg"
)

model = tf.keras.models.load_model("models/enhancer.h5")

LR_image = HR_image.resize(
    (
        HR_image.size[0] // upscale_factor,  # Donwscaling to LR occurs here.
        HR_image.size[1] // upscale_factor,
    ),
    PIL.Image.BICUBIC,
)

ycbcr = LR_image.convert("YCbCr")  # Here, we converted the image to YCbCr because we
y, cb, cr = ycbcr.split()  # need the Y channel for our model. We did not convert
y = tf.keras.preprocessing.image.img_to_array(
    y
)  # it to YUV because we can't in PIL and Y in YCbCr is
y = y.astype("float32") / 255.0  # the same as in YUV.

input = y.reshape(
    1, y.shape[0], y.shape[1], y.shape[2]
)  # (Batch, Width, Height, Channels)

output = model.predict(input)
output = output[0]  # (Width, Height, Channels)
output *= 255.0
output = output.clip(0, 255)
output = output.reshape((output.shape[0], output.shape[1]))  # (Width, Height)
output = PIL.Image.fromarray(np.uint8(output))
output = output.resize(
    HR_image.size, PIL.Image.Resampling.NEAREST
)  # This step is necessary to
# fill missing pixels from
# the prediction process.

cb = cb.resize(output.size, PIL.Image.Resampling.BICUBIC)
cr = cr.resize(output.size, PIL.Image.Resampling.BICUBIC)

ER_image = PIL.Image.merge("YCbCr", (output, cb, cr))
ER_image = ER_image.convert("RGB")

LR_image = LR_image.resize(
    ER_image.size, PIL.Image.Resampling.BICUBIC
)  # Resizing LR for plotting.
plt.imshow(ER_image)
plt.show()
