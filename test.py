
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

model=load_model('predd.h5')
class_2_indices = {'melanoma': 0, 'nevus': 1, 'seborrheic_keratoses': 2}
indices_2_class = {v: k for k, v in class_2_indices.items()}

img = load_img('download.png', target_size=(224,224))
img = img_to_array(img)/255
img_expand = np.expand_dims(img, axis=0)

# Make a prediction
prediction = model.predict(img_expand, steps=1)
image_idx = np.argmax(prediction[0])
prediction_string = indices_2_class[image_idx]
print("Prediction: {}".format(prediction_string))

# Get the real label's name

# Plot predictions
title = "Prediction: {}".format(prediction_string)

plt.imshow(img)
plt.title(title)
pred_df = pd.DataFrame({'Cancer type':['melanoma', 'nevus', 'seborrheic keratosis'], 'val':prediction[0]})
ax = pred_df.plot.barh(x='Cancer type', y='val', title="Predictions", grid=True)
plt.show()
