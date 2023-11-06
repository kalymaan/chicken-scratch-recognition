import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.layers import StringLookup

# Dependencies and Functions
###############################################################

# constants

batch_size = 64
characters= ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
AUTOTUNE = tf.data.AUTOTUNE
image_width, image_height = 128, 32
max_len = 21
padding_token = 99


# CTCLayer
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred
    
# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
untokenize = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


# distortion_free_resize function
def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image, img_size=(image_width, image_height)):
    #image = tf.io.read_file(image_path)
    #image = tf.image.decode_png(image, 1)
    image = tf.convert_to_tensor(image)
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def prepare_dataset(pil_images, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((pil_images, labels))

    def process_images_labels(pil_image, label):
        image = preprocess_image(pil_image)
        label = vectorize_label(label)
        return {"image": image, "label": label}

    def vectorize_label(label):
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = max_len - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
        return label

    dataset = dataset.map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    ).batch(batch_size).cache().prefetch(AUTOTUNE)

    return dataset

def decode_batch_predictions(pred, max_len = 21):
    '''
    Converts the model.predict() sequences into meaningful text sequences
    *REQUIRES* the 'untokenize' StringLookup() object created earlier in the notebook
    
    Input: pred - predictions from the model
           max_len - max sequence length used when creating the tokenize/untokenize objects - default is 21
    
    Output:  returns a list of the decoded text for each input sequence in the batch
    
    '''
    
    # Decode the sequences from model.predict()
    results = keras.backend.ctc_decode(
        pred, 
        input_length = np.ones(pred.shape[0]) * pred.shape[1], # a NumPy array of ones multiplied by the length of the predictions 
        greedy=True)[0][0][:, :max_len] # ensures that the sequences don't exceed the max_len
    
    
    # Iterate over the decoded results and obtain the text.
    output_text = []
    for res in results:
        
        # remove -1 (blank label) from the decoded result
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        
        # uses the untokenize object(from where we tokenized the labels) to turn the integers into strings
        res = tf.strings.reduce_join(untokenize(res)).numpy().decode("utf-8")
        
        output_text.append(res)
        
    return output_text

def pil_predict(model,image, input_label = None, batch_size=64):
    
    if input_label == None:
        input_label = ''
    
    
    dataset = prepare_dataset([image], [input_label], batch_size)
    
    for data in dataset.take(1):
        images = data['image']

        preds = model.predict(images)
        pred_texts = decode_batch_predictions(preds)


    return pred_texts[0]

# Loading the model
###############################################################
loaded_model= keras.models.load_model("../models_pickles/prediction_model_2.keras", custom_objects={
  'CTCLayer': CTCLayer
})

# Streamlit app
###############################################################
st.title("Handwritten Word Recognition")

uploaded_image = st.file_uploader("Upload a handwritten word image (PNG)", type=["png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Recognize Word"):
        # Open and process the uploaded image
        image = Image.open(uploaded_image).convert('L')
        array_image = np.asarray(image)

        # Call the prediction function
        predicted_text = pil_predict(model=loaded_model,image=array_image)

        # Display the predicted text
        st.write("Predicted Word:", predicted_text)