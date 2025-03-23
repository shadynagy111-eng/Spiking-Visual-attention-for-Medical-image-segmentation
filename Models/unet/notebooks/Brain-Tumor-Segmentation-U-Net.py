import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K


images_dir = 'E:/Code/Spiking-Visual-attention-for-Medical-image-segmentation/datasets/Brain Tumor Segmentation/images'
val_images_dir = 'E:/Code/Spiking-Visual-attention-for-Medical-image-segmentation/datasets/Brain Tumor Segmentation/val_images'
test_images_dir = 'E:/Code/Spiking-Visual-attention-for-Medical-image-segmentation/datasets/Brain Tumor Segmentation/test_images'

masks_dir = 'E:/Code/Spiking-Visual-attention-for-Medical-image-segmentation/datasets/Brain Tumor Segmentation/masks'
val_masks_dir = 'E:/Code/Spiking-Visual-attention-for-Medical-image-segmentation/datasets/Brain Tumor Segmentation/valid_masks'
test_masks_dir = 'E:/Code/Spiking-Visual-attention-for-Medical-image-segmentation/datasets/Brain Tumor Segmentation/test_masks'


# In[17]:


train_coco = COCO(train_annotation_file)
val_coco = COCO(val_annotation_file)
test_coco = COCO(test_annotation_file)


############################################################################################################

# In[18]:


def load_image_and_mask(coco, image_dir, image_id):
    image_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(image_dir, image_info['file_name'])
    image = Image.open(image_path)
    image = np.array(image)

    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros((image_info['height'], image_info['width']))
    for ann in anns:
        mask = np.maximum(mask, coco.annToMask(ann))

    return image, mask

############################################################################################################

# In[19]:


def create_tf_dataset(coco, image_dir, image_ids):
    def generator():
        for image_id in image_ids:
            yield load_image_and_mask(coco, image_dir, image_id)

    return tf.data.Dataset.from_generator(generator,
                                          output_signature=(tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8), 
                                                            tf.TensorSpec(shape=(None, None), dtype=tf.uint8)))

train_dataset = create_tf_dataset(train_coco, train_dir, train_coco.getImgIds())
val_dataset = create_tf_dataset(val_coco, val_dir, val_coco.getImgIds())
test_dataset = create_tf_dataset(test_coco, test_dir, test_coco.getImgIds())

############################################################################################################
# In[21]:


def preprocess(image, mask):

    image = tf.image.resize(image, (256, 256))

    mask = tf.expand_dims(mask, axis=-1)  
    mask = tf.image.resize(mask, (256, 256))

    image = tf.cast(image, tf.float32) / 255.0

    return image, mask

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

############################################################################################################
# In[22]:


def visualize_dataset(dataset, num_samples=5):
    for i, (image, mask) in enumerate(dataset.take(num_samples)):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image.numpy())
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask.numpy().squeeze(), cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        plt.show()

visualize_dataset(train_dataset)
visualize_dataset(val_dataset)


# In[24]:


def downsampling_block(x, n_filters):
    c = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(x)
    c = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(c)
    p = layers.MaxPooling2D((2, 2))(c)
    return c, p

def upsampling_block(x, skip_connection, n_filters):
    u = layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(x)
    u = layers.concatenate([u, skip_connection])
    c = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(u)
    c = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(c)
    return c

def unet_model(input_size=(256, 256, 3), n_filters=32, n_classes=1):
    inputs = layers.Input(input_size)

    dblock1, p1 = downsampling_block(inputs, n_filters)
    dblock2, p2 = downsampling_block(p1, n_filters * 2)
    dblock3, p3 = downsampling_block(p2, n_filters * 4)
    dblock4, p4 = downsampling_block(p3, n_filters * 8)

    bottleneck = layers.Conv2D(n_filters * 16, (3, 3), activation='relu', padding='same')(p4)
    bottleneck = layers.Conv2D(n_filters * 16, (3, 3), activation='relu', padding='same')(bottleneck)

    u6 = upsampling_block(bottleneck, dblock4, n_filters * 8)
    u7 = upsampling_block(u6, dblock3, n_filters * 4)
    u8 = upsampling_block(u7, dblock2, n_filters * 2)
    u9 = upsampling_block(u8, dblock1, n_filters)

    outputs = layers.Conv2D(n_classes, (1, 1), activation='sigmoid' if n_classes == 1 else 'softmax')(u9)

    model = models.Model(inputs, outputs)

    return model

model = unet_model(input_size=(256, 256, 3), n_filters=32)


############################################################################################################
# In[25]:


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.6 * dice + 0.4 * bce

metrics = ["accuracy", dice_coef]

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss=combined_loss, metrics=metrics)
model.summary()


############################################################################################################
# In[26]:


BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

steps_per_epoch = len(train_coco.getImgIds()) // BATCH_SIZE
validation_steps = len(val_coco.getImgIds()) // BATCH_SIZE
test_steps = len(test_coco.getImgIds()) // BATCH_SIZE


############################################################################################################
# In[27]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)


############################################################################################################

def plot_loss(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracy(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_loss(history)
plot_accuracy(history)


# In[29]:


test_loss, test_accuracy, test_coef = model.evaluate(test_dataset, steps=test_steps)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test Dice Coefficient: {test_coef}")


# In[30]:


def visualize_predictions(model, dataset, num_samples=5, threshold=0.5):

    random_batch = random.choice(list(dataset))
    images, masks = random_batch

    indices = random.sample(range(len(images)), min(num_samples, len(images)))

    predictions = model.predict(images)

    binary_predictions = (predictions > threshold).astype('uint8')

    for i in indices:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(images[i])
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(masks[i].numpy().squeeze(), cmap='gray')
        plt.title('True Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(binary_predictions[i].squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.show()


# In[33]:


import random


# In[34]:


visualize_predictions(model, test_dataset, num_samples=10)


# In[35]:


from tensorflow.keras import layers, models
from tensorflow.keras.layers import MultiHeadAttention

def downsampling_block(x, n_filters):
    c = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(x)
    c = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(c)
    p = layers.MaxPooling2D((2, 2))(c)
    return c, p

def upsampling_block(x, skip_connection, n_filters):
    u = layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(x)
    u = layers.concatenate([u, skip_connection])
    c = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(u)
    c = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(c)
    return c

def multihead_attention_block(query, key, value, num_heads, key_dim):
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(query, key, value)
    return attention

def unet_model_with_multihead_attention(input_size=(256, 256, 3), n_filters=32, n_classes=1, num_heads=4, key_dim=64):
    inputs = layers.Input(input_size)

    dblock1, p1 = downsampling_block(inputs, n_filters)
    dblock2, p2 = downsampling_block(p1, n_filters * 2)
    dblock3, p3 = downsampling_block(p2, n_filters * 4)
    dblock4, p4 = downsampling_block(p3, n_filters * 8)


    bottleneck = layers.Conv2D(n_filters * 16, (3, 3), activation='relu', padding='same')(p4)
    bottleneck = layers.Conv2D(n_filters * 16, (3, 3), activation='relu', padding='same')(bottleneck)

    attention = multihead_attention_block(bottleneck, bottleneck, bottleneck, num_heads, key_dim)
    attention = layers.LayerNormalization()(attention) 

    u6 = upsampling_block(attention, dblock4, n_filters * 8)
    u7 = upsampling_block(u6, dblock3, n_filters * 4)
    u8 = upsampling_block(u7, dblock2, n_filters * 2)
    u9 = upsampling_block(u8, dblock1, n_filters)

    outputs = layers.Conv2D(n_classes, (1, 1), activation='sigmoid' if n_classes == 1 else 'softmax')(u9)

    model = models.Model(inputs, outputs)

    return model

model_with_mha = unet_model_with_multihead_attention(input_size=(256, 256, 3), n_filters=32)

############################################################################################################

from tensorflow.keras import backend as K
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.6 * dice + 0.4 * bce

metrics = ["accuracy", dice_coef]

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model_with_mha.compile(optimizer=optimizer, loss=combined_loss, metrics=metrics)
model_with_mha.summary()


# In[39]:


history = model_with_mha.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)


# In[40]:


plot_loss(history)
plot_accuracy(history)


# In[41]:


test_loss, test_accuracy, test_coef = model_with_mha.evaluate(test_dataset, steps=test_steps)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test Dice Coefficient: {test_coef}")

