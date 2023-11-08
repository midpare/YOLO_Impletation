import utils
import generate

import tensorflow as tf

f = open('./VOC2012/ImageSets/Segmentation/train.txt')
train_dataset_list = f.readlines()
f.close()

f = open('./VOC2012/ImageSets/Segmentation/val.txt')
test_dataset_list = f.readlines()
f.close()

train_xml_path = list(map(lambda x: "./VOC2012/Annotations/" + x.replace("\n", ".xml"), train_dataset_list))
test_xml_path = list(map(lambda x: "./VOC2012/Annotations/" + x.replace("\n", ".xml"), test_dataset_list))

classes = utils.get_classes(train_xml_path)

train_image_path, train_labels = utils.get_data(train_xml_path, classes)
test_image_path, test_labels = utils.get_data(test_xml_path, classes)

max_num = len(tf.keras.applications.VGG16(weights='imagenet', include_top=False,  input_shape=(224, 224, 3)).layers) # 레이어 최대 개수

train_ds = tf.data.Dataset.from_generator(
  generate.gen,
  (tf.float32, tf.int32),
  (tf.TensorShape([448, 448, 3]), tf.TensorShape([])),
  args=[train_image_path, train_labels]
)

test_ds = tf.data.Dataset.from_generator(
  generate.gen,
  (tf.float32, tf.int32),
  (tf.TensorShape([448, 448, 3]), tf.TensorShape([])),
  args=[test_image_path, test_labels]
)
YOLO = tf.keras.models.Sequential()

initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=42)
lrelu = tf.keras.layers.LeakyReLU(alpha=0.01)  
l2 = tf.keras.regularizers.l2(5e-4)

YOLO.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), input_shape =(448, 448, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

YOLO.add(tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

YOLO.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

YOLO.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

YOLO.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same'))
YOLO.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

YOLO.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2))

YOLO.add(tf.keras.layers.Flatten())
YOLO.add(tf.keras.layers.Dense(512, kernel_initializer=initializer, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Dense(1024, kernel_initializer=initializer, kernel_regularizer=l2))
YOLO.add(tf.keras.layers.Dropout(0.5))

YOLO.add(tf.keras.layers.Dense(1470, kernel_initializer=initializer, kernel_regularizer=l2)) 
YOLO.add(tf.keras.layers.Reshape((7, 7, 30), name = 'output', dtype='float32'))

YOLO.summary()
BATCH_SIZE = 64
EPOCH = 135

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=AUTOTUNE)


def lr_schedule(epoch, lr):
  if epoch >=0 and epoch < 75 :
    lr = 0.001 + 0.009 * (float(epoch)/(75.0)) 
    return lr
  elif epoch >= 75 and epoch < 105 :
    lr = 0.001
    return lr
  else : 
    lr = 0.0001
    return lr

filename = 'yolo.h5'

checkpoint = tf.keras.callbacks.ModelCheckpoint(
  filename,             
  verbose=1,           
  save_best_only=True  
)

optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.0001, momentum = 0.9)

YOLO.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer=optimizer, run_eagerly=True)

YOLO.fit(
  train_ds,
  batch_size=BATCH_SIZE,
  validation_data = test_ds,
  epochs=EPOCH,
  verbose=1,
  callbacks=[checkpoint, tf.keras.callbacks.LearningRateScheduler(lr_schedule)]
)
