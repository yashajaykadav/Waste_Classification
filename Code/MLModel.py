import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

Test_dir = 'C:/Users/YASH KADAV/Desktop/ORE/ORE/Waste/Test'
Train_dir = 'C:/Users/YASH KADAV/Desktop/ORE/ORE/Waste/Train'

Img_size=(128,128)
Batch_size=32

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(Train_dir,target_size=Img_size,
batch_size=Batch_size,class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory(Test_dir,batch_size = Batch_size,
target_size = Img_size,class_mode = 'categorical')

def Build_model():
    model = Sequential([
        
        Conv2D(32,(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),

        
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),

        Flatten(),

        Dense(128,activation='relu'),
        Dropout(0.5),
        Dense(3,activation='softmax')

    ])

    model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = Build_model()
model.summary()


EarlyStopping = EarlyStopping( monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint('Best_Model.h5',save_best_only=True)

History = model.fit(train_generator,epochs=15,validation_data=test_generator,callbacks=[EarlyStopping,checkpoint])

model.save('Model.h5')

plt.plot(History.history['accuracy'],label = 'Training Accuracy')
plt.plot(History.history['val_accuracy'],label = 'Validation Accuracy')
plt.legend()
plt.show()