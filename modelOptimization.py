#AUTOMATING OPTIMIZATION


            


tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))



# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0



dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))


            model = Sequential()
            model.add(  Conv2D(64, (3,3), input_shape = X.shape[1:])  )
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size = (2,2)))
            
            #CONV LAYER (Minus 1 because there is one Conv Layer Above)
            for l in range(conv_layer-1):

                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size = (2,2)))
            
            model.add(Flatten())
            
            #DENSE LAYER
            for l in range(dense_layer):
           

                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss="binary_crossentropy", optimizer="adam",metrics = ['accuracy'])


            model.fit(X, y, batch_size=32,epochs = 10, validation_split=0.3, callbacks= [tensorboard])


