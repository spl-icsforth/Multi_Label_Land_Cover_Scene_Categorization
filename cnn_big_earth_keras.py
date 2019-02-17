######################################################
# Training & Testing of the CNN for the multilabel   #
# classification of the provided dataset (BigEarth). #
######################################################


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization

np.set_printoptions(threshold=np.inf) # PRINT THE FULL NUMPY ARRAYS, NO MATTER WHAT THEIR SIZE IS


#############################################################################################
# VARIABLES IN FULL-CAPITAL LETTERS, NEED TO BE ADJUSTED ACCORDINGLY IN EVERY EXECUTION     #
# OF THIS PROGRAM, REGARDING WHAT MODEL WE WANT TO PRODUCE, WITH WHICH HYPERPARAMETERS ETC. #
#############################################################################################

TESTING_DATA_NUM   = 500  # How many of the samples will be used for testing.

VALIDATION_PORTION = 0    # This portion will determine the percentage split of the train
                          # set, into Train Data and Validation Data.

SOFTMAX_THRESHOLD  = 0.1   # The threshold that defines when a prediction is considered positive (If softmax is used for the probabilities).
SIGMOID_THRESHOLD  = 0.45  # The threshold that defines when a prediction is considered positive (If sigmoid is used for the probabilities).

INPUT_SHAPE        = (247,242,3) # The dimensions of each data sample.

OBSERVATIONS_FILE  = 'BigEarthImages.npy' # The file containing the data samples.
LABELS_FILE        = 'BigEarthLabels.npy' # The file containing the labels.
ASSOC_TYPE_FILE    = 'Assoc_Types.npy'    # The file containing the type of the image  (e.g. agricultural)
ASSOC_IND_FILE     = 'Assoc_Index.npy'    # The file containing the index of the image (e.g. agricultural34)

# The possible choices for the loss function.
LOSS_FUNCTION_CHOICES = [ 'mean_squared_error',        'poisson',
                          'categorical_crossentropy',  'binary_crossentropy',  'kullback_leibler_divergence' ] # more choices can be added.

OPTIMIZER_CHOICES = [ 'adagrad', 'adadelta', 'adam', 'adamax', 'Nadam' ] # The possible choices for the optimizer (more choices can be added).

# The below parameters will be combined column-wise, for the training of the different CNNs.
CONV_LAYERS       = [  3,      3,      3,      3,     3,     3    ] # Depth of the Network.
RELU_DENSE_LAYERS = [  1,      2,      1,      2,     1,     2    ] # How many ReLU Dense Layers will be added. A Softmax/Sigmoid Dense Layer will always be added in the end of the ReLU Dense Layers.
LOSS_FUNCTION     = [  3,      3,      1,      1,     4,     4    ] # The chosen loss function from the list above.
OPTIMIZER         = [  0,      0,      0,      0,     0,     0    ] # The chosen optimizer from the list above.
DROPOUT           = [  0.25,   0.25,   0.25,   0.25,  0.25,  0.25 ] # If 0, no Dropout will  be used.
DROP_ALL_LAYERS   = [  0,      0,      0,      0,     0,     0    ] # If 0, then Dropout will be applied only on the Fully-Connected Layer, else on ALL applicable layers.
BATCH_NORM        = [  0,      0,      0,      0,     0,     0    ] # If non-zero, adds Batch Normalization.
BATCH_SIZE        = [  10,     10,     10,     10,    10,    10   ] # The size of each batch.
EPOCHS            = [  300,    300,    300,    300,   300,   300  ] # Train for this number of epochs.
INIT_EPOCH        = [  0,      0,      0,      0,     0,     0    ] # The number of epochs that the network has already been trained for. (0 if new network).
NEW_MODEL         = [  1,      1,      1,      1,     1,     1    ] # If 0, the model specified in MODEL_LOAD_PATH will be loaded.
SAVE_MODEL        = [  1,      1,      1,      1,     1,     1    ] # If 0, model will not be saved.

# Filters (Kernels) per Conv. Layer
NUM_FILTERS  =     [ (128, 256, 512),
                     (128, 256, 512),
                     (128, 256, 512),
                     (128, 256, 512),
                     (128, 256, 512),
                     (128, 256, 512) ]

KERNEL_SIZE  =     [ ( (3,3), (3,3), (3,3) ),
                     ( (3,3), (3,3), (3,3) ),
                     ( (3,3), (3,3), (3,3) ),
                     ( (3,3), (3,3), (3,3) ),
                     ( (3,3), (3,3), (3,3) ),
                     ( (3,3), (3,3), (3,3) ) ]

STRIDE           = [ ( (2,2), (2,2), (2,2) ),
                     ( (2,2), (2,2), (2,2) ),
                     ( (2,2), (2,2), (2,2) ),
                     ( (2,2), (2,2), (2,2) ),
                     ( (2,2), (2,2), (2,2) ),
                     ( (2,2), (2,2), (2,2) )  ] # The stride of the operation of Convolution

POOLING          = [ ( (2,2), (2,2), (2,2) ),
                     ( (2,2), (2,2), (2,2) ),
                     ( (2,2), (2,2), (2,2) ),
                     ( (2,2), (2,2), (2,2) ),
                     ( (2,2), (2,2), (2,2) ),
                     ( (2,2), (2,2), (2,2) )  ]

# Units (Neurons) per ReLU Dense Layer.
RELU_DENSE_UNITS = [ (256, 256),
                     (256, 256),
                     (256, 256),
                     (256, 256),
                     (256, 256),
                     (256, 256)  ]

DISCARD                 = 5  # Discard some of the above combinations (columnwise and starting from the last one).
ADD_POOLING             = 1  # If 0, no pooling will be used.
PERFORM_AUGMENTATION    = 1  # If 0, data will stay as is, else dataset size will increase using different transformations.
TEST_ONLY               = 0  # If 0, training will take place, else only the testing set will be evaluated using the model in MODEL_LOAD_PATH.
FIND_MISCLASSIFICATIONS = 1  # If not 0, find which samples of the Test Set have at least one mislabeling.
ACTIVATION_CODE         = 1  # A code that will define, which will be the final activation layer.
                             ########   If  0 softmax_only will be used.
                             ########   If -1 (or any negative) sigmoid (first) and softmax (second and final layer) will be used.
                             ########   If  1 sigmoid_only will be used.
                             ########   If  2 (or any other positive except 1) both softmax (first) and sigmoid (second and final layer) will be used.


#########################################
#  DATA AUGMENTATION RELATED VARIABLES  #
#########################################
ROTATION_RANGE  = 45
SHIFT_FRACTION  = 0.2
SHEAR_RANGE     = 0.0
ZOOM_RANGE      = 0.0
HORIZONTAL_FLIP = True
VERTICAL_FILP   = True
#########################################


K_M = [ 'Xception',   'ResNet50', 'InceptionV3', 'InceptionResNetV2' ] # Don't change this list.
# The name specified here, will be used as a base for the naming of the figures, the text files and the '*.h5' models.
# If you want to save a model named 'cnn_9.h5'  or 'cnn_my_network.h5' etc. then below you should specify the name
# '9' or 'my_network' respectively.
MODEL_CHARACTERISTIC_NAME = [  '51',      '52',     '53',     '54',      '55',      '56'     ]
# If you want to load a keras pretrained model, use the names stated in the list above (K_M). Please DO NOT use these
# stated names (in K_M) for saving a model.
MODEL_LOAD_NAME           = [  '2',       K_M[2],   K_M[0],   K_M[2],   K_M[0],    K_M[0]   ]


SAVE_FOLDER = "cnn_results" # The folder where the figures will be saved.
                            # IF THE FOLDER DOESN'T EXIST, YOU NEED TO CREATE IT BEFOREHAND (in the same path as this file).

#############################################################################################


#########################
#  FUNCTIONS & CLASSES  #
#########################


# Calculates the precision, recall, F Measure and accuracy of the predicted outcome using 3 different methodologies.
# One on the whole dataset (overall), one on average per sample and one on average per label.
#
# yTrue: The ground truth labels.
# yPred: The predicted labels.
#
def findMetrics(yTrue, yPred):

    # precision overall
    positive_predictions = np.count_nonzero(yPred) # denominator
    true_positives       = np.sum( np.logical_and(yTrue == 1, yPred == 1) ) # numerator

    if positive_predictions == 0:
        precision = 0
    else:
        precision = true_positives / positive_predictions


    # recall overall
    relevant_positives = np.count_nonzero(yTrue) # denominator

    recall = true_positives / relevant_positives


    # F Measure overall
    numerator   = precision * recall
    denominator = precision + recall

    if denominator == 0:
        f_measure = 0
    else:
        f_measure = (2 * numerator) / denominator


    # classic accuracy overall
    dimensions = yTrue.shape

    accuracy1 = ( np.sum(yTrue == yPred) ) / ( dimensions[0] * dimensions[1] )


    # multi_label accuracy overall
    accuracy2 = true_positives / ( np.sum( np.logical_or(yTrue == 1, yPred == 1) ) )


    # precision per row/column
    positive_predictions_row = np.count_nonzero(yPred, axis=1) # denominators
    positive_predictions_col = np.count_nonzero(yPred, axis=0) # denominators

    true_positives_row       = np.sum( np.logical_and(yTrue == 1, yPred == 1), axis=1 ) # numerators
    true_positives_col       = np.sum( np.logical_and(yTrue == 1, yPred == 1), axis=0 ) # numerators

    positive_predictions_row = positive_predictions_row.astype('float')
    positive_predictions_col = positive_predictions_col.astype('float')

    true_positives_row       = true_positives_row.astype('float')
    true_positives_col       = true_positives_col.astype('float')

    precision_per_row        = np.true_divide( true_positives_row, positive_predictions_row, out=np.zeros_like(true_positives_row), where=positive_predictions_row!=0 )
    precision_per_col        = np.true_divide( true_positives_col, positive_predictions_col, out=np.zeros_like(true_positives_col), where=positive_predictions_col!=0 )

    avrg_precision_row       = np.mean(precision_per_row)
    avrg_precision_col       = np.mean(precision_per_col)


    # recall per row/column
    relevant_positives_row   = np.count_nonzero(yTrue, axis=1) # denominators
    relevant_positives_col   = np.count_nonzero(yTrue, axis=0) # denominators

    relevant_positives_row   = relevant_positives_row.astype('float')
    relevant_positives_col   = relevant_positives_col.astype('float')

    recall_per_row           = np.true_divide( true_positives_row, relevant_positives_row, out=np.zeros_like(true_positives_row), where=relevant_positives_row!=0 )
    recall_per_col           = np.true_divide( true_positives_col, relevant_positives_col, out=np.zeros_like(true_positives_col), where=relevant_positives_col!=0 )

    avrg_recall_row          = np.mean(recall_per_row)
    avrg_recall_col          = np.mean(recall_per_col)


    # F Measure per row/column
    numerator_row   = avrg_precision_row * avrg_recall_row
    denominator_row = avrg_precision_row + avrg_recall_row

    if denominator_row == 0:
        f_measure_row = 0
    else:
        f_measure_row = (2 * numerator_row) / denominator_row

    numerator_col   = avrg_precision_col * avrg_recall_col
    denominator_col = avrg_precision_col + avrg_recall_col

    if denominator_col == 0:
        f_measure_col = 0
    else:
        f_measure_col = (2 * numerator_col) / denominator_col


    # classic accuracy per row/column
    accuracy1_row = ( np.sum(yTrue == yPred, axis=1) ) / dimensions[1]
    accuracy1_col = ( np.sum(yTrue == yPred, axis=0) ) / dimensions[0]

    avrg_acc1_row = np.mean(accuracy1_row)
    avrg_acc1_col = np.mean(accuracy1_col)


    # multi_label accuracy per row/column
    acc2_denominator_row = np.sum( np.logical_or(yTrue == 1, yPred == 1), axis=1 )
    acc2_denominator_col = np.sum( np.logical_or(yTrue == 1, yPred == 1), axis=0 )

    acc2_denominator_row = acc2_denominator_row.astype('float')
    acc2_denominator_col = acc2_denominator_col.astype('float')

    accuracy2_row        = np.true_divide( true_positives_row, acc2_denominator_row, out=np.zeros_like(true_positives_row), where=acc2_denominator_row!=0 )
    accuracy2_col        = np.true_divide( true_positives_col, acc2_denominator_col, out=np.zeros_like(true_positives_col), where=acc2_denominator_col!=0 )

    avrg_acc2_row        = np.mean(accuracy2_row)
    avrg_acc2_col        = np.mean(accuracy2_col)


    return precision, recall, f_measure, accuracy1, accuracy2, avrg_precision_row, avrg_recall_row, f_measure_row, avrg_acc1_row, avrg_acc2_row, avrg_precision_col, avrg_recall_col, f_measure_col, avrg_acc1_col, avrg_acc2_col


# Implement a callback at the end of each epoch for the recording of metrics such as precision, recall, etc...
class Metrics(keras.callbacks.Callback):

    def on_epoch_end(self, batch, logs={}):

        if VALIDATION_PORTION != 0:
            y_true = self.validation_data[1]
            y_pred = np.asarray( self.model.predict(self.validation_data[0]) )

            if ACTIVATION_CODE <= 0:
                y_pred = [ y_pred > SOFTMAX_THRESHOLD ]
            else:
                y_pred = [ y_pred > SIGMOID_THRESHOLD ]

            y_pred     = y_pred[0]
            y_pred     = y_pred * 1.0

            calculated_metrics = findMetrics(y_true, y_pred)

            precision_history.append(calculated_metrics[0])
            recall_history.append(calculated_metrics[1])
            f_measure_history.append(calculated_metrics[2])
            accuracy1_history.append(calculated_metrics[3])
            accuracy2_history.append(calculated_metrics[4])

            precision_row_history.append(calculated_metrics[5])
            recall_row_history.append(calculated_metrics[6])
            f_measure_row_history.append(calculated_metrics[7])
            accuracy1_row_history.append(calculated_metrics[8])
            accuracy2_row_history.append(calculated_metrics[9])

            precision_col_history.append(calculated_metrics[10])
            recall_col_history.append(calculated_metrics[11])
            f_measure_col_history.append(calculated_metrics[12])
            accuracy1_col_history.append(calculated_metrics[13])
            accuracy2_col_history.append(calculated_metrics[14])

            print( "\n" + (80 * "-") )
            print( "\n Validation Set (Whole):     Precision = {:06.4f},  Recall = {:06.4f}, F Measure = {:06.4f}, AccuracyCl = {:06.4f}, Accuracy_ML = {:06.4f}\n".format(calculated_metrics[0],  calculated_metrics[1],  calculated_metrics[2],  calculated_metrics[3],  calculated_metrics[4])  )
            print( "\n Validation Set (Row Avrg):  Precision = {:06.4f},  Recall = {:06.4f}, F Measure = {:06.4f}, AccuracyCl = {:06.4f}, Accuracy_ML = {:06.4f}\n".format(calculated_metrics[5],  calculated_metrics[6],  calculated_metrics[7],  calculated_metrics[8],  calculated_metrics[9])  )
            print( "\n Validation Set (Col Avrg):  Precision = {:06.4f},  Recall = {:06.4f}, F Measure = {:06.4f}, AccuracyCl = {:06.4f}, Accuracy_ML = {:06.4f}\n".format(calculated_metrics[10], calculated_metrics[11], calculated_metrics[12], calculated_metrics[13], calculated_metrics[14]) )

            return


# Creates Epochs(x) - Metric(y) type plots.
#
# plotHistory:   The history list to be plotted.
# yLab:          The label of the y-axis.
# fileNamePart1: Part of the name that will be used for the .png file of the saved figure.
# fileNamePart2: Part of the name that will be used for the .png file of the saved figure.
# endEpoch:      The number of the final Epoch.
#
def createBasicPlot(plotHistory, yLab, fileNamePart1, fileNamePart2, endEpoch):

    plt.figure()
    plt.ylim(0,1)
    plt.xlim(1, endEpoch)

    plt.plot(plotHistory)
    plt.ylabel(yLab)
    plt.xlabel('Epoch')

    plt.legend( ['validation'], loc='lower right' )

    plt.savefig(SAVE_FOLDER + fileNamePart1 + fileNamePart2 + '.png', bbox_inches='tight')


label_strings = [ 'airplane',    'bare soil', 'buildings', 'cars', 'chaparral', 'court', 'dock',  'field', 'grass',
                  'mobile home', 'pavement',  'sand',      'sea',  'ship',      'tanks', 'trees', 'water' ]


# Given a label vector, this function outputs its active labels (strings).
#
# labelVector: The label vector.
#
def interpreteLabels(labelVector):

    func_output = ''

    for k in range(labelVector.shape[0]):

        if labelVector[k] == 1:
            func_output = func_output + label_strings[k] + '  '

    return func_output


##################
#  MAIN PROGRAM  #
##################


if not isinstance(ACTIVATION_CODE, int):
    raise RuntimeError("You have given an invalid value for the variable named 'ACTIVATION CODE'.")

metrics = Metrics()

if ( len(NUM_FILTERS) - DISCARD <= 0 ) or ( DISCARD < 0 ):
    raise RuntimeError("You have used an invalid value for the variable named 'DISCARD'.")

# Start loading the needed data (and shuffle them).
images = np.load(OBSERVATIONS_FILE)
labels = np.load(LABELS_FILE)

random_indices = np.arange( images.shape[0] )
np.random.shuffle(random_indices)

labels = labels[random_indices]
images = images[random_indices]

test_set    = images[:TESTING_DATA_NUM]
test_labels = labels[:TESTING_DATA_NUM]

num_test_samples   = test_set.shape[0]
num_unique_classes = test_labels.shape[1]

# If we don't want just to test some data with an existing classifier, we need to load the training data.
if TEST_ONLY == 0:
    train_set    = images[TESTING_DATA_NUM:]
    train_labels = labels[TESTING_DATA_NUM:]

    num_train_samples = train_set.shape[0]
    num_validation    = int( num_train_samples * VALIDATION_PORTION )

    cv_set       = train_set[:num_validation]
    cv_labels    = train_labels[:num_validation]

    train_set    = train_set[num_validation:]
    train_labels = train_labels[num_validation:]


images = None
labels = None


if PERFORM_AUGMENTATION != 0:

    data_generator = ImageDataGenerator( rotation_range=ROTATION_RANGE, width_shift_range=SHIFT_FRACTION, height_shift_range=SHIFT_FRACTION,
                                         shear_range=SHEAR_RANGE,       zoom_range=ZOOM_RANGE,            horizontal_flip=HORIZONTAL_FLIP,
                                         vertical_flip=VERTICAL_FILP)

    data_generator.fit(train_set)


# Start processing the different classifiers columnwise, as defined with the capital letter hyperparameters above.
for i in range( len(NUM_FILTERS) - DISCARD ):

    if MODEL_CHARACTERISTIC_NAME[i] in K_M:
        raise RuntimeError("You have chosen a restricted name to save your model. All the restricted names are contained in the list named 'K_M'.")


    precision_history = []
    recall_history    = []
    f_measure_history = []
    accuracy1_history = []
    accuracy2_history = []

    precision_row_history = []
    recall_row_history    = []
    f_measure_row_history = []
    accuracy1_row_history = []
    accuracy2_row_history = []

    precision_col_history = []
    recall_col_history    = []
    f_measure_col_history = []
    accuracy1_col_history = []
    accuracy2_col_history = []

    chosen_loss_function  = LOSS_FUNCTION_CHOICES[ LOSS_FUNCTION[i] ]
    chosen_optimizer      = OPTIMIZER_CHOICES[ OPTIMIZER[i] ]

    text_file = open(SAVE_FOLDER + '/results_and_parameters_of_network_named_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', "w")

    text_file.write("Total Epochs: %s\n" % EPOCHS[i] )
    text_file.write("Init Epoch: %s\n\n" % INIT_EPOCH[i])

    if TEST_ONLY == 0:
        text_file.write("Training Examples: %s\n" % num_train_samples)

    text_file.write("Testing Examples: %s\n\n" % num_test_samples)

    text_file.write("Validation Portion: %s\n" % VALIDATION_PORTION)
    text_file.write("Softmax Threshold: %s\n" % SOFTMAX_THRESHOLD)
    text_file.write("Sigmoid Threshold: %s\n\n" % SIGMOID_THRESHOLD)

    text_file.write("Number of ReLU Dense Layers: %s\n" % RELU_DENSE_LAYERS[i])
    text_file.write("Number of units per ReLU Dense Layer: {}\n\n".format(RELU_DENSE_UNITS[i]))

    text_file.write("Dropout: %s\n" % DROPOUT[i])
    text_file.write("Dropout used on all applicable layers?: %s\n" % DROP_ALL_LAYERS[i])
    text_file.write("Batch Normalization used?: %s\n" % BATCH_NORM[i])
    text_file.write("Batch Size: %s\n\n" % BATCH_SIZE[i])

    text_file.write("Chosen Loss Function: {}\n".format(chosen_loss_function))
    text_file.write("Chosen Optimizer: {}\n".format(chosen_optimizer))
    text_file.write("Activation Code (0 for softmax, 1 for sigmoid, any other number for both): %s\n\n" % ACTIVATION_CODE)

    text_file.write("Data Augmentation used?: %s\n\n" % PERFORM_AUGMENTATION)

    # Clear GPU Memory to train next model, else program might eventually crush.
    if i > 0:
        del model
        K.clear_session()

    # We will not train a new model. We will load and train (or test) an existing one.
    if NEW_MODEL[i] == 0:

        if MODEL_LOAD_NAME[i] in K_M: # Load existing Keras pre-trained models.

            text_file.write( "Keras Pre-Trained Model: {}\n\n".format(MODEL_LOAD_NAME[i]) )

            if   MODEL_LOAD_NAME[i] == 'Xception':
                base_model = Xception(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
            elif MODEL_LOAD_NAME[i] == 'ResNet50':
                base_model = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
            elif MODEL_LOAD_NAME[i] == 'InceptionV3':
                base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
            else:
                base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

            # We have not loaded any FC layers so we need to keep adding the layers we want.
            base_model_out = base_model.output
            base_model_out = GlobalAveragePooling2D()(base_model_out)

            if DROPOUT[i] != 0: base_model_out = Dropout(DROPOUT[i])(base_model_out)

            j = 0

            while j < RELU_DENSE_LAYERS[i]:
                base_model_out = Dense( (RELU_DENSE_UNITS[i])[j] )(base_model_out)
                base_model_out = Activation('relu')(base_model_out)
                if DROPOUT[i] != 0: base_model_out = Dropout(DROPOUT[i])(base_model_out)
                j += 1

            base_model_out = Dense( num_unique_classes )(base_model_out)

            if ACTIVATION_CODE == 0:
                base_model_out = Activation('softmax')(base_model_out)
            elif ACTIVATION_CODE == 1:
                base_model_out = Activation('sigmoid')(base_model_out)
            else:
                if ACTIVATION_CODE < 0:
                    base_model_out = Activation('sigmoid')(base_model_out)
                    base_model_out = Dense( num_unique_classes )(base_model_out)
                    base_model_out = Activation('softmax')(base_model_out)
                else:
                    base_model_out = Activation('softmax')(base_model_out)
                    base_model_out = Dense(num_unique_classes)(base_model_out)
                    base_model_out = Activation('sigmoid')(base_model_out)

            model = Model(inputs=base_model.input, outputs=base_model_out)

            for layer in base_model.layers: # Freeze all layers of the pre-trained model. Only the final layers will be trained.
                layer.trainable = False

            model.compile(loss=chosen_loss_function, optimizer=chosen_optimizer)

        else:
            model = load_model('cnn_' + MODEL_LOAD_NAME[i] + '.h5')
            text_file.write("This file corresponds to a custom pre-trained model (not Keras pre-trained).\n\n")

    # We will train a new model.
    else:

        if TEST_ONLY != 0:
            raise RuntimeError("You cannot be in 'TEST_ONLY' mode, when you create a new model that hasn't be trained yet.")

        if INIT_EPOCH[i] != 0:
            raise RuntimeError("You just created a new model. The 'INIT_EPOCH' variable should be zero.")

        text_file.write("Number of Convolutional Layers: %s\n" % CONV_LAYERS[i])
        text_file.write("Number of filters per layer: {}\n".format(NUM_FILTERS[i]) )
        text_file.write("Kernel Size per layer: {}\n".format(KERNEL_SIZE[i]) )
        text_file.write("Stride: {}\n".format(STRIDE[i]))
        text_file.write("Pooling Used?: %s\n" % ADD_POOLING)
        text_file.write("If Pooling Used, what is its width: {}\n\n".format(POOLING[i]))

        # Define and compile the new model.
        model = Sequential()

        j = 0
        model.add( Conv2D( (NUM_FILTERS[i])[j], (KERNEL_SIZE[i])[j], strides=(STRIDE[i])[j], input_shape=INPUT_SHAPE ) )
        if ADD_POOLING != 0: model.add(MaxPooling2D(pool_size=(POOLING[i])[j]))
        model.add( Activation('relu') )
        if BATCH_NORM[i] != 0: model.add(BatchNormalization())
        if ( DROP_ALL_LAYERS[i] != 0 ) and ( DROPOUT[i] != 0 ): model.add(Dropout(DROPOUT[i]))


        j += 1
        while j < CONV_LAYERS[i]:
            model.add(Conv2D((NUM_FILTERS[i])[j], (KERNEL_SIZE[i])[j], strides=(STRIDE[i])[j] ) )
            if ADD_POOLING != 0: model.add(MaxPooling2D(pool_size=(POOLING[i])[j]))
            model.add( Activation('relu') )
            if BATCH_NORM[i] != 0: model.add(BatchNormalization())
            if ( DROP_ALL_LAYERS[i] != 0 ) and ( DROPOUT[i] != 0 ): model.add(Dropout(DROPOUT[i]))
            j += 1

        # If DROP_ALL_LAYERS !=0 then we have already added a DROPOUT LAYER associated with the DENSE LAYER.
        if ( DROP_ALL_LAYERS[i] == 0 ) and ( DROPOUT[i] != 0 ):    model.add(Dropout(DROPOUT[i]))

        # Fully Connected Layers
        model.add( Flatten() )

        j = 0
        while j < RELU_DENSE_LAYERS[i]:
            model.add( Dense( (RELU_DENSE_UNITS[i])[j] ) )
            model.add( Activation('relu') )
            if DROPOUT[i] != 0: model.add( Dropout(DROPOUT[i]) )
            j += 1

        model.add( Dense(num_unique_classes) )

        if ACTIVATION_CODE == 0:
            model.add( Activation('softmax') )
        elif ACTIVATION_CODE == 1:
            model.add( Activation('sigmoid') )
        else:
            if ACTIVATION_CODE < 0:
                model.add(Activation('sigmoid'))
                model.add(Dense(num_unique_classes))
                model.add(Activation('softmax'))
            else:
                model.add(Activation('softmax'))
                model.add(Dense(num_unique_classes))
                model.add(Activation('sigmoid'))

        model.compile(loss=chosen_loss_function, optimizer=chosen_optimizer)

    # If not in Test Mode, fit the given model.
    if TEST_ONLY == 0:

        if PERFORM_AUGMENTATION == 0:
            if VALIDATION_PORTION == 0:
                history = model.fit(train_set, train_labels, batch_size=BATCH_SIZE[i], epochs=EPOCHS[i], verbose=2,
                                   initial_epoch=INIT_EPOCH[i], callbacks=[metrics])
            else:
                history = model.fit(train_set, train_labels, batch_size=BATCH_SIZE[i], epochs=EPOCHS[i], verbose=2,
                                   initial_epoch=INIT_EPOCH[i], validation_data=(cv_set, cv_labels), callbacks=[metrics])


        else:
            if VALIDATION_PORTION == 0:
                history = model.fit_generator( data_generator.flow(train_set, train_labels, batch_size=BATCH_SIZE[i]),
                                               initial_epoch=INIT_EPOCH[i], epochs=EPOCHS[i],
                                               steps_per_epoch=(num_train_samples - num_validation) // BATCH_SIZE[i],
                                               verbose=2, callbacks=[metrics] )
            else:
                history = model.fit_generator( data_generator.flow(train_set, train_labels, batch_size=BATCH_SIZE[i]),
                                               initial_epoch=INIT_EPOCH[i], epochs=EPOCHS[i],
                                               steps_per_epoch=(num_train_samples - num_validation) // BATCH_SIZE[i],
                                               verbose=2, validation_data=(cv_set, cv_labels), callbacks=[metrics] )


        # Creating Different Plots.

        # Loss History
        fig1 = plt.figure()
        plt.xlim(1, EPOCHS[i] - INIT_EPOCH[i])
        plt.plot( history.history['loss'] )
        if VALIDATION_PORTION != 0: plt.plot( history.history['val_loss'] )
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        if VALIDATION_PORTION != 0:
            plt.legend( ['training', 'validation'], loc='upper right' )
        else:
            plt.legend(['training'], loc='upper right')

        plt.savefig(SAVE_FOLDER + '/lss_' + MODEL_CHARACTERISTIC_NAME[i] + '.png', bbox_inches='tight')

        if VALIDATION_PORTION != 0:

            # Accuracy History.
            createBasicPlot(accuracy1_history, 'Classic Accuracy',     '/accuracy1_', MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])
            createBasicPlot(accuracy2_history, 'Multi-Label Accuracy', '/accuracy2_', MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])

            # Precision, Recall and F Measure History
            createBasicPlot(precision_history, 'Precision', '/precision_', MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])
            createBasicPlot(recall_history,    'Recall',    '/recall_',    MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])
            createBasicPlot(f_measure_history, 'F Measure', '/f_measure_', MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])

            # Accuracy (Average per row) History.
            createBasicPlot(accuracy1_row_history, 'Classic Accuracy',     '/accuracy1_per_row', MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])
            createBasicPlot(accuracy2_row_history, 'Multi-Label Accuracy', '/accuracy2_per_row', MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])

            # (Average per row) Precision, Recall and F Measure History
            createBasicPlot(precision_row_history, 'Precision', '/precision_per_row', MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])
            createBasicPlot(recall_row_history,    'Recall',    '/recall_per_row',    MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])
            createBasicPlot(f_measure_row_history, 'F Measure', '/f_measure_per_row', MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])

            # Accuracy (Average per column) History.
            createBasicPlot(accuracy1_col_history, 'Classic Accuracy',     '/accuracy1_per_col', MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])
            createBasicPlot(accuracy2_col_history, 'Multi-Label Accuracy', '/accuracy2_per_col', MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])

            # (Average per column) Precision, Recall and F Measure History
            createBasicPlot(precision_col_history, 'Precision', '/precision_per_col', MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])
            createBasicPlot(recall_col_history,    'Recall',    '/recall_per_col',    MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])
            createBasicPlot(f_measure_col_history, 'F Measure', '/f_measure_per_col', MODEL_CHARACTERISTIC_NAME[i], EPOCHS[i] - INIT_EPOCH[i])

        plt.close('all')

        loss    = history.history["loss"]
        loss    = np.array(loss)
        np.savetxt(SAVE_FOLDER + '/lss_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', loss, delimiter=",")

        if VALIDATION_PORTION != 0:
            loss    = history.history["val_loss"]
            loss    = np.array(loss)
            np.savetxt(SAVE_FOLDER + '/CV_lss_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', loss, delimiter=",")
            loss    = None
            history = None

            accuracy1_history = np.array(accuracy1_history)
            accuracy2_history = np.array(accuracy2_history)
            precision_history = np.array(precision_history)
            recall_history    = np.array(recall_history)
            f_measure_history = np.array(f_measure_history)
            np.savetxt(SAVE_FOLDER + '/precision_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', precision_history, delimiter=",")
            np.savetxt(SAVE_FOLDER + '/recall_history_'    + MODEL_CHARACTERISTIC_NAME[i] + '.txt', recall_history,    delimiter=",")
            np.savetxt(SAVE_FOLDER + '/f_measure_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', f_measure_history, delimiter=",")
            np.savetxt(SAVE_FOLDER + '/accuracy1_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', accuracy1_history, delimiter=",")
            np.savetxt(SAVE_FOLDER + '/accuracy2_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', accuracy2_history, delimiter=",")
            precision_history = None
            recall_history    = None
            f_measure_history = None
            accuracy1_history = None
            accuracy2_history = None

            accuracy1_row_history = np.array(accuracy1_row_history)
            accuracy2_row_history = np.array(accuracy2_row_history)
            precision_row_history = np.array(precision_row_history)
            recall_row_history    = np.array(recall_row_history)
            f_measure_row_history = np.array(f_measure_row_history)
            np.savetxt(SAVE_FOLDER + '/precision_row_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', precision_row_history, delimiter=",")
            np.savetxt(SAVE_FOLDER + '/recall_row_history_'    + MODEL_CHARACTERISTIC_NAME[i] + '.txt', recall_row_history,    delimiter=",")
            np.savetxt(SAVE_FOLDER + '/f_measure_row_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', f_measure_row_history, delimiter=",")
            np.savetxt(SAVE_FOLDER + '/accuracy1_row_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', accuracy1_row_history, delimiter=",")
            np.savetxt(SAVE_FOLDER + '/accuracy2_row_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', accuracy2_row_history, delimiter=",")
            precision_row_history = None
            recall_row_history    = None
            f_measure_row_history = None
            accuracy1_row_history = None
            accuracy2_row_history = None

            accuracy1_col_history = np.array(accuracy1_col_history)
            accuracy2_col_history = np.array(accuracy2_col_history)
            precision_col_history = np.array(precision_col_history)
            recall_col_history    = np.array(recall_col_history)
            f_measure_col_history = np.array(f_measure_col_history)
            np.savetxt(SAVE_FOLDER + '/precision_col_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', precision_col_history, delimiter=",")
            np.savetxt(SAVE_FOLDER + '/recall_col_history_'    + MODEL_CHARACTERISTIC_NAME[i] + '.txt', recall_col_history,    delimiter=",")
            np.savetxt(SAVE_FOLDER + '/f_measure_col_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', f_measure_col_history, delimiter=",")
            np.savetxt(SAVE_FOLDER + '/accuracy1_col_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', accuracy1_col_history, delimiter=",")
            np.savetxt(SAVE_FOLDER + '/accuracy2_col_history_' + MODEL_CHARACTERISTIC_NAME[i] + '.txt', accuracy2_col_history, delimiter=",")
            precision_col_history = None
            recall_col_history    = None
            f_measure_col_history = None
            accuracy1_col_history = None
            accuracy2_col_history = None


    # Perform the evaluation on the Test Set
    predictions = model.predict(test_set)

    print("\n" + (80 * "-"))

    if ACTIVATION_CODE <= 0:
        predictions = [predictions > SOFTMAX_THRESHOLD]
    else:
        predictions = [predictions > SIGMOID_THRESHOLD]

    predictions = predictions[0]
    predictions = predictions * 1.0

    other_metrics = findMetrics(test_labels, predictions)

    print( "\n Test Set (Whole):     Precision = {:06.4f},  Recall = {:06.4f}, F Measure = {:06.4f}, AccuracyCl = {:06.4f}, Accuracy_ML = {:06.4f}\n".format(other_metrics[0],  other_metrics[1],  other_metrics[2],  other_metrics[3],  other_metrics[4])  )
    print( "\n Test Set (Row Avrg):  Precision = {:06.4f},  Recall = {:06.4f}, F Measure = {:06.4f}, AccuracyCl = {:06.4f}, Accuracy_ML = {:06.4f}\n".format(other_metrics[5],  other_metrics[6],  other_metrics[7],  other_metrics[8],  other_metrics[9])  )
    print( "\n Test Set (Col Avrg):  Precision = {:06.4f},  Recall = {:06.4f}, F Measure = {:06.4f}, AccuracyCl = {:06.4f}, Accuracy_ML = {:06.4f}\n".format(other_metrics[10], other_metrics[11], other_metrics[12], other_metrics[13], other_metrics[14]) )

    text_file.write("WHOLE DATASET\n-------------\n")
    text_file.write("Test Precision: %s\n"    % other_metrics[0])
    text_file.write("Test Recall: %s\n"       % other_metrics[1])
    text_file.write("Test F Measure: %s\n"    % other_metrics[2])
    text_file.write("Test AccuracyCl: %s\n"   % other_metrics[3])
    text_file.write("Test AccuracyML: %s\n\n" % other_metrics[4])

    text_file.write("PER ROW AVERAGE\n---------------\n")
    text_file.write("Test Precision: %s\n"    % other_metrics[5])
    text_file.write("Test Recall: %s\n"       % other_metrics[6])
    text_file.write("Test F Measure: %s\n"    % other_metrics[7])
    text_file.write("Test AccuracyCl: %s\n"   % other_metrics[8])
    text_file.write("Test AccuracyML: %s\n\n" % other_metrics[9])

    text_file.write("PER COLUMN AVERAGE\n------------------\n")
    text_file.write("Test Precision: %s\n"    % other_metrics[10])
    text_file.write("Test Recall: %s\n"       % other_metrics[11])
    text_file.write("Test F Measure: %s\n"    % other_metrics[12])
    text_file.write("Test AccuracyCl: %s\n"   % other_metrics[13])
    text_file.write("Test AccuracyML: %s\n\n" % other_metrics[14])


    if FIND_MISCLASSIFICATIONS != 0:
        print( "\n\n\nMISCLASSIFIED SAMPLES\n" + (21 * "-") )
        associated_type  = np.load(ASSOC_TYPE_FILE)
        associated_index = np.load(ASSOC_IND_FILE)

        associated_type = associated_type[random_indices]
        associated_index = associated_index[random_indices]

        associated_type_test  = associated_type[:TESTING_DATA_NUM]
        associated_index_test = associated_index[:TESTING_DATA_NUM]

        for current in range(num_test_samples):

            missed_labels_num = np.sum( np.abs(test_labels[current,:] - predictions[current,:]) )

            if missed_labels_num != 0:

                print_this = "\n\n" + associated_type_test[current] + str( associated_index_test[current] ) + "\n\nTrue: "
                print_this = print_this + interpreteLabels( test_labels[current,:] )  + "\n\nPred: "
                print_this = print_this + interpreteLabels( predictions[current, :] ) + "\n\n" + (21 * "-")

                print( print_this )


    if SAVE_MODEL[i] != 0:
        model.save('cnn_' + MODEL_CHARACTERISTIC_NAME[i] + '.h5')

    text_file.close()
