import logging

import keras.backend as K

alpha = 0.2  # used in FaceNet https://arxiv.org/pdf/1503.03832.pdf


def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction

    """
x1.shape,x2.shape (300, 200) (300, 200)
dot <bound method Tensor.eval of <tf.Tensor 'loss/embeddings_loss/Squeeze_1:0' shape=(300,) dtype=float32>>
********************************************************************************************************************************************************************************************************
x1.shape,x2.shape (300, 200) (300, 200)
dot <bound method Tensor.eval of <tf.Tensor 'loss/embeddings_loss/Squeeze_3:0' shape=(300,) dtype=float32>>
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (900, 390)                0
_________________________________________________________________
fc1 (Dense)                  (900, 200)                78200
_________________________________________________________________
normalization (Lambda)       (900, 200)                0
_________________________________________________________________
embeddings (Lambda)          (900, 200)                0
_________________________________________________________________
softmax (Dense)              (900, 49)                 9849
=================================================================
    """

    print("*"*200)
    print("x1.shape,x2.shape",x1.shape,x2.shape)
    dot = K.squeeze(K.batch_dot(x1, x2, axes=1), axis=1)
    print("dot",dot.eval)
    logging.info('dot: {}'.format(dot))
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return dot


def deep_speaker_loss(y_true, y_pred):
    logging.info('y_true={}'.format(y_true))
    logging.info('y_pred={}'.format(y_pred))
    # y_true.shape = (batch_size, embedding_size)
    # y_pred.shape = (batch_size, embedding_size)
    # CONVENTION: Input is:
    # concat(BATCH_SIZE * [ANCHOR, POSITIVE_EX, NEGATIVE_EX] * NUM_FRAMES)
    # EXAMPLE:
    # BATCH_NUM_TRIPLETS = 3, NUM_FRAMES = 2
    # _____________________________________________________
    # ANCHOR 1 (512,)
    # ANCHOR 2 (512,)
    # ANCHOR 3 (512,)
    # POS EX 1 (512,)
    # POS EX 2 (512,)
    # POS EX 3 (512,)
    # NEG EX 1 (512,)
    # NEG EX 2 (512,)
    # NEG EX 3 (512,)
    # _____________________________________________________

    elements = int(K.int_shape(y_pred)[0] / 3)
    logging.info('elements={}'.format(elements))

    anchor = y_pred[0:elements]
    positive_ex = y_pred[elements:2 * elements]
    negative_ex = y_pred[2 * elements:]
    logging.info('anchor={}'.format(anchor))
    logging.info('positive_ex={}'.format(positive_ex))
    logging.info('negative_ex={}'.format(negative_ex))

    sap = batch_cosine_similarity(anchor, positive_ex)
    logging.info('sap={}'.format(sap))
    san = batch_cosine_similarity(anchor, negative_ex)
    logging.info('san={}'.format(san))
    loss = K.maximum(san - sap + alpha, 0.0)
    logging.info('loss={}'.format(loss))
    # total_loss = K.sum(loss)
    total_loss = K.mean(loss)
    logging.info('total_loss={}'.format(total_loss))
    return total_loss
