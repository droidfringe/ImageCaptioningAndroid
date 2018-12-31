import vocabulary

import heapq
import math
import tensorflow as tf

import numpy as np

from tensorflow.python.platform import gfile
import cv2

class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, state, logprob, score, metadata=None):
        """Initializes the Caption.

        Args:
          sentence: List of word ids in the caption.
          state: Model state after generating the previous word.
          logprob: Log-probability of the caption.
          score: Score of the caption.
          metadata: Optional metadata associated with the partial sentence. If not
            None, a list of strings with the same length as 'sentence'.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.metadata = metadata

    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score


class TopN(object):
  """Maintains the top n elements of an incrementally provided set."""

  def __init__(self, n):
    self._n = n
    self._data = []

  def size(self):
    assert self._data is not None
    return len(self._data)

  def push(self, x):
    """Pushes a new element."""
    assert self._data is not None
    if len(self._data) < self._n:
      heapq.heappush(self._data, x)
    else:
      heapq.heappushpop(self._data, x)

  def extract(self, sort=False):
    """Extracts all elements from the TopN. This is a destructive operation.

    The only method that can be called immediately after extract() is reset().

    Args:
      sort: Whether to return the elements in descending sorted order.

    Returns:
      A list of data; the top n elements provided to the set.
    """
    assert self._data is not None
    data = self._data
    self._data = None
    if sort:
      data.sort(reverse=True)
    return data

  def reset(self):
    """Returns the TopN to an empty state."""
    self._data = []



def beam_search(interpreter, image, vocab, input_name2index, output_name2index, beam_size=1, max_caption_length=20):
    """Runs beam search caption generation on a single image.

    Args:
      interpreter: tensorflow.lite.python.interpreter.Interpreter object.
      image: An image of size 346x346x3, dtype=float32, range=[0,1).
      vocab: Vocabulary object
      input_name2index: dict for input_tensor_name -> tflite model tensor index
      output_name2index: dict for input_tensor_name -> tflite model tensor index

    Returns:
      A list of Caption sorted by descending score.
    """
    # Feed in the image to get the initial state.
    #beam_size = 3
    #max_caption_length = 20
    length_normalization_factor = 0.0
    print('input image shape=', image.shape)
    #initial_state = sess.run(fetches="import/lstm/initial_state:0", feed_dict={"import/image_feed:0": encoded_image})
    interpreter.set_tensor(input_name2index['import/image_feed'], image)
    interpreter.invoke()
    initial_state = interpreter.get_tensor(output_name2index['import/lstm/initial_state'])


    initial_beam = Caption(
        sentence=[vocab.start_id],
        state=initial_state[0],
        logprob=0.0,
        score=0.0,
        metadata=[""])
    partial_captions = TopN(beam_size)
    partial_captions.push(initial_beam)
    complete_captions = TopN(beam_size)

    # Run beam search.
    for _ in range(max_caption_length - 1):
        partial_captions_list = partial_captions.extract()
        partial_captions.reset()
        input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
        state_feed = np.array([c.state for c in partial_captions_list])

        #print('input_feed: shape=', input_feed.shape, ', state_feed: shape=', state_feed.shape, ' input_feed=', input_feed)
        #print()
        """
        softmax, new_states = sess.run(
        fetches=["import/softmax:0", "import/lstm/state:0"],
        feed_dict={
            "import/input_feed:0": input_feed,
            "import/lstm/state_feed:0": state_feed,
        })
        """

        interpreter.set_tensor(input_name2index['import/input_feed'], input_feed)
        interpreter.set_tensor(input_name2index['import/lstm/state_feed'], state_feed)
        interpreter.invoke()
        softmax = interpreter.get_tensor(output_name2index['import/softmax'])
        new_states = interpreter.get_tensor(output_name2index['import/lstm/state'])

        metadata = None

        for i, partial_caption in enumerate(partial_captions_list):
            word_probabilities = softmax[i]
            state = new_states[i]
            # For this partial caption, get the beam_size most probable next words.
            words_and_probs = list(enumerate(word_probabilities))
            words_and_probs.sort(key=lambda x: -x[1])
            words_and_probs = words_and_probs[0:beam_size]
            # Each next word gives a new partial caption.
            for w, p in words_and_probs:
                if p < 1e-12:
                    continue  # Avoid log(0).
                sentence = partial_caption.sentence + [w]
                logprob = partial_caption.logprob + math.log(p)
                score = logprob
                if metadata:
                    metadata_list = partial_caption.metadata + [metadata[i]]
                else:
                    metadata_list = None
                if w == vocab.end_id:
                    if length_normalization_factor > 0:
                        score /= len(sentence )**length_normalization_factor
                    beam = Caption(sentence, state, logprob, score, metadata_list)
                    complete_captions.push(beam)
                else:
                    beam = Caption(sentence, state, logprob, score, metadata_list)
                    partial_captions.push(beam)
        if partial_captions.size() == 0:
            # We have run out of partial candidates; happens when beam_size = 1.
            break

    # If we have no complete captions then fall back to the partial captions.
    # But never output a mixture of complete and partial captions because a
    # partial caption could have a higher score than all the complete captions.
    if not complete_captions.size():
        complete_captions = partial_captions

    captions = complete_captions.extract(sort=True)
    sentences = []
    for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        sentences.append(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
    return sentences




def read_image(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, (346,346))
    image = image.astype(np.float32)
    image = image / 256.0
    return image






def main():

    #sess = tf.Session()
    GRAPH_LOCATION = '/home/droid/show_and_tell/im2txt/im2txt/data/output_graph_2.pb'
    TFLITE_MODEL_LOCATION = "/home/droid/show_and_tell/im2txt/im2txt/data/show_and_tell_converted_model.tflite"
    VOCAB_FILE = '/home/droid/show_and_tell/im2txt/im2txt/data/word_counts.txt'
    IMAGE_FILE = '/home/droid/show_and_tell/im2txt/im2txt/data/cat_dog.jpg.346.jpg'
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_LOCATION)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_name2index = {}
    output_name2index = {}

    for i in input_details:
        input_name2index[i['name']] = i['index']

    for o in output_details:
        output_name2index[o['name']] = o['index']

    vocab = vocabulary.Vocabulary(VOCAB_FILE)
    image = read_image(IMAGE_FILE)

    captions = beam_search(interpreter, image, vocab, input_name2index, output_name2index, beam_size=1)
    print(captions)

if __name__ == '__main__':
    main()
