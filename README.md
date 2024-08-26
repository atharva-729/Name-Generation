## Name Generation Using Deep Learning
This has been inspired by Andrej Karpathy's makemore series. This project generated Indian names, using the kaggle dataset: [Indian Names (Boys & Girls)](https://www.kaggle.com/datasets/meemr5/indian-names-boys-girls)

The notebooks for this repository are:
1. [Using bigram](https://www.kaggle.com/code/atharva729/name-gen-bigram/)
2. [Using Softmax MLP](https://www.kaggle.com/code/atharva729/fork-of-name-gen-softmax-mlp)
3. [Using Batch Normalization](https://www.kaggle.com/code/atharva729/name-gen-batch-norm)
4. [Using Custom Backprop](https://www.kaggle.com/code/atharva729/name-gen-backprop)
5. [Finalising](https://www.kaggle.com/code/atharva729/name-gen-finishing)

Explanations:

### Using Bigram
**1. Data Preprocessing:**

* Reads names from a file and converts them to lowercase.
Creates a dictionary fr to store bigram frequencies (frequency of two consecutive characters appearing together).
* Encodes characters using a dictionary stoi that maps characters to unique integer indices.

**2. Building the Bigram Matrix (N):**

* Creates a 29x29 matrix N where each cell represents the count of a bigram (two consecutive characters) based on the fr dictionary.
* Visualizes the matrix using matplotlib to understand character transition patterns. This gives a beautiful plot:
  ![image](https://github.com/atharva-729/Name-Generation/assets/118381293/d608c5a3-302b-4e67-8f20-63863824353f)


**3. Smoothing and Negative Log-Likelihood (NLL):**

* Applies Laplace smoothing (adding 1 to all counts) to the N matrix to avoid zeros and improve model stability.
* Calculates the negative log-likelihood (NLL) for the entire dataset, which measures how well the current bigram probabilities predict the actual character sequences in the names.

**4. Transition from Bigrams to Neural Network:**

* Shifts from bigrams to using a neural network to model character probabilities.
* Creates a training set of input-output pairs, where the input is a one-hot encoded character and the output is the next character in the sequence.

**5. Neural Network Architecture:**

* Defines a simple neural network with one hidden layer and 29 neurons (number of characters).
* Uses one-hot encoding for the input character.
* Calculates logits (unnormalized probabilities) through a matrix multiplication of the input and the weight matrix W.
* Applies the softmax function to convert logits to normalized probabilities, representing the likelihood of each character following the input character.

**6. Training and Evaluation:**

* Calculates the negative log-likelihood loss, which measures the discrepancy between predicted and actual probabilities for a given bigram.
* Performs gradient descent optimization for 40 epochs, updating the weight matrix W to minimize the loss.
* Evaluates the trained network by generating name samples using the learned character probabilities.

**7. Observations and Next Steps:**

* The initial name generation results are not very realistic. This is likely due to the limitations of a simple bigram model and a small training dataset.
* The explanation mentions exploring more complex network architectures and regularization techniques for better performance.

error i achieved using only bigram approach: negative log likelihood: 2.188798666000366
with neural network approach (90 epochs): loss was 2.2484030723571777

and i could generate names were: not good to say the least
too small and too long names like "n" and "uthanyesathishumanudhacfugahoovi" were generated. I could not find a single 'Indian-sounding' name.
this was a bad model.


### Softmax MLP
**Building the Dataset**

* This part is similar to the previous notebook. We're taking the list of names and chopping them up into smaller pieces (of size 3 in this case), considering the previous characters to predict the next one.
* We use a special function `build_dataset` to achieve this. It creates two separate datasets: training and development. We'll use the training data to train our model and the development data to see how well it's doing during training. There's also a testing data set created that we'll use later to evaluate the final model's performance.

**Training, Learning and Remembering**

* This is where things get interesting! We've defined a bunch of variables with letters like `C`, `W1`, `b1`, and so on. These represent different parts of our special math model, the neural network.
* We're setting random starting values for these variables, like giving our network a clean slate to learn on.
* Here comes the training loop:
    * We grab a small chunk of data (32 names in this case) from the training set. This is called a minibatch.
    * We pass the minibatch through the neural network. This involves calculations using the values in `C`, `W1`, and others. Think of it like the network is making predictions about the next letter based on what it sees in the minibatch.
    * We then compare the network's predictions with the actual letters in the minibatch. This tells us how wrong the network was in its guesses.
    * The magic happens here: we use this information to update the values in `C`, `W1`, and the others. It's like the network is learning from its mistakes and adjusting itself to get better at predicting the next letter.
    * This loop runs for a long time (200,000 times in this case) so the network can learn effectively.

**Keeping Track of Learning**

* We plot a graph to see how well the network is doing over time. Ideally, the line should keep going down, which means the network's predictions are getting better (the loss is decreasing). The plot we get here looks like a hockey stick, which is a good sign!

**Looking at the Hidden World**

* Remember those variables like `C` that we used to train the network? `C` is interesting because it stores information about each character. We can plot this information to see how the network perceives the relationships between characters. The plot you'll see might show some characters clustered together and others further apart. This reflects how often the network sees these characters appearing together in the names.

**Generating Names!**

* Finally, the fun part! We use the trained network to generate new names.
* We start with a sequence of special characters (like "..." in this case) and keep feeding that into the network. The network predicts the next letter, and we add that to our sequence. We keep doing this until we predict a special character that means "end of name" (like a period).
* This way, the network creates a brand new name, letter by letter, based on what it learned from the training data.

**The Results**

* The generated names are looking much better now! We're getting some realistic-looking Indian names like "Nishan" and "Vindha".
* the loss i got here was reduced a lot actually, it came down to 1.9452 on the validation set.  


## 3. with batch norm
probelms:
1. initial loss is 27.~, if we assume unfirom distribution, it should be around 3.4~
   * the problem is that the model is confidently wrong: the initial context is ... which can have any letter after it, that is why we see such a bad loss
   * we want the logits to be nearly equal, or closer to zero to not give probablities that are far apart, to get the loss to a lower, expected value
   * fix: make b2 a zero tensor, and make W2 very small (multiply by 0.1 or 0.01, do not make it 0)
   * this makes a better algorithm: we do not waste the first small no of iterations on the hockey stick appearance, we start optimising from the start
2. problem with h: the activations:
   * tanh activations are mostly -1 or 1. happens due to nature of tanh (it requires small values of x to output a value that is not 1 or -1)
   * this makes the backward pass (for that neuron) nearly useless. the grad vanishes because of +-1 values of tanh.
   *  this is not that big of a probelm in our case. would have been if we got a neuron with all the examples having a tanh value of over 0.99. in this case, we would call that neuron a **dead neuron**. this neuron will never learn anything.
   *  sigmoid and relu would suffer from this problem if something similar happens.
   *  we can use leaky relu, or elu instead of these to get rid of the problem of dead neurons.
   *  
