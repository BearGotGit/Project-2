# Plan (Approach)

## Main Plan

1. Data

   - Dataset can be downloaded from Foundation repo

   - You can compile your own dataset using books downloaded from Project Gutenberg;
     make a data curator for them;
     finally, create test.jsonl and train.jsonl.
     - Lines of json with "prompt" and "completion" keys
     - Insert `<bos>` and `<eos>` where appropriate in data (ad hoc, before capitalizations, after `...`, `.`, `!`, `?`)

2. Tokenizer

   - Sentencepiece model, 10000 tokens
   - Train model with user-def tokens: `<bos>` and `<eos>`
     - Transformer models require `<pad>` token too

3. Data & Tokenizer Marriage

   - Parse books for random sequences of words (seperated by spaces, punctuation) of lengths in [short, long).
   - Create non-intersecting set of test and train sequences
   - store in jsonl file

   - For all models, use one-hot encoding as features for tokens (will be okay using cross-entropy)

   - Training converges more quickly and accurately when tokens are mapped to an embedding space;
     - Use an embedding layer

4. Models

   - RNN

     - Learn to use nn.RNN
     - Input jsonl data (described as sequences of features)
     - At timestep t (t = 1, 2, ..., s), model predicts the next token (t+1) given tokens 1, 2, ... t

   - LSTM

     - Learn to use nn.LSTM
     - ""
     - ""

   - Transformer

     - Learn to use nn.Transformer
     - ""
     - ""; except, instead of given tokens 1, 2, ..., t, you always consider the window of context size before the token you want to predict; when no tokens exist, buffer with prefix of `<pad>` tokens.

     - Transformer: One or more transformer encoders with 2 or more attention heads between
       embedding and output layer. The maximum input sequence length should be 512.

   - Each model's architecture will include a fully connected output layer that predicts token
     probabilities.

   * Models need only be "based" on these types: experiment with different model architectures.

5. Measurement Criteria

   - See latter half of lecture 6b

   - BLEU Score
     - Use `ntlk`

  - Perplexity (PPL) Score
    - Average cross entropy and raise e to that

6. Train

   - Training Recommendation:

     - Loss Function: CrossEntropyLoss.
     - Optimizer: AdamW
     - Batch Size: Start at 128.
     - Epochs: 30 epochs with early stopping and learning rate scheduler

   - LONI:

     - Train a small sample of data to start. For each model (RNN, LSTM, Transformer), use teacher-forcing method.

       - Verify training occurs for small data set. Training is successful when tokens are generated similar to the training corpus.

     - Learn to use LONI

       - For each model type (RNN, LSTM, Transformer)

         - Make a single batch-job (simple as possible test)
         - Observe the output of a batch-job
         - Terminate a batch-job before it completes

         - Compare batch-job to training result of small sample
         - Make single batch-job satisfactorably efficient

     - Use LONI for each model

7. Inference

   - Each model class (code) must have a forward method that predicts the vocabulary token
     probabilities and samples the next token in the sequence, returning that token ID.
     - WARN: Made RNN first... The model `my-rnn_4-14-25_1108pm_45epochs_vocab10000_hidden512_m-token`
        was trained with poorly interfaced forward method that must accept onehot vectors instead of just token IDs.

   - Grad Students: Your forward method must allow one to specify temperature for sampling (see second half lecture 6b).

   - Each model class (code) must have a prompt method that takes a textual prompt as input,
     tokenizes it, processes it through the model, and returns the model’s reponse:
     a. The response should be autoregressively generated and stop when the model output an
     end of sequence token OR the sequence hits a maximum length (optional argument to the
     function, max_seq_length).

8. Compare & Deliverables

   - Compare the performance of the models, discussing trade-offs in performance and computational requirements.

   Deliverables:

   - You will prepare a short report in 12 point Calibri font (1/2” margins), composed of the following sections:
     Abstract: In one paragraph, summarize your approach and key results.

   - Methodology: 2 to 5 pages describing your approach to designing, training, and evaluating each model.

   - You must include diagrams for the architecture of each model you design, I recommend using draw.io,
     but you are welcome to use other tools.
     Results: For both applications, use matplotlib to generate plots of the loss curves (both training and
     validation).
   - The following should be presented:
     - Training/Validation Loss Curve plots for each of the models
     - Table containing evaluation metrics for each model on the test dataset
     - For each model, show the response for the following prompt: “Which do you prefer? Dogs or
       cats?”
     - Select a prompt of your choosing and show each model’s response to the prompt.
       Code Repo Link: Provide a link to the Github repository containing all of the code for the project. There
       should be a README.md file with instructions on how to run the code (this should not be complicated).
       Discussion & Conclusion: One half page or less discussing the results and what you learned from the
       project.

#### Need to do
9. Transformer still needed

## Lagniappe
