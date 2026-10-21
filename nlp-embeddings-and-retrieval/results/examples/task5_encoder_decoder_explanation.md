**When would you choose an encoder model like BERT over a decoder model like GPT-2?**

Choose an encoder model when the task is to understand an entire input at once. BERT reads text bidirectionally, so each token representation can use both left and right context. That makes it a natural fit for classification, sentiment analysis, named entity recognition, sentence similarity, reranking, and embedding/search workflows. In this project, Task 4 sentiment classification is a better match for DistilBERT than GPT-2 because the goal is to assign a label to a complete review, not continue the review.

**Can you use GPT-2 for classification?**

Yes, but it is not the most direct architecture for it. GPT-2 is a decoder-only model trained for next-token prediction, so classification usually requires an adaptation: add a classification head, prompt it to produce a label token, or score candidate label strings and choose the highest-probability one. This can work, but it is less natural than using an encoder because GPT-2 only conditions left-to-right and is optimized for continuation.

**Can you use BERT for generation?**

Not for normal open-ended left-to-right text generation in the same way as GPT-2. BERT is trained with masked language modeling, where it fills in missing tokens using both left and right context. That is useful for understanding and infilling, but it does not define a clean next-token generation process for producing long text one token at a time. Encoder-decoder models such as T5 or BART are better for conditional generation, while decoder-only models like GPT-2 are better for free-form continuation.

**Which architecture did the GPT-3, GPT-4, and Llama family inherit, and why did it win?**

They inherited the decoder-only, autoregressive architecture. It won for large language models because next-token prediction gives a simple and scalable training objective: train on massive text corpora by predicting the next token. The same objective directly supports open-ended generation, chat-style continuation, tool-use text formats, summarization, coding, and many other tasks through prompting. Decoder-only models also scale efficiently with model size and data, which made them the dominant architecture for general-purpose generative LLMs.