from transformers import pipeline

# Load summarization model
model = pipeline("summarization")

def generate_chunks(inp_str, max_chunk=500):
    inp_str = inp_str.replace('.', '.<eos>')
    inp_str = inp_str.replace('?', '?<eos>')
    inp_str = inp_str.replace('!', '!<eos>')

    sentences = inp_str.split('<eos>')
    current_chunk = 0
    chunks = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk].split()) + len(sentence.split()) <= max_chunk:
                chunks[current_chunk] += " " + sentence
            else:
                current_chunk += 1
                chunks.append(sentence)
        else:
            chunks.append(sentence)

    return chunks

# Get user input
sentence = input("Enter the text to summarize:")
max_len = int(input("Select max length [50 - 500]: "))
min_len = int(input("Select min length [10 - 450]: "))

# Generate text chunks
chunks = generate_chunks(sentence)

# Summarize each chunk
res = model(chunks, max_length=max_len, min_length=min_len)

# Combine summaries
summary_text = ' '.join([summ['summary_text'] for summ in res])

print("\nFinal Summary:\n", summary_text)
