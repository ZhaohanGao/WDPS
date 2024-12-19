from transformers import pipeline, BertTokenizer, BertModel
import wikipedia
import torch
from scipy.spatial.distance import cosine
from concurrent.futures import ThreadPoolExecutor
import warnings
from bs4 import BeautifulSoup
from llama_cpp import Llama
import re

# Ignore specific warnings from BeautifulSoup
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

# Wikipedia request header setup
user_agent = "WDP_lecture_demo/1.0 (s.wang15@student.vu.nl)"
wikipedia.headers = {'User-Agent': user_agent}

# Load NER model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Zero-shot classification model
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Global thread pool for multi-threaded processing
global_thread_pool = ThreadPoolExecutor(max_workers=10)

# Caching dictionaries
embedding_cache = {}
page_cache = {}
search_cache = {}


# Unified classification for questions and answers
def classify_question(question):
    """
    Determine the type of question based on its content: yes/no or entity
    """
    # Remove the prefix 'Question: ' or other unnecessary markers
    question_cleaned = re.sub(r'^\s*Question:\s*', '', question, flags=re.IGNORECASE).strip()

    question_lower = question_cleaned.lower()

    # Prioritize cases starting with 'Who', 'What', 'Where', 'When', 'Why', 'How'
    open_question_keywords = ["who", "what", "where", "when", "why", "how"]
    if any(question_lower.startswith(keyword) for keyword in open_question_keywords):
        return "entity"

    # Directly determine if it is a yes/no type based on keywords
    yes_no_keywords = ["is", "are", "do", "does", "can", "will", "has", "have", "is it true that"]
    if any(question_lower.startswith(keyword) for keyword in yes_no_keywords):
        labels = ["yes", "no"]
        result = zero_shot_classifier(f"Question: {question_cleaned}", candidate_labels=labels)
        return result["labels"][0]

    # If no rule matches, call zero-shot classification for question classification
    labels = ["yes", "no", "entity"]
    result = zero_shot_classifier(f"Question: {question_cleaned}", candidate_labels=labels)
    # return "yes/no" if result["labels"][0] in ["yes", "no"] else "entity"
    return result["labels"][0]


# Compute sentence embedding using BERT
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    return torch.mean(embeddings, dim=1).squeeze()


# Cache and reuse sentence embeddings
def get_cached_sentence_embedding(sentence):
    if sentence in embedding_cache:
        return embedding_cache[sentence]
    embedding = get_sentence_embedding(sentence)
    embedding_cache[sentence] = embedding
    return embedding


# Cache and reuse Wikipedia page data
def get_cached_page(title):
    if title in page_cache:
        return page_cache[title]
    try:
        page = wikipedia.page(title, auto_suggest=False)
        page_cache[title] = page
        return page
    except (wikipedia.PageError, wikipedia.DisambiguationError):
        return None


# Perform Wikipedia search with caching
def cached_search(entity):
    if entity in search_cache:
        return search_cache[entity]
    results = wikipedia.search(entity)
    search_cache[entity] = results
    return results


# Calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    embedding1_np = embedding1.numpy()
    embedding2_np = embedding2.numpy()
    return 1 - cosine(embedding1_np, embedding2_np)


# Process a single Wikipedia candidate page
def process_candidate(option, context_embedding):
    try:
        candidate_page = get_cached_page(option)
        if not candidate_page:
            return None, 0

        candidate_title_embedding = get_cached_sentence_embedding(candidate_page.title)
        candidate_summary_embedding = get_cached_sentence_embedding(candidate_page.summary[:500])
        title_similarity = cosine_similarity(context_embedding, candidate_title_embedding) * 0.3
        summary_similarity = cosine_similarity(context_embedding, candidate_summary_embedding) * 0.7
        similarity_score = title_similarity + summary_similarity

        return candidate_page, similarity_score
    except Exception as e:
        print(f"Error processing candidate '{option}': {e}")
        return None, 0


# Get Wikipedia link for an entity
def get_wikipedia_link(entity, context_sentence, context_embedding=None, threshold=0.5):
    try:
        if context_embedding is None:
            context_embedding = get_cached_sentence_embedding(context_sentence)

        # Try to directly get the page
        try:
            page = wikipedia.page(entity, auto_suggest=False)
            page_cache[entity] = page
        except wikipedia.PageError:
            search_results = cached_search(entity)
            if search_results:
                page = get_cached_page(search_results[0])
            else:
                return None
        except wikipedia.DisambiguationError as e:
            disambiguation_pages = e.options
            best_match, best_similarity = None, 0

            # Multi-thread processing for disambiguation candidate pages
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = executor.map(
                    lambda option: process_candidate(option, context_embedding),
                    disambiguation_pages
                )
                for candidate_page, similarity_score in results:
                    if candidate_page and similarity_score > best_similarity:
                        best_match = candidate_page
                        best_similarity = similarity_score

            return best_match.url if best_match and best_similarity > threshold else None

        # BeautifulSoup check for top-level hatnote
        page_html = page.html()
        soup = BeautifulSoup(page_html, "html.parser")
        hatnote = soup.find("div", {"class": "hatnote"})
        best_hatnote_match, best_hatnote_similarity = None, 0

        if hatnote:
            # Extract each link's text and URL
            links_in_hatnote = [(a.get_text(), a['href']) for a in hatnote.find_all("a", href=True)]

            # Multi-thread processing for hatnote links
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = executor.map(
                    lambda link: process_candidate(link[0], context_embedding),
                    links_in_hatnote
                )
                for candidate_page, similarity_score in results:
                    if candidate_page and similarity_score > best_hatnote_similarity:
                        best_hatnote_match = candidate_page
                        best_hatnote_similarity = similarity_score

        # Check similarity of the current page
        title_embedding = get_cached_sentence_embedding(page.title)
        summary_embedding = get_cached_sentence_embedding(page.summary[:500])
        title_similarity = cosine_similarity(context_embedding, title_embedding) * 0.3
        summary_similarity = cosine_similarity(context_embedding, summary_embedding) * 0.7
        original_page_similarity = title_similarity + summary_similarity

        # Compare hatnote and original page similarity
        if best_hatnote_match and best_hatnote_similarity > original_page_similarity:
            return best_hatnote_match.url

        # If no better hatnote page, return original page
        if original_page_similarity > threshold:
            return page.url

        return None

    except Exception as e:
        print(f"Error processing entity '{entity}': {e}")
        return None


# Extract Wikipedia link for a single entity
def process_entity(entity, context_sentence, context_embedding):
    try:
        wiki_url = get_wikipedia_link(entity, context_sentence, context_embedding)
        return entity, wiki_url
    except Exception as e:
        print(f"Error processing entity '{entity}': {e}")
        return entity, None


# Multi-threaded entity extraction and Wikipedia linking
def extract_entities_with_wiki_multithread(text):
    entities = ner_pipeline(text)
    context_embedding = get_cached_sentence_embedding(text)

    futures = [
        global_thread_pool.submit(process_entity, ent["word"], text, context_embedding)
        for ent in entities
    ]

    result = {}
    for future in futures:
        entity, wiki_url = future.result()
        if wiki_url:
            result[entity] = wiki_url

    return result


def verify_yes_no_with_question_context(question, yes_no_answer, entities_with_links):
    """
    Verify the yes/no answer based on the question and cached Wikipedia page data
    """
    try:
        for entity, url in entities_with_links.items():
            if not url:
                continue  # Skip invalid links

            # Retrieve page content (from the cache)
            cached_page = get_cached_page(entity)
            if not cached_page:
                continue

            page_content = cached_page.content[:300]  # Limit to the first 2000 characters
            if not page_content:
                continue

            # Combine the question with the page content as context
            combined_input = f"Question: {question} Context: {page_content}"

            # Use zero-shot classification to check the match between the question and page content
            validation_result = zero_shot_classifier(
                combined_input,
                candidate_labels=["yes", "no", "neutral"]
            )

            # Take the classification result with the highest confidence
            top_label = validation_result["labels"][0]

            # Handle logic based on different answers
            if yes_no_answer == "no":
                # If the page clearly supports 'yes,' directly return 'incorrect'
                if top_label == "yes":
                    return "incorrect"
                # If the page clearly supports 'no,' continue checking the next page
                elif top_label == "no":
                    continue
                # If the page is neutral, ignore this page
                elif top_label == "neutral":
                    continue

            elif yes_no_answer == "yes":
                # If the page clearly supports 'yes,' return 'correct'
                if top_label == "yes":
                    return "correct"
                # If the page clearly supports 'no,' return 'incorrect'
                elif top_label == "no":
                    return "incorrect"
                # If the page is neutral, ignore this page
                elif top_label == "neutral":
                    continue

        # If no definitive supporting information is found, return 'unknown'
        return "unknown"

    except Exception as e:
        print(f"Error during verification: {e}")
        return "unknown"


# Specify the path to the text file
file_path = 'example_input.txt'

# Read input file and process questions
with open(file_path, 'r', encoding='utf-8') as file:
    # Read lines, strip newline characters, and remove the "question-XXX" prefixes
    questions_array = [re.sub(r'^question-\d+\s*', '', line.strip()) for line in file]

filtered_questions_array = [item for item in questions_array if item]

# Modify your model path here
llm_model = Llama("models/llama-2-7b.Q4_K_M.gguf", verbose=False)
with open("example_output.txt", "w") as file:
    count = 1
    for question in filtered_questions_array:
        # 1. Classify the question
        classification = classify_question(question)

        # 2. Use Llama model to generate the answer
        output = llm_model(
            question,
            max_tokens=30
        )
        llm_return_text = output['choices'][0]['text']
        clean_llm_return_text = llm_return_text.replace("\n", " ").strip()
        file.write(f'question-00{count}    R"{clean_llm_return_text}"\n')

        # 3. Handle yes/no and entity classification
        if classification in ["yes", "no"]:
            yes_no_answer = classification
            file.write(f'question-00{count}    A"{classification}"\n')

        # 4. Extract entities and cache Wikipedia links
        entities_with_links = extract_entities_with_wiki_multithread(question + clean_llm_return_text)

        # 5. If it's a yes/no question, verify the answer
        if classification in ["yes", "no"]:
            # Directly use cached entities for verification
            verification_result = verify_yes_no_with_question_context(
                question, classification, entities_with_links
            )
            file.write(f'question-00{count}    C"{verification_result}"\n')

        for entity, url in entities_with_links.items():
            file.write(f'question-00{count}    E"{entity}"    "{url}"\n')

        count += 1
