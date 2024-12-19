from transformers import pipeline, BertTokenizer, BertModel
import wikipedia
import torch
from scipy.spatial.distance import cosine
from concurrent.futures import ThreadPoolExecutor
import warnings
from bs4 import BeautifulSoup
from llama_cpp import Llama
import spacy
import re

# Ignore specific warnings from BeautifulSoup
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

# Wikipedia request header setup
user_agent = "WDP_lecture_demo/1.0 (s.wang15@student.vu.nl)"
wikipedia.headers = {'User-Agent': user_agent}

# Load NER model
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple"
)

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Global thread pool
global_thread_pool = ThreadPoolExecutor(max_workers=10)

# Caching dictionaries
embedding_cache = {}
page_cache = {}
search_cache = {}


# 使用zero-shot分类模型
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# spaCy加载
nlp = spacy.load("en_core_web_sm")

# 判断问题是否为yes/no类型
def classify_question(question):
    yes_no_keywords = ["is", "are", "do", "does", "can", "will", "has", "have"]
    # 判断问题是否包含yes/no类型的关键词
    question = question.strip().lower()
    
    # 检查问题开头是否包含这些助动词
    if any(question.startswith(keyword) for keyword in yes_no_keywords):
        return "yes/no"
    else:
        return "entity"  # 其他的都视为实体类型

def classify_question_type2(question):
    # 使用模型判断问题属于哪个标签
    labels = ["yes/no", "entity"]
    result = zero_shot_classifier(question, candidate_labels=labels)
    
    # 输出预测的标签，选择得分最高的标签作为类型
    predicted_label = result['labels'][0]
    return predicted_label




# 使用zero-shot分类模型判断是yes/no
def classify_answer_as_yes_no(answer):
    result = zero_shot_classifier(answer, candidate_labels=["yes", "no"])
    if result['labels'][0] == "yes":
        return "yes"
    elif result['labels'][0] == "no":
        return "no"
    return None  # 如果没有明确的yes/no

# 判断回答类型的函数
def classify_answer_type(answer_text):
    # 定义候选标签：yes/no
    candidate_labels = ["yes", "no", "entity"]

    # 使用 zero-shot 分类进行推理，判断回答是 yes/no 还是包含实体
    result = zero_shot_classifier(answer_text, candidate_labels)

    # 获取最匹配的标签
    predicted_label = result['labels'][0]

    return predicted_label




# 综合判断问题和回答的类型
def get_answer_type_based_on_question_and_answer(question, answer):
    question_type = classify_question(question)
    if  question_type == "entity":
        question_type = classify_question_type2(question)

    # 如果问题是yes/no类型，则直接从回答中判断yes/no
    if question_type == "yes/no":
        answer_yes_no = classify_answer_as_yes_no(answer)
        if answer_yes_no:
            return answer_yes_no  # 如果是yes/no，直接返回

    # 如果问题不是yes/no类型，在对answer继续分类
    answer_type = classify_answer_type(answer)

    if  answer_type == "entity":
        return extract_entities_with_wiki_multithread(answer)
    else:
         return answer_type

# Compute sentence embedding using BERT
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    return torch.mean(embeddings, dim=1).squeeze()


# Get or cache the sentence embedding
def get_cached_sentence_embedding(sentence):
    if sentence in embedding_cache:
        return embedding_cache[sentence]
    embedding = get_sentence_embedding(sentence)
    embedding_cache[sentence] = embedding
    return embedding


# Get or cache Wikipedia page
def get_cached_page(title):
    if title in page_cache:
        return page_cache[title]
    try:
        page = wikipedia.page(title, auto_suggest=True)
        page_cache[title] = page
        return page
    except (wikipedia.PageError, wikipedia.DisambiguationError):
        return None


# Wikipedia search and cache results
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


# Logic to process a single candidate page
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


# Global multi-thread processing for multiple entities
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


# Specify the path to the text file
file_path = 'example_input.txt'

# Open the file and read its content
with open(file_path, 'r', encoding='utf-8') as file:
    # Read lines, strip newline characters, and remove the "question-XXX" prefixes
    questions_array = [re.sub(r'^question-\d+\s*', '', line.strip()) for line in file]

filtered_questions_array = [item for item in questions_array if item]

# Modify your model path here
llm_model = Llama("models/llama-2-7b.Q4_K_M.gguf", verbose=False)
with open("example_output.txt", "w") as file:
    count = 1
    for question in filtered_questions_array:

        output = llm_model(
            question,
            max_tokens=30
        )
        llm_return_text = output['choices'][0]['text']
        file.write(f'question-00{count}    R"{llm_return_text}"\n')

        # task 2
        # yes_or_no = get_answer_type_based_on_question_and_answer(question, llm_return_text)
        # if yes_or_no == 'yes' or yes_or_no == 'no':
        #     file.write(f'question-00{count}    A"{yes_or_no}"\n')
        # # if the yes_or_no list is not empty, return the first element
        # elif len(yes_or_no) > 0:
        #     file.write(f'question-00{count}    A"{list(yes_or_no.values())[0]}"\n')
        # else:
        #     file.write(f'question-00{count}    A"Cannot extract entities from the answer!"\n')

        # Extract entities and print links
        entities_with_links = extract_entities_with_wiki_multithread(question + llm_return_text)
        for entity, url in entities_with_links.items():
            file.write(f'question-00{count}    E"{entity}"    "{url}"\n')
        count+=1
