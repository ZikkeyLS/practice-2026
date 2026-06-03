import os
import fitz  # PyMuPDF
import re
import json
import subprocess
from typing import Dict, List, Any


def build_metadata_prompt(text: str) -> str:
    """
    Prompt для извлечения title и author через Qwen2.5
    """

    text = text[:6000]

    return f"""
Ты анализируешь текст научного PDF документа.

Нужно определить:
1. title — название документа
2. author — автора или авторов

Правила:
- title обычно находится в начале документа
- не включай лишний текст
- author — находится прямо в тексте, только фамилия и инициалы
- если данных нет, верни из текста что-то наиболее похожее на author
- ответ ТОЛЬКО в JSON
- без пояснений
- без markdown

Формат ответа:

{{
  "title": "",
  "author": ""
}}

Текст документа:

{text}
"""


def clean_ansi_escapes(text: str) -> str:
    """Удаляет ANSI escape последовательности"""

    ansi_escape = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
    return ansi_escape.sub('', text)


def extract_metadata_with_qwen(text: str) -> Dict[str, str]:
    """
    Извлечение title/author через локальную модель Qwen
    """

    prompt = build_metadata_prompt(text)

    try:
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:7b"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )

        response = result.stdout.strip()
        response = clean_ansi_escapes(response)

        match = re.search(r'\{.*\}', response, re.DOTALL)

        if match:
            json_str = match.group()
            json_str = re.sub(
                r'[\x00-\x1f\x7f-\x9f]',
                '',
                json_str
            )

            parsed = json.loads(json_str)
            title = re.sub(
                r'[\x00-\x1f\x7f-\x9f]',
                '',
                parsed.get("title", "")
            )
            author = re.sub(
                r'[\x00-\x1f\x7f-\x9f]',
                '',
                parsed.get("author", "")
            )

            return {
                "title": title.strip(),
                "author": author.strip()
            }
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
    except Exception as e:
        print(f"Ошибка Qwen metadata extraction: {e}")

    return {
        "title": "",
        "author": ""
    }


def is_good_context(context: str) -> bool:
    """
    Проверка на хороший контекст для формулы/предложения
    """

    context_lower = context.lower()

    bad_words = [
        "список литературы",
        "литература",
        "references",
        "bibliography",
        "оглавление",
        "содержание"
    ]
    
    math_words = [
        "неравен",
        "оцен",
        "границ",
        "верхн",
        "нижн",
        "теорем",
        "лемм",
        "доказ",
        "следует",
        "получаем",
        "констант",
        "функц"
    ]

    for bad_word in bad_words:
        if bad_word in context_lower:
            return False

    if len(context.strip()) < 80:
        return False

    math_signs = [
        "≤", "≥", "<=", ">=", "<", ">", "="
    ]

    signs_count = 0
    for sign in math_signs:
        if sign in context:
            signs_count += 1

    words_count = 0
    for word in math_words:
        if word in context_lower:
            words_count += 1

    if words_count >= 1 and signs_count >= 1:
        return True

    return False


def extract_text_with_positions(pdf_path: str) -> Dict[str, Any]:
    """
    Извлечение текста из PDF
    """

    result = {
        "full_text": "",
        "full_text_raw": "",
        "pages": [],
        "metadata": {}
    }

    try:
        doc = fitz.open(pdf_path)

        result["metadata"] = {
            "file": os.path.basename(pdf_path),
            "pages": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", "")
        }

        full_text_parts = []
        full_text_raw_parts = []

        global_pos = 0
        first_pages_text = ""

        for page_num, page in enumerate(doc, 1):
            raw_text = page.get_text()

            if not raw_text:
                raw_text = ""

            if page_num <= 2:
                first_pages_text += raw_text + "\n"

            page_data = {
                "page": page_num,
                "start_pos": global_pos,
                "raw_text": raw_text
            }

            cleaned_chars = []

            for char in raw_text:
                if char == '\n':
                    cleaned_chars.append('\n')
                    global_pos += 1

                elif ord(char) < 32 and char not in '\n\r\t':
                    continue

                else:
                    cleaned_chars.append(char)
                    global_pos += 1

            cleaned_text = ''.join(cleaned_chars)

            # Разделение слипшихся слов
            cleaned_text = re.sub(
                r'([а-яА-Яa-zA-Z])([А-ЯA-Z])',
                r'\1 \2',
                cleaned_text
            )
            cleaned_text = re.sub(
                r'([а-яА-Яa-zA-Z])([0-9])',
                r'\1 \2',
                cleaned_text
            )
            cleaned_text = re.sub(
                r'([0-9])([а-яА-Яa-zA-Z])',
                r'\1 \2',
                cleaned_text
            )
            cleaned_text = re.sub(
                r'([.,:;!?])([а-яА-Яa-zA-Z0-9])',
                r'\1 \2',
                cleaned_text
            )
            cleaned_text = re.sub(
                r'[ \t]+',
                ' ',
                cleaned_text
            )

            cleaned_text = cleaned_text.strip()

            page_data["cleaned_text"] = cleaned_text
            page_data["end_pos"] = global_pos

            result["pages"].append(page_data)

            full_text_parts.append(cleaned_text)
            full_text_raw_parts.append(raw_text)

        result["full_text"] = '\n'.join(full_text_parts)
        result["full_text_raw"] = ''.join(full_text_raw_parts)

        # AI metadata extraction
        ai_metadata = extract_metadata_with_qwen(
            first_pages_text
        )

        result["metadata"]["title"] = ai_metadata["title"]
        result["metadata"]["author"] = ai_metadata["author"]

        doc.close()

    except Exception as e:
        print(f"Ошибка: {e}")
        result["error"] = str(e)

    return result


def extract_sentences_with_keywords(
    text_data: Dict[str, Any],
    keywords: List[str]
) -> List[Dict]:
    """
    Поиск предложений с ключевыми словами + контекст
    """

    results = []
    full_text = text_data["full_text"]

    # Убираем лишние переносы, чтобы предложения не рвались
    text = re.sub(
        r'\s+',
        ' ',
        full_text
    )

    # Разбиваем на предложения
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    current_pos = 0

    for i, sentence in enumerate(sentences):

        sentence_clean = sentence.strip()

        if len(sentence_clean) < 20:
            current_pos += len(sentence) + 1
            continue

        sentence_lower = sentence_clean.lower()

        found_keywords = [
            kw for kw in keywords
            if kw.lower() in sentence_lower
        ]

        if found_keywords:
            context_parts = []

            if i > 0:
                context_parts.append(sentences[i - 1].strip())

            context_parts.append(sentence_clean)

            if i < len(sentences) - 1:
                context_parts.append(sentences[i + 1].strip())

            context = ' '.join(context_parts).strip()

            if not is_good_context(context):
                current_pos += len(sentence) + 1
                continue

            start_pos = full_text.find(
                sentence_clean
            )

            if start_pos == -1:
                start_pos = current_pos

            end_pos = start_pos + len(sentence_clean)

            page = 1

            for page_info in text_data["pages"]:
                if start_pos >= page_info["start_pos"]:
                    page = page_info["page"]
                else:
                    break

            results.append({
                "text": context,
                "page": page,
                "start_position": start_pos,
                "end_position": end_pos,
                "length": len(context),
                "matched_keywords": found_keywords
            })

        current_pos += len(sentence) + 1

    return results


def process_files(
    input_dir: str,
    output_dir: str,
    keywords: List[str]
):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(
            input_dir,
            filename
        )

        print(f"\nОбработка: {filename}")

        text_data = extract_text_with_positions(
            pdf_path
        )

        if "error" in text_data:
            print(f"Ошибка: {text_data['error']}")
            continue

        print(
            f"Страниц: "
            f"{text_data['metadata']['pages']}"
            f"\n"
            f"Название: "
            f"{text_data['metadata']['title']}"
            f"\n"
            f"Автор: "
            f"{text_data['metadata']['author']}"
            f"\n"
            f"Длина текста: "
            f"{len(text_data['full_text'])}"
        )

        sentences = extract_sentences_with_keywords(
            text_data,
            keywords
        )

        output_file = os.path.join(
            output_dir,
            filename.replace('.pdf', '.json')
        )

        output_data = {
            "source_file": filename,
            "metadata": text_data["metadata"],
            "extraction_info": {
                "keywords_used": keywords,
                "total_sentences_found": len(sentences),
                "total_text_length": len(text_data["full_text"])
            },
            "sentences": sentences
        }

        with open(
            output_file,
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(
                output_data,
                f,
                ensure_ascii=False,
                indent=2
            )

        txt_file = os.path.join(
            output_dir,
            filename.replace('.pdf', '.txt')
        )

        with open(
            txt_file,
            'w',
            encoding='utf-8'
        ) as f:
            f.write(f"Файл: {filename}\n")
            f.write(
                f"Название: "
                f"{text_data['metadata']['title']}\n"
                f"Автор: "
                f"{text_data['metadata']['author']}\n"
                f"Страниц: "
                f"{text_data['metadata']['pages']}\n"
                f"Общая длина текста: "
                f"{len(text_data['full_text'])}\n"
            )
            f.write("=" * 70 + "\n\n")

            for i, sent in enumerate(sentences, 1):
                f.write(
                    f"{i}. "
                    f"[Страница {sent['page']}, "
                    f"позиции "
                    f"{sent['start_position']}-"
                    f"{sent['end_position']}]\n"
                )
                f.write(f"{sent['text']}\n")
                f.write(
                    f"Ключевые слова: "
                    f"{', '.join(sent['matched_keywords'])}\n"
                )
                f.write(
                    f"Длина: "
                    f"{sent['length']} символов\n\n"
                )

        print(f"Сохранено: {output_file}")


if __name__ == "__main__":
    input_dir = "input"
    output_dir = "output"
    keywords = [
        "неравенств",
        "неравенства",
        "неравенство",
        "оценка",
        "оценки",
        "оценку",
        "оценивается",
        "верхняя оценка",
        "нижняя оценка",
        "граница",
        "верхняя граница",
        "нижняя граница",
        "≤",
        "≥",
        "<=",
        ">="
    ]

    process_files(
        input_dir,
        output_dir,
        keywords
    )

