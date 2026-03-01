import re
from difflib import SequenceMatcher

def remove_duplicate_lines(text: str) -> str:
    """Hapus baris duplikat persis."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    seen = set()
    result = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            result.append(line)
    return ' '.join(result)

def remove_near_duplicates(text: str, threshold: float = 0.85) -> str:
    """Hapus baris yang hampir sama (untuk slide 2 kolom)."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    result = []
    for line in lines:
        is_dup = False
        for kept in result:
            ratio = SequenceMatcher(None, line.lower(), kept.lower()).ratio()
            if ratio > threshold:
                is_dup = True
                break
        if not is_dup:
            result.append(line)
    return ' '.join(result)

def fix_broken_words(text: str) -> str:
    """Fix spasi yang hilang antar kata."""
    # Fix camelCase yang terjadi karena text box nempel
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Fix angka nempel huruf
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    return text

def clean_slide_text(text: str) -> str:
    """Pipeline cleaning untuk slide deck."""
    text = remove_duplicate_lines(text)
    text = remove_near_duplicates(text)
    text = fix_broken_words(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_document_text(text: str) -> str:
    """Pipeline cleaning untuk dokumen biasa."""
    text = remove_duplicate_lines(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{3,}', '', text)
    return text.strip()
