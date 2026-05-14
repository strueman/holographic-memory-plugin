# Cygnet — Multi-Language WordNet for Query Expansion

> Cygnet provides access to WordNet in multiple languages. Useful for future multi-language query expansion.
> Date: 2026-05-10

## What It Is

Cygnet is a collection of WordNets from many languages, providing a unified interface for multi-language synonym lookup.

## Languages Available

English, German, French, Spanish, Portuguese, Italian, Dutch, Polish, Russian, Chinese, Japanese, Korean, and more (varies by implementation).

## Use Case

For query expansion in non-English queries:
```python
# Instead of NLTK WordNet (English only)
from cygnet import wordnet  # hypothetical API

# Get synonyms in any supported language
syns = wordnet.synsets("schnell", lang="de")  # German: fast
syns = wordnet.synsets("rapide", lang="fr")   # French: fast
```

## Tradeoffs vs NLTK WordNet

| Factor | NLTK WordNet | Cygnet |
|--------|-------------|--------|
| Languages | English only | Many |
| Size | ~15MB | Larger (varies) |
| Coverage per language | Dense (English) | Sparse (other langs) |
| API | Standard wordnet API | Varies by implementation |
| Dependencies | nltk | cygnet + language data |

## Recommendation

- **Now (English only):** Use NLTK WordNet for query expansion Phase 1
- **Future (multi-language):** Switch to Cygnet when non-English queries appear
- **Hybrid:** Cygnet for common words + custom dict for domain terms, regardless of language

## References

- NLTK WordNet: https://www.nltk.org/howto/wordnet.html
- Cygnet project: https://github.com/ (check for active multi-language WordNet project)
