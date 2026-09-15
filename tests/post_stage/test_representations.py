"""Tests CPU rapides pour src/post_stage/representations.py (baselines E01,
plan §6.3). Textes synthetiques minimaux, pas de modele/tenseur reel --
embed_bge_m3_documents est teste via un tokenizer/modele FACTICE injecte
(aucun poids HuggingFace charge, interdit sur le noeud frontal)."""
import torch

from src.post_stage.representations import (
    build_tfidf_representation,
    embed_bge_m3_documents,
    TfidfConfig,
)


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    """Compte les "tokens" comme le nombre de mots (espaces) -- suffisant
    pour tester la logique de troncature/batching sans vrai tokenizer."""

    def __call__(self, texts, truncation=None, padding=None, max_length=None, return_tensors=None):
        if truncation is False:
            return {"input_ids": [t.split() for t in texts]}
        # Chemin "encodage reel" (padding=True, truncation=True, max_length=N) :
        # taille de sequence = min(n_mots, max_length), batch pad a la longueur max.
        lengths = [min(len(t.split()), max_length) for t in texts]
        seq_len = max(lengths) if lengths else 1
        return _FakeBatchEncoding(
            input_ids=torch.zeros(len(texts), seq_len, dtype=torch.long),
            attention_mask=torch.ones(len(texts), seq_len, dtype=torch.long),
        )


class _FakeModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeModel:
    def __call__(self, input_ids=None, attention_mask=None):
        batch, seq_len = input_ids.shape
        hidden = 8
        # Deterministe : chaque doc a un vecteur different (base sur son index de batch).
        cls = torch.stack([
            torch.full((hidden,), float(i + 1)) for i in range(batch)
        ])
        return _FakeModelOutput(last_hidden_state=cls.unsqueeze(1).expand(-1, seq_len, -1))


FIT_TEXTS = [
    "le client conteste sa facture d'electricite",
    "demande de mise en service du compteur",
    "coupure reseau signalee par le client",
    "resiliation du contrat energie demandee",
]
DEV_TEXTS = ["le client demande une explication du montant facture"]
CONFIRM_TEXTS = ["mot_unique_confirm_jamais_vu_ailleurs apparait seulement ici"]


def test_vectorizer_fit_only_on_fit_texts():
    vec, X_by_split, config_dict = build_tfidf_representation(
        FIT_TEXTS, {"dev": DEV_TEXTS, "confirm": CONFIRM_TEXTS}
    )
    # Un mot present UNIQUEMENT dans confirm ne doit jamais entrer au vocabulaire.
    assert "mot_unique_confirm_jamais_vu_ailleurs" not in vec.vocabulary_
    assert config_dict["vocabulary_size"] == len(vec.vocabulary_)


def test_all_splits_present_and_correct_shape():
    vec, X_by_split, _ = build_tfidf_representation(
        FIT_TEXTS, {"dev": DEV_TEXTS, "confirm": CONFIRM_TEXTS}
    )
    assert set(X_by_split.keys()) == {"fit", "dev", "confirm"}
    n_vocab = len(vec.vocabulary_)
    assert X_by_split["fit"].shape == (len(FIT_TEXTS), n_vocab)
    assert X_by_split["dev"].shape == (len(DEV_TEXTS), n_vocab)
    assert X_by_split["confirm"].shape == (len(CONFIRM_TEXTS), n_vocab)


def test_config_dict_reflects_all_explicit_parameters():
    config = TfidfConfig(max_features=500, ngram_range=(1, 2), min_df=2)
    _, _, config_dict = build_tfidf_representation(FIT_TEXTS, {}, config=config)
    assert config_dict["max_features"] == 500
    assert config_dict["ngram_range"] == [1, 2]
    assert config_dict["min_df"] == 2
    assert config_dict["sublinear_tf"] is True  # defaut du plan, meme avec d'autres params surcharges


def test_default_config_matches_existing_probe_script_convention():
    # scripts/intent_urgency_probe.py : TfidfVectorizer(max_features=20000, sublinear_tf=True)
    config = TfidfConfig()
    assert config.max_features == 20000
    assert config.sublinear_tf is True


def test_empty_eval_splits_returns_only_fit():
    vec, X_by_split, _ = build_tfidf_representation(FIT_TEXTS, {})
    assert list(X_by_split.keys()) == ["fit"]


def test_embed_bge_m3_documents_returns_normalized_embeddings():
    texts = ["mot1 mot2 mot3", "un seul mot ici encore un peu plus"]
    embs, config_dict = embed_bge_m3_documents(
        texts, max_length=100, tokenizer=_FakeTokenizer(), model=_FakeModel(),
    )
    assert embs.shape == (2, 8)
    norms = embs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)
    assert config_dict["n_docs"] == 2
    assert config_dict["pooling"] == "cls_normalized"


def test_embed_bge_m3_documents_reports_truncation_rate():
    # 3 documents : 2 dépassent max_length=3 mots, 1 non.
    texts = ["un deux trois quatre cinq", "a b c d", "juste trois mots"]
    _, config_dict = embed_bge_m3_documents(
        texts, max_length=3, tokenizer=_FakeTokenizer(), model=_FakeModel(),
    )
    assert config_dict["n_truncated"] == 2
    assert abs(config_dict["truncation_rate"] - 2 / 3) < 1e-9


def test_embed_bge_m3_documents_zero_truncation_when_all_texts_short():
    texts = ["court", "aussi court"]
    _, config_dict = embed_bge_m3_documents(
        texts, max_length=2048, tokenizer=_FakeTokenizer(), model=_FakeModel(),
    )
    assert config_dict["n_truncated"] == 0
    assert config_dict["truncation_rate"] == 0.0


def test_embed_bge_m3_documents_empty_input():
    embs, config_dict = embed_bge_m3_documents(
        [], tokenizer=_FakeTokenizer(), model=_FakeModel(),
    )
    assert embs.shape == (0,)
    assert config_dict["n_docs"] == 0
    assert config_dict["truncation_rate"] == 0.0
