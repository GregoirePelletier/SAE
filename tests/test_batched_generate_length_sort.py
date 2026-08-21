"""Teste src/sae/judge.py::_batched_generate -- vérifie que le tri par
longueur de prompt introduit pour amortir le padding (AUDIT_SAE_2026-08.md,
§2 Performance : "Juge -- pas de tri par longueur avant batching") ne casse
pas le ré-alignement `responses[i] <-> list_of_messages[i]`. Aucun test
existant n'exerçait l'implémentation réelle de cette fonction (le seul test
voisin, test_judge_batching_orchestration.py, la mocke entièrement) --
fake tokenizer/model minimal ici plutôt qu'un MagicMock, pour pouvoir vérifier
que chaque réponse revient bien à l'index d'origine après ré-ordonnancement
interne, indépendamment de la composition des lots."""
import re

import torch

from src.sae.judge import _batched_generate


class _FakeEncoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    """`apply_chat_template` encode juste l'index d'origine dans le texte, avec
    un préfixe de longueur variable pour que le tri par longueur produise un
    ordre différent de l'ordre d'entrée. Le "tokenizer" lit cet index en
    retour plutôt que de tokeniser réellement -- suffisant pour vérifier le
    ré-alignement, pas le contenu généré."""

    def __init__(self):
        self.pad_token_id = 0
        self.padding_side = "right"

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        idx = msgs[0]["content"]  # l'appelant y met l'index d'origine (str)
        pad_len = int(idx) % 5  # longueur de prompt variable et déterministe
        return ("x" * pad_len) + f"##IDX{idx}##"

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        idxs = [int(re.search(r"##IDX(\d+)##", t).group(1)) for t in texts]
        return _FakeEncoding(input_ids=torch.tensor([[i] for i in idxs]),
                              attention_mask=torch.ones(len(idxs), 1, dtype=torch.long))

    def decode(self, token_row, skip_special_tokens=True):
        return f"resp_for_{int(token_row[-1].item())}"


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample):
        # "Génère" en ré-émettant l'index d'entrée comme unique token produit.
        return torch.cat([input_ids, input_ids], dim=1)


def test_batched_generate_realigns_responses_after_length_sort():
    n = 23
    list_of_messages = [[{"role": "user", "content": str(i)}] for i in range(n)]
    responses = _batched_generate(_FakeModel(), _FakeTokenizer(), list_of_messages,
                                   max_new_tokens=1, batch_size=4)
    assert responses == [f"resp_for_{i}" for i in range(n)]


def test_batched_generate_restores_original_padding_side():
    tok = _FakeTokenizer()
    tok.padding_side = "right"
    list_of_messages = [[{"role": "user", "content": str(i)}] for i in range(5)]
    _batched_generate(_FakeModel(), tok, list_of_messages, max_new_tokens=1, batch_size=2)
    assert tok.padding_side == "right"


def test_batched_generate_single_item():
    list_of_messages = [[{"role": "user", "content": "0"}]]
    responses = _batched_generate(_FakeModel(), _FakeTokenizer(), list_of_messages,
                                   max_new_tokens=1, batch_size=16)
    assert responses == ["resp_for_0"]
